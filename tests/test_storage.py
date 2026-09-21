from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from src.jev.models import (
    JevCallResult,
    JevCallStatus,
    JevSafetyAction,
    ParsedJevDecision,
    QuestionMode,
)
from src.storage.database import Database, async_database_url, postgres_requires_ssl
from src.storage.hashing import market_state_hash
from src.storage.models import JevDecisionRow, MarketSnapshotRow, SystemEventRow
from src.storage.repository import Repository
from tests.helpers import make_market_state
from tests.test_jev_parse import ENTRY_PAYLOAD


def _ok_result(symbol: str = "ETHUSDT") -> JevCallResult:
    decision = ParsedJevDecision(
        mode=QuestionMode.ENTRY,
        question_set_version="q.v1.h300",
        model="typesafe/jev-1.13-20260917",
        request_id="gen-dec-test",
        latency_ms=250.0,
        long_probability=0.2,
        short_probability=0.3,
        evidence_probability=0.1,
        exposure_choice="FLAT",
        exposure_choice_probabilities={"LONG": 0.2, "FLAT": 0.7, "SHORT": 0.1},
    )
    return JevCallResult(
        status=JevCallStatus.OK,
        symbol=symbol,
        exchange="binance_usdm",
        mode=QuestionMode.ENTRY,
        question_set_version="q.v1.h300",
        latency_ms=250.0,
        decision=decision,
        raw_response=ENTRY_PAYLOAD,
        safety_action=JevSafetyAction.NONE,
    )


def test_async_url_normalization() -> None:
    assert async_database_url("sqlite:///./jev_trader.db").startswith("sqlite+aiosqlite://")
    assert async_database_url("postgresql://u:p@localhost/db").startswith("postgresql+asyncpg://")
    railway = async_database_url(
        "postgres://u:p@maglev.proxy.rlwy.net:1234/railway?sslmode=require&channel_binding=require"
    )
    assert railway.startswith("postgresql+asyncpg://")
    assert "sslmode" not in railway
    assert "channel_binding" not in railway
    assert postgres_requires_ssl(
        "postgresql://u:p@maglev.proxy.rlwy.net:1234/railway?sslmode=require"
    )
    assert not postgres_requires_ssl("postgresql://jev:jev@postgres.railway.internal:5432/railway")


@pytest.mark.asyncio
async def test_persist_and_read_back(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'research.db'}")
    await db.create_all()
    repo = Repository(db)
    state = make_market_state(symbol="ETHUSDT", price=3501.25)
    decision_id = await repo.persist_tick(state, _ok_result())
    await repo.record_system_event(event_type="TEST", symbol="ETHUSDT", exchange="binance_usdm")

    async with db.session() as session:
        snap = (await session.execute(select(MarketSnapshotRow).limit(1))).scalar_one()
        decision = await session.get(JevDecisionRow, decision_id)
        events = (await session.execute(select(SystemEventRow))).scalars().all()
    assert snap.symbol == "ETHUSDT"
    assert snap.exchange == "binance_usdm"
    assert snap.price == 3501.25
    assert decision is not None
    assert decision.raw_jev_response is not None
    assert decision.raw_jev_response["answers"]["desired_exposure"]["choice"] == "FLAT"
    assert decision.parsed_probabilities is not None
    assert decision.parsed_probabilities["exposure_choice"] == "FLAT"
    assert decision.final_action == "NO_ACTION"
    assert decision.market_state_hash == market_state_hash(state.canonical_dict())
    assert any(event.event_type == "TEST" for event in events)
    await db.aclose()


@pytest.mark.asyncio
async def test_wipe_all_operator_data_clears_every_table(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'wipe.db'}")
    await db.create_all()
    repo = Repository(db)
    state = make_market_state(symbol="ETHUSDT", price=3501.25)
    await repo.persist_tick(state, _ok_result())
    await repo.record_system_event(event_type="TEST", symbol="ETHUSDT", exchange="binance_usdm")
    counts = await repo.wipe_all_operator_data()
    assert counts["market_snapshots"] >= 1
    assert counts["jev_decisions"] >= 1
    assert counts["system_events"] >= 1
    assert await repo.list_recent_events(20) == []
    assert await repo.list_recent_trades(20) == []
    async with db.session() as session:
        assert (await session.execute(select(MarketSnapshotRow))).scalars().first() is None
        assert (await session.execute(select(JevDecisionRow))).scalars().first() is None
        assert (await session.execute(select(SystemEventRow))).scalars().first() is None
    await db.aclose()


@pytest.mark.asyncio
async def test_outcome_enrichment_uses_later_snapshots(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'outcomes.db'}")
    await db.create_all()
    repo = Repository(db)
    t0 = 1_700_000_000_000
    state = make_market_state(symbol="SOLUSDT", price=150.0).model_copy(
        update={"timestamp_ms": t0, "timestamp": datetime.now(UTC).isoformat()}
    )
    await repo.persist_tick(state, _ok_result("SOLUSDT"))

    later = make_market_state(symbol="SOLUSDT", price=151.5).model_copy(
        update={"timestamp_ms": t0 + 12_000, "timestamp": datetime.now(UTC).isoformat()}
    )
    await repo.persist_tick(later, _ok_result("SOLUSDT"))

    updated = await repo.enrich_outcomes(now_ms=t0 + 15_000)
    assert updated >= 1
    async with db.session() as session:
        first = (
            await session.execute(select(JevDecisionRow).order_by(JevDecisionRow.timestamp_ms))
        ).scalars().first()
    assert first is not None
    assert first.price_after_10s == 151.5
    assert first.return_after_10s == pytest.approx(151.5 / 150.0 - 1.0)
    assert first.price_after_30s is None
    await db.aclose()


POSTGRES_SMOKE_URL = "postgresql://jev:jev@127.0.0.1:5433/jev_trader_test"


@pytest.mark.asyncio
async def test_postgres_persist_jsonb_and_bigint_timestamps() -> None:
    db = Database(POSTGRES_SMOKE_URL)
    try:
        await db.drop_all()
        await db.create_all()
    except Exception as exc:
        await db.aclose()
        pytest.skip(f"Postgres in OrbStack is not reachable: {exc}")
    repo = Repository(db)
    far_future_ms = 1_790_003_765_175
    state = make_market_state(symbol="BTCUSDT", price=86_133.15).model_copy(
        update={
            "timestamp_ms": far_future_ms,
            "timestamp": datetime.fromtimestamp(far_future_ms / 1000, tz=UTC).isoformat(),
        }
    )
    decision_id = await repo.persist_tick(state, _ok_result("BTCUSDT"))
    later_ms = far_future_ms + 12_000
    later = state.model_copy(
        update={
            "price": 86_108.35,
            "timestamp_ms": later_ms,
            "timestamp": datetime.fromtimestamp(later_ms / 1000, tz=UTC).isoformat(),
        }
    )
    await repo.persist_tick(later, _ok_result("BTCUSDT"))
    updated = await repo.enrich_outcomes(now_ms=far_future_ms + 15_000)
    async with db.session() as session:
        snap = (
            await session.execute(
                select(MarketSnapshotRow).order_by(MarketSnapshotRow.timestamp_ms)
            )
        ).scalars().first()
        decision = await session.get(JevDecisionRow, decision_id)
    assert updated >= 1
    assert snap is not None
    assert snap.timestamp_ms == far_future_ms
    assert snap.market_state["symbol"] == "BTCUSDT"
    assert decision is not None
    assert decision.price_after_10s == pytest.approx(86_108.35)
    assert decision.return_after_10s == pytest.approx(86_108.35 / 86_133.15 - 1.0)
    await db.drop_all()
    await db.aclose()
