from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from src.decision.actions import Action
from src.storage.database import Database
from src.storage.models import JevDecisionRow
from src.storage.repository import Repository
from tests.helpers import make_market_state
from tests.paper_helpers import execute_action, fresh_account
from tests.test_storage import POSTGRES_SMOKE_URL, _ok_result


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


@pytest.mark.asyncio
async def test_latency_and_scalp_payload_persist_sqlite(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'scalp.db'}")
    await db.create_all()
    repo = Repository(db)
    state = make_market_state(symbol="BTCUSDT", price=100.0, bid=100.0, ask=100.10)
    decision_id = await repo.persist_tick(
        state,
        _ok_result("BTCUSDT"),
        candidate_action="NO_ACTION",
        final_action="NO_ACTION",
        scalp_payload={
            "best": {"side": "SHORT", "horizon_seconds": 30, "net_ev_bps": 2.4},
            "latency": {
                "market_state_created_at_ms": 1.0,
                "jev_latency_ms": 180.0,
                "decision_calculated_at_ms": 2.0,
            },
            "cost": {"taker_fee_bps": 5.0, "slippage_bps": 1.0},
        },
    )
    async with db.session() as session:
        row = await session.get(JevDecisionRow, decision_id)
    assert row is not None
    assert row.jev_latency_ms == pytest.approx(250.0)
    assert row.scalp_payload is not None
    assert row.scalp_payload["latency"]["jev_latency_ms"] == 180.0
    await db.aclose()


@pytest.mark.asyncio
async def test_counterfactual_long_short_sqlite(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'cf.db'}")
    await db.create_all()
    repo = Repository(db)
    t0 = 1_700_000_000_000
    entry = make_market_state(symbol="BTCUSDT", price=100.0, bid=100.0, ask=100.10).model_copy(
        update={"timestamp_ms": t0, "timestamp": _iso(t0)}
    )
    await repo.persist_tick(
        entry,
        _ok_result("BTCUSDT"),
        scalp_payload={"cost": {"taker_fee_bps": 5.0, "slippage_bps": 1.0}, "counterfactual": {}},
    )
    later = make_market_state(symbol="BTCUSDT", price=100.4, bid=100.3, ask=100.5).model_copy(
        update={"timestamp_ms": t0 + 16_000, "timestamp": _iso(t0 + 16_000)}
    )
    await repo.persist_tick(later, _ok_result("BTCUSDT"))
    updated = await repo.enrich_outcomes(now_ms=t0 + 20_000)
    assert updated >= 1
    async with db.session() as session:
        row = (
            await session.execute(
                select(JevDecisionRow).order_by(JevDecisionRow.timestamp_ms)
            )
        ).scalars().first()
    assert row is not None
    assert row.scalp_payload is not None
    cf = row.scalp_payload["counterfactual"]["15s"]
    assert "long_net_pnl" in cf
    assert "short_net_pnl" in cf
    await db.aclose()


@pytest.mark.asyncio
async def test_restart_with_scalp_position(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'restart.db'}")
    await db.create_all()
    repo = Repository(db)
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    report = execute_action(Action.OPEN_SHORT, account, state, now_ms=1_000_000, decision_id="s1")
    assert report is not None
    assert report.opened is not None
    report.opened.entry_horizon_seconds = 30
    report.opened.entry_net_ev_bps = 2.4
    await repo.persist_execution(
        report=report, now_ms=1_000_000, symbol=state.symbol, exchange=state.exchange
    )
    await repo.save_paper_account(account)
    restored = await repo.load_paper_account()
    await db.aclose()
    assert restored is not None
    position = restored.get_position(state.symbol, state.exchange)
    assert position is not None
    assert position.side.value == "SHORT"
    assert position.entry_horizon_seconds == 30
    assert position.entry_net_ev_bps == pytest.approx(2.4)


@pytest.mark.asyncio
async def test_postgres_scalp_payload() -> None:
    db = Database(POSTGRES_SMOKE_URL)
    try:
        await db.drop_all()
        await db.create_all()
    except Exception as exc:
        await db.aclose()
        pytest.skip(f"Postgres in OrbStack is not reachable: {exc}")
    repo = Repository(db)
    state = make_market_state(symbol="BTCUSDT", price=100.0, bid=99.9, ask=100.1)
    decision_id = await repo.persist_tick(
        state,
        _ok_result("BTCUSDT"),
        scalp_payload={
            "best": {"side": "LONG", "net_ev_bps": 1.1},
            "latency": {"jev_latency_ms": 90.0},
        },
    )
    async with db.session() as session:
        row = await session.get(JevDecisionRow, decision_id)
    assert row is not None
    assert row.scalp_payload is not None
    assert row.scalp_payload["best"]["side"] == "LONG"
    await db.drop_all()
    await db.aclose()
