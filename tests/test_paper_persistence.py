from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from src.decision.actions import Action
from src.execution.paper import PaperExecutionEngine
from src.market.models import Side
from src.storage.database import Database
from src.storage.models import EquitySnapshotRow, FillRow, OrderRow, TradeRow
from src.storage.repository import Repository
from tests.helpers import make_market_state
from tests.paper_helpers import execute_action, fresh_account, paper_settings
from tests.test_storage import POSTGRES_SMOKE_URL, _ok_result


@pytest.mark.asyncio
async def test_restart_restores_open_paper_position(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'paper.db'}")
    await db.create_all()
    repo = Repository(db)
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    report = execute_action(Action.OPEN_LONG, account, state, now_ms=1_000_000, decision_id="dec-1")
    assert report is not None
    await repo.persist_execution(
        report=report, now_ms=1_000_000, symbol="ETHUSDT", exchange="binance_usdm"
    )
    await repo.save_paper_account(account)
    await repo.upsert_open_position(
        account.get_position("ETHUSDT", "binance_usdm"),
        symbol="ETHUSDT",
        exchange="binance_usdm",
        now_ms=1_000_000,
    )
    restored = await repo.load_paper_account()
    await db.aclose()
    assert restored is not None
    position = restored.get_position("ETHUSDT", "binance_usdm")
    assert position is not None
    assert position.side is Side.LONG
    assert position.entry_price == pytest.approx(report.opened.entry_price if report.opened else 0)
    assert position.quantity == pytest.approx(report.opened.quantity if report.opened else 0)
    assert restored.fees_paid == pytest.approx(account.fees_paid)


@pytest.mark.asyncio
async def test_sqlite_persists_orders_fills_trades_equity(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'paper2.db'}")
    await db.create_all()
    repo = Repository(db)
    account = fresh_account()
    state = make_market_state(symbol="BTCUSDT", price=100.0, bid=100.0, ask=100.10)
    opened = execute_action(Action.OPEN_LONG, account, state, now_ms=1, decision_id="o1")
    assert opened is not None
    later = make_market_state(symbol="BTCUSDT", price=101.0, bid=101.0, ask=101.10, side=Side.LONG)
    closed = execute_action(Action.CLOSE, account, later, now_ms=20_000, decision_id="c1")
    assert closed is not None
    await repo.persist_execution(report=opened, now_ms=1, symbol="BTCUSDT", exchange="binance_usdm")
    await repo.persist_execution(
        report=closed, now_ms=20_000, symbol="BTCUSDT", exchange="binance_usdm"
    )
    await repo.save_paper_account(account)
    await repo.persist_equity_snapshot(account, 20_000)
    async with db.session() as session:
        orders = (await session.execute(select(OrderRow))).scalars().all()
        fills = (await session.execute(select(FillRow))).scalars().all()
        trades = (await session.execute(select(TradeRow))).scalars().all()
        equity = (await session.execute(select(EquitySnapshotRow))).scalars().all()
    assert len(orders) == 2
    assert len(fills) == 2
    assert len(trades) == 1
    assert trades[0].entry_jev_decision_id == "o1"
    assert trades[0].exit_jev_decision_id == "c1"
    assert trades[0].fees == pytest.approx(closed.closed.total_fees if closed.closed else 0)
    assert equity
    assert await repo.has_orders_for_decision("o1") is True
    await db.aclose()


@pytest.mark.asyncio
async def test_postgres_paper_entities() -> None:
    db = Database(POSTGRES_SMOKE_URL)
    try:
        await db.drop_all()
        await db.create_all()
    except Exception as exc:
        await db.aclose()
        pytest.skip(f"Postgres in OrbStack is not reachable: {exc}")
    repo = Repository(db)
    account = fresh_account()
    state = make_market_state(symbol="BTCUSDT", price=100.0, bid=100.0, ask=100.10)
    report = execute_action(
        Action.OPEN_LONG, account, state, now_ms=1_790_003_765_175, decision_id="pg1"
    )
    assert report is not None
    state_snap = make_market_state(symbol="BTCUSDT", price=100.0, bid=100.0, ask=100.10)
    await repo.persist_tick(
        state_snap,
        _ok_result("BTCUSDT"),
        candidate_action="OPEN_LONG",
        final_action="OPEN_LONG",
    )
    await repo.persist_execution(
        report=report, now_ms=1_790_003_765_175, symbol="BTCUSDT", exchange="binance_usdm"
    )
    await repo.save_paper_account(account)
    restored = await repo.load_paper_account()
    assert restored is not None
    assert restored.get_position("BTCUSDT", "binance_usdm") is not None
    await db.drop_all()
    await db.aclose()


@pytest.mark.asyncio
async def test_duplicate_persist_does_not_double_open(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'idemp.db'}")
    await db.create_all()
    repo = Repository(db)
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    engine = PaperExecutionEngine(paper_settings())
    from src.decision.actions import ActionSource
    from src.risk.engine import RiskDecision

    risk = RiskDecision(Action.OPEN_LONG, Action.OPEN_LONG, None, ActionSource.JEV)
    first = engine.execute(
        account=account,
        risk=risk,
        symbol=state.symbol,
        exchange=state.exchange,
        bid=100.0,
        ask=100.10,
        now_ms=1,
        decision_id="once",
        already_processed=False,
    )
    assert first is not None
    await repo.persist_execution(
        report=first, now_ms=1, symbol=state.symbol, exchange=state.exchange
    )
    assert await repo.has_orders_for_decision("once") is True
    second = engine.execute(
        account=account,
        risk=risk,
        symbol=state.symbol,
        exchange=state.exchange,
        bid=100.0,
        ask=100.10,
        now_ms=2,
        decision_id="once",
        already_processed=True,
    )
    assert second is not None
    assert second.skipped_duplicate is True
    assert account.open_position_count == 1
    await db.aclose()


@pytest.mark.asyncio
async def test_new_experiment_resets_paper_without_deleting_research(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'oos.db'}")
    await db.create_all()
    repo = Repository(db)
    repo.bind_run(run_id="run-002", experiment_id="directional-v1-manual-002", config_hash="h")
    stale = fresh_account()
    stale.consecutive_losses = 5
    stale.cash = 9_000.0
    stale.realized_pnl = -10.0
    await repo.save_paper_account(stale)
    await repo.persist_tick(
        make_market_state(symbol="BTCUSDT", price=100.0),
        _ok_result("BTCUSDT"),
        directional_payload={
            "forecast": {"p_up": {"1m": 0.3}, "research_sample_valid": True},
            "policy": {"candidate": "NO_ACTION"},
            "research_sample_valid": True,
        },
    )
    assert await repo.paper_account_experiment_id() == "directional-v1-manual-002"
    assert len(await repo.list_directional_forecasts("run-002")) == 1

    repo.bind_run(
        run_id="run-003", experiment_id="directional-v1-prediction-003", config_hash="h3"
    )
    assert await repo.paper_account_experiment_id() != "directional-v1-prediction-003"
    fresh = await repo.reset_paper_account(starting_balance=10_000.0, currency="USDT")
    assert fresh.consecutive_losses == 0
    assert fresh.equity == pytest.approx(10_000.0)
    assert fresh.open_position_count == 0
    assert await repo.paper_account_experiment_id() == "directional-v1-prediction-003"
    leftover = await repo.list_directional_forecasts("run-002")
    assert len(leftover) == 1
    assert leftover[0].forecast["p_up"]["1m"] == 0.3

    fresh.consecutive_losses = 2
    await repo.save_paper_account(fresh)
    restored = await repo.load_paper_account()
    assert restored is not None
    assert await repo.paper_account_experiment_id() == "directional-v1-prediction-003"
    assert restored.consecutive_losses == 2
    await db.aclose()


@pytest.mark.asyncio
async def test_paper_persist_timeout_marks_degraded_and_blocks_opens(tmp_path: Path) -> None:
    from src.config.settings import Settings
    from src.decision.actions import ActionSource
    from src.risk.engine import RiskDecision
    from src.trader.engine import TraderEngine

    engine = TraderEngine(
        Settings(
            trading_mode="PAPER",
            database_url=f"sqlite:///{tmp_path / 'degraded.db'}",
            _env_file=None,
        )
    )
    await engine.db.create_all()
    orig_save = engine.repo.save_paper_account
    orig_snap = engine.repo.persist_equity_snapshot

    async def boom(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError()

    engine.repo.save_paper_account = boom  # type: ignore[method-assign]
    engine.repo.persist_equity_snapshot = boom  # type: ignore[method-assign]
    assert await engine._persist_paper_runtime(1) is False
    assert engine.paper_persistence == "DEGRADED"

    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    risk = RiskDecision(Action.OPEN_LONG, Action.OPEN_LONG, None, ActionSource.JEV)
    await engine._maybe_execute(state, risk, "dec-degraded", 1, 100.0, 100.10, None, 1, None)
    assert engine.account.open_position_count == 0

    engine.repo.save_paper_account = orig_save
    engine.repo.persist_equity_snapshot = orig_snap
    assert await engine._persist_paper_runtime(2) is True
    assert engine.paper_persistence == "HEALTHY"
    await engine.db.aclose()
