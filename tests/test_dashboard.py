from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from src.api.app import create_app
from src.api.snapshot import _session_view
from src.config.settings import Settings
from src.market.models import Side
from src.portfolio.account import ClosedTrade
from src.trader.engine import TraderEngine


def _engine(tmp_path: Path, *, mode: str = "PAPER") -> TraderEngine:
    settings = Settings(
        trading_mode=mode,
        database_url=f"sqlite:///{tmp_path / 'operator.db'}",
        openrouter_api_key="sk-secret-must-not-leak",
        binance_api_key="binance-secret-must-not-leak",
        _env_file=None,
    )
    return TraderEngine(settings)


@pytest.mark.asyncio
async def test_operator_status_hides_secrets_and_shows_paper(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    await engine.db.create_all()
    engine.lifecycle = "RUNNING"
    payload = await engine.operator_snapshot()
    blob = json.dumps(payload)
    assert "sk-secret-must-not-leak" not in blob
    assert "binance-secret-must-not-leak" not in blob
    assert "OPENROUTER_API_KEY" not in blob
    assert payload["safety"]["live"] == "BLOCKED"
    assert payload["safety"]["trading_mode"] == "PAPER"
    assert payload["system"]["strategy"] == "BASELINE"
    assert payload["cost_model"]["taker_fee_bps"] == 0.0
    assert payload["position"]["side"] == Side.FLAT.value
    assert payload["position"]["open"] is False
    assert payload["recent_trades"] == []
    assert payload["system"]["engine"] == "RUNNING"
    assert payload["system"]["strategy_running"] is True
    assert payload["system"]["can_start"] is False
    assert payload["system"]["can_stop"] is True
    await engine.db.aclose()


@pytest.mark.asyncio
async def test_shadow_engine_still_exposes_status(tmp_path: Path) -> None:
    engine = _engine(tmp_path, mode="SHADOW")
    await engine.db.create_all()
    payload = await engine.operator_snapshot()
    assert payload["system"]["trading_mode"] == "SHADOW"
    assert payload["system"]["live"] == "BLOCKED"
    await engine.db.aclose()


@pytest.mark.asyncio
async def test_dashboard_and_health_routes(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    await engine.db.create_all()
    app = create_app(engine)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/dashboard")
        assert page.status_code == 200
        assert "PAPER TRADING" in page.text
        assert "NO REAL ORDERS" in page.text
        assert "LIVE TRADING DISABLED" in page.text
        assert "PAPER P&amp;L" in page.text or "PAPER P&L" in page.text
        assert "PROFIT" in page.text
        assert "MARK-TO-MARKET ACCOUNT METRICS" in page.text
        assert "REALIZED TRADE METRICS" in page.text
        assert "Raw predictions N" in page.text
        assert "Valid research N" in page.text
        assert "PREDICTION RESEARCH" in page.text
        assert "BEST OPPORTUNITY NOW" in page.text
        assert "OPPORTUNITY MATRIX" in page.text
        assert "CONTROL DE ESTRATEGIA" in page.text
        assert "Encender" in page.text
        assert "Parar" in page.text
        assert "/api/strategy/start" in page.text
        assert "/api/strategy/stop" in page.text
        assert "Resetear data" in page.text
        assert "/api/strategy/reset" in page.text
        root = await client.get("/")
        assert root.status_code == 200
        health = await client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["live"] == "BLOCKED"
        status = await client.get("/api/status")
        assert status.status_code == 200
        assert "sk-secret-must-not-leak" not in status.text
        system = status.json()["system"]
        assert system["engine"] == "STOPPED"
        assert system["can_start"] is True
        assert system["can_stop"] is False
        assert system["strategy_running"] is False
    await engine.db.aclose()


@pytest.mark.asyncio
async def test_strategy_start_stop_routes_without_binance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    await engine.db.create_all()

    async def fake_session() -> None:
        engine.lifecycle = "RUNNING"
        try:
            await engine._stop.wait()
        finally:
            engine.lifecycle = "STOPPED"
            engine._strategy_task = None

    monkeypatch.setattr(engine, "_strategy_session", fake_session)
    app = create_app(engine)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        idle = await client.post("/api/strategy/stop")
        assert idle.status_code == 200
        assert idle.json()["detail"] == "already_stopped"

        started = await client.post("/api/strategy/start")
        assert started.status_code == 200
        assert started.json()["ok"] is True
        assert started.json()["engine"] == "STARTING"

        system = {"engine": "STARTING"}
        for _ in range(50):
            system = (await client.get("/api/status")).json()["system"]
            if system["engine"] == "RUNNING":
                break
            await asyncio.sleep(0.02)
        assert system["engine"] == "RUNNING"
        assert system["can_start"] is False
        assert system["can_stop"] is True
        assert system["strategy_running"] is True

        again = await client.post("/api/strategy/start")
        assert again.status_code == 200
        assert again.json()["changed"] is False
        assert again.json()["detail"] == "already_running"

        stopped = await client.post("/api/strategy/stop")
        assert stopped.status_code == 200
        assert stopped.json()["engine"] == "STOPPED"
        after = (await client.get("/api/status")).json()["system"]
        assert after["engine"] == "STOPPED"
        assert after["can_start"] is True
        assert after["can_stop"] is False
        assert after["strategy_running"] is False
    await engine.db.aclose()


@pytest.mark.asyncio
async def test_strategy_reset_wipes_db_and_restores_flat_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    await engine.db.create_all()
    engine.account.cash = 1.0
    engine.account.realized_pnl = -9_999.0
    engine.last_jev_at_ms = 1
    await engine.repo.save_paper_account(engine.account)
    await engine.repo.record_system_event(event_type="NOISE")
    assert await engine.repo.list_recent_events(20)
    dirty = await engine.repo.load_paper_account()
    assert dirty is not None
    assert dirty.equity == pytest.approx(1.0)

    async def fake_session() -> None:
        engine.lifecycle = "RUNNING"
        try:
            await engine._stop.wait()
        finally:
            engine.lifecycle = "STOPPED"
            engine._strategy_task = None

    monkeypatch.setattr(engine, "_strategy_session", fake_session)
    app = create_app(engine)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        started = await client.post("/api/strategy/start")
        assert started.status_code == 200
        for _ in range(50):
            if (await client.get("/api/status")).json()["system"]["engine"] == "RUNNING":
                break
            await asyncio.sleep(0.02)

        reset = await client.post("/api/strategy/reset")
        assert reset.status_code == 200
        body = reset.json()
        assert body["ok"] is True
        assert body["engine"] == "STOPPED"
        assert body["stopped_strategy"] is True
        assert body["paper_equity"] == 10_000.0
        assert body["deleted"]["system_events"] >= 1
        assert body["deleted"]["paper_accounts"] >= 1

        status = (await client.get("/api/status")).json()
        assert status["system"]["engine"] == "STOPPED"
        assert status["system"]["can_start"] is True
        assert status["system"]["can_reset"] is True
        assert status["account"]["equity"] == 10_000.0
        assert status["position"]["side"] == Side.FLAT.value
        assert status["recent_trades"] == []
        assert status["recent_events"] == []
        assert status["recent_decisions"] == []
    assert engine.last_jev_at_ms is None
    assert engine.last_result is None
    restored = await engine.repo.load_paper_account()
    assert restored is not None
    assert restored.equity == 10_000.0
    assert restored.closed_trades == []
    await engine.db.aclose()


def _closed_trade(net_pnl: float = -1.5) -> ClosedTrade:
    return ClosedTrade(
        symbol="BTCUSDT",
        exchange="binance_usdm",
        side=Side.LONG,
        quantity=0.01,
        entry_price=100.0,
        exit_price=99.0,
        entry_notional=1000.0,
        entry_timestamp_ms=1,
        exit_timestamp_ms=2,
        holding_seconds=10.0,
        gross_pnl=net_pnl,
        entry_fee=0.0,
        exit_fee=0.0,
        total_fees=0.0,
        slippage_cost=0.0,
        funding_cost=0.0,
        net_pnl=net_pnl,
        gross_pnl_bps=-15.0,
        net_pnl_bps=-15.0,
        mfe_bps=0.0,
        mae_bps=0.0,
        entry_jev_decision_id=None,
        exit_jev_decision_id=None,
        entry_source="JEV",
        exit_source="JEV",
    )


def test_session_view_keeps_in_memory_trades_when_db_lists_are_none(tmp_path: Path) -> None:
    settings = Settings(
        trading_mode="PAPER",
        trading_strategy="DIRECTIONAL",
        database_url=f"sqlite:///{tmp_path / 'session.db'}",
        openrouter_api_key="sk-secret-must-not-leak",
        _env_file=None,
    )
    engine = TraderEngine(settings)
    engine.account.closed_trades.append(_closed_trade(-3.93))
    view = _session_view(engine, now_ms=1_000, run_trades=None, experiment_trades=None)
    assert view["trades"] == 1
    assert view["net_pnl"] == pytest.approx(-3.93)
    assert view["experiment_total"]["trades"] == 1


def test_session_view_empty_db_list_is_real_zero(tmp_path: Path) -> None:
    settings = Settings(
        trading_mode="PAPER",
        trading_strategy="DIRECTIONAL",
        database_url=f"sqlite:///{tmp_path / 'session-zero.db'}",
        openrouter_api_key="sk-secret-must-not-leak",
        _env_file=None,
    )
    engine = TraderEngine(settings)
    engine.account.closed_trades.append(_closed_trade())
    view = _session_view(engine, now_ms=1_000, run_trades=[], experiment_trades=[])
    assert view["trades"] == 0
    assert view["experiment_total"]["trades"] == 0


@pytest.mark.asyncio
async def test_failed_snapshot_reuses_cached_experiment_totals(tmp_path: Path) -> None:
    settings = Settings(
        trading_mode="PAPER",
        trading_strategy="DIRECTIONAL",
        database_url=f"sqlite:///{tmp_path / 'cache.db'}",
        openrouter_api_key="sk-secret-must-not-leak",
        _env_file=None,
    )
    engine = TraderEngine(settings)
    await engine.db.create_all()
    engine._cached_experiment_trades = [_closed_trade(-19.50), _closed_trade(1.0)]
    engine._cached_run_trades = [_closed_trade(-3.93)]

    async def boom() -> None:
        raise TimeoutError("snapshot blocked")

    engine.repo.ping = boom  # type: ignore[method-assign]
    payload = await engine.operator_snapshot()
    assert payload["system"]["database"] == "ERROR"
    assert payload["session"]["trades"] == 1
    assert payload["session"]["experiment_total"]["trades"] == 2
    assert payload["session"]["experiment_total"]["net_pnl"] == pytest.approx(-18.50)
    await engine.db.aclose()
