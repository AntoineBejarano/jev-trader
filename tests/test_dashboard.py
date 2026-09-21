from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from src.api.app import create_app
from src.config.settings import Settings
from src.market.models import Side
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
        assert "BEST OPPORTUNITY NOW" in page.text
        assert "OPPORTUNITY MATRIX" in page.text
        assert "CONTROL DE ESTRATEGIA" in page.text
        assert "Encender" in page.text
        assert "Parar" in page.text
        assert "/api/strategy/start" in page.text
        assert "/api/strategy/stop" in page.text
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
