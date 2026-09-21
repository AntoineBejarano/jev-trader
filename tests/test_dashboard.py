from __future__ import annotations

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
        root = await client.get("/")
        assert root.status_code == 200
        health = await client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["live"] == "BLOCKED"
        status = await client.get("/api/status")
        assert status.status_code == 200
        assert "sk-secret-must-not-leak" not in status.text
    await engine.db.aclose()
