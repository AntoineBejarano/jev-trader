from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.trader.engine import TraderEngine

DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")


def _dashboard_html() -> str:
    return DASHBOARD_PATH.read_text(encoding="utf-8")


def create_app(engine: TraderEngine) -> FastAPI:
    app = FastAPI(title="Jev Trader", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        snapshot = await engine.operator_snapshot()
        system = snapshot["system"]
        return {
            "ok": system["engine"] == "RUNNING" and system["database"] == "CONNECTED",
            "trading_mode": system["trading_mode"],
            "strategy": system.get("strategy"),
            "live": "BLOCKED",
            "engine": system["engine"],
            "strategy_running": system.get("strategy_running"),
            "database": system["database"],
            "binance_websocket": system["binance_websocket"],
            "jev": system["jev"],
        }

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return await engine.operator_snapshot()

    @app.post("/api/strategy/start")
    async def strategy_start() -> dict[str, Any]:
        result = await engine.start_strategy()
        if not result.get("ok"):
            raise HTTPException(status_code=409, detail=result)
        return result

    @app.post("/api/strategy/stop")
    async def strategy_stop() -> dict[str, Any]:
        result = await engine.stop_strategy()
        if not result.get("ok"):
            raise HTTPException(status_code=409, detail=result)
        return result

    @app.post("/api/strategy/reset")
    async def strategy_reset() -> dict[str, Any]:
        result = await engine.reset_operator_data()
        if not result.get("ok"):
            raise HTTPException(status_code=409, detail=result)
        return result

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        return HTMLResponse(_dashboard_html())

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(_dashboard_html())

    return app
