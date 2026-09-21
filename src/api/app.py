from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
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
            "database": system["database"],
            "binance_websocket": system["binance_websocket"],
            "jev": system["jev"],
        }

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return await engine.operator_snapshot()

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        return HTMLResponse(_dashboard_html())

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(_dashboard_html())

    return app
