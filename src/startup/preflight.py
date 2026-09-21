from __future__ import annotations

import logging

import httpx

from src.config.settings import Settings, TradingMode
from src.storage.database import Database, database_kind
from src.storage.repository import Repository

logger = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    """Local operator start was refused before the engine launched."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


async def run_preflight(settings: Settings, *, check_network: bool = True) -> None:
    _require(
        settings.trading_mode is not TradingMode.LIVE,
        "LIVE trading is blocked. Set TRADING_MODE=PAPER.",
    )
    _require(
        settings.live_trading_confirmation == "",
        "LIVE_TRADING_CONFIRMATION must stay empty for this local validation.",
    )
    _require(
        settings.trading_mode is TradingMode.PAPER,
        "TRADING_MODE must be PAPER for this local operator run.",
    )
    _require(bool(settings.openrouter_api_key.strip()), "OPENROUTER_API_KEY is missing.")
    _require(bool(settings.symbols), "TRADING_SYMBOLS must contain at least one symbol.")
    _require(
        database_kind(settings.database_url) == "PostgreSQL",
        "DATABASE_URL must point at local PostgreSQL for this validation.",
    )

    db = Database(settings.database_url)
    try:
        await db.create_all()
        await Repository(db).ping()
    except Exception as exc:
        raise PreflightError(
            "PostgreSQL is not reachable. Start OrbStack Postgres and retry."
        ) from exc
    finally:
        await db.aclose()
    logger.info("Preflight database=PostgreSQL connected")

    if not check_network:
        return

    async with httpx.AsyncClient(timeout=10.0) as http:
        try:
            ping = await http.get(f"{settings.binance_futures_rest_base}/fapi/v1/ping")
            ping.raise_for_status()
            ticker = await http.get(
                f"{settings.binance_futures_rest_base}/fapi/v1/ticker/bookTicker",
                params={"symbol": settings.symbols[0]},
            )
            ticker.raise_for_status()
        except Exception as exc:
            raise PreflightError("Binance USD-M public REST is not reachable.") from exc
        logger.info("Preflight Binance public REST reachable")

        try:
            key_check = await http.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            )
            key_check.raise_for_status()
        except Exception as exc:
            raise PreflightError("OpenRouter is not reachable or the API key was refused.") from exc
        logger.info("Preflight OpenRouter reachable")
