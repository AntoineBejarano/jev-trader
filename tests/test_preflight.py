from __future__ import annotations

import pytest
from src.config.settings import LIVE_CONFIRMATION_PHRASE, Settings
from src.startup.preflight import PreflightError, run_preflight
from src.trader.engine import LIVE_PHASE4_MESSAGE, TraderEngine


@pytest.mark.asyncio
async def test_preflight_rejects_live() -> None:
    settings = Settings(
        trading_mode="LIVE",
        live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
        _env_file=None,
    )
    with pytest.raises(PreflightError, match="LIVE"):
        await run_preflight(settings, check_network=False)


@pytest.mark.asyncio
async def test_preflight_rejects_shadow() -> None:
    settings = Settings(trading_mode="SHADOW", _env_file=None)
    with pytest.raises(PreflightError, match="PAPER"):
        await run_preflight(settings, check_network=False)


@pytest.mark.asyncio
async def test_preflight_rejects_sqlite() -> None:
    settings = Settings(
        trading_mode="PAPER",
        database_url="sqlite:///./jev_trader.db",
        openrouter_api_key="present",
        _env_file=None,
    )
    with pytest.raises(PreflightError, match="PostgreSQL"):
        await run_preflight(settings, check_network=False)


def test_live_engine_still_blocked() -> None:
    settings = Settings(
        trading_mode="LIVE",
        live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
        _env_file=None,
    )
    with pytest.raises(RuntimeError, match="Phase 4"):
        TraderEngine(settings)
    assert "LIVE" in LIVE_PHASE4_MESSAGE
