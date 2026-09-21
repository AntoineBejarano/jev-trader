from __future__ import annotations

from src.config.settings import Settings
from src.trader.engine import TraderEngine


def test_shadow_engine_still_constructs() -> None:
    settings = Settings(trading_mode="SHADOW", _env_file=None)
    engine = TraderEngine(settings)
    assert engine.paper is None


def test_live_engine_refuses_to_start_even_with_confirmation() -> None:
    settings = Settings(
        trading_mode="LIVE",
        live_trading_confirmation="I_UNDERSTAND_REAL_MONEY_IS_AT_RISK",
        _env_file=None,
    )
    try:
        TraderEngine(settings)
        raise AssertionError("LIVE engine must not construct in Phase 4")
    except RuntimeError as exc:
        assert "Phase 4" in str(exc)
