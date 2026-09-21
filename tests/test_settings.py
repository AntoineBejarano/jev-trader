from __future__ import annotations

import pytest
from src.config.settings import LIVE_CONFIRMATION_PHRASE, Settings, TradingMode


def test_default_mode_is_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRADING_MODE", raising=False)
    settings = Settings(_env_file=None)
    assert settings.trading_mode is TradingMode.SHADOW
    assert settings.symbols == ("BTCUSDT",)


def test_symbols_are_independent_and_uppercased(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(trading_symbols="btcusdt, ethusdt", _env_file=None)
    assert settings.symbols == ("BTCUSDT", "ETHUSDT")


def test_live_requires_explicit_confirmation() -> None:
    with pytest.raises(ValueError, match="LIVE_TRADING_CONFIRMATION"):
        Settings(trading_mode="LIVE", live_trading_confirmation="", _env_file=None)


def test_live_accepted_only_with_exact_phrase() -> None:
    settings = Settings(
        trading_mode="LIVE",
        live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
        _env_file=None,
    )
    assert settings.trading_mode is TradingMode.LIVE


def test_binance_aliases_to_usdm() -> None:
    settings = Settings(trading_exchange="binance", _env_file=None)
    assert settings.trading_exchange == "binance_usdm"


def test_legacy_exchange_env_name_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXCHANGE", "binance")
    monkeypatch.delenv("TRADING_EXCHANGE", raising=False)
    settings = Settings(_env_file=None)
    assert settings.trading_exchange == "binance_usdm"


def test_railway_binds_all_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("DASHBOARD_HOST", raising=False)
    settings = Settings(_env_file=None)
    assert settings.dashboard_host == "0.0.0.0"


def test_paper_fee_and_host_aliases_match_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_TAKER_FEE_BPS", "4")
    monkeypatch.setenv("PAPER_MAKER_FEE_BPS", "2")
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("MAX_POSITION_DURATION_SECONDS", "0")
    monkeypatch.delenv("TAKER_FEE_BPS", raising=False)
    monkeypatch.delenv("MAKER_FEE_BPS", raising=False)
    monkeypatch.delenv("DASHBOARD_HOST", raising=False)
    monkeypatch.delenv("DASHBOARD_PORT", raising=False)
    settings = Settings(_env_file=None)
    assert settings.taker_fee_bps == 4
    assert settings.maker_fee_bps == 2
    assert settings.dashboard_host == "0.0.0.0"
    assert settings.dashboard_port == 9000
    assert settings.max_position_duration_or_none is None
    assert settings.jev_decisions_url.endswith("/api/alpha/decisions")


def test_scalp_uses_assumed_taker_not_paper_zero() -> None:
    settings = Settings(
        trading_strategy="SCALP",
        taker_fee_bps=0.0,
        maker_fee_bps=0.0,
        _env_file=None,
    )
    assert settings.taker_fee_bps == 0.0
    assert settings.execution_taker_fee_bps == 5.0
    assert settings.execution_maker_fee_bps == 2.0
    assert settings.trading_strategy.value == "SCALP"


def test_jev_latest_is_rejected() -> None:
    with pytest.raises(ValueError, match="pinned"):
        Settings(jev_model="jev-latest", _env_file=None)
