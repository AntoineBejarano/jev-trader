from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Self

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(StrEnum):
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    LIVE = "LIVE"


class TradingStrategy(StrEnum):
    BASELINE = "BASELINE"
    SCALP = "SCALP"
    DIRECTIONAL = "DIRECTIONAL"


LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_REAL_MONEY_IS_AT_RISK"
JEV_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


class Settings(BaseSettings):
    """Process configuration. Field names match the existing .env contract where possible."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    openrouter_api_key: str = ""
    jev_model: str = "typesafe/jev-1.13"
    jev_base_url: str = "https://openrouter.ai/api/v1"
    jev_timeout_seconds: float = 10.0
    jev_failure_timeout_seconds: float = 30.0

    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = False
    binance_futures_rest_base: str = "https://fapi.binance.com"
    binance_ws_public_base: str = "wss://fstream.binance.com/public"
    binance_ws_market_base: str = "wss://fstream.binance.com/market"

    database_url: str = "sqlite:///./jev_trader.db"

    trading_mode: TradingMode = TradingMode.SHADOW
    trading_strategy: TradingStrategy = TradingStrategy.BASELINE
    live_trading_confirmation: str = ""

    trading_exchange: str = Field(
        default="binance_usdm",
        validation_alias=AliasChoices("TRADING_EXCHANGE", "EXCHANGE", "trading_exchange"),
    )
    trading_symbols: str = "BTCUSDT"

    decision_interval_seconds: float = 5.0
    primary_horizon_seconds: int = 300
    stale_market_data_timeout_seconds: float = 10.0
    open_interest_poll_seconds: float = 5.0
    history_seconds: int = 3600
    mid_sample_interval_ms: int = 100
    order_book_levels: int = 20

    paper_initial_balance: float = 10_000.0
    paper_currency: str = "USDT"
    maker_fee_bps: float = Field(
        default=0.0,
        validation_alias=AliasChoices("PAPER_MAKER_FEE_BPS", "MAKER_FEE_BPS", "maker_fee_bps"),
    )
    taker_fee_bps: float = Field(
        default=0.0,
        validation_alias=AliasChoices("PAPER_TAKER_FEE_BPS", "TAKER_FEE_BPS", "taker_fee_bps"),
    )
    slippage_bps: float = Field(
        default=1.0,
        validation_alias=AliasChoices("PAPER_SLIPPAGE_BPS", "SLIPPAGE_BPS", "slippage_bps"),
    )

    entry_threshold: float = 0.80
    evidence_threshold: float = 0.75
    hold_threshold: float = 0.55
    close_threshold: float = 0.70
    reverse_threshold: float = 0.88
    min_position_age_seconds: float = 15.0
    min_flat_cooldown_seconds: float = 10.0
    min_seconds_between_orders: float = 10.0

    max_position_notional: float = 1000.0
    max_leverage: float = 1.0
    max_loss_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_daily_drawdown_pct: float = 3.0
    max_consecutive_losses: int = 5
    max_orders_per_minute: int = 6
    max_position_duration_seconds: int = 0
    max_total_exposure: float = 1000.0
    max_simultaneous_positions: int = 1
    paper_position_notional: float = 0.0
    target_position_equity_pct: float = 0.0

    scalp_horizons_seconds: str = "15,30,60,180,300"
    scalp_min_net_ev_bps: float = 1.0
    scalp_min_probability_edge: float = 0.08
    scalp_max_noise: float = 0.85
    scalp_min_position_age_seconds: float = 8.0
    scalp_min_seconds_between_orders: float = 8.0
    scalp_hold_min_net_ev_bps: float = 0.0
    scalp_close_ev_bps: float = 0.0
    scalp_reverse_advantage_bps: float = 1.5
    scalp_min_target_bps: float = 1.5
    scalp_min_stop_bps: float = 1.0
    scalp_profit_buffer_bps: float = 0.5
    scalp_target_vol_mult: float = 1.0
    scalp_stop_vol_mult: float = 0.7
    scalp_taker_fee_bps: float = 5.0
    scalp_maker_fee_bps: float = 2.0
    scalp_cost_model_assumed: bool = True
    scalp_cost_model_source: str = "binance_usdm_vip0_regular_no_bnb"
    scalp_expected_return_min_bps: float = -100.0
    scalp_expected_return_max_bps: float = 100.0
    paper_funding_enabled: bool = False

    dir_entry_confidence: float = 0.70
    dir_exit_opposite_confidence: float = 0.65
    dir_reverse_confidence: float = 0.75
    dir_confirmations_required: int = 2

    experiment_id: str = "scalp-v3-calibration"
    experiment_frozen: bool = False

    kill_switch: bool = False

    log_level: str = "INFO"
    environment: str = "development"
    dashboard_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("DASHBOARD_HOST", "HOST", "dashboard_host"),
    )
    dashboard_port: int = Field(
        default=8000,
        validation_alias=AliasChoices("DASHBOARD_PORT", "PORT", "dashboard_port"),
    )

    @field_validator("jev_model")
    @classmethod
    def reject_jev_latest(cls, value: str) -> str:
        if value.strip().lower() in {"jev-latest", "typesafe/jev-latest", "~typesafe/jev-latest"}:
            raise ValueError(
                "JEV_MODEL must be a pinned model such as typesafe/jev-1.13, not jev-latest"
            )
        return value.strip()

    @field_validator("trading_mode", "trading_strategy", mode="before")
    @classmethod
    def normalize_mode(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("trading_exchange")
    @classmethod
    def normalize_exchange(cls, value: str) -> str:
        aliases = {
            "binance": "binance_usdm",
            "binance_usdm": "binance_usdm",
            "binance-usdm": "binance_usdm",
            "binance-futures": "binance_usdm",
            "usdm": "binance_usdm",
        }
        return aliases.get(value.strip().lower(), value.strip())

    @field_validator("trading_symbols")
    @classmethod
    def normalize_symbols(cls, value: str) -> str:
        symbols = [part.strip().upper() for part in value.split(",") if part.strip()]
        if not symbols:
            raise ValueError("TRADING_SYMBOLS must contain at least one symbol")
        return ",".join(symbols)

    @model_validator(mode="after")
    def refuse_unconfirmed_live(self) -> Self:
        if (
            self.trading_mode is TradingMode.LIVE
            and self.live_trading_confirmation != LIVE_CONFIRMATION_PHRASE
        ):
            raise ValueError(
                "LIVE mode requires LIVE_TRADING_CONFIRMATION="
                f"{LIVE_CONFIRMATION_PHRASE}"
            )
        return self

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.trading_symbols.split(","))

    @property
    def history_ms(self) -> int:
        return self.history_seconds * 1000

    @property
    def stale_timeout_ms(self) -> int:
        return int(self.stale_market_data_timeout_seconds * 1000)

    @property
    def jev_decisions_url(self) -> str:
        """Jev is not a chat model. Official Decisions API, not /api/v1 chat completions."""
        return JEV_DECISIONS_URL

    @property
    def max_position_duration_or_none(self) -> int | None:
        if self.max_position_duration_seconds <= 0:
            return None
        return self.max_position_duration_seconds

    @property
    def scalp_horizons(self) -> tuple[int, ...]:
        parts = [part.strip() for part in self.scalp_horizons_seconds.split(",") if part.strip()]
        return tuple(int(part) for part in parts)

    @property
    def execution_taker_fee_bps(self) -> float:
        if self.trading_strategy in {TradingStrategy.SCALP, TradingStrategy.DIRECTIONAL}:
            return self.scalp_taker_fee_bps
        return self.taker_fee_bps

    @property
    def execution_maker_fee_bps(self) -> float:
        if self.trading_strategy in {TradingStrategy.SCALP, TradingStrategy.DIRECTIONAL}:
            return self.scalp_maker_fee_bps
        return self.maker_fee_bps

    @property
    def execution_notional(self) -> float:
        if self.paper_position_notional > 0:
            return self.paper_position_notional
        if self.target_position_equity_pct > 0:
            return self.paper_initial_balance * self.target_position_equity_pct
        return self.max_position_notional

    @property
    def active_min_position_age_seconds(self) -> float:
        if self.trading_strategy is TradingStrategy.SCALP:
            return self.scalp_min_position_age_seconds
        return self.min_position_age_seconds

    @property
    def active_min_seconds_between_orders(self) -> float:
        if self.trading_strategy is TradingStrategy.SCALP:
            return self.scalp_min_seconds_between_orders
        return self.min_seconds_between_orders


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()
