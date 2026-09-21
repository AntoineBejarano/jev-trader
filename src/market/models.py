from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Side(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Returns(BaseModel):
    model_config = ConfigDict(frozen=True)

    r_5s: float | None = None
    r_15s: float | None = None
    r_30s: float | None = None
    r_1m: float | None = None
    r_3m: float | None = None
    r_5m: float | None = None
    r_15m: float | None = None
    r_1h: float | None = None


class Momentum(BaseModel):
    """Signed path measurements. Not a trading signal."""

    model_config = ConfigDict(frozen=True)

    acceleration: float | None = None
    short_term_direction: float | None = None
    medium_term_direction: float | None = None


class Microstructure(BaseModel):
    model_config = ConfigDict(frozen=True)

    spread_bps: float | None = None
    book_imbalance_1: float | None = None
    book_imbalance_5: float | None = None
    book_imbalance_10: float | None = None
    aggressive_flow_5s: float | None = None
    aggressive_flow_30s: float | None = None
    aggressive_flow_1m: float | None = None
    trade_velocity: float | None = None
    volume_velocity: float | None = None
    cvd_1m: float | None = None
    cvd_5m: float | None = None


class Volatility(BaseModel):
    model_config = ConfigDict(frozen=True)

    rv_30s: float | None = None
    rv_1m: float | None = None
    rv_5m: float | None = None
    rv_15m: float | None = None
    range_5m_bps: float | None = None
    range_15m_bps: float | None = None
    atr_like_5m: float | None = None


class Derivatives(BaseModel):
    model_config = ConfigDict(frozen=True)

    open_interest: float | None = None
    oi_change_1m: float | None = None
    oi_change_5m: float | None = None
    funding: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    basis: float | None = None
    long_liquidations: float | None = None
    short_liquidations: float | None = None


class Ranges(BaseModel):
    model_config = ConfigDict(frozen=True)

    high_1m: float | None = None
    low_1m: float | None = None
    high_5m: float | None = None
    low_5m: float | None = None
    high_15m: float | None = None
    low_15m: float | None = None
    high_1h: float | None = None
    low_1h: float | None = None
    dist_from_high_1m_bps: float | None = None
    dist_from_low_1m_bps: float | None = None
    dist_from_high_5m_bps: float | None = None
    dist_from_low_5m_bps: float | None = None
    dist_from_high_15m_bps: float | None = None
    dist_from_low_15m_bps: float | None = None
    range_position_1m: float | None = None
    range_position_5m: float | None = None


class Regime(BaseModel):
    """Compact independently meaningful regime dimensions. Context for Jev, not rules."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["regime.v1"] = "regime.v1"
    direction: Literal["UPTREND", "DOWNTREND", "RANGE"] | None = None
    volatility: Literal["EXPANDING", "NORMAL", "COMPRESSING"] | None = None
    liquidity: Literal["NORMAL", "STRESSED"] | None = None


class IntelligenceContext(BaseModel):
    """Compact derived context. Not a trading signal soup."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["intel.v1"] = "intel.v1"
    price_vs_vwap_1m_bps: float | None = None
    price_vs_vwap_5m_bps: float | None = None
    vol_ratio_short_long: float | None = None
    flow_accel_5s_vs_30s: float | None = None
    flow_accel_30s_vs_1m: float | None = None
    imbalance_1_change: float | None = None
    expected_exit_spread_bps: float | None = None
    regime: Regime = Field(default_factory=Regime)


class PositionState(BaseModel):
    """Compact position view for Jev. Marked at the conservative executable price.

    LONG is marked at best bid (exitable now). SHORT is marked at best ask.
    MFE/MAE are in bps versus entry on that same mark.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    exchange: str
    side: Side = Side.FLAT
    quantity: float = 0.0
    entry_price: float | None = None
    current_price: float | None = None
    entry_timestamp: str | None = None
    entry_notional: float | None = None
    age_seconds: float = 0.0
    unrealized_pnl: float | None = None
    unrealized_pnl_bps: float | None = None
    realized_pnl: float = 0.0
    gross_pnl: float | None = None
    fees_paid: float = 0.0
    net_pnl: float | None = None
    max_favorable_excursion_bps: float | None = None
    max_adverse_excursion_bps: float | None = None
    max_favorable_excursion: float | None = None
    max_adverse_excursion: float | None = None
    profit_giveback_bps: float | None = None


class MarketHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    trading_allowed: bool
    stale: bool
    warmup_ready: bool
    book_age_ms: int | None = None
    trade_age_ms: int | None = None
    mark_age_ms: int | None = None
    ws_public_connected: bool = False
    ws_market_connected: bool = False
    reasons: tuple[str, ...] = ()


class CrossAssetContext(BaseModel):
    """Optional cross-market context. Empty in V1; schema reserved for later."""

    model_config = ConfigDict(frozen=True, extra="allow")

    btc: dict[str, float] | None = None
    eth: dict[str, float] | None = None
    market: dict[str, float] | None = None


class MarketState(BaseModel):
    """Compact, instrument-tagged snapshot consumed by Jev in later phases."""

    model_config = ConfigDict(frozen=True)

    timestamp: str
    timestamp_ms: int
    symbol: str
    exchange: str
    price: float | None = None
    last_trade_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    returns: Returns = Field(default_factory=Returns)
    momentum: Momentum = Field(default_factory=Momentum)
    microstructure: Microstructure = Field(default_factory=Microstructure)
    volatility: Volatility = Field(default_factory=Volatility)
    derivatives: Derivatives = Field(default_factory=Derivatives)
    ranges: Ranges = Field(default_factory=Ranges)
    intelligence: IntelligenceContext = Field(default_factory=IntelligenceContext)
    position: PositionState
    health: MarketHealth
    context: CrossAssetContext | None = None
    feature_schema_version: Literal["ms.v1", "ms.v2"] = "ms.v2"

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


RETURN_WINDOWS_MS: Mapping[str, int] = {
    "r_5s": 5_000,
    "r_15s": 15_000,
    "r_30s": 30_000,
    "r_1m": 60_000,
    "r_3m": 180_000,
    "r_5m": 300_000,
    "r_15m": 900_000,
    "r_1h": 3_600_000,
}

RV_WINDOWS_MS: Mapping[str, int] = {
    "rv_30s": 30_000,
    "rv_1m": 60_000,
    "rv_5m": 300_000,
    "rv_15m": 900_000,
}

FLOW_WINDOWS_MS: Mapping[str, int] = {
    "aggressive_flow_5s": 5_000,
    "aggressive_flow_30s": 30_000,
    "aggressive_flow_1m": 60_000,
}

CVD_WINDOWS_MS: Mapping[str, int] = {
    "cvd_1m": 60_000,
    "cvd_5m": 300_000,
}

RANGE_WINDOWS_MS: Sequence[int] = (300_000, 900_000, 3_600_000)
PRIMARY_WARMUP_MS = 300_000
