from __future__ import annotations

from collections import deque
from statistics import median
from typing import Literal

from src.market.buffer import Trade
from src.market.models import IntelligenceContext, Regime, Returns, Volatility

BPS = 10_000.0
SPREAD_SAMPLE_LIMIT = 120
DIRECTION_TREND_BPS = 8.0
VOL_EXPAND_RATIO = 1.25
VOL_COMPRESS_RATIO = 0.80
LIQUIDITY_STRESS_SPREAD_MULT = 2.5


def vwap_bps(price: float, trades: list[Trade]) -> float | None:
    notional = sum(trade.price * trade.qty for trade in trades)
    qty = sum(trade.qty for trade in trades)
    if qty <= 0 or price <= 0 or notional <= 0:
        return None
    vwap = notional / qty
    return (price - vwap) / price * BPS


def vol_ratio_short_long(vol: Volatility) -> float | None:
    """Short-horizon RV versus 5m RV, both scaled to 1m. >1 expanding."""
    if vol.rv_30s is None or vol.rv_5m is None or vol.rv_5m <= 0:
        return None
    short_1m = vol.rv_30s * (60.0 / 30.0) ** 0.5
    long_1m = vol.rv_5m * (60.0 / 300.0) ** 0.5
    if long_1m <= 0:
        return None
    return float(short_1m / long_1m)


def flow_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def classify_direction(returns: Returns) -> Literal["UPTREND", "DOWNTREND", "RANGE"] | None:
    medium = returns.r_5m
    longer = returns.r_15m if returns.r_15m is not None else returns.r_1h
    if medium is None:
        return None
    medium_bps = medium * BPS
    longer_bps = None if longer is None else longer * BPS
    if medium_bps >= DIRECTION_TREND_BPS and (longer_bps is None or longer_bps >= 0):
        return "UPTREND"
    if medium_bps <= -DIRECTION_TREND_BPS and (longer_bps is None or longer_bps <= 0):
        return "DOWNTREND"
    return "RANGE"


def classify_volatility(
    ratio: float | None,
) -> Literal["EXPANDING", "NORMAL", "COMPRESSING"] | None:
    if ratio is None:
        return None
    if ratio >= VOL_EXPAND_RATIO:
        return "EXPANDING"
    if ratio <= VOL_COMPRESS_RATIO:
        return "COMPRESSING"
    return "NORMAL"


def classify_liquidity(
    *,
    spread_bps: float | None,
    typical_spread_bps: float | None,
    trade_velocity: float | None,
) -> Literal["NORMAL", "STRESSED"] | None:
    if spread_bps is None:
        return None
    stressed = False
    if typical_spread_bps is not None and typical_spread_bps > 0:
        stressed = spread_bps >= typical_spread_bps * LIQUIDITY_STRESS_SPREAD_MULT
    if trade_velocity is not None and trade_velocity <= 0:
        stressed = True
    return "STRESSED" if stressed else "NORMAL"


def rolling_spread_bps(samples: deque[float]) -> float | None:
    if not samples:
        return None
    return float(median(samples))


def build_intelligence(
    *,
    price: float,
    trades_1m: list[Trade],
    trades_5m: list[Trade],
    returns: Returns,
    vol: Volatility,
    spread_bps: float | None,
    spread_samples: deque[float],
    flow_5s: float | None,
    flow_30s: float | None,
    flow_1m: float | None,
    imbalance_1: float | None,
    prev_imbalance_1: float | None,
    trade_velocity: float | None,
) -> IntelligenceContext:
    ratio = vol_ratio_short_long(vol)
    typical = rolling_spread_bps(spread_samples)
    expected_exit = typical if typical is not None else spread_bps
    return IntelligenceContext(
        price_vs_vwap_1m_bps=vwap_bps(price, trades_1m),
        price_vs_vwap_5m_bps=vwap_bps(price, trades_5m),
        vol_ratio_short_long=ratio,
        flow_accel_5s_vs_30s=flow_delta(flow_5s, flow_30s),
        flow_accel_30s_vs_1m=flow_delta(flow_30s, flow_1m),
        imbalance_1_change=flow_delta(imbalance_1, prev_imbalance_1),
        expected_exit_spread_bps=expected_exit,
        regime=Regime(
            direction=classify_direction(returns),
            volatility=classify_volatility(ratio),
            liquidity=classify_liquidity(
                spread_bps=spread_bps,
                typical_spread_bps=typical,
                trade_velocity=trade_velocity,
            ),
        ),
    )
