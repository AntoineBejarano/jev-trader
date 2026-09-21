from __future__ import annotations

import math

from src.features.returns import mid_at_or_before
from src.market.models import RV_WINDOWS_MS, Volatility


def _log_return(current: float, previous: float) -> float | None:
    if previous <= 0 or current <= 0:
        return None
    return math.log(current / previous)


def realized_variance(log_returns: list[float]) -> float | None:
    if not log_returns:
        return None
    return math.sqrt(sum(value * value for value in log_returns))


def sampled_log_returns(
    timestamps: list[int],
    values: list[float],
    now_ms: int,
    window_ms: int,
    step_ms: int = 1_000,
) -> list[float]:
    cutoff = now_ms - window_ms
    if not timestamps or timestamps[0] > cutoff:
        return []
    previous: float | None = None
    out: list[float] = []
    t = cutoff
    while t <= now_ms:
        price = mid_at_or_before(timestamps, values, t)
        if price is not None and previous is not None:
            ret = _log_return(price, previous)
            if ret is not None:
                out.append(ret)
        if price is not None:
            previous = price
        t += step_ms
    return out


def compute_volatility(
    timestamps: list[int],
    values: list[float],
    now_ms: int,
    *,
    high_5m: float | None,
    low_5m: float | None,
    high_15m: float | None,
    low_15m: float | None,
    current_price: float,
) -> Volatility:
    payload: dict[str, float | None] = {}
    for field_name, window_ms in RV_WINDOWS_MS.items():
        payload[field_name] = realized_variance(
            sampled_log_returns(timestamps, values, now_ms, window_ms)
        )
    range_5m_bps = None
    if high_5m is not None and low_5m is not None and current_price > 0:
        range_5m_bps = ((high_5m - low_5m) / current_price) * 10_000.0
    range_15m_bps = None
    if high_15m is not None and low_15m is not None and current_price > 0:
        range_15m_bps = ((high_15m - low_15m) / current_price) * 10_000.0
    atr_like_5m = None
    if range_5m_bps is not None:
        atr_like_5m = range_5m_bps / 10_000.0
    return Volatility(
        rv_30s=payload.get("rv_30s"),
        rv_1m=payload.get("rv_1m"),
        rv_5m=payload.get("rv_5m"),
        rv_15m=payload.get("rv_15m"),
        range_5m_bps=range_5m_bps,
        range_15m_bps=range_15m_bps,
        atr_like_5m=atr_like_5m,
    )
