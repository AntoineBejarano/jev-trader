from __future__ import annotations

import bisect

from src.market.models import RETURN_WINDOWS_MS, Returns


def simple_return(current: float, previous: float) -> float | None:
    if previous == 0.0:
        return None
    return current / previous - 1.0


def mid_at_or_before(
    timestamps: list[int],
    values: list[float],
    target_ms: int,
    tolerance_ms: int = 250,
) -> float | None:
    if not timestamps:
        return None
    index = bisect.bisect_right(timestamps, target_ms) - 1
    if index >= 0:
        return values[index]
    if timestamps[0] - target_ms <= tolerance_ms:
        return values[0]
    return None


def compute_returns(
    timestamps: list[int],
    values: list[float],
    now_ms: int,
    current_price: float,
) -> Returns:
    payload: dict[str, float | None] = {}
    for field_name, window_ms in RETURN_WINDOWS_MS.items():
        previous = mid_at_or_before(timestamps, values, now_ms - window_ms)
        payload[field_name] = None if previous is None else simple_return(current_price, previous)
    return Returns(**payload)


def window_high_low(
    timestamps: list[int],
    values: list[float],
    now_ms: int,
    window_ms: int,
) -> tuple[float | None, float | None]:
    cutoff = now_ms - window_ms
    start = bisect.bisect_left(timestamps, cutoff)
    if start >= len(values):
        return None, None
    window = values[start:]
    if not window:
        return None, None
    return max(window), min(window)
