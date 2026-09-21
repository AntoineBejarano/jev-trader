from __future__ import annotations

BPS = 10_000.0
UP_FIRST = "UP_FIRST"
DOWN_FIRST = "DOWN_FIRST"
NEITHER = "NEITHER"
AMBIGUOUS = "AMBIGUOUS"
LONG_TARGET_FIRST = "LONG_TARGET_FIRST"
LONG_STOP_FIRST = "LONG_STOP_FIRST"
LONG_NEITHER = "LONG_NEITHER"
LONG_AMBIGUOUS = "LONG_AMBIGUOUS"
SHORT_TARGET_FIRST = "SHORT_TARGET_FIRST"
SHORT_STOP_FIRST = "SHORT_STOP_FIRST"
SHORT_NEITHER = "SHORT_NEITHER"
SHORT_AMBIGUOUS = "SHORT_AMBIGUOUS"


def signed_return_bps(reference: float, price: float) -> float:
    if reference <= 0:
        return 0.0
    return (price / reference - 1.0) * BPS


def _crossed(previous: float, current: float, level: float) -> bool:
    return (previous < level <= current) or (previous > level >= current)


def barrier_first_touch(
    *,
    reference_mid: float,
    path: tuple[tuple[int, float], ...],
    start_ms: int,
    horizon_ms: int,
    target_bps: float | None = None,
    up_bps: float | None = None,
    down_bps: float | None = None,
    up_label: str = UP_FIRST,
    down_label: str = DOWN_FIRST,
    neither_label: str = NEITHER,
    ambiguous_label: str = AMBIGUOUS,
) -> str:
    """Classify which signed level was reached first on the MID path.

    Default labels are the symmetric UP/DOWN names. Side-specific races pass
    +target vs -stop (LONG) or +stop vs -target (SHORT) with their own labels.

    AMBIGUOUS: both levels are crossed inside one sample interval.
    """
    resolved_up = target_bps if up_bps is None else up_bps
    resolved_down = -target_bps if down_bps is None and target_bps is not None else down_bps
    if (
        reference_mid <= 0
        or resolved_up is None
        or resolved_down is None
        or resolved_up <= resolved_down
    ):
        return ambiguous_label
    up_level = resolved_up
    down_level = resolved_down
    deadline = start_ms + horizon_ms
    previous_ts = start_ms
    previous_bps = 0.0
    started = False
    for ts_ms, mid in path:
        if ts_ms < start_ms:
            continue
        current_bps = signed_return_bps(reference_mid, mid)
        if not started:
            started = True
            if ts_ms == start_ms:
                previous_ts = ts_ms
                previous_bps = current_bps
                continue
        if ts_ms > deadline:
            span = ts_ms - previous_ts
            if previous_ts < deadline <= ts_ms and span > 0:
                frac = (deadline - previous_ts) / span
                current_bps = previous_bps + frac * (current_bps - previous_bps)
            else:
                break
            ts_ms = deadline
        hit_up = _crossed(previous_bps, current_bps, up_level)
        hit_down = _crossed(previous_bps, current_bps, down_level)
        if hit_up and hit_down:
            return ambiguous_label
        if hit_up:
            return up_label
        if hit_down:
            return down_label
        previous_ts = ts_ms
        previous_bps = current_bps
        if ts_ms >= deadline:
            break
    return neither_label


def long_barrier_outcome(
    *,
    reference_mid: float,
    path: tuple[tuple[int, float], ...],
    target_bps: float,
    stop_bps: float,
    start_ms: int,
    horizon_ms: int,
) -> str:
    """LONG race: +target vs -stop."""
    return barrier_first_touch(
        reference_mid=reference_mid,
        path=path,
        start_ms=start_ms,
        horizon_ms=horizon_ms,
        up_bps=target_bps,
        down_bps=-stop_bps,
        up_label=LONG_TARGET_FIRST,
        down_label=LONG_STOP_FIRST,
        neither_label=LONG_NEITHER,
        ambiguous_label=LONG_AMBIGUOUS,
    )


def short_barrier_outcome(
    *,
    reference_mid: float,
    path: tuple[tuple[int, float], ...],
    target_bps: float,
    stop_bps: float,
    start_ms: int,
    horizon_ms: int,
) -> str:
    """SHORT race: -target vs +stop."""
    return barrier_first_touch(
        reference_mid=reference_mid,
        path=path,
        start_ms=start_ms,
        horizon_ms=horizon_ms,
        up_bps=stop_bps,
        down_bps=-target_bps,
        up_label=SHORT_STOP_FIRST,
        down_label=SHORT_TARGET_FIRST,
        neither_label=SHORT_NEITHER,
        ambiguous_label=SHORT_AMBIGUOUS,
    )
