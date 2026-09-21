from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.analytics.counterfactual import hypothetical_side_net_pnl_bps
from src.directional.candles import (
    WINDOW_MS,
    Candle,
    aligned_start,
    candle_from_mids,
    rolling_direction,
    rolling_return_bps,
    streak_hit,
)
from src.directional.constants import HORIZON_LABELS, HORIZON_SECONDS, MAGNITUDE_BUCKETS


def magnitude_bucket(abs_move_bps: float, sigma_bps: float | None) -> str:
    if sigma_bps is None or sigma_bps <= 0:
        return "NORMAL"
    ratio = abs_move_bps / sigma_bps
    if ratio < 0.5:
        return "NOISE"
    if ratio < 1.0:
        return "NORMAL"
    if ratio < 2.0:
        return "LARGE"
    return "EXTREME"


def remaining_segment(remaining_frac: float | None) -> str | None:
    if remaining_frac is None:
        return None
    if remaining_frac > 0.80:
        return "gt_80"
    if remaining_frac >= 0.50:
        return "50_80"
    if remaining_frac >= 0.20:
        return "20_50"
    return "lt_20"


def rolling_outcome(
    *,
    reference_mid: float,
    future_mid: float,
    sigma_bps: float | None,
    path: Sequence[float] = (),
) -> dict[str, Any]:
    direction = rolling_direction(reference_mid=reference_mid, future_mid=future_mid)
    ret = rolling_return_bps(reference_mid=reference_mid, future_mid=future_mid)
    abs_move = abs(ret)
    mfe = 0.0
    mae = 0.0
    for mid in path:
        move = rolling_return_bps(reference_mid=reference_mid, future_mid=mid)
        mfe = max(mfe, move)
        mae = min(mae, move)
    return {
        "direction": direction,
        "future_mid": future_mid,
        "return_bps": ret,
        "abs_move_bps": abs_move,
        "magnitude_bucket": magnitude_bucket(abs_move, sigma_bps),
        "mfe_bps": mfe,
        "mae_bps": mae,
    }


def counterfactual_horizon(
    *,
    predicted_side: str,
    entry_bid: float | None,
    entry_ask: float | None,
    exit_bid: float | None,
    exit_ask: float | None,
    reference_mid: float,
    future_mid: float,
    taker_fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    raw = rolling_return_bps(reference_mid=reference_mid, future_mid=future_mid)
    signed_raw = raw if predicted_side == "UP" else -raw
    trade_side = "LONG" if predicted_side == "UP" else "SHORT"
    net = None
    if None not in {entry_bid, entry_ask, exit_bid, exit_ask}:
        net = hypothetical_side_net_pnl_bps(
            side=trade_side,
            entry_bid=float(entry_bid or 0.0),
            entry_ask=float(entry_ask or 0.0),
            exit_bid=float(exit_bid or 0.0),
            exit_ask=float(exit_ask or 0.0),
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        )
    return {
        "side": trade_side,
        "raw_market_return_bps": signed_raw,
        "net_executable_return_bps": net,
        "direction_correct": (predicted_side == "UP" and raw > 0)
        or (predicted_side == "DOWN" and raw <= 0),
        "correct_but_unprofitable": bool(
            ((predicted_side == "UP" and raw > 0) or (predicted_side == "DOWN" and raw <= 0))
            and net is not None
            and net <= 0
        ),
    }


def next_completed_directions(
    timestamps: Sequence[int],
    values: Sequence[float],
    *,
    timeframe: str,
    after_ms: int,
    count: int,
    now_ms: int,
) -> list[str] | None:
    window_ms = WINDOW_MS[timeframe]
    start = aligned_start(after_ms, window_ms)
    if after_ms >= start:
        start += window_ms
    directions: list[str] = []
    for step in range(count):
        candle_start = start + step * window_ms
        candle = candle_from_mids(
            timestamps,
            values,
            timeframe=timeframe,
            start_ms=candle_start,
            now_ms=now_ms,
        )
        if candle is None or not candle.complete:
            return None
        directions.append(candle.direction)
    return directions


def persistence_hit(directions: Sequence[str] | None, *, count: int, side: str) -> bool | None:
    if directions is None:
        return None
    return streak_hit(directions, count=count, side=side)


def candle_outcome(candle: Candle) -> dict[str, Any]:
    return {
        "window_start": candle.start_ms,
        "window_end": candle.end_ms,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "direction": candle.direction,
    }


def baseline_side(return_value: float | None) -> str:
    if return_value is None or return_value == 0:
        return "DOWN"
    return "UP" if return_value > 0 else "DOWN"


def enrich_forecast_row(
    forecast: dict[str, Any],
    *,
    timestamps: Sequence[int],
    values: Sequence[float],
    now_ms: int,
    reference_mid: float,
    reference_ts: int,
    sigma_bps: dict[str, float | None],
    entry_bid: float | None,
    entry_ask: float | None,
    exit_books: dict[str, tuple[float | None, float | None]],
    taker_fee_bps: float,
    slippage_bps: float,
    returns_at_forecast: dict[str, float | None],
) -> dict[str, Any]:
    rolling: dict[str, Any] = {}
    counterfactuals: dict[str, Any] = {}
    p_up = dict(forecast.get("p_up") or {})
    for label, seconds in zip(HORIZON_LABELS, HORIZON_SECONDS, strict=True):
        target = reference_ts + seconds * 1000
        if now_ms < target:
            continue
        future = _mid_at_or_after(timestamps, values, target)
        if future is None:
            continue
        path = _path(timestamps, values, reference_ts, target)
        outcome = rolling_outcome(
            reference_mid=reference_mid,
            future_mid=future,
            sigma_bps=sigma_bps.get(label),
            path=path,
        )
        rolling[label] = outcome
        predicted = "UP" if float(p_up.get(label) or 0.5) >= 0.5 else "DOWN"
        exit_bid, exit_ask = exit_books.get(label, (None, None))
        counterfactuals[label] = counterfactual_horizon(
            predicted_side=predicted,
            entry_bid=entry_bid,
            entry_ask=entry_ask,
            exit_bid=exit_bid,
            exit_ask=exit_ask,
            reference_mid=reference_mid,
            future_mid=future,
            taker_fee_bps=taker_fee_bps,
            slippage_bps=slippage_bps,
        )
        recent = returns_at_forecast.get("r_1m") if label == "1m" else returns_at_forecast.get(
            {"5m": "r_5m", "10m": "r_10m", "30m": "r_30m"}.get(label, "r_5m")
        )
        momentum = returns_at_forecast.get("r_5s")
        rolling[label]["baselines"] = {
            "fifty_fifty": "UP",
            "recent_direction": baseline_side(recent),
            "momentum": baseline_side(momentum),
        }
    candles: dict[str, Any] = {}
    raw_active = forecast.get("active_candles")
    active = raw_active if isinstance(raw_active, dict) else {}
    candle_up = dict(forecast.get("candle_up") or {})
    for tf in HORIZON_LABELS:
        meta = active.get(tf) if isinstance(active, dict) else None
        start = None if not isinstance(meta, dict) else meta.get("window_start")
        if not isinstance(start, int):
            start = aligned_start(reference_ts, WINDOW_MS[tf])
        candle = candle_from_mids(
            timestamps, values, timeframe=tf, start_ms=int(start), now_ms=now_ms
        )
        if candle is None or not candle.complete:
            continue
        candles[tf] = {
            **candle_outcome(candle),
            "p_up": candle_up.get(tf),
            "remaining_frac": None if not isinstance(meta, dict) else meta.get("remaining_frac"),
            "elapsed_seconds": None if not isinstance(meta, dict) else meta.get("elapsed_seconds"),
            "remaining_seconds": (
                None if not isinstance(meta, dict) else meta.get("remaining_seconds")
            ),
            "remaining_segment": remaining_segment(
                None if not isinstance(meta, dict) else meta.get("remaining_frac")
            ),
        }
    persist_out: dict[str, Any] = {}
    persist = dict(forecast.get("persistence") or {})
    dirs_1m = next_completed_directions(
        timestamps, values, timeframe="1m", after_ms=reference_ts, count=5, now_ms=now_ms
    )
    dirs_5m = next_completed_directions(
        timestamps, values, timeframe="5m", after_ms=reference_ts, count=3, now_ms=now_ms
    )
    for key, count, side, dirs in (
        ("1m_2_up", 2, "UP", dirs_1m),
        ("1m_3_up", 3, "UP", dirs_1m),
        ("1m_5_up", 5, "UP", dirs_1m),
        ("1m_2_down", 2, "DOWN", dirs_1m),
        ("1m_3_down", 3, "DOWN", dirs_1m),
        ("1m_5_down", 5, "DOWN", dirs_1m),
        ("5m_2_up", 2, "UP", dirs_5m),
        ("5m_3_up", 3, "UP", dirs_5m),
        ("5m_2_down", 2, "DOWN", dirs_5m),
        ("5m_3_down", 3, "DOWN", dirs_5m),
    ):
        hit = persistence_hit(dirs, count=count, side=side)
        if hit is None:
            continue
        persist_out[key] = {"hit": hit, "p": persist.get(key)}
    return {
        "rolling": rolling,
        "candles": candles,
        "persistence": persist_out,
        "counterfactuals": counterfactuals,
    }


def _mid_at_or_after(
    timestamps: Sequence[int], values: Sequence[float], target: int
) -> float | None:
    for ts_ms, price in zip(timestamps, values, strict=True):
        if ts_ms >= target and price > 0:
            return price
    return None


def _path(timestamps: Sequence[int], values: Sequence[float], start: int, end: int) -> list[float]:
    return [
        price
        for ts_ms, price in zip(timestamps, values, strict=True)
        if start <= ts_ms <= end and price > 0
    ]


def valid_magnitude(name: str) -> bool:
    return name in MAGNITUDE_BUCKETS
