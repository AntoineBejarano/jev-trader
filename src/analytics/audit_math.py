# ruff: noqa: E501
"""Read-only DIRECTIONAL audit math. No trading side effects."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from src.analytics.calibration import nonoverlapping_indices
from src.directional.calibration import brier_binary, directional_accuracy, log_loss_binary
from src.directional.candles import WINDOW_MS, aligned_start
from src.execution.attribution import attribute_round_trip
from src.execution.pricing import long_gross_pnl, short_gross_pnl
from src.market.models import Side

AUDIT_TOLERANCE_USDT = 1e-6
CONFIDENCE_BINS: tuple[tuple[str, float, float], ...] = (
    ("0.50-0.55", 0.50, 0.55),
    ("0.55-0.60", 0.55, 0.60),
    ("0.60-0.65", 0.60, 0.65),
    ("0.65-0.70", 0.65, 0.70),
    ("0.70-0.75", 0.70, 0.75),
    ("0.75-0.80", 0.75, 0.80),
    ("0.80-0.90", 0.80, 0.90),
    ("0.90-1.00", 0.90, 1.00),
)
PROB_BINS: tuple[tuple[str, float, float], ...] = tuple(
    (f"{lo:.2f}-{lo + 0.10:.2f}", lo, lo + 0.10) for lo in (i / 10 for i in range(10))
)
HORIZON_GAP_MS = {"1m": 60_000, "5m": 300_000, "10m": 600_000, "30m": 1_800_000}
LATENCY_BUCKETS: tuple[tuple[str, float, float | None], ...] = (
    ("lt_1s", 0.0, 1_000.0),
    ("1_2s", 1_000.0, 2_000.0),
    ("2_5s", 2_000.0, 5_000.0),
    ("5_10s", 5_000.0, 10_000.0),
    ("10_30s", 10_000.0, 30_000.0),
    ("30s_plus", 30_000.0, None),
)
LOW_SAMPLE_N = 30


def as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def percentiles(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": None if not values else max(values),
        "n": float(len(values)),
    }


def predicted_side_and_confidence(p_up: float) -> tuple[str, float]:
    if p_up >= 0.5:
        return "UP", p_up
    return "DOWN", 1.0 - p_up


def signed_future_return(*, p_up: float, reference_mid: float, future_mid: float) -> float:
    """Forecast-oriented return. UP uses future/ref-1; DOWN uses ref/future-1."""
    if reference_mid <= 0 or future_mid <= 0:
        return 0.0
    if p_up >= 0.5:
        return future_mid / reference_mid - 1.0
    return reference_mid / future_mid - 1.0


def signed_excursions(
    *,
    predicted_side: str,
    mfe_up_bps: float | None,
    mae_up_bps: float | None,
) -> tuple[float | None, float | None]:
    """Stored rolling MFE/MAE are mid-path vs reference (UP-signed). Flip for DOWN."""
    if mfe_up_bps is None or mae_up_bps is None:
        return None, None
    if predicted_side == "UP":
        return mfe_up_bps, mae_up_bps
    return -mae_up_bps, -mfe_up_bps


def in_bin(value: float, lo: float, hi: float, *, last: bool = False) -> bool:
    if last:
        return lo <= value <= hi
    return lo <= value < hi


def ece_binary(predicted: Sequence[float], hits: Sequence[bool], bins: Sequence[tuple[str, float, float]]) -> float | None:
    if not predicted or len(predicted) != len(hits):
        return None
    n = len(predicted)
    total = 0.0
    for index, (_label, lo, hi) in enumerate(bins):
        last = index == len(bins) - 1
        paired = [
            (pred, hit)
            for pred, hit in zip(predicted, hits, strict=True)
            if in_bin(pred, lo, hi, last=last)
        ]
        if not paired:
            continue
        avg_p = sum(item[0] for item in paired) / len(paired)
        freq = sum(1 for item in paired if item[1]) / len(paired)
        total += abs(avg_p - freq) * (len(paired) / n)
    return total


def expected_round_trip_cost_bps(*, taker_fee_bps: float, slippage_bps: float, spread_bps: float) -> float:
    return 2.0 * taker_fee_bps + 2.0 * slippage_bps + spread_bps


def recover_reference_mids(
    *,
    side: str,
    quantity: float,
    entry_fill: float,
    exit_fill: float,
    entry_slippage: float,
    exit_slippage: float,
    entry_spread: float,
    exit_spread: float,
) -> tuple[float | None, float | None]:
    if quantity <= 0:
        return None, None
    if side == "LONG":
        entry_ask = entry_fill - entry_slippage / quantity
        entry_mid = entry_ask - entry_spread / quantity
        exit_bid = exit_fill + exit_slippage / quantity
        exit_mid = exit_bid + exit_spread / quantity
        return entry_mid, exit_mid
    if side == "SHORT":
        entry_bid = entry_fill + entry_slippage / quantity
        entry_mid = entry_bid + entry_spread / quantity
        exit_ask = exit_fill - exit_slippage / quantity
        exit_mid = exit_ask - exit_spread / quantity
        return entry_mid, exit_mid
    return None, None


def decompose_round_trip(
    *,
    side: str,
    quantity: float,
    entry_fill: float,
    exit_fill: float,
    entry_bid: float | None,
    entry_ask: float | None,
    exit_bid: float | None,
    exit_ask: float | None,
    entry_fee: float,
    exit_fee: float,
    stored_market_move: float | None,
    stored_entry_slip: float,
    stored_exit_slip: float,
    stored_entry_spread: float,
    stored_exit_spread: float,
    stored_gross: float,
    stored_fees: float,
    stored_net: float,
) -> dict[str, Any]:
    books = None not in {entry_bid, entry_ask, exit_bid, exit_ask}
    entry_mid: float | None
    exit_mid: float | None
    if books:
        attr = attribute_round_trip(
            side=Side(side),
            quantity=quantity,
            entry_fill=entry_fill,
            entry_bid=float(entry_bid or 0.0),
            entry_ask=float(entry_ask or 0.0),
            exit_fill=exit_fill,
            exit_bid=float(exit_bid or 0.0),
            exit_ask=float(exit_ask or 0.0),
            entry_fee=entry_fee,
            exit_fee=exit_fee,
        )
        entry_mid = (float(entry_bid or 0.0) + float(entry_ask or 0.0)) / 2.0
        exit_mid = (float(exit_bid or 0.0) + float(exit_ask or 0.0)) / 2.0
        ideal = attr.market_move_pnl
        entry_slip = attr.entry_slippage_cost
        exit_slip = attr.exit_slippage_cost
        spread = attr.total_spread_cost
        fill_gross = attr.gross_pnl
        fees = attr.total_fees
        net = attr.net_pnl
    else:
        recovered = recover_reference_mids(
            side=side,
            quantity=quantity,
            entry_fill=entry_fill,
            exit_fill=exit_fill,
            entry_slippage=stored_entry_slip,
            exit_slippage=stored_exit_slip,
            entry_spread=stored_entry_spread,
            exit_spread=stored_exit_spread,
        )
        entry_mid, exit_mid = recovered
        if stored_market_move is not None:
            ideal = stored_market_move
        elif entry_mid is None or exit_mid is None:
            ideal = 0.0
        elif side == "LONG":
            ideal = (exit_mid - entry_mid) * quantity
        else:
            ideal = (entry_mid - exit_mid) * quantity
        if side == "LONG":
            fill_gross = long_gross_pnl(quantity, entry_fill, exit_fill)
        else:
            fill_gross = short_gross_pnl(quantity, entry_fill, exit_fill)
        entry_slip = stored_entry_slip
        exit_slip = stored_exit_slip
        spread = stored_entry_spread + stored_exit_spread
        fees = stored_fees
        net = stored_net
    execution_vs_mid = spread + entry_slip + exit_slip
    user_slip_identity = abs((ideal - execution_vs_mid) - fill_gross) <= AUDIT_TOLERANCE_USDT
    fee_identity = abs((fill_gross - fees) - net) <= AUDIT_TOLERANCE_USDT
    stored_gross_match = abs(fill_gross - stored_gross) <= AUDIT_TOLERANCE_USDT
    stored_net_match = abs(net - stored_net) <= AUDIT_TOLERANCE_USDT
    return {
        "reference_mid_at_entry": entry_mid,
        "reference_mid_at_exit": exit_mid,
        "ideal_mid_pnl_usdt": ideal,
        "entry_slippage_usdt": entry_slip,
        "exit_slippage_usdt": exit_slip,
        "total_slippage_usdt": entry_slip + exit_slip,
        "total_spread_usdt": spread,
        "execution_vs_mid_usdt": execution_vs_mid,
        "fill_to_fill_gross_pnl_usdt": fill_gross,
        "fees_usdt": fees,
        "net_pnl_usdt": net,
        "identity_ideal_minus_execution_eq_gross": user_slip_identity,
        "identity_gross_minus_fees_eq_net": fee_identity,
        "stored_gross_matches_recompute": stored_gross_match,
        "stored_net_matches_recompute": stored_net_match,
        "books_available": books,
    }


def check_rolling_label(
    *,
    reference_mid: float | None,
    future_mid: float | None,
    stored_direction: str | None,
    stored_return_bps: float | None,
) -> dict[str, Any]:
    issues: list[str] = []
    if reference_mid is None or future_mid is None:
        return {"ok": False, "issues": ["missing_mids"], "expected_direction": None}
    expected = "UP" if future_mid > reference_mid else "DOWN"
    if stored_direction not in {None, expected}:
        issues.append("direction_mismatch")
    if stored_return_bps is not None and reference_mid > 0:
        expected_ret = (future_mid / reference_mid - 1.0) * 10_000.0
        if abs(expected_ret - stored_return_bps) > 1e-4:
            issues.append("return_bps_mismatch")
    return {"ok": not issues, "issues": issues, "expected_direction": expected}


def check_candle_label(outcome: dict[str, Any], *, timeframe: str) -> dict[str, Any]:
    issues: list[str] = []
    open_px = as_float(outcome.get("open"))
    close_px = as_float(outcome.get("close"))
    start = outcome.get("window_start")
    end = outcome.get("window_end")
    direction = outcome.get("direction")
    if open_px is None or close_px is None:
        issues.append("missing_ohlc")
        return {"ok": False, "issues": issues}
    expected = "UP" if close_px > open_px else "DOWN"
    if direction != expected:
        issues.append("direction_mismatch")
    if not isinstance(start, int) or not isinstance(end, int):
        issues.append("missing_window")
    else:
        window = WINDOW_MS[timeframe]
        if start != aligned_start(start, window):
            issues.append("window_not_utc_aligned")
        if end - start != window:
            issues.append("window_width_mismatch")
        if start % window != 0:
            issues.append("window_start_not_epoch_aligned")
    return {"ok": not issues, "issues": issues, "expected_direction": expected}


def vote_from_p_up(p_1m: float, entry_confidence: float) -> str:
    if p_1m >= entry_confidence:
        return "UP"
    if (1.0 - p_1m) >= entry_confidence:
        return "DOWN"
    return "NONE"


def capture_ratio_bps(mfe_bps: float, realized_gross_bps: float) -> float | None:
    if mfe_bps <= 0:
        return None
    return realized_gross_bps / mfe_bps


def summarize_numeric(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "n": float(len(values)),
        "mean": mean(values),
        "median": quantile(values, 0.50),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
    }


def binary_quality(
    predicted: Sequence[float],
    hits: Sequence[bool],
    timestamps_ms: Sequence[int],
    *,
    min_gap_ms: int,
) -> dict[str, Any]:
    n = len(predicted)
    if n == 0:
        return {
            "n": 0,
            "n_nonoverlapping": 0,
            "accuracy": None,
            "brier": None,
            "log_loss": None,
            "ece": None,
            "accuracy_nonoverlapping": None,
            "brier_nonoverlapping": None,
            "log_loss_nonoverlapping": None,
            "ece_nonoverlapping": None,
        }
    spaced = nonoverlapping_indices(timestamps_ms, min_gap_ms=min_gap_ms)
    spaced_p = [predicted[i] for i in spaced]
    spaced_h = [hits[i] for i in spaced]
    return {
        "n": n,
        "n_nonoverlapping": len(spaced),
        "accuracy": directional_accuracy(predicted, hits),
        "brier": brier_binary(predicted, hits),
        "log_loss": log_loss_binary(predicted, hits),
        "ece": ece_binary(predicted, hits, PROB_BINS),
        "accuracy_nonoverlapping": directional_accuracy(spaced_p, spaced_h),
        "brier_nonoverlapping": brier_binary(spaced_p, spaced_h),
        "log_loss_nonoverlapping": log_loss_binary(spaced_p, spaced_h),
        "ece_nonoverlapping": ece_binary(spaced_p, spaced_h, PROB_BINS),
    }


def low_sample(n_independent: int) -> bool:
    return n_independent < LOW_SAMPLE_N
