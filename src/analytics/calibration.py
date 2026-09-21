from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from src.analytics.barriers import (
    AMBIGUOUS,
    DOWN_FIRST,
    LONG_AMBIGUOUS,
    LONG_NEITHER,
    LONG_STOP_FIRST,
    LONG_TARGET_FIRST,
    NEITHER,
    SHORT_AMBIGUOUS,
    SHORT_NEITHER,
    SHORT_STOP_FIRST,
    SHORT_TARGET_FIRST,
    UP_FIRST,
)

CLASSES = (UP_FIRST, DOWN_FIRST, NEITHER)
LONG_CLASSES = (LONG_TARGET_FIRST, LONG_STOP_FIRST, LONG_NEITHER)
SHORT_CLASSES = (SHORT_TARGET_FIRST, SHORT_STOP_FIRST, SHORT_NEITHER)
EV_BINS: tuple[tuple[str, float | None, float | None], ...] = (
    ("lt_0", None, 0.0),
    ("0_1", 0.0, 1.0),
    ("1_2", 1.0, 2.0),
    ("2_4", 2.0, 4.0),
    ("4_plus", 4.0, None),
)
MIN_DASHBOARD_N = 20
PROB_BIN_WIDTH = 0.10
LOG_LOSS_FLOOR = 1e-15


def brier_multiclass(
    predicted: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
    *,
    classes: tuple[str, str, str] = CLASSES,
) -> float | None:
    if not predicted or len(predicted) != len(outcomes):
        return None
    total = 0.0
    n = 0
    for probs_raw, outcome in zip(predicted, outcomes, strict=True):
        if outcome not in classes:
            continue
        probs = _normalize(probs_raw[0], probs_raw[1], probs_raw[2], classes=classes)
        observed = {label: 1.0 if outcome == label else 0.0 for label in classes}
        total += sum((probs[key] - observed[key]) ** 2 for key in classes)
        n += 1
    if n == 0:
        return None
    return total / n


def multiclass_log_loss(
    predicted: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
    *,
    classes: tuple[str, str, str] = CLASSES,
) -> float | None:
    if not predicted or len(predicted) != len(outcomes):
        return None
    total = 0.0
    n = 0
    for probs_raw, outcome in zip(predicted, outcomes, strict=True):
        if outcome not in classes:
            continue
        probs = _normalize(probs_raw[0], probs_raw[1], probs_raw[2], classes=classes)
        p_true = max(LOG_LOSS_FLOOR, probs[outcome])
        total += -math.log(p_true)
        n += 1
    if n == 0:
        return None
    return total / n


def reliability_bins(
    predicted: Sequence[float],
    hits: Sequence[bool],
    *,
    width: float = PROB_BIN_WIDTH,
) -> list[dict[str, float | int | None]]:
    bins: list[dict[str, float | int | None]] = []
    edge = 0.0
    while edge < 1.0 - 1e-12:
        high = min(1.0, edge + width)
        preds: list[float] = []
        actuals: list[bool] = []
        for probability, hit in zip(predicted, hits, strict=True):
            if edge <= probability < high or (high == 1.0 and probability == 1.0):
                preds.append(probability)
                actuals.append(hit)
        count = len(preds)
        bins.append(
            {
                "lo": edge,
                "hi": high,
                "n": count,
                "predicted_avg": None if count == 0 else sum(preds) / count,
                "observed_frequency": None if count == 0 else sum(actuals) / count,
            }
        )
        edge = high
    return bins


def ev_realized_bins(
    predicted_ev: Sequence[float],
    realized_pnl: Sequence[float | None],
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    for label, lo, hi in EV_BINS:
        paired = [
            (pred, realized)
            for pred, realized in zip(predicted_ev, realized_pnl, strict=True)
            if realized is not None and _in_bin(pred, lo, hi)
        ]
        n = len(paired)
        pnls = [item[1] for item in paired]
        wins = [pnl for pnl in pnls if pnl > 0]
        rows.append(
            {
                "bin": label,
                "n": n,
                "avg_predicted_ev": None if n == 0 else sum(item[0] for item in paired) / n,
                "avg_realized_pnl": None if n == 0 else sum(pnls) / n,
                "median_realized_pnl": None if n == 0 else _median(pnls),
                "win_frequency": None if n == 0 else len(wins) / n,
            }
        )
    return rows


def nonoverlapping_indices(timestamps_ms: Sequence[int], *, min_gap_ms: int) -> list[int]:
    chosen: list[int] = []
    last = None
    for index, stamp in enumerate(timestamps_ms):
        if last is None or stamp - last >= min_gap_ms:
            chosen.append(index)
            last = stamp
    return chosen


def summarize_horizon(
    rows: list[dict[str, Any]],
    *,
    min_n: int = MIN_DASHBOARD_N,
    classes: tuple[str, str, str] = CLASSES,
    ambiguous: str = AMBIGUOUS,
    p_target_key: str = "p_up_first",
    p_stop_key: str = "p_down_first",
    p_neither_key: str = "p_neither",
    predicted_ev_key: str = "predicted_entry_net_ev",
    realized_key: str = "realized_selected_net_pnl",
) -> dict[str, Any]:
    class_set = set(classes)
    scored = [row for row in rows if row.get("outcome") in class_set]
    predicted = [
        (
            float(row[p_target_key]),
            float(row[p_stop_key]),
            float(row[p_neither_key]),
        )
        for row in scored
    ]
    outcomes = [str(row["outcome"]) for row in scored]
    n = len(outcomes)
    insufficient = n < min_n
    target_pred = [item[0] for item in predicted]
    stop_pred = [item[1] for item in predicted]
    neither_pred = [item[2] for item in predicted]
    target_label, stop_label, neither_label = classes
    target_block = {
        "predicted_avg": None if n == 0 else sum(target_pred) / n,
        "observed_frequency": None
        if n == 0
        else sum(1 for outcome in outcomes if outcome == target_label) / n,
    }
    stop_block = {
        "predicted_avg": None if n == 0 else sum(stop_pred) / n,
        "observed_frequency": None
        if n == 0
        else sum(1 for outcome in outcomes if outcome == stop_label) / n,
    }
    neither_block = {
        "predicted_avg": None if n == 0 else sum(neither_pred) / n,
        "observed_frequency": None
        if n == 0
        else sum(1 for outcome in outcomes if outcome == neither_label) / n,
    }
    summary: dict[str, Any] = {
        "n": n,
        "n_raw": len(rows),
        "n_valid": n,
        "n_ambiguous": sum(1 for row in rows if row.get("outcome") == ambiguous),
        "insufficient": insufficient,
        "brier": None if insufficient else brier_multiclass(predicted, outcomes, classes=classes),
        "log_loss": None
        if insufficient
        else multiclass_log_loss(predicted, outcomes, classes=classes),
        "target": target_block,
        "stop": stop_block,
        "neither": neither_block,
        "up": target_block,
        "down": stop_block,
        "reliability_target": reliability_bins(
            target_pred, [outcome == target_label for outcome in outcomes]
        )
        if not insufficient
        else [],
        "reliability_stop": reliability_bins(
            stop_pred, [outcome == stop_label for outcome in outcomes]
        )
        if not insufficient
        else [],
        "reliability_down": reliability_bins(
            stop_pred, [outcome == stop_label for outcome in outcomes]
        )
        if not insufficient
        else [],
    }
    predicted_ev = [
        float(row[predicted_ev_key])
        for row in scored
        if row.get(predicted_ev_key) is not None and row.get(realized_key) is not None
    ]
    realized = [
        float(row[realized_key])
        for row in scored
        if row.get(predicted_ev_key) is not None and row.get(realized_key) is not None
    ]
    summary["ev_bins"] = ev_realized_bins(predicted_ev, realized)
    return summary


def summarize_long_race(
    rows: list[dict[str, Any]], *, min_n: int = MIN_DASHBOARD_N
) -> dict[str, Any]:
    return summarize_horizon(
        rows,
        min_n=min_n,
        classes=LONG_CLASSES,
        ambiguous=LONG_AMBIGUOUS,
        p_target_key="p_long_target",
        p_stop_key="p_long_stop",
        p_neither_key="p_long_neither",
        predicted_ev_key="predicted_entry_net_long_bps",
        realized_key="realized_long_net_pnl",
    )


def summarize_short_race(
    rows: list[dict[str, Any]], *, min_n: int = MIN_DASHBOARD_N
) -> dict[str, Any]:
    return summarize_horizon(
        rows,
        min_n=min_n,
        classes=SHORT_CLASSES,
        ambiguous=SHORT_AMBIGUOUS,
        p_target_key="p_short_target",
        p_stop_key="p_short_stop",
        p_neither_key="p_short_neither",
        predicted_ev_key="predicted_entry_net_short_bps",
        realized_key="realized_short_net_pnl",
    )


def _in_bin(value: float, lo: float | None, hi: float | None) -> bool:
    if lo is not None and value < lo:
        return False
    return not (hi is not None and value >= hi)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _normalize(
    p_target: float,
    p_stop: float,
    p_neither: float,
    *,
    classes: tuple[str, str, str],
) -> dict[str, float]:
    total = max(0.0, p_target) + max(0.0, p_stop) + max(0.0, p_neither)
    if total <= 0:
        return {label: 1.0 / 3.0 for label in classes}
    return {
        classes[0]: max(0.0, p_target) / total,
        classes[1]: max(0.0, p_stop) / total,
        classes[2]: max(0.0, p_neither) / total,
    }
