from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from src.analytics.calibration import nonoverlapping_indices, reliability_bins

LOG_LOSS_FLOOR = 1e-15
UP_BINS: tuple[tuple[str, float, float | None], ...] = (
    ("0.50_0.60", 0.50, 0.60),
    ("0.60_0.70", 0.60, 0.70),
    ("0.70_0.80", 0.70, 0.80),
    ("0.80_0.90", 0.80, 0.90),
    ("0.90_plus", 0.90, None),
)


def brier_binary(predicted: Sequence[float], hits: Sequence[bool]) -> float | None:
    if not predicted or len(predicted) != len(hits):
        return None
    total = 0.0
    for probability, hit in zip(predicted, hits, strict=True):
        target = 1.0 if hit else 0.0
        total += (probability - target) ** 2
    return total / len(predicted)


def log_loss_binary(predicted: Sequence[float], hits: Sequence[bool]) -> float | None:
    if not predicted or len(predicted) != len(hits):
        return None
    total = 0.0
    for probability, hit in zip(predicted, hits, strict=True):
        p_true = probability if hit else 1.0 - probability
        total += -math.log(max(LOG_LOSS_FLOOR, min(1.0, p_true)))
    return total / len(predicted)


def directional_accuracy(predicted_up: Sequence[float], actual_up: Sequence[bool]) -> float | None:
    if not predicted_up or len(predicted_up) != len(actual_up):
        return None
    correct = 0
    for probability, hit in zip(predicted_up, actual_up, strict=True):
        guess = probability >= 0.5
        if guess == hit:
            correct += 1
    return correct / len(predicted_up)


def up_probability_bins(
    predicted_up: Sequence[float], actual_up: Sequence[bool]
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    for label, lo, hi in UP_BINS:
        paired = [
            (pred, hit)
            for pred, hit in zip(predicted_up, actual_up, strict=True)
            if pred >= lo and (hi is None or pred < hi)
        ]
        n = len(paired)
        rows.append(
            {
                "bin": label,
                "lo": lo,
                "hi": hi,
                "n": n,
                "predicted_avg": None if n == 0 else sum(item[0] for item in paired) / n,
                "observed_up_frequency": None
                if n == 0
                else sum(1 for item in paired if item[1]) / n,
            }
        )
    return rows


def summarize_binary(
    rows: list[dict[str, Any]],
    *,
    p_key: str,
    hit_key: str,
    timestamp_key: str = "timestamp_ms",
    min_gap_ms: int = 0,
) -> dict[str, Any]:
    scored = [
        row
        for row in rows
        if isinstance(row.get(p_key), int | float) and isinstance(row.get(hit_key), bool)
    ]
    predicted = [float(row[p_key]) for row in scored]
    hits = [bool(row[hit_key]) for row in scored]
    n = len(scored)
    summary: dict[str, Any] = {
        "n": n,
        "n_nonoverlapping": n,
        "brier": None if n == 0 else brier_binary(predicted, hits),
        "log_loss": None if n == 0 else log_loss_binary(predicted, hits),
        "accuracy": None if n == 0 else directional_accuracy(predicted, hits),
        "calibration_bins": [] if n == 0 else up_probability_bins(predicted, hits),
        "reliability": [] if n == 0 else reliability_bins(predicted, hits),
    }
    if min_gap_ms > 0 and n:
        stamps = [int(row.get(timestamp_key) or 0) for row in scored]
        indexes = nonoverlapping_indices(stamps, min_gap_ms=min_gap_ms)
        spaced_p = [predicted[i] for i in indexes]
        spaced_h = [hits[i] for i in indexes]
        summary["n_nonoverlapping"] = len(indexes)
        summary["brier_nonoverlapping"] = brier_binary(spaced_p, spaced_h)
        summary["accuracy_nonoverlapping"] = directional_accuracy(spaced_p, spaced_h)
    return summary


def baseline_accuracy(actual_up: Sequence[bool], predicted_side: Sequence[str]) -> float | None:
    if not actual_up or len(actual_up) != len(predicted_side):
        return None
    correct = 0
    for hit, side in zip(actual_up, predicted_side, strict=True):
        guess = side == "UP"
        if guess == hit:
            correct += 1
    return correct / len(actual_up)
