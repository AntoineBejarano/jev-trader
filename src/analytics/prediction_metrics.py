# ruff: noqa: E501
"""Pure directional prediction metrics. No execution, fees, or risk."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from src.analytics.audit_math import (
    CONFIDENCE_BINS,
    PROB_BINS,
    ece_binary,
    in_bin,
    mean,
    predicted_side_and_confidence,
    quantile,
    signed_future_return,
)
from src.analytics.calibration import nonoverlapping_indices
from src.directional.calibration import brier_binary, directional_accuracy, log_loss_binary

LOG_LOSS_FLOOR = 1e-15
MIN_INFERENTIAL_N = 30
MAX_ASSOCIATION_LATENCY_MS = 5_000.0
BOOTSTRAP_N_FULL = 400
BOOTSTRAP_N_DASH = 80
BOOTSTRAP_SEED = 42
EPISODE_THRESHOLDS = (0.60, 0.65, 0.70, 0.75)
ROLLING_WINDOWS_MS = {
    "last_1h": 3_600_000,
    "last_6h": 21_600_000,
    "last_24h": 86_400_000,
    "full": None,
}
HORIZON_GAP_MS = {"1m": 60_000, "5m": 300_000, "10m": 600_000, "30m": 1_800_000}

# Expanding-prior climatology: P(UP) using only outcomes strictly before the sample.
CLIMATOLOGY_DEFINITION = (
    "expanding_prior: climatology p_up at t is the mean of matured UP labels with "
    "timestamp < t. The first sample uses 0.5. No future class balance is used."
)


def balanced_accuracy(predicted_up: Sequence[bool], actual_up: Sequence[bool]) -> float | None:
    if not predicted_up or len(predicted_up) != len(actual_up):
        return None
    tp = fp = tn = fn = 0
    for guess, hit in zip(predicted_up, actual_up, strict=True):
        if guess and hit:
            tp += 1
        elif guess and not hit:
            fp += 1
        elif not guess and not hit:
            tn += 1
        else:
            fn += 1
    tpr = None if (tp + fn) == 0 else tp / (tp + fn)
    tnr = None if (tn + fp) == 0 else tn / (tn + fp)
    if tpr is None or tnr is None:
        return None
    return (tpr + tnr) / 2.0


def confusion(predicted_up: Sequence[bool], actual_up: Sequence[bool]) -> dict[str, float | None]:
    tp = fp = tn = fn = 0
    for guess, hit in zip(predicted_up, actual_up, strict=True):
        if guess and hit:
            tp += 1
        elif guess and not hit:
            fp += 1
        elif not guess and not hit:
            tn += 1
        else:
            fn += 1
    return {
        "up_precision": None if (tp + fp) == 0 else tp / (tp + fp),
        "up_recall": None if (tp + fn) == 0 else tp / (tp + fn),
        "down_precision": None if (tn + fn) == 0 else tn / (tn + fn),
        "down_recall": None if (tn + fp) == 0 else tn / (tn + fp),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    if not scores or len(scores) != len(labels):
        return None
    n_pos = sum(1 for label in labels if label)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for pos in range(index, end):
            if ordered[pos][1]:
                rank_sum += avg_rank
        index = end
    u_stat = rank_sum - n_pos * (n_pos + 1) / 2.0
    return u_stat / (n_pos * n_neg)


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        avg = (index + end + 1) / 2.0
        for pos in range(index, end):
            out[indexed[pos][0]] = avg
        index = end
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(_ranks(xs), _ranks(ys))


def expanding_climatology(actual_up: Sequence[bool]) -> list[float]:
    """Point-in-time P(UP) using only earlier labels. First sample is 0.5."""
    out: list[float] = []
    seen_up = 0
    for index, hit in enumerate(actual_up):
        out.append(0.5 if index == 0 else seen_up / index)
        if hit:
            seen_up += 1
    return out


def hard_from_probability(values: Sequence[float]) -> list[bool]:
    return [value >= 0.5 for value in values]


def brier_skill_score(model_brier: float | None, ref_brier: float | None) -> float | None:
    if model_brier is None or ref_brier is None or ref_brier == 0.0:
        return None
    return 1.0 - model_brier / ref_brier


def _block_starts(n: int, block: int, rng: random.Random) -> list[int]:
    starts: list[int] = []
    filled = 0
    while filled < n:
        starts.append(rng.randrange(0, n))
        filled += block
    return starts


def block_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = BOOTSTRAP_N_FULL,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    n = len(values)
    if n == 0:
        return None, None
    if n == 1:
        return float(values[0]), float(values[0])
    block = max(1, min(8, n // 8 or 1))
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample: list[float] = []
        for start in _block_starts(n, block, rng):
            sample.extend(values[start : start + block])
            if len(sample) >= n:
                break
        sample = sample[:n]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    lo_i = int(0.025 * (n_boot - 1))
    hi_i = int(0.975 * (n_boot - 1))
    return stats[lo_i], stats[hi_i]


def block_bootstrap_stat_ci(
    items: Sequence[Any],
    stat: Any,
    *,
    n_boot: int = BOOTSTRAP_N_FULL,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    n = len(items)
    if n == 0:
        return None, None
    block = max(1, min(8, n // 8 or 1))
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample: list[Any] = []
        for start in _block_starts(n, block, rng):
            sample.extend(items[start : start + block])
            if len(sample) >= n:
                break
        sample = sample[:n]
        value = stat(sample)
        if value is not None:
            stats.append(float(value))
    if not stats:
        return None, None
    stats.sort()
    lo_i = int(0.025 * (len(stats) - 1))
    hi_i = int(0.975 * (len(stats) - 1))
    return stats[lo_i], stats[hi_i]


def research_status(
    *,
    n_nonoverlap: int,
    auc: float | None,
    auc_lo: float | None,
    brier_skill: float | None,
    delta_vs_momentum: float | None,
) -> str:
    if n_nonoverlap < MIN_INFERENTIAL_N:
        return "INSUFFICIENT_SAMPLE"
    auc_point = auc is not None and auc > 0.5
    auc_ci = auc_lo is not None and auc_lo > 0.5
    bss_ok = brier_skill is not None and brier_skill > 0.0
    mom_ok = delta_vs_momentum is not None and delta_vs_momentum > 0.0
    if auc_point and bss_ok and mom_ok and auc_ci:
        return "MEASURED_LIFT_PENDING_REPLICATION"
    if auc_point and (bss_ok or mom_ok):
        return "WEAK_MEASURED_LIFT"
    return "NO_MEASURED_LIFT"


def classification_bundle(
    predicted: Sequence[float],
    actual_up: Sequence[bool],
) -> dict[str, Any]:
    guesses = hard_from_probability(predicted)
    conf = confusion(guesses, actual_up)
    n = len(predicted)
    up_n = sum(1 for hit in actual_up if hit)
    return {
        "n": n,
        "up_base_rate": None if n == 0 else up_n / n,
        "down_base_rate": None if n == 0 else 1.0 - up_n / n,
        "accuracy": directional_accuracy(predicted, actual_up),
        "balanced_accuracy": balanced_accuracy(guesses, actual_up),
        "roc_auc": roc_auc(predicted, actual_up),
        "brier": brier_binary(predicted, actual_up),
        "log_loss": log_loss_binary(predicted, actual_up),
        "ece": ece_binary(list(predicted), list(actual_up), PROB_BINS),
        "mean_p_up": mean(list(predicted)),
        "realized_up_rate": None if n == 0 else up_n / n,
        **conf,
    }


def evaluate_stream(
    samples: list[dict[str, Any]],
    *,
    min_gap_ms: int,
    n_boot: int = BOOTSTRAP_N_FULL,
) -> dict[str, Any]:
    predicted = [float(item["p_up"]) for item in samples]
    actual = [bool(item["actual_up"]) for item in samples]
    stamps = [int(item["timestamp_ms"]) for item in samples]
    overlapping = classification_bundle(predicted, actual)
    indexes = nonoverlapping_indices(stamps, min_gap_ms=min_gap_ms)
    spaced = [samples[i] for i in indexes]
    spaced_p = [float(item["p_up"]) for item in spaced]
    spaced_y = [bool(item["actual_up"]) for item in spaced]
    research = classification_bundle(spaced_p, spaced_y)
    clim = expanding_climatology(spaced_y)
    clim_brier = brier_binary(clim, spaced_y)
    clim_log = log_loss_binary(clim, spaced_y)
    clim_guess = hard_from_probability(clim)
    fifty = [0.5] * len(spaced_p)
    mom_acc = _hard_accuracy(spaced, "momentum")
    recent_acc = _hard_accuracy(spaced, "recent_direction")
    jev_acc = research["accuracy"]
    bss = brier_skill_score(research["brier"], clim_brier)
    signed = [float(item["signed_move_bps"]) for item in spaced if item.get("signed_move_bps") is not None]
    returns = [float(item["return_bps"]) for item in spaced if item.get("return_bps") is not None]
    edges = [float(item["p_up"]) - 0.5 for item in spaced if item.get("return_bps") is not None]
    paired_ret = [float(item["return_bps"]) for item in spaced if item.get("return_bps") is not None]
    acc_hits = [1.0 if (item["p_up"] >= 0.5) == item["actual_up"] else 0.0 for item in spaced]
    bal_values = _per_sample_balanced_proxy(spaced)
    auc_ci: tuple[float | None, float | None]
    bss_ci: tuple[float | None, float | None]
    acc_ci: tuple[float | None, float | None]
    bal_ci: tuple[float | None, float | None]
    if n_boot <= 0:
        auc_ci = (None, None)
        bss_ci = (None, None)
        acc_ci = (None, None)
        bal_ci = (None, None)
    else:
        auc_ci = block_bootstrap_stat_ci(
            list(zip(spaced_p, spaced_y, strict=True)),
            lambda pairs: roc_auc([p for p, _y in pairs], [y for _p, y in pairs]),
            n_boot=n_boot,
        )
        bss_ci = block_bootstrap_stat_ci(
            list(zip(spaced_p, clim, spaced_y, strict=True)),
            _bss_from_triples,
            n_boot=n_boot,
        )
        acc_ci = block_bootstrap_mean_ci(acc_hits, n_boot=n_boot)
        bal_ci = block_bootstrap_mean_ci(bal_values, n_boot=n_boot) if bal_values else (None, None)
    return {
        "overlapping_descriptive": overlapping,
        "nonoverlapping_research": {
            **research,
            "n_nonoverlapping": len(spaced),
            "climatology_brier": clim_brier,
            "climatology_log_loss": clim_log,
            "climatology_accuracy": directional_accuracy(clim, spaced_y),
            "climatology_balanced_accuracy": balanced_accuracy(clim_guess, spaced_y),
            "brier_skill_score": bss,
            "baseline_50_50_accuracy": 0.5,
            "baseline_50_50_brier": brier_binary(fifty, spaced_y),
            "baseline_momentum_accuracy": mom_acc,
            "baseline_recent_return_sign_accuracy": recent_acc,
            "delta_accuracy_vs_50_50": None if jev_acc is None else jev_acc - 0.5,
            "delta_accuracy_vs_climatology": _delta(jev_acc, directional_accuracy(clim, spaced_y)),
            "delta_accuracy_vs_momentum": _delta(jev_acc, mom_acc),
            "pearson_edge_vs_return": pearson(edges, paired_ret),
            "spearman_edge_vs_return": spearman(edges, paired_ret),
            "avg_signed_move_bps": mean(signed),
            "median_signed_move_bps": quantile(signed, 0.50),
            "avg_return_bps": mean(returns),
            "ci95": {
                "accuracy": {"lo": acc_ci[0], "hi": acc_ci[1]},
                "balanced_accuracy": {"lo": bal_ci[0], "hi": bal_ci[1]},
                "roc_auc": {"lo": auc_ci[0], "hi": auc_ci[1]},
                "brier_skill_score": {"lo": bss_ci[0], "hi": bss_ci[1]},
            },
        },
        "climatology_definition": CLIMATOLOGY_DEFINITION,
    }


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _hard_accuracy(samples: list[dict[str, Any]], key: str) -> float | None:
    if not samples:
        return None
    correct = 0
    for item in samples:
        side = str(item.get(key) or "DOWN")
        guess = side == "UP"
        if guess == bool(item["actual_up"]):
            correct += 1
    return correct / len(samples)


def _bss_from_triples(triples: Sequence[tuple[float, float, bool]]) -> float | None:
    predicted = [item[0] for item in triples]
    clim = [item[1] for item in triples]
    actual = [item[2] for item in triples]
    return brier_skill_score(brier_binary(predicted, actual), brier_binary(clim, actual))


def _per_sample_balanced_proxy(samples: list[dict[str, Any]]) -> list[float]:
    """0/1 hits; CI of their mean is not balanced-accuracy, used only if class-wise CI is empty."""
    return [1.0 if (item["p_up"] >= 0.5) == item["actual_up"] else 0.0 for item in samples]


def calibration_bins(samples: list[dict[str, Any]], *, min_gap_ms: int) -> list[dict[str, Any]]:
    stamps = [int(item["timestamp_ms"]) for item in samples]
    keep = set(nonoverlapping_indices(stamps, min_gap_ms=min_gap_ms))
    rows: list[dict[str, Any]] = []
    for index, (label, lo, hi) in enumerate(PROB_BINS):
        last = index == len(PROB_BINS) - 1
        group = [item for item in samples if in_bin(float(item["p_up"]), lo, hi, last=last)]
        spaced = [item for i, item in enumerate(samples) if i in keep and in_bin(float(item["p_up"]), lo, hi, last=last)]
        rets = [float(item["return_bps"]) for item in group if item.get("return_bps") is not None]
        n = len(group)
        avg_p = mean([float(item["p_up"]) for item in group])
        freq = None if n == 0 else sum(1 for item in group if item["actual_up"]) / n
        rows.append(
            {
                "bin": label,
                "n": n,
                "n_nonoverlapping": len(spaced),
                "mean_predicted_p_up": avg_p,
                "realized_up_frequency": freq,
                "absolute_calibration_error": None if avg_p is None or freq is None else abs(avg_p - freq),
                "avg_future_return_bps": mean(rets),
                "median_future_return_bps": quantile(rets, 0.50),
            }
        )
    return rows


def confidence_buckets(samples: list[dict[str, Any]], *, min_gap_ms: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (label, lo, hi) in enumerate(CONFIDENCE_BINS):
        last = index == len(CONFIDENCE_BINS) - 1
        group = [item for item in samples if in_bin(float(item["confidence"]), lo, hi, last=last)]
        stamps = [int(item["timestamp_ms"]) for item in group]
        n_star = len(nonoverlapping_indices(stamps, min_gap_ms=min_gap_ms))
        predicted = [float(item["p_up"]) for item in group]
        actual = [bool(item["actual_up"]) for item in group]
        signed = [float(item["signed_move_bps"]) for item in group if item.get("signed_move_bps") is not None]
        guesses = hard_from_probability(predicted)
        rows.append(
            {
                "bucket": label,
                "n": len(group),
                "n_nonoverlapping": n_star,
                "low_sample": n_star < MIN_INFERENTIAL_N,
                "accuracy": directional_accuracy(predicted, actual),
                "balanced_accuracy": balanced_accuracy(guesses, actual),
                "avg_signed_move_bps": mean(signed),
                "median_signed_move_bps": quantile(signed, 0.50),
            }
        )
    return rows


def segment_episodes(
    samples: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    ordered = sorted(samples, key=lambda item: int(item["timestamp_ms"]))
    for item in ordered:
        side, conf = predicted_side_and_confidence(float(item["p_up"]))
        active = conf >= threshold
        if current is None:
            if active:
                current = _new_episode(item, side, conf, threshold)
            continue
        if active and side == current["side"]:
            current["end_ms"] = int(item["timestamp_ms"])
            current["n_ticks"] += 1
            current["conf_sum"] += conf
            current["max_confidence"] = max(float(current["max_confidence"]), conf)
            continue
        episodes.append(_close_episode(current, item))
        current = _new_episode(item, side, conf, threshold) if active else None
    if current is not None:
        episodes.append(_close_episode(current, None))
    return episodes


def _new_episode(item: dict[str, Any], side: str, conf: float, threshold: float) -> dict[str, Any]:
    return {
        "side": side,
        "threshold": threshold,
        "start_ms": int(item["timestamp_ms"]),
        "end_ms": int(item["timestamp_ms"]),
        "n_ticks": 1,
        "conf_sum": conf,
        "max_confidence": conf,
        "start_p_up": float(item["p_up"]),
        "start_outcomes": dict(item.get("horizon_outcomes") or {}),
    }


def _close_episode(current: dict[str, Any], _next: dict[str, Any] | None) -> dict[str, Any]:
    duration = max(0, int(current["end_ms"]) - int(current["start_ms"]))
    mean_conf = current["conf_sum"] / max(1, int(current["n_ticks"]))
    outcomes = dict(current.get("start_outcomes") or {})
    row = {
        "side": current["side"],
        "threshold": current["threshold"],
        "start_ms": current["start_ms"],
        "end_ms": current["end_ms"],
        "duration_ms": duration,
        "duration_seconds": duration / 1000.0,
        "n_ticks": current["n_ticks"],
        "max_confidence": current["max_confidence"],
        "mean_confidence": mean_conf,
    }
    for label in ("1m", "5m", "10m", "30m"):
        block = dict(outcomes.get(label) or {})
        row[f"future_direction_{label}"] = block.get("direction")
        signed = None
        future = block.get("future_mid")
        ref = block.get("reference_mid")
        if isinstance(future, int | float) and isinstance(ref, int | float) and ref > 0:
            p_up = 0.7 if current["side"] == "UP" else 0.3
            signed = signed_future_return(p_up=p_up, reference_mid=float(ref), future_mid=float(future)) * 10_000.0
        row[f"signed_return_bps_{label}"] = signed
    return row


def scorecard_row(horizon: str, research: dict[str, Any]) -> dict[str, Any]:
    ci = dict(research.get("ci95") or {})
    auc_ci = dict(ci.get("roc_auc") or {})
    status = research_status(
        n_nonoverlap=int(research.get("n_nonoverlapping") or 0),
        auc=research.get("roc_auc"),
        auc_lo=auc_ci.get("lo"),
        brier_skill=research.get("brier_skill_score"),
        delta_vs_momentum=research.get("delta_accuracy_vs_momentum"),
    )
    return {
        "horizon": horizon,
        "n_star": research.get("n_nonoverlapping"),
        "up_base_rate": research.get("up_base_rate"),
        "accuracy": research.get("accuracy"),
        "balanced_accuracy": research.get("balanced_accuracy"),
        "roc_auc": research.get("roc_auc"),
        "brier": research.get("brier"),
        "brier_skill": research.get("brier_skill_score"),
        "log_loss": research.get("log_loss"),
        "ece": research.get("ece"),
        "momentum_accuracy": research.get("baseline_momentum_accuracy"),
        "delta_vs_momentum": research.get("delta_accuracy_vs_momentum"),
        "status": status,
        "accuracy_ci95": ci.get("accuracy"),
        "balanced_accuracy_ci95": ci.get("balanced_accuracy"),
        "roc_auc_ci95": ci.get("roc_auc"),
        "brier_skill_ci95": ci.get("brier_skill_score"),
    }
