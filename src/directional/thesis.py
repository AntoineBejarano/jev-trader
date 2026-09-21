from __future__ import annotations

from typing import Any

from src.config.settings import Settings
from src.directional.constants import (
    FEATURE_SCHEMA_VERSION,
    HORIZON_LABELS,
    HORIZON_SECONDS_BY_LABEL,
    QUESTION_VERSION,
)
from src.execution.attribution import profit_giveback_bps


def same_side_probability(p_up: dict[str, Any], side: str, label: str) -> float:
    raw = p_up.get(label)
    p_up_h = 0.5 if not isinstance(raw, int | float) else float(raw)
    p_up_h = min(1.0, max(0.0, p_up_h))
    return p_up_h if side == "UP" else 1.0 - p_up_h


def classify_horizons(
    p_up: dict[str, Any],
    side: str,
    *,
    entry_confidence: float,
    support_confidence: float,
) -> tuple[str, list[str], list[str]]:
    """Primary = longest strong horizon. Distant context cannot become primary later."""
    strong = [
        label
        for label in HORIZON_LABELS
        if same_side_probability(p_up, side, label) >= entry_confidence
    ]
    supporting = [
        label
        for label in HORIZON_LABELS
        if same_side_probability(p_up, side, label) >= support_confidence
    ]
    if strong:
        primary = strong[-1]
    elif supporting:
        primary = supporting[-1]
    else:
        primary = "1m"
    if primary not in supporting:
        supporting = [primary, *supporting]
    context = [label for label in HORIZON_LABELS if label not in supporting]
    return primary, supporting, context


def horizon_weight(label: str, primary: str, supporting: list[str], settings: Settings) -> float:
    if label == primary:
        return settings.dir_horizon_weight_primary
    if label in supporting:
        return settings.dir_horizon_weight_support
    return settings.dir_horizon_weight_context


def elapsed_multiple(age_seconds: float, primary: str) -> float:
    horizon_s = float(HORIZON_SECONDS_BY_LABEL.get(primary) or 60)
    if horizon_s <= 0:
        return 0.0
    return max(0.0, age_seconds / horizon_s)


def capture_ratio(mfe_bps: float, pnl_bps: float) -> float | None:
    if mfe_bps <= 0:
        return None
    return pnl_bps / mfe_bps


def score_thesis(
    p_up: dict[str, Any],
    side: str,
    *,
    primary: str,
    supporting: list[str],
    settings: Settings,
    consensus: str,
    conflict: bool,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Interpretable same-side thesis. PnL is not an input."""
    weights: dict[str, float] = {}
    same: dict[str, float] = {}
    weighted_num = 0.0
    weighted_den = 0.0
    for label in HORIZON_LABELS:
        p_same = same_side_probability(p_up, side, label)
        weight = horizon_weight(label, primary, supporting, settings)
        same[label] = p_same
        weights[label] = weight
        weighted_num += weight * p_same
        weighted_den += weight
    p_weighted = 0.5 if weighted_den <= 0 else weighted_num / weighted_den
    primary_p = same.get(primary, 0.5)
    if consensus == side:
        consensus_adj = 0.03
    elif conflict or consensus == "MIXED":
        consensus_adj = -0.05
    else:
        consensus_adj = 0.0
    strengthening = float(diagnostics.get("momentum_strengthening") or 0.5)
    weakening = float(diagnostics.get("momentum_weakening") or 0.5)
    reversal = float(diagnostics.get("reversal_60s") or 0.5)
    momentum_adj = 0.04 * (strengthening - weakening)
    reversal_adj = -0.12 * max(0.0, reversal - 0.5)
    score = min(1.0, max(0.0, p_weighted + consensus_adj + momentum_adj + reversal_adj))
    return {
        "score": score,
        "p_weighted": p_weighted,
        "primary_p": primary_p,
        "same_side": same,
        "weights": weights,
        "consensus_adj": consensus_adj,
        "momentum_adj": momentum_adj,
        "reversal_adj": reversal_adj,
        "strengthening": strengthening,
        "weakening": weakening,
        "reversal_60s": reversal,
        "consensus": consensus,
        "conflict": conflict,
        "market_character": diagnostics.get("market_character"),
    }


def build_entry_snapshot(
    *,
    side: str,
    p_up: dict[str, Any],
    timestamp_ms: int,
    forecast: dict[str, Any],
    directional_state: dict[str, Any] | None,
    settings: Settings,
    confirmations: int,
) -> dict[str, Any]:
    diagnostics = _diagnostics(forecast)
    regime = _regime(directional_state)
    consensus = str(forecast.get("direction_consensus") or "MIXED")
    conflict = bool(forecast.get("direction_conflict"))
    primary, supporting, context = classify_horizons(
        p_up,
        side,
        entry_confidence=settings.dir_entry_confidence,
        support_confidence=settings.dir_support_confidence,
    )
    scored = score_thesis(
        p_up,
        side,
        primary=primary,
        supporting=supporting,
        settings=settings,
        consensus=consensus,
        conflict=conflict,
        diagnostics=diagnostics,
    )
    persist = forecast.get("persistence")
    persistence = persist if isinstance(persist, dict) else {}
    return {
        "entry_direction": side,
        "entry_timestamp": timestamp_ms,
        "entry_primary_horizon": primary,
        "entry_supporting_horizons": supporting,
        "entry_context_horizons": context,
        "entry_p_up_1m": _f(p_up.get("1m")),
        "entry_p_up_5m": _f(p_up.get("5m")),
        "entry_p_up_10m": _f(p_up.get("10m")),
        "entry_p_up_30m": _f(p_up.get("30m")),
        "entry_consensus": consensus,
        "entry_conflict": conflict,
        "entry_regime_direction": regime.get("direction"),
        "entry_market_character": diagnostics.get("market_character"),
        "entry_persistence": {
            "1m_2_up": _f(persistence.get("1m_2_up")),
            "1m_2_down": _f(persistence.get("1m_2_down")),
            "5m_2_up": _f(persistence.get("5m_2_up")),
            "5m_2_down": _f(persistence.get("5m_2_down")),
        },
        "entry_reversal_60s": scored["reversal_60s"],
        "entry_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "entry_question_version": QUESTION_VERSION,
        "entry_thesis_score": scored["score"],
        "entry_thesis_components": {
            "p_weighted": scored["p_weighted"],
            "primary_p": scored["primary_p"],
            "consensus_adj": scored["consensus_adj"],
            "momentum_adj": scored["momentum_adj"],
            "reversal_adj": scored["reversal_adj"],
        },
        "entry_confirmations": confirmations,
        "primary_horizon_seconds": HORIZON_SECONDS_BY_LABEL[primary],
    }


def current_position_thesis(
    *,
    entry: dict[str, Any],
    p_up: dict[str, Any],
    forecast: dict[str, Any],
    settings: Settings,
    age_seconds: float,
    mfe_bps: float,
    mae_bps: float,
    unrealized_bps: float,
) -> dict[str, Any]:
    side = str(entry.get("entry_direction") or "UP")
    primary = str(entry.get("entry_primary_horizon") or "1m")
    raw_support = entry.get("entry_supporting_horizons")
    supporting = [str(item) for item in raw_support] if isinstance(raw_support, list) else [primary]
    diagnostics = _diagnostics(forecast)
    consensus = str(forecast.get("direction_consensus") or "MIXED")
    conflict = bool(forecast.get("direction_conflict"))
    scored = score_thesis(
        p_up,
        side,
        primary=primary,
        supporting=supporting,
        settings=settings,
        consensus=consensus,
        conflict=conflict,
        diagnostics=diagnostics,
    )
    entry_score = float(entry.get("entry_thesis_score") or scored["score"])
    decay = entry_score - float(scored["score"])
    giveback = profit_giveback_bps(mfe_bps, unrealized_bps)
    multiple = elapsed_multiple(age_seconds, primary)
    return {
        "side": side,
        "primary_horizon": primary,
        "supporting_horizons": supporting,
        "context_horizons": entry.get("entry_context_horizons") or [],
        "entry_p_up": {
            "1m": entry.get("entry_p_up_1m"),
            "5m": entry.get("entry_p_up_5m"),
            "10m": entry.get("entry_p_up_10m"),
            "30m": entry.get("entry_p_up_30m"),
        },
        "current_p_up": {
            "1m": _f(p_up.get("1m")),
            "5m": _f(p_up.get("5m")),
            "10m": _f(p_up.get("10m")),
            "30m": _f(p_up.get("30m")),
        },
        "entry_consensus": entry.get("entry_consensus"),
        "current_consensus": consensus,
        "entry_conflict": entry.get("entry_conflict"),
        "current_conflict": conflict,
        "entry_thesis_score": entry_score,
        "current_thesis_score": scored["score"],
        "thesis_decay": decay,
        "components": scored,
        "primary_horizon_elapsed_multiple": multiple,
        "primary_horizon_seconds": HORIZON_SECONDS_BY_LABEL.get(primary),
        "momentum_strengthening": scored["strengthening"],
        "momentum_weakening": scored["weakening"],
        "reversal_risk": scored["reversal_60s"],
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "unrealized_bps": unrealized_bps,
        "profit_giveback_bps": giveback,
        "age_seconds": age_seconds,
        "stale_horizon": multiple >= settings.dir_stale_horizon_multiple,
    }


def _diagnostics(forecast: dict[str, Any]) -> dict[str, Any]:
    raw = forecast.get("diagnostics")
    return raw if isinstance(raw, dict) else {}


def _regime(directional_state: dict[str, Any] | None) -> dict[str, Any]:
    if not directional_state:
        return {}
    raw = directional_state.get("regime")
    return raw if isinstance(raw, dict) else {}


def _f(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
