from __future__ import annotations

import hashlib
import json
from typing import Any

from src.config.settings import Settings, TradingStrategy
from src.directional.constants import (
    FEATURE_SCHEMA_VERSION as DIRECTIONAL_FEATURE_SCHEMA,
)
from src.directional.constants import (
    QUESTION_VERSION as DIRECTIONAL_QUESTION_VERSION,
)
from src.directional.constants import (
    STRATEGY_VERSION as DIRECTIONAL_STRATEGY_VERSION,
)

QUESTION_VERSION = "q.scalp.v3"
MARKET_STATE_SCHEMA_VERSION = "ms.v2"
INTEL_SCHEMA_VERSION = "intel.v1"
REGIME_SCHEMA_VERSION = "regime.v1"
SCORE_MAPPING_VERSION = "jev.score.return.v1"
STRATEGY_VERSION = "scalp.action_values.v2"
ACTION_VALUE_REFERENCE = "MID_INCREMENTAL_BPS_FROM_NOW"

HASH_EXCLUDED_KEYS = frozenset({"jev_resolved_model"})

SECRET_FIELD_MARKERS = (
    "key",
    "secret",
    "password",
    "token",
    "dsn",
    "confirmation",
)


def active_question_version(settings: Settings) -> str:
    if settings.trading_strategy is TradingStrategy.DIRECTIONAL:
        return DIRECTIONAL_QUESTION_VERSION
    return QUESTION_VERSION


def active_strategy_version(settings: Settings) -> str:
    if settings.trading_strategy is TradingStrategy.DIRECTIONAL:
        return DIRECTIONAL_STRATEGY_VERSION
    return STRATEGY_VERSION


def active_feature_schema_version(settings: Settings) -> str:
    if settings.trading_strategy is TradingStrategy.DIRECTIONAL:
        return DIRECTIONAL_FEATURE_SCHEMA
    return MARKET_STATE_SCHEMA_VERSION


def research_fingerprint(
    settings: Settings,
    *,
    resolved_model: str | None,
    question_version: str | None = None,
) -> dict[str, Any]:
    if settings.trading_strategy is TradingStrategy.DIRECTIONAL:
        return directional_fingerprint(settings, resolved_model=resolved_model)
    version = QUESTION_VERSION if question_version is None else question_version
    return {
        "trading_mode": settings.trading_mode.value,
        "trading_strategy": settings.trading_strategy.value,
        "trading_exchange": settings.trading_exchange,
        "trading_symbols": settings.trading_symbols,
        "jev_configured_model": settings.jev_model,
        "jev_resolved_model": resolved_model,
        "question_version": version,
        "market_state_schema_version": MARKET_STATE_SCHEMA_VERSION,
        "intel_schema_version": INTEL_SCHEMA_VERSION,
        "regime_schema_version": REGIME_SCHEMA_VERSION,
        "score_mapping_version": SCORE_MAPPING_VERSION,
        "strategy_version": STRATEGY_VERSION,
        "action_value_reference": ACTION_VALUE_REFERENCE,
        "decision_interval_seconds": settings.decision_interval_seconds,
        "scalp_horizons_seconds": list(settings.scalp_horizons),
        "scalp_min_net_ev_bps": settings.scalp_min_net_ev_bps,
        "scalp_hold_min_net_ev_bps": settings.scalp_hold_min_net_ev_bps,
        "scalp_close_ev_bps": settings.scalp_close_ev_bps,
        "scalp_reverse_advantage_bps": settings.scalp_reverse_advantage_bps,
        "scalp_min_position_age_seconds": settings.scalp_min_position_age_seconds,
        "scalp_min_target_bps": settings.scalp_min_target_bps,
        "scalp_min_stop_bps": settings.scalp_min_stop_bps,
        "scalp_profit_buffer_bps": settings.scalp_profit_buffer_bps,
        "scalp_target_vol_mult": settings.scalp_target_vol_mult,
        "scalp_stop_vol_mult": settings.scalp_stop_vol_mult,
        "execution_taker_fee_bps": settings.execution_taker_fee_bps,
        "execution_maker_fee_bps": settings.execution_maker_fee_bps,
        "execution_slippage_bps": settings.slippage_bps,
        "paper_position_notional": settings.execution_notional,
        "paper_equity": settings.paper_initial_balance,
        "paper_funding_enabled": settings.paper_funding_enabled,
        "experiment_id": settings.experiment_id,
        "experiment_frozen": settings.experiment_frozen,
    }


def directional_fingerprint(
    settings: Settings, *, resolved_model: str | None
) -> dict[str, Any]:
    return {
        "trading_mode": settings.trading_mode.value,
        "trading_strategy": settings.trading_strategy.value,
        "trading_exchange": settings.trading_exchange,
        "trading_symbols": settings.trading_symbols,
        "jev_configured_model": settings.jev_model,
        "jev_resolved_model": resolved_model,
        "question_version": DIRECTIONAL_QUESTION_VERSION,
        "strategy_version": DIRECTIONAL_STRATEGY_VERSION,
        "feature_state_schema_version": DIRECTIONAL_FEATURE_SCHEMA,
        "market_state_schema_version": MARKET_STATE_SCHEMA_VERSION,
        "intel_schema_version": INTEL_SCHEMA_VERSION,
        "regime_schema_version": REGIME_SCHEMA_VERSION,
        "decision_interval_seconds": settings.decision_interval_seconds,
        "dir_entry_confidence": settings.dir_entry_confidence,
        "dir_exit_opposite_confidence": settings.dir_exit_opposite_confidence,
        "dir_reverse_confidence": settings.dir_reverse_confidence,
        "dir_confirmations_required": settings.dir_confirmations_required,
        "dir_hold_floor": settings.dir_hold_floor,
        "dir_close_thesis_floor": settings.dir_close_thesis_floor,
        "dir_thesis_decay_close": settings.dir_thesis_decay_close,
        "dir_reverse_margin": settings.dir_reverse_margin,
        "dir_reverse_confirmations": settings.dir_reverse_confirmations,
        "dir_close_confirmations": settings.dir_close_confirmations,
        "dir_support_confidence": settings.dir_support_confidence,
        "dir_horizon_weight_primary": settings.dir_horizon_weight_primary,
        "dir_horizon_weight_support": settings.dir_horizon_weight_support,
        "dir_horizon_weight_context": settings.dir_horizon_weight_context,
        "dir_reversal_risk_close": settings.dir_reversal_risk_close,
        "dir_momentum_weakening_close": settings.dir_momentum_weakening_close,
        "dir_giveback_context_bps": settings.dir_giveback_context_bps,
        "dir_stale_horizon_multiple": settings.dir_stale_horizon_multiple,
        "execution_taker_fee_bps": settings.execution_taker_fee_bps,
        "execution_maker_fee_bps": settings.execution_maker_fee_bps,
        "execution_slippage_bps": settings.slippage_bps,
        "paper_position_notional": settings.execution_notional,
        "paper_equity": settings.paper_initial_balance,
        "paper_funding_enabled": settings.paper_funding_enabled,
        "experiment_id": settings.experiment_id,
        "experiment_frozen": settings.experiment_frozen,
    }


def config_hash(fingerprint: dict[str, Any]) -> str:
    payload = {key: value for key, value in fingerprint.items() if key not in HASH_EXCLUDED_KEYS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def excluded_secret_fields(settings: Settings) -> tuple[str, ...]:
    excluded: list[str] = []
    for name in type(settings).model_fields:
        lowered = name.lower()
        if any(marker in lowered for marker in SECRET_FIELD_MARKERS):
            excluded.append(name)
        if lowered.endswith("_url") or lowered == "database_url":
            excluded.append(name)
    return tuple(sorted(set(excluded)))
