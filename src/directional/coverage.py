from __future__ import annotations

from typing import Any

# Intended directional.state.v1 fields. Classification only — does not
# change how features are computed. OPTIONAL missingness is explicit.
# Near-touch refill/cancel is not derived in v1.

REQUIRED_PATHS: tuple[tuple[str, ...], ...] = (
    ("microstructure", "mid"),
    ("microstructure", "best_bid"),
    ("microstructure", "best_ask"),
    ("microstructure", "spread_bps"),
    ("price_action", "returns", "r_5s"),
    ("price_action", "returns", "r_15s"),
    ("price_action", "returns", "r_30s"),
    ("price_action", "returns", "r_1m"),
    ("price_action", "returns", "r_3m"),
    ("price_action", "returns", "r_5m"),
    ("price_action", "returns", "r_10m"),
    ("price_action", "returns", "r_15m"),
    ("price_action", "returns", "r_30m"),
    ("price_action", "returns", "r_1h"),
    ("price_action", "active_candles", "1m"),
    ("price_action", "active_candles", "5m"),
    ("health", "book_valid"),
    ("health", "has_mid"),
)

OPTIONAL_PATHS: tuple[tuple[str, ...], ...] = (
    ("microstructure", "microprice"),
    ("microstructure", "depth", "bid_20"),
    ("microstructure", "depth", "ask_20"),
    ("derivatives", "open_interest"),
    ("derivatives", "oi_delta_1m"),
    ("derivatives", "oi_delta_5m"),
    ("derivatives", "oi_delta_15m"),
    ("derivatives", "funding"),
    ("derivatives", "liquidation_imbalance"),
    ("cross_market", "btc_spot_perp_basis_bps"),
    ("cross_market", "ethusdt"),
    ("price_action", "realized_vol", "rv_30m"),
    ("price_action", "recent_candles", "30m"),
    ("technical", "1m", "adx14"),
    ("technical", "5m", "adx14"),
)

OPTIONAL_UNIMPLEMENTED: tuple[str, ...] = (
    "near_touch_refill_rate",
    "near_touch_cancellation_pressure",
)


def _walk(payload: Any, path: tuple[str, ...]) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list)):
        return bool(value)
    return True


def missing_required(state: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for path in REQUIRED_PATHS:
        if not _present(_walk(state, path)):
            missing.append(".".join(path))
    return missing


def missing_optional(state: dict[str, Any]) -> list[str]:
    missing = [name for name in OPTIONAL_UNIMPLEMENTED]
    for path in OPTIONAL_PATHS:
        if not _present(_walk(state, path)):
            missing.append(".".join(path))
    declared = state.get("missing")
    if isinstance(declared, list):
        for item in declared:
            if isinstance(item, str) and item not in missing:
                missing.append(item)
    return missing


def feature_coverage(state: dict[str, Any]) -> dict[str, Any]:
    required_missing = missing_required(state)
    optional_missing = missing_optional(state)
    groups: dict[str, str] = {}
    for name, group in (
        ("MICROSTRUCTURE", state.get("microstructure")),
        ("PRICE ACTION", state.get("price_action")),
        ("TECHNICAL", state.get("technical")),
        ("DERIVATIVES", state.get("derivatives")),
        ("CROSS MARKET", state.get("cross_market")),
        ("REGIME", state.get("regime")),
    ):
        prefix = {
            "MICROSTRUCTURE": "microstructure",
            "PRICE ACTION": "price_action",
            "TECHNICAL": "technical",
            "DERIVATIVES": "derivatives",
            "CROSS MARKET": "cross_market",
            "REGIME": "regime",
        }[name]
        req_hit = [item for item in required_missing if item.startswith(prefix)]
        opt_hit = [item for item in optional_missing if item.startswith(prefix)]
        if req_hit:
            groups[name] = "INCOMPLETE"
        elif not isinstance(group, dict) or not group:
            groups[name] = "EMPTY"
        elif opt_hit:
            groups[name] = "PARTIAL"
        else:
            groups[name] = "POPULATED"
    returns = (state.get("price_action") or {}).get("returns") or {}
    return {
        "required_missing": required_missing,
        "optional_missing": optional_missing,
        "groups": groups,
        "fully_populated": not required_missing,
        "r_15m": None if not isinstance(returns, dict) else returns.get("r_15m"),
        "r_1h": None if not isinstance(returns, dict) else returns.get("r_1h"),
        "unimplemented_optional": list(OPTIONAL_UNIMPLEMENTED),
    }
