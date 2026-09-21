from __future__ import annotations

import json
from typing import Any

from src.directional.constants import FEATURE_SCHEMA_VERSION
from src.directional.features import feature_groups
from src.jev.state_serializer import MAX_STATE_CHARS, _round

FORBIDDEN_KEY_MARKERS = (
    "fee",
    "slip",
    "pnl",
    "profit",
    "required_ev",
    "net_ev",
    "gross_ev",
    "min_net_ev",
    "scalp_min",
    "unrealized",
    "realized_pnl",
    "notional",
    "equity",
)


def _forbidden(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in FORBIDDEN_KEY_MARKERS)


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if not _forbidden(str(key))}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def serialize_directional_state(state: dict[str, Any]) -> dict[str, Any]:
    """Compact forecast state. No fees, PnL, or execution economics."""
    payload = _round(_strip(state))
    if not isinstance(payload, dict):
        raise TypeError("directional state must be an object")
    payload["schema"] = FEATURE_SCHEMA_VERSION
    encoded = json.dumps(payload, separators=(",", ":"), default=str)
    if len(encoded) > MAX_STATE_CHARS:
        price_action = dict(payload.get("price_action") or {})
        recent = dict(price_action.get("recent_candles") or {})
        for key in list(recent):
            rows = recent[key]
            if isinstance(rows, list) and len(rows) > 2:
                recent[key] = rows[-2:]
        price_action["recent_candles"] = recent
        payload["price_action"] = price_action
    return payload


def compact_inspector(state: dict[str, Any]) -> dict[str, Any]:
    groups = feature_groups(state)
    compact: dict[str, Any] = {}
    for name, group in groups.items():
        if not isinstance(group, dict):
            compact[name] = group
            continue
        compact[name] = {
            key: _summarize(value) for key, value in group.items() if not _forbidden(str(key))
        }
    return compact


def _summarize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _summarize(item) for key, item in list(value.items())[:12]}
    if isinstance(value, list):
        return [_summarize(item) for item in value[:4]]
    return value
