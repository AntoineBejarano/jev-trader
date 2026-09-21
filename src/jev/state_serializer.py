from __future__ import annotations

import json
from typing import Any

from src.market.models import MarketState

MAX_STATE_CHARS = 8_000


def _round(value: Any) -> Any:
    if isinstance(value, float):
        if abs(value) >= 1:
            return round(value, 6)
        return round(value, 8)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def serialize_market_state(state: MarketState) -> dict[str, Any]:
    """Compact numeric state. Never include raw books or trade prints."""
    payload = _round(state.model_dump(mode="json", exclude_none=True))
    if not isinstance(payload, dict):
        raise TypeError("serialized market state must be an object")
    health = dict(payload.get("health") or {})
    payload["health"] = {
        "trading_allowed": health.get("trading_allowed"),
        "stale": health.get("stale"),
        "warmup_ready": health.get("warmup_ready"),
    }
    encoded = json.dumps(payload, separators=(",", ":"))
    if len(encoded) > MAX_STATE_CHARS:
        payload.pop("context", None)
    return payload
