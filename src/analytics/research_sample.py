from __future__ import annotations

from src.jev.models import JevCallResult, JevCallStatus
from src.market.models import MarketState

INVALID_WARMUP = "warmup_incomplete"
INVALID_CONTEXT = "missing_required_context"
INVALID_STALE = "stale_market"
INVALID_JEV = "jev_failure"


def research_sample_status(
    state: MarketState, result: JevCallResult
) -> tuple[bool, str | None]:
    """Official calibration gate. Raw predictions are still persisted.

    r_15m / r_1h are intended Jev context and are bootstrapped from public
    USD-M klines. Their absence after a failed bootstrap does not invalidate
    a sample; warmup, stale market, missing executable context, and Jev
    failure do.
    """
    if result.status is not JevCallStatus.OK:
        return False, INVALID_JEV
    if state.health.stale:
        return False, INVALID_STALE
    if not state.health.warmup_ready or INVALID_WARMUP in state.health.reasons:
        return False, INVALID_WARMUP
    if state.price is None or state.price <= 0:
        return False, INVALID_CONTEXT
    if "order_book_invalid" in state.health.reasons:
        return False, INVALID_CONTEXT
    if not state.health.ws_public_connected or not state.health.ws_market_connected:
        return False, INVALID_CONTEXT
    return True, None


def annotate_research_sample(
    payload: dict[str, object] | None,
    *,
    valid: bool,
    reason: str | None,
) -> dict[str, object]:
    annotated: dict[str, object] = {} if payload is None else dict(payload)
    annotated["research_sample_valid"] = valid
    annotated["invalid_reason"] = reason
    return annotated
