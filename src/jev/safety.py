from __future__ import annotations

from src.jev.models import JevSafetyAction
from src.market.models import Side


def safety_action_on_jev_failure(
    *,
    side: Side,
    unavailable_for_seconds: float,
    failure_timeout_seconds: float,
) -> JevSafetyAction:
    """Deterministic policy when Jev is unavailable. Jev cannot override this later."""
    if side is Side.FLAT:
        return JevSafetyAction.BLOCK_OPEN
    if unavailable_for_seconds >= failure_timeout_seconds:
        return JevSafetyAction.FLATTEN
    return JevSafetyAction.NONE
