from __future__ import annotations

from src.jev.models import JevSafetyAction
from src.jev.safety import safety_action_on_jev_failure
from src.market.models import Side


def test_flat_never_opens_when_jev_fails() -> None:
    action = safety_action_on_jev_failure(
        side=Side.FLAT,
        unavailable_for_seconds=120,
        failure_timeout_seconds=30,
    )
    assert action is JevSafetyAction.BLOCK_OPEN


def test_positioned_flattens_only_after_timeout() -> None:
    waiting = safety_action_on_jev_failure(
        side=Side.LONG,
        unavailable_for_seconds=10,
        failure_timeout_seconds=30,
    )
    flatten = safety_action_on_jev_failure(
        side=Side.SHORT,
        unavailable_for_seconds=30,
        failure_timeout_seconds=30,
    )
    assert waiting is JevSafetyAction.NONE
    assert flatten is JevSafetyAction.FLATTEN
