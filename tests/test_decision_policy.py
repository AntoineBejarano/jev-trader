from __future__ import annotations

from src.decision.actions import Action
from src.decision.policy import DecisionPolicy
from src.jev.models import JevCallStatus
from src.market.models import Side
from tests.helpers import make_market_state
from tests.paper_helpers import jev_result, paper_settings


def test_entry_requires_threshold_and_evidence() -> None:
    policy = DecisionPolicy(paper_settings())
    state = make_market_state()
    weak = jev_result(state, exposure="LONG", long_p=0.90, evidence=0.20)
    strong = jev_result(state, exposure="LONG", long_p=0.90, evidence=0.80)
    assert policy.propose(state, weak) is Action.NO_ACTION
    assert policy.propose(state, strong) is Action.OPEN_LONG


def test_short_entry() -> None:
    policy = DecisionPolicy(paper_settings())
    state = make_market_state()
    result = jev_result(state, exposure="SHORT", short_p=0.90, evidence=0.80)
    assert policy.propose(state, result) is Action.OPEN_SHORT


def test_position_actions() -> None:
    policy = DecisionPolicy(paper_settings())
    state = make_market_state(side=Side.LONG, age_seconds=30)
    assert policy.propose(state, jev_result(state, position_action="HOLD")) is Action.HOLD
    assert policy.propose(state, jev_result(state, position_action="CLOSE")) is Action.CLOSE
    assert policy.propose(state, jev_result(state, position_action="REDUCE")) is Action.REDUCE
    gated = jev_result(state, position_action="REVERSE", reversal=0.50)
    assert policy.propose(state, gated) is Action.HOLD
    flip = jev_result(state, position_action="REVERSE", reversal=0.90)
    assert policy.propose(state, flip) is Action.REVERSE


def test_failed_jev_is_no_action() -> None:
    policy = DecisionPolicy(paper_settings())
    state = make_market_state()
    result = jev_result(state, status=JevCallStatus.TIMEOUT)
    assert policy.propose(state, result) is Action.NO_ACTION
