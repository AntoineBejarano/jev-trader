from __future__ import annotations

from src.config.settings import Settings
from src.decision.actions import Action
from src.jev.models import JevCallResult, JevCallStatus, ParsedJevDecision
from src.market.models import MarketState, Side


class DecisionPolicy:
    """Maps Jev probabilities to a candidate action. No orders. Thresholds stay frozen."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def propose(self, state: MarketState, result: JevCallResult) -> Action:
        if result.status is not JevCallStatus.OK or result.decision is None:
            return Action.NO_ACTION
        if state.position.side is Side.FLAT:
            return self._entry(result.decision)
        return self._position(result.decision)

    def _entry(self, decision: ParsedJevDecision) -> Action:
        evidence = decision.evidence_probability or 0.0
        if evidence < self.settings.evidence_threshold:
            return Action.NO_ACTION
        if (
            decision.exposure_choice == "LONG"
            and (decision.long_probability or 0.0) >= self.settings.entry_threshold
        ):
            return Action.OPEN_LONG
        if (
            decision.exposure_choice == "SHORT"
            and (decision.short_probability or 0.0) >= self.settings.entry_threshold
        ):
            return Action.OPEN_SHORT
        return Action.NO_ACTION

    def _position(self, decision: ParsedJevDecision) -> Action:
        requested = (decision.position_action or "HOLD").upper()
        if requested == "REVERSE":
            if (decision.reversal_probability or 0.0) >= self.settings.reverse_threshold:
                return Action.REVERSE
            return Action.HOLD
        if requested == "CLOSE":
            return Action.CLOSE
        if requested == "REDUCE":
            return Action.REDUCE
        return Action.HOLD
