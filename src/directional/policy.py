from __future__ import annotations

from dataclasses import dataclass, field

from src.config.settings import Settings
from src.decision.actions import Action
from src.jev.models import JevCallResult, JevCallStatus
from src.market.models import MarketState, Side


@dataclass
class DirectionalPlan:
    candidate: Action
    policy_reason: str | None
    side: str | None
    p_up: dict[str, float]
    confirmations: int
    direction_consensus: str
    direction_conflict: bool
    entry_reason: str | None = None
    hold_reason: str | None = None
    close_reason: str | None = None
    reverse_reason: str | None = None
    forecast: dict[str, object] = field(default_factory=dict)


class DirectionalPolicy:
    """Threshold + hysteresis policy. Does not use Jev EV or fees."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._history: list[str] = []

    def evaluate(self, state: MarketState, result: JevCallResult) -> DirectionalPlan:
        empty = DirectionalPlan(
            candidate=Action.NO_ACTION,
            policy_reason="jev_unavailable",
            side=None,
            p_up={},
            confirmations=0,
            direction_consensus="MIXED",
            direction_conflict=True,
        )
        if result.status is not JevCallStatus.OK or result.decision is None:
            self._history.clear()
            return empty
        forecast = result.decision.directional_forecast or {}
        p_up = dict(forecast.get("p_up") or {})
        if not p_up:
            return empty
        p_1m = float(p_up.get("1m") or 0.5)
        p_5m = float(p_up.get("5m") or 0.5)
        side = "UP" if p_1m >= 0.5 else "DOWN"
        if side == "UP" and p_1m >= self.settings.dir_entry_confidence:
            voted = "UP"
        elif side == "DOWN" and (1.0 - p_1m) >= self.settings.dir_entry_confidence:
            voted = "DOWN"
        else:
            voted = "NONE"
        if voted != "NONE":
            self._history.append(voted)
        else:
            self._history.append("NONE")
        needed = max(1, self.settings.dir_confirmations_required)
        self._history = self._history[-needed:]
        confirmations = sum(1 for item in self._history if item == voted) if voted != "NONE" else 0
        confirmed = voted != "NONE" and confirmations >= needed
        consensus = str(forecast.get("direction_consensus") or "MIXED")
        conflict = bool(forecast.get("direction_conflict"))
        position = state.position.side
        if position is Side.FLAT:
            if not confirmed:
                return DirectionalPlan(
                    candidate=Action.NO_ACTION,
                    policy_reason="insufficient_confirmation",
                    side=None,
                    p_up=p_up,
                    confirmations=confirmations,
                    direction_consensus=consensus,
                    direction_conflict=conflict,
                    forecast=forecast,
                )
            action = Action.OPEN_LONG if voted == "UP" else Action.OPEN_SHORT
            return DirectionalPlan(
                candidate=action,
                policy_reason="directional_entry",
                side=voted,
                p_up=p_up,
                confirmations=confirmations,
                direction_consensus=consensus,
                direction_conflict=conflict,
                entry_reason=(
                    f"{voted} p_1m={p_1m:.3f} p_5m={p_5m:.3f} "
                    f"confirmations={confirmations} consensus={consensus}"
                ),
                forecast=forecast,
            )
        long = position is Side.LONG
        p_same = p_1m if long else 1.0 - p_1m
        p_opp = 1.0 - p_1m if long else p_1m
        if p_opp >= self.settings.dir_reverse_confidence and confirmed and voted != (
            "UP" if long else "DOWN"
        ):
            return DirectionalPlan(
                candidate=Action.REVERSE,
                policy_reason="directional_reverse",
                side="DOWN" if long else "UP",
                p_up=p_up,
                confirmations=confirmations,
                direction_consensus=consensus,
                direction_conflict=conflict,
                reverse_reason=f"opposite p_1m={p_opp:.3f} confirmed={confirmed}",
                forecast=forecast,
            )
        if p_opp >= self.settings.dir_exit_opposite_confidence:
            return DirectionalPlan(
                candidate=Action.CLOSE,
                policy_reason="directional_close",
                side="UP" if long else "DOWN",
                p_up=p_up,
                confirmations=confirmations,
                direction_consensus=consensus,
                direction_conflict=conflict,
                close_reason=f"opposite p_1m={p_opp:.3f}",
                forecast=forecast,
            )
        return DirectionalPlan(
            candidate=Action.HOLD,
            policy_reason="directional_hold",
            side="UP" if long else "DOWN",
            p_up=p_up,
            confirmations=confirmations,
            direction_consensus=consensus,
            direction_conflict=conflict,
            hold_reason=f"same-side p_1m={p_same:.3f}",
            forecast=forecast,
        )
