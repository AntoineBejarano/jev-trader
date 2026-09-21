from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.settings import Settings
from src.decision.actions import Action
from src.directional.thesis import (
    build_entry_snapshot,
    classify_horizons,
    current_position_thesis,
    score_thesis,
)
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
    entry_thesis: dict[str, Any] | None = None
    position_thesis: dict[str, Any] | None = None
    entry_signal: bool = False
    hold_signal: bool = False
    close_signal: bool = False
    reverse_signal: bool = False
    recommended: str | None = None


class DirectionalPolicy:
    """ENTRY / HOLD / CLOSE / REVERSE are distinct. No fees, no Jev EV."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._history: list[str] = []
        self._close_streak = 0

    def evaluate(
        self,
        state: MarketState,
        result: JevCallResult,
        *,
        entry_thesis: dict[str, Any] | None = None,
        directional_state: dict[str, Any] | None = None,
    ) -> DirectionalPlan:
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
            if state.position.side is Side.FLAT:
                self._history.clear()
                self._close_streak = 0
                return empty
            return self._hold_unavailable(state, entry_thesis)
        forecast = result.decision.directional_forecast or {}
        raw_p = forecast.get("p_up")
        p_up = {str(key): float(value) for key, value in dict(raw_p or {}).items()}
        if not p_up:
            return empty
        consensus = str(forecast.get("direction_consensus") or "MIXED")
        conflict = bool(forecast.get("direction_conflict"))
        voted, confirmations, confirmed = self._vote(p_up)
        position = state.position.side
        if position is Side.FLAT:
            self._close_streak = 0
            return self._entry(
                p_up=p_up,
                forecast=forecast,
                directional_state=directional_state,
                timestamp_ms=state.timestamp_ms,
                voted=voted,
                confirmations=confirmations,
                confirmed=confirmed,
                consensus=consensus,
                conflict=conflict,
            )
        long = position is Side.LONG
        current_side = "UP" if long else "DOWN"
        opposite = "DOWN" if long else "UP"
        thesis = self._live_thesis(
            state=state,
            p_up=p_up,
            forecast=forecast,
            entry_thesis=entry_thesis,
            current_side=current_side,
            consensus=consensus,
            conflict=conflict,
        )
        reverse_ok = self._reverse_ready(
            p_up=p_up,
            forecast=forecast,
            voted=voted,
            confirmed=confirmed,
            opposite=opposite,
            current=thesis,
        )
        close_reason = self._close_reason(thesis)
        close_ok = close_reason is not None
        if close_ok:
            self._close_streak += 1
        else:
            self._close_streak = 0
        close_confirmed = close_ok and self._close_streak >= max(
            1, self.settings.dir_close_confirmations
        )
        if reverse_ok:
            snapshot = build_entry_snapshot(
                side=opposite,
                p_up=p_up,
                timestamp_ms=state.timestamp_ms,
                forecast=forecast,
                directional_state=directional_state,
                settings=self.settings,
                confirmations=confirmations,
            )
            thesis["recommended"] = "REVERSE"
            thesis["reason"] = "reverse_confirmed"
            return DirectionalPlan(
                candidate=Action.REVERSE,
                policy_reason="directional_reverse",
                side=opposite,
                p_up=p_up,
                confirmations=confirmations,
                direction_consensus=consensus,
                direction_conflict=conflict,
                reverse_reason="reverse_confirmed",
                close_reason=close_reason,
                forecast=forecast,
                entry_thesis=snapshot,
                position_thesis=thesis,
                reverse_signal=True,
                close_signal=close_ok,
                recommended="REVERSE",
            )
        if close_confirmed:
            reason = close_reason or "thesis_decayed"
            if voted == opposite and not reverse_ok:
                reason = "opposite_signal_not_strong_enough_to_reverse"
            thesis["recommended"] = "CLOSE"
            thesis["reason"] = reason
            return DirectionalPlan(
                candidate=Action.CLOSE,
                policy_reason="directional_close",
                side=current_side,
                p_up=p_up,
                confirmations=confirmations,
                direction_consensus=consensus,
                direction_conflict=conflict,
                close_reason=reason,
                forecast=forecast,
                entry_thesis=entry_thesis,
                position_thesis=thesis,
                close_signal=True,
                recommended="CLOSE",
            )
        hold_reason = (
            "same_side_thesis_intact"
            if float(thesis.get("current_thesis_score") or 0.0) >= self.settings.dir_hold_floor
            else "thesis_watch"
        )
        thesis["recommended"] = "HOLD"
        thesis["reason"] = hold_reason
        return DirectionalPlan(
            candidate=Action.HOLD,
            policy_reason="directional_hold",
            side=current_side,
            p_up=p_up,
            confirmations=confirmations,
            direction_consensus=consensus,
            direction_conflict=conflict,
            hold_reason=hold_reason,
            forecast=forecast,
            entry_thesis=entry_thesis,
            position_thesis=thesis,
            hold_signal=True,
            recommended="HOLD",
        )

    def _vote(self, p_up: dict[str, float]) -> tuple[str, int, bool]:
        p_1m = float(p_up.get("1m") or 0.5)
        if p_1m >= self.settings.dir_entry_confidence:
            voted = "UP"
        elif (1.0 - p_1m) >= self.settings.dir_entry_confidence:
            voted = "DOWN"
        else:
            voted = "NONE"
        self._history.append(voted)
        needed = max(1, self.settings.dir_confirmations_required)
        self._history = self._history[-max(needed, self.settings.dir_reverse_confirmations) :]
        confirmations = sum(1 for item in self._history if item == voted) if voted != "NONE" else 0
        confirmed = voted != "NONE" and confirmations >= needed
        return voted, confirmations, confirmed

    def _entry(
        self,
        *,
        p_up: dict[str, float],
        forecast: dict[str, Any],
        directional_state: dict[str, Any] | None,
        timestamp_ms: int,
        voted: str,
        confirmations: int,
        confirmed: bool,
        consensus: str,
        conflict: bool,
    ) -> DirectionalPlan:
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
                entry_signal=False,
            )
        action = Action.OPEN_LONG if voted == "UP" else Action.OPEN_SHORT
        snapshot = build_entry_snapshot(
            side=voted,
            p_up=p_up,
            timestamp_ms=timestamp_ms,
            forecast=forecast,
            directional_state=directional_state,
            settings=self.settings,
            confirmations=confirmations,
        )
        p_1m = float(p_up.get("1m") or 0.5)
        p_5m = float(p_up.get("5m") or 0.5)
        return DirectionalPlan(
            candidate=action,
            policy_reason="directional_entry",
            side=voted,
            p_up=p_up,
            confirmations=confirmations,
            direction_consensus=consensus,
            direction_conflict=conflict,
            entry_reason=(
                f"{voted} primary={snapshot['entry_primary_horizon']} "
                f"p_1m={p_1m:.3f} p_5m={p_5m:.3f} "
                f"confirmations={confirmations} consensus={consensus}"
            ),
            forecast=forecast,
            entry_thesis=snapshot,
            entry_signal=True,
            recommended="ENTRY",
        )

    def _live_thesis(
        self,
        *,
        state: MarketState,
        p_up: dict[str, float],
        forecast: dict[str, Any],
        entry_thesis: dict[str, Any] | None,
        current_side: str,
        consensus: str,
        conflict: bool,
    ) -> dict[str, Any]:
        entry = entry_thesis or build_entry_snapshot(
            side=current_side,
            p_up=p_up,
            timestamp_ms=state.timestamp_ms,
            forecast=forecast,
            directional_state=None,
            settings=self.settings,
            confirmations=0,
        )
        position = state.position
        return current_position_thesis(
            entry=entry,
            p_up=p_up,
            forecast=forecast,
            settings=self.settings,
            age_seconds=float(position.age_seconds or 0.0),
            mfe_bps=float(position.max_favorable_excursion_bps or 0.0),
            mae_bps=float(position.max_adverse_excursion_bps or 0.0),
            unrealized_bps=float(position.unrealized_pnl_bps or 0.0),
        )

    def _close_reason(self, thesis: dict[str, Any]) -> str | None:
        settings = self.settings
        score = float(thesis.get("current_thesis_score") or 0.0)
        decay = float(thesis.get("thesis_decay") or 0.0)
        raw_components = thesis.get("components")
        components = raw_components if isinstance(raw_components, dict) else {}
        primary_p = float(components.get("primary_p") or 0.5)
        conflict = bool(thesis.get("current_conflict"))
        weakening = float(thesis.get("momentum_weakening") or 0.0)
        reversal = float(thesis.get("reversal_risk") or 0.0)
        giveback = thesis.get("profit_giveback_bps")
        giveback_v = float(giveback) if isinstance(giveback, int | float) else 0.0
        stale = bool(thesis.get("stale_horizon"))
        thesis_weak = (
            score < settings.dir_hold_floor or decay >= settings.dir_thesis_decay_close / 2.0
        )
        if primary_p < 0.5:
            return "short_horizon_flipped"
        if score < settings.dir_close_thesis_floor:
            return "thesis_decayed"
        if decay >= settings.dir_thesis_decay_close and score < settings.dir_hold_floor:
            return "thesis_decayed"
        if conflict and primary_p < settings.dir_hold_floor:
            return "term_structure_conflict"
        if stale and score < settings.dir_hold_floor:
            return "stale_short_horizon_thesis"
        weakening_hit = weakening >= settings.dir_momentum_weakening_close
        reversal_hit = reversal >= settings.dir_reversal_risk_close
        if thesis_weak and weakening_hit and (reversal_hit or conflict):
            if (
                isinstance(giveback, int | float)
                and giveback_v >= settings.dir_giveback_context_bps
            ):
                return "momentum_weakening_with_giveback"
            if score < settings.dir_hold_floor:
                return "momentum_weakening"
        if (
            thesis_weak
            and isinstance(giveback, int | float)
            and giveback_v >= settings.dir_giveback_context_bps
            and (weakening_hit or conflict)
        ):
            return "momentum_weakening_with_giveback"
        return None

    def _reverse_ready(
        self,
        *,
        p_up: dict[str, float],
        forecast: dict[str, Any],
        voted: str,
        confirmed: bool,
        opposite: str,
        current: dict[str, Any],
    ) -> bool:
        if voted != opposite:
            return False
        reverse_needed = max(1, self.settings.dir_reverse_confirmations)
        reverse_confirms = sum(1 for item in self._history if item == opposite)
        if reverse_confirms < reverse_needed or not confirmed:
            return False
        diagnostics = (
            forecast.get("diagnostics") if isinstance(forecast.get("diagnostics"), dict) else {}
        )
        primary, supporting, _context = classify_horizons(
            p_up,
            opposite,
            entry_confidence=self.settings.dir_entry_confidence,
            support_confidence=self.settings.dir_support_confidence,
        )
        opp = score_thesis(
            p_up,
            opposite,
            primary=primary,
            supporting=supporting,
            settings=self.settings,
            consensus=str(forecast.get("direction_consensus") or "MIXED"),
            conflict=bool(forecast.get("direction_conflict")),
            diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
        )
        current_score = float(current.get("current_thesis_score") or 0.0)
        if float(opp["score"]) < self.settings.dir_reverse_confidence:
            return False
        if float(opp["score"]) - current_score < self.settings.dir_reverse_margin:
            return False
        persist = forecast.get("persistence")
        persistence = persist if isinstance(persist, dict) else {}
        persist_key = "1m_2_up" if opposite == "UP" else "1m_2_down"
        persist_p = persistence.get(persist_key)
        if not isinstance(persist_p, int | float):
            return True
        return float(persist_p) >= 0.55

    def _hold_unavailable(
        self, state: MarketState, entry_thesis: dict[str, Any] | None
    ) -> DirectionalPlan:
        side = "UP" if state.position.side is Side.LONG else "DOWN"
        return DirectionalPlan(
            candidate=Action.HOLD,
            policy_reason="jev_unavailable",
            side=side,
            p_up={},
            confirmations=0,
            direction_consensus="MIXED",
            direction_conflict=True,
            hold_reason="jev_unavailable",
            entry_thesis=entry_thesis,
            hold_signal=True,
            recommended="HOLD",
        )
