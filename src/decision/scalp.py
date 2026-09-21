from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import Settings
from src.decision.action_values import (
    ACTION_VALUE_REFERENCE,
    ACTION_VALUE_UNIT,
    action_margin_bps,
    compute_position_action_values,
)
from src.decision.actions import Action
from src.decision.barriers import Barrier
from src.decision.costs import CostSnapshot, cost_snapshot
from src.decision.ev import (
    COST_KIND_ENTRY,
    COST_KIND_HOLD,
    Opportunity,
    opportunities_for_forecast,
    rank_opportunities,
)
from src.jev.models import JevCallResult, JevCallStatus, ScalpHorizonForecast
from src.jev.scalp_questions import SCORE_MAPPING_VERSION, score_mapping_payload
from src.market.models import MarketState, Side


@dataclass(frozen=True, slots=True)
class ScalpPlan:
    barriers: tuple[Barrier, ...]
    forecasts: tuple[ScalpHorizonForecast, ...]
    opportunities: tuple[Opportunity, ...]
    hold_opportunities: tuple[Opportunity, ...]
    best: Opportunity | None
    candidate: Action
    policy_reason: str | None
    current_side_ev_bps: float | None
    opposite_ev_bps: float | None
    hold_net_ev_bps: float | None
    close_now_value_bps: float | None
    reverse_net_ev_bps: float | None
    entry_net_ev_if_flat_bps: float | None
    hold_advantage_bps: float | None
    action_margin_bps: float | None
    noise: float | None
    cost: CostSnapshot


class ScalpPolicy:
    """NET-EV scalper. Does not use BASELINE evidence/entry thresholds."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self,
        state: MarketState,
        result: JevCallResult,
        barriers: tuple[Barrier, ...],
    ) -> ScalpPlan:
        expected_exit = None
        if state.intelligence is not None:
            expected_exit = state.intelligence.expected_exit_spread_bps
        cost = cost_snapshot(
            taker_fee_bps=self.settings.execution_taker_fee_bps,
            maker_fee_bps=self.settings.execution_maker_fee_bps,
            spread_bps=state.microstructure.spread_bps,
            slippage_bps=self.settings.slippage_bps,
            assumed=self.settings.scalp_cost_model_assumed,
            source=self.settings.scalp_cost_model_source,
            expected_exit_spread_bps=expected_exit,
        )
        empty = ScalpPlan(
            barriers=barriers,
            forecasts=(),
            opportunities=(),
            hold_opportunities=(),
            best=None,
            candidate=Action.NO_ACTION,
            policy_reason=result.status.value,
            current_side_ev_bps=None,
            opposite_ev_bps=None,
            hold_net_ev_bps=None,
            close_now_value_bps=None,
            reverse_net_ev_bps=None,
            entry_net_ev_if_flat_bps=None,
            hold_advantage_bps=None,
            action_margin_bps=None,
            noise=None,
            cost=cost,
        )
        if result.status is not JevCallStatus.OK or result.decision is None:
            return empty
        forecasts = tuple(result.decision.scalp_forecasts)
        entry_opps = rank_opportunities(
            tuple(
                opp
                for forecast in forecasts
                for opp in opportunities_for_forecast(
                    forecast, cost_bps=cost.round_trip_cost_bps, cost_kind=COST_KIND_ENTRY
                )
            )
        )
        hold_opps = rank_opportunities(
            tuple(
                opp
                for forecast in forecasts
                for opp in opportunities_for_forecast(
                    forecast, cost_bps=cost.exit_leg_cost_bps, cost_kind=COST_KIND_HOLD
                )
            )
        )
        noise = result.decision.noise_probability
        best = entry_opps[0] if entry_opps else None
        if state.position.side is Side.FLAT:
            candidate, reason = self._entry(best, noise)
            return ScalpPlan(
                barriers=barriers,
                forecasts=forecasts,
                opportunities=entry_opps,
                hold_opportunities=hold_opps,
                best=best,
                candidate=candidate,
                policy_reason=reason,
                current_side_ev_bps=None,
                opposite_ev_bps=None,
                hold_net_ev_bps=None,
                close_now_value_bps=None,
                reverse_net_ev_bps=None,
                entry_net_ev_if_flat_bps=None if best is None else best.net_ev_bps,
                hold_advantage_bps=None,
                action_margin_bps=None,
                noise=noise,
                cost=cost,
            )
        current_side = "LONG" if state.position.side is Side.LONG else "SHORT"
        hold_value, close_value, reverse_value, entry_if_flat, _hold_gross, _rev_gross = (
            compute_position_action_values(
                current_side=current_side,
                cost=cost,
                hold_opportunities=hold_opps,
                entry_opportunities=entry_opps,
            )
        )
        hold_advantage = None if hold_value is None else hold_value - close_value
        candidate, reason = self._manage(
            hold_net=hold_value,
            close_now=close_value,
            reverse_net=reverse_value,
            noise=noise,
            age=state.position.age_seconds,
        )
        margin = action_margin_bps(
            candidate, hold=hold_value, close=close_value, reverse=reverse_value
        )
        return ScalpPlan(
            barriers=barriers,
            forecasts=forecasts,
            opportunities=entry_opps,
            hold_opportunities=hold_opps,
            best=best,
            candidate=candidate,
            policy_reason=reason,
            current_side_ev_bps=hold_value,
            opposite_ev_bps=reverse_value,
            hold_net_ev_bps=hold_value,
            close_now_value_bps=close_value,
            reverse_net_ev_bps=reverse_value,
            entry_net_ev_if_flat_bps=entry_if_flat,
            hold_advantage_bps=hold_advantage,
            action_margin_bps=margin,
            noise=noise,
            cost=cost,
        )

    def propose(self, plan: ScalpPlan) -> Action:
        return plan.candidate

    def _entry(
        self, best: Opportunity | None, noise: float | None
    ) -> tuple[Action, str | None]:
        if best is None:
            return Action.NO_ACTION, "no_forecast"
        if noise is not None and noise > self.settings.scalp_max_noise:
            return Action.NO_ACTION, "high_noise"
        if best.net_ev_bps < self.settings.scalp_min_net_ev_bps:
            return Action.NO_ACTION, "net_ev_below_edge"
        if best.probability_edge < self.settings.scalp_min_probability_edge:
            return Action.NO_ACTION, "probability_edge_too_small"
        if best.side == "LONG":
            return Action.OPEN_LONG, None
        return Action.OPEN_SHORT, None

    def _manage(
        self,
        *,
        hold_net: float | None,
        close_now: float,
        reverse_net: float | None,
        noise: float | None,
        age: float,
    ) -> tuple[Action, str | None]:
        if age < self.settings.scalp_min_position_age_seconds:
            return Action.HOLD, "min_position_age"
        if noise is not None and noise > self.settings.scalp_max_noise:
            return Action.CLOSE, "high_noise"
        hold = -1e9 if hold_net is None else hold_net
        reverse = -1e9 if reverse_net is None else reverse_net
        keep = max(hold, close_now)
        reverse_needed = (
            reverse >= self.settings.scalp_min_net_ev_bps
            and reverse >= keep + self.settings.scalp_reverse_advantage_bps
        )
        if reverse_needed:
            return Action.REVERSE, None
        hold_advantage = hold - close_now
        if hold_advantage < self.settings.scalp_close_ev_bps:
            return Action.CLOSE, "forward_ev_gone"
        if hold_advantage < self.settings.scalp_hold_min_net_ev_bps:
            return Action.CLOSE, "hold_ev_below_min"
        return Action.HOLD, None


def opportunity_dict(item: Opportunity) -> dict[str, float | int | str | bool | None]:
    return {
        "horizon_seconds": item.horizon_seconds,
        "side": item.side,
        "target_bps": item.target_bps,
        "stop_bps": item.stop_bps,
        "p_up": item.p_up,
        "p_down": item.p_down,
        "p_neither": item.p_neither,
        "p_target": item.p_target,
        "p_stop": item.p_stop,
        "expected_return_bps": item.expected_return_bps,
        "neither_return_bps": item.neither_return_bps,
        "neither_payoff_method": item.neither_payoff_method,
        "gross_ev_bps": item.gross_ev_bps,
        "cost_bps": item.cost_bps,
        "net_ev_bps": item.net_ev_bps,
        "probability_edge": item.probability_edge,
        "cost_kind": item.cost_kind,
        "volatility_target_bps": item.volatility_target_bps,
        "cost_floor_bps": item.cost_floor_bps,
        "final_target_bps": item.target_bps,
        "target_bound_by_cost_floor": item.target_bound_by_cost_floor,
        "horizon_cost_constrained": item.horizon_cost_constrained,
        "score_raw": item.score_raw,
        "score_mapping_version": item.score_mapping_version,
    }


def predictions_payload(
    plan: ScalpPlan, *, reference_price: float, timestamp: str
) -> list[dict[str, object]]:
    by_horizon: dict[int, dict[str, Opportunity]] = {}
    for item in plan.opportunities:
        by_horizon.setdefault(item.horizon_seconds, {})[item.side] = item
    rows: list[dict[str, object]] = []
    for forecast in plan.forecasts:
        long_opp = by_horizon.get(forecast.horizon_seconds, {}).get("LONG")
        short_opp = by_horizon.get(forecast.horizon_seconds, {}).get("SHORT")
        predicted_long = None if long_opp is None else long_opp.net_ev_bps
        predicted_short = None if short_opp is None else short_opp.net_ev_bps
        selected_side = None
        selected_ev = None
        if predicted_long is not None and predicted_short is not None:
            if predicted_long >= predicted_short:
                selected_side, selected_ev = "LONG", predicted_long
            else:
                selected_side, selected_ev = "SHORT", predicted_short
        elif predicted_long is not None:
            selected_side, selected_ev = "LONG", predicted_long
        elif predicted_short is not None:
            selected_side, selected_ev = "SHORT", predicted_short
        rows.append(
            {
                "horizon_seconds": forecast.horizon_seconds,
                "prediction_timestamp": timestamp,
                "reference_price": reference_price,
                "target_bps": forecast.target_bps,
                "stop_bps": forecast.stop_bps,
                "p_up_first": forecast.p_long_target,
                "p_down_first": forecast.p_long_stop,
                "p_neither": forecast.p_long_neither,
                "p_long_target": forecast.p_long_target,
                "p_long_stop": forecast.p_long_stop,
                "p_long_neither": forecast.p_long_neither,
                "p_short_target": forecast.p_short_target,
                "p_short_stop": forecast.p_short_stop,
                "p_short_neither": forecast.p_short_neither,
                "neither_score_raw": forecast.long_neither_score_raw or forecast.score_raw,
                "long_neither_score_raw": forecast.long_neither_score_raw,
                "short_neither_score_raw": forecast.short_neither_score_raw,
                "interpreted_neither_bps": forecast.long_neither_return_bps
                or forecast.neither_return_bps,
                "long_neither_return_bps": forecast.long_neither_return_bps,
                "short_neither_return_bps": forecast.short_neither_return_bps,
                "neither_payoff_method": forecast.neither_payoff_method,
                "predicted_gross_long_bps": None if long_opp is None else long_opp.gross_ev_bps,
                "predicted_gross_short_bps": None if short_opp is None else short_opp.gross_ev_bps,
                "predicted_entry_net_long_bps": predicted_long,
                "predicted_entry_net_short_bps": predicted_short,
                "predicted_entry_net_ev": selected_ev,
                "predicted_entry_net_side": selected_side,
            }
        )
    return rows


def plan_payload(
    plan: ScalpPlan,
    *,
    latency: dict[str, float | None],
    reference_price: float | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    best = None if plan.best is None else opportunity_dict(plan.best)
    predictions = []
    if reference_price is not None and timestamp is not None:
        predictions = predictions_payload(
            plan, reference_price=reference_price, timestamp=timestamp
        )
    return {
        "barriers": [
            {
                "horizon_seconds": barrier.horizon_seconds,
                "target_bps": barrier.target_bps,
                "stop_bps": barrier.stop_bps,
                "sigma_bps": barrier.sigma_bps,
                "cost_floor_bps": barrier.cost_floor_bps,
                "volatility_target_bps": barrier.volatility_target_bps,
                "final_target_bps": barrier.target_bps,
                "target_bound_by_cost_floor": barrier.target_bound_by_cost_floor,
                "horizon_cost_constrained": barrier.horizon_cost_constrained,
            }
            for barrier in plan.barriers
        ],
        "opportunities": [opportunity_dict(item) for item in plan.opportunities],
        "hold_opportunities": [opportunity_dict(item) for item in plan.hold_opportunities],
        "best": best,
        "candidate": plan.candidate.value,
        "policy_reason": plan.policy_reason,
        "current_side_ev_bps": plan.current_side_ev_bps,
        "opposite_ev_bps": plan.opposite_ev_bps,
        "hold_net_ev_bps": plan.hold_net_ev_bps,
        "close_now_value_bps": plan.close_now_value_bps,
        "reverse_net_ev_bps": plan.reverse_net_ev_bps,
        "hold_value_bps": plan.hold_net_ev_bps,
        "close_value_bps": plan.close_now_value_bps,
        "reverse_value_bps": plan.reverse_net_ev_bps,
        "entry_net_ev_if_flat_bps": plan.entry_net_ev_if_flat_bps,
        "hold_advantage_bps": plan.hold_advantage_bps,
        "action_margin_bps": plan.action_margin_bps,
        "chosen_action": plan.candidate.value,
        "reason": plan.policy_reason,
        "action_value_reference": ACTION_VALUE_REFERENCE,
        "action_value_unit": ACTION_VALUE_UNIT,
        "noise": plan.noise,
        "score_mapping": score_mapping_payload(),
        "score_mapping_version": SCORE_MAPPING_VERSION,
        "predictions": predictions,
        "cost": {
            "maker_fee_bps": plan.cost.maker_fee_bps,
            "taker_fee_bps": plan.cost.taker_fee_bps,
            "slippage_bps": plan.cost.slippage_bps,
            "spread_bps": plan.cost.spread_bps,
            "entry_half_spread_bps": plan.cost.entry_half_spread_bps,
            "expected_exit_half_spread_bps": plan.cost.expected_exit_half_spread_bps,
            "entry_leg_cost_bps": plan.cost.entry_leg_cost_bps,
            "exit_leg_cost_bps": plan.cost.exit_leg_cost_bps,
            "round_trip_cost_bps": plan.cost.round_trip_cost_bps,
            "assumed": plan.cost.assumed,
            "source": plan.cost.source,
        },
        "latency": latency,
    }
