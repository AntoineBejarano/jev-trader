from __future__ import annotations

from dataclasses import dataclass

from src.decision.actions import Action
from src.decision.costs import CostSnapshot
from src.decision.ev import Opportunity, best_for_side

ACTION_VALUE_REFERENCE = "MID_INCREMENTAL_BPS_FROM_NOW"
ACTION_VALUE_UNIT = "bps_on_position_notional_from_now"


@dataclass(frozen=True, slots=True)
class PositionActionValues:
    """Incremental action values from NOW, referenced to current MID.

    Forecast payoffs are signed price bps versus the current mid. Closing a
    position therefore pays the *remaining* executable friction from mid
    (half-spread + slippage + taker), not a second copy of a mark that is
    already at bid/ask.

    Sunk entry fee/slip/spread are excluded from HOLD and CLOSE.
    """

    reference: str
    hold_value_bps: float | None
    close_value_bps: float | None
    reverse_value_bps: float | None
    entry_net_ev_if_flat_bps: float | None
    chosen_action: Action
    action_margin_bps: float | None
    reason: str | None
    hold_gross_ev_bps: float | None
    reverse_gross_ev_bps: float | None
    exit_now_bps: float
    expected_future_exit_bps: float
    opposite_entry_bps: float


def close_value_from_mid(*, exit_now_bps: float) -> float:
    """VALUE_CLOSE: pay current executable exit versus mid. One half-spread only."""
    return -exit_now_bps


def hold_value_from_mid(*, expected_payoff_bps: float, expected_future_exit_bps: float) -> float:
    """VALUE_HOLD: remaining market payoff versus mid, minus future exit friction."""
    return expected_payoff_bps - expected_future_exit_bps


def reverse_value_from_mid(
    *,
    opposite_payoff_bps: float,
    exit_now_bps: float,
    opposite_entry_bps: float,
    opposite_future_exit_bps: float,
) -> float:
    """VALUE_REVERSE: close now + open opposite now + eventual opposite exit."""
    return (
        opposite_payoff_bps - exit_now_bps - opposite_entry_bps - opposite_future_exit_bps
    )


def action_margin_bps(
    chosen: Action,
    *,
    hold: float | None,
    close: float | None,
    reverse: float | None,
) -> float | None:
    values = {
        Action.HOLD: hold,
        Action.CLOSE: close,
        Action.REVERSE: reverse,
    }
    chosen_value = values.get(chosen)
    if chosen_value is None:
        return None
    others = [
        value
        for action, value in values.items()
        if action is not chosen and value is not None
    ]
    if not others:
        return None
    return chosen_value - max(others)


def compute_position_action_values(
    *,
    current_side: str,
    cost: CostSnapshot,
    hold_opportunities: tuple[Opportunity, ...],
    entry_opportunities: tuple[Opportunity, ...],
) -> tuple[float | None, float, float | None, float | None, float | None, float | None]:
    opposite = "SHORT" if current_side == "LONG" else "LONG"
    hold_best = best_for_side(hold_opportunities, current_side)
    opposite_best = best_for_side(hold_opportunities, opposite)
    same_entry = best_for_side(entry_opportunities, current_side)
    exit_now = cost.exit_leg_cost_bps
    future_exit = cost.exit_leg_cost_bps
    opposite_entry = cost.entry_leg_cost_bps
    hold_gross = None if hold_best is None else hold_best.gross_ev_bps
    hold_value = (
        None
        if hold_gross is None
        else hold_value_from_mid(
            expected_payoff_bps=hold_gross, expected_future_exit_bps=future_exit
        )
    )
    close_value = close_value_from_mid(exit_now_bps=exit_now)
    reverse_gross = None if opposite_best is None else opposite_best.gross_ev_bps
    reverse_value = (
        None
        if reverse_gross is None
        else reverse_value_from_mid(
            opposite_payoff_bps=reverse_gross,
            exit_now_bps=exit_now,
            opposite_entry_bps=opposite_entry,
            opposite_future_exit_bps=future_exit,
        )
    )
    entry_if_flat = None if same_entry is None else same_entry.net_ev_bps
    return hold_value, close_value, reverse_value, entry_if_flat, hold_gross, reverse_gross
