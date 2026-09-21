from __future__ import annotations

from dataclasses import dataclass

from src.decision.barriers import Barrier
from src.jev.models import ScalpHorizonForecast

NEITHER_JEV_CONDITIONAL = "JEV_CONDITIONAL"
NEITHER_ZERO_CONSERVATIVE = "ZERO_CONSERVATIVE"
COST_KIND_ENTRY = "ENTRY_ROUND_TRIP"
COST_KIND_HOLD = "FORWARD_HOLD_EXIT"


@dataclass(frozen=True, slots=True)
class Opportunity:
    horizon_seconds: int
    side: str
    target_bps: float
    stop_bps: float
    p_up: float
    p_down: float
    p_neither: float
    p_target: float
    p_stop: float
    expected_return_bps: float
    neither_return_bps: float
    neither_payoff_method: str
    gross_ev_bps: float
    cost_bps: float
    net_ev_bps: float
    probability_edge: float
    cost_kind: str = COST_KIND_ENTRY
    volatility_target_bps: float = 0.0
    cost_floor_bps: float = 0.0
    target_bound_by_cost_floor: bool = False
    horizon_cost_constrained: bool = False
    score_raw: float | None = None
    score_mapping_version: str = "jev.score.return.v1"


def normalize_barrier_probs(
    p_target: float, p_stop: float, p_neither: float
) -> tuple[float, float, float]:
    total = p_target + p_stop + p_neither
    if total <= 0:
        return 0.0, 0.0, 1.0
    return p_target / total, p_stop / total, p_neither / total


def neither_branch_payoff_bps(
    *,
    neither_return_bps: float,
    neither_payoff_method: str,
) -> float:
    """Payoff used only in the p_neither branch. Never an unconditional E[R]."""
    if neither_payoff_method == NEITHER_JEV_CONDITIONAL:
        return neither_return_bps
    return 0.0


def long_gross_ev_bps(
    *,
    p_up: float | None = None,
    p_down: float | None = None,
    p_neither: float | None = None,
    p_long_target: float | None = None,
    p_long_stop: float | None = None,
    p_long_neither: float | None = None,
    target_bps: float,
    stop_bps: float,
    expected_return_bps: float,
    neither_payoff_method: str = NEITHER_JEV_CONDITIONAL,
) -> float:
    """LONG_GROSS_EV = p_target*target - p_stop*stop + p_neither*long_neither_return."""
    p_target = p_long_target if p_long_target is not None else (0.0 if p_up is None else p_up)
    p_stop = p_long_stop if p_long_stop is not None else (0.0 if p_down is None else p_down)
    if p_long_neither is not None:
        p_miss = p_long_neither
    else:
        p_miss = 0.0 if p_neither is None else p_neither
    target_p, stop_p, neither_p = normalize_barrier_probs(p_target, p_stop, p_miss)
    neither_payoff = neither_branch_payoff_bps(
        neither_return_bps=expected_return_bps,
        neither_payoff_method=neither_payoff_method,
    )
    return target_p * target_bps - stop_p * stop_bps + neither_p * neither_payoff


def short_gross_ev_bps(
    *,
    p_up: float | None = None,
    p_down: float | None = None,
    p_neither: float | None = None,
    p_short_target: float | None = None,
    p_short_stop: float | None = None,
    p_short_neither: float | None = None,
    target_bps: float,
    stop_bps: float,
    expected_return_bps: float,
    neither_payoff_method: str = NEITHER_JEV_CONDITIONAL,
) -> float:
    """SHORT_GROSS_EV = p_target*target - p_stop*stop - p_neither*short_neither_return."""
    p_target = p_short_target if p_short_target is not None else (0.0 if p_down is None else p_down)
    p_stop = p_short_stop if p_short_stop is not None else (0.0 if p_up is None else p_up)
    if p_short_neither is not None:
        p_miss = p_short_neither
    else:
        p_miss = 0.0 if p_neither is None else p_neither
    target_p, stop_p, neither_p = normalize_barrier_probs(p_target, p_stop, p_miss)
    neither_payoff = neither_branch_payoff_bps(
        neither_return_bps=expected_return_bps,
        neither_payoff_method=neither_payoff_method,
    )
    return target_p * target_bps - stop_p * stop_bps - neither_p * neither_payoff


def opportunities_for_forecast(
    forecast: ScalpHorizonForecast,
    *,
    cost_bps: float,
    cost_kind: str = COST_KIND_ENTRY,
) -> tuple[Opportunity, Opportunity]:
    method = forecast.neither_payoff_method
    long_neither = forecast.long_neither_return_bps
    if long_neither == 0.0:
        long_neither = forecast.neither_return_bps or forecast.expected_return_bps
    short_neither = forecast.short_neither_return_bps
    if short_neither == 0.0:
        short_neither = forecast.neither_return_bps or forecast.expected_return_bps
    p_long_t, p_long_s, p_long_n = normalize_barrier_probs(
        forecast.p_long_target, forecast.p_long_stop, forecast.p_long_neither
    )
    p_short_t, p_short_s, p_short_n = normalize_barrier_probs(
        forecast.p_short_target, forecast.p_short_stop, forecast.p_short_neither
    )
    long_gross = long_gross_ev_bps(
        p_long_target=p_long_t,
        p_long_stop=p_long_s,
        p_long_neither=p_long_n,
        target_bps=forecast.target_bps,
        stop_bps=forecast.stop_bps,
        expected_return_bps=long_neither,
        neither_payoff_method=method,
    )
    short_gross = short_gross_ev_bps(
        p_short_target=p_short_t,
        p_short_stop=p_short_s,
        p_short_neither=p_short_n,
        target_bps=forecast.target_bps,
        stop_bps=forecast.stop_bps,
        expected_return_bps=short_neither,
        neither_payoff_method=method,
    )

    def _make(
        *,
        side: str,
        gross: float,
        p_target: float,
        p_stop: float,
        p_neither: float,
        neither_return: float,
        score_raw: float | None,
        p_up: float,
        p_down: float,
    ) -> Opportunity:
        return Opportunity(
            horizon_seconds=forecast.horizon_seconds,
            side=side,
            target_bps=forecast.target_bps,
            stop_bps=forecast.stop_bps,
            p_up=p_up,
            p_down=p_down,
            p_neither=p_neither,
            p_target=p_target,
            p_stop=p_stop,
            expected_return_bps=neither_return,
            neither_return_bps=neither_return,
            neither_payoff_method=method,
            gross_ev_bps=gross,
            cost_bps=cost_bps,
            net_ev_bps=gross - cost_bps,
            probability_edge=p_target - p_stop,
            cost_kind=cost_kind,
            volatility_target_bps=forecast.volatility_target_bps,
            cost_floor_bps=forecast.cost_floor_bps,
            target_bound_by_cost_floor=forecast.target_bound_by_cost_floor,
            horizon_cost_constrained=forecast.horizon_cost_constrained,
            score_raw=score_raw,
            score_mapping_version=forecast.score_mapping_version,
        )

    return (
        _make(
            side="LONG",
            gross=long_gross,
            p_target=p_long_t,
            p_stop=p_long_s,
            p_neither=p_long_n,
            neither_return=long_neither,
            score_raw=forecast.long_neither_score_raw or forecast.score_raw,
            p_up=p_long_t,
            p_down=p_long_s,
        ),
        _make(
            side="SHORT",
            gross=short_gross,
            p_target=p_short_t,
            p_stop=p_short_s,
            p_neither=p_short_n,
            neither_return=short_neither,
            score_raw=forecast.short_neither_score_raw or forecast.score_raw,
            p_up=p_short_s,
            p_down=p_short_t,
        ),
    )


def rank_opportunities(opportunities: tuple[Opportunity, ...]) -> tuple[Opportunity, ...]:
    return tuple(sorted(opportunities, key=lambda item: item.net_ev_bps, reverse=True))


def best_for_side(opportunities: tuple[Opportunity, ...], side: str) -> Opportunity | None:
    sided = [item for item in opportunities if item.side == side]
    if not sided:
        return None
    return max(sided, key=lambda item: item.net_ev_bps)


def barrier_for(barriers: tuple[Barrier, ...], horizon: int) -> Barrier | None:
    for barrier in barriers:
        if barrier.horizon_seconds == horizon:
            return barrier
    return None


def opportunity_matrix(
    opportunities: tuple[Opportunity, ...],
) -> list[dict[str, float | int | str | bool | None]]:
    rows: dict[int, dict[str, float | int | str | bool | None]] = {}
    for item in opportunities:
        row = rows.setdefault(
            item.horizon_seconds,
            {
                "horizon_seconds": item.horizon_seconds,
                "long_ev": None,
                "short_ev": None,
                "best": None,
                "target_bps": item.target_bps,
                "stop_bps": item.stop_bps,
                "volatility_target_bps": item.volatility_target_bps,
                "cost_floor_bps": item.cost_floor_bps,
                "final_target_bps": item.target_bps,
                "target_bound_by_cost_floor": item.target_bound_by_cost_floor,
                "horizon_cost_constrained": item.horizon_cost_constrained,
                "p_up": item.p_up,
                "p_down": item.p_down,
                "p_neither": item.p_neither,
                "neither_return_bps": item.neither_return_bps,
                "neither_payoff_method": item.neither_payoff_method,
                "gross_long_ev": None,
                "gross_short_ev": None,
                "cost_bps": item.cost_bps,
                "cost_kind": item.cost_kind,
                "p_long_target": None,
                "p_long_stop": None,
                "p_long_neither": None,
                "p_short_target": None,
                "p_short_stop": None,
                "p_short_neither": None,
            },
        )
        if item.side == "LONG":
            row["long_ev"] = item.net_ev_bps
            row["gross_long_ev"] = item.gross_ev_bps
            row["p_long_target"] = item.p_target
            row["p_long_stop"] = item.p_stop
            row["p_long_neither"] = item.p_neither
        else:
            row["short_ev"] = item.net_ev_bps
            row["gross_short_ev"] = item.gross_ev_bps
            row["p_short_target"] = item.p_target
            row["p_short_stop"] = item.p_stop
            row["p_short_neither"] = item.p_neither
    ordered: list[dict[str, float | int | str | bool | None]] = []
    for horizon in sorted(rows):
        row = rows[horizon]
        long_ev = row["long_ev"]
        short_ev = row["short_ev"]
        if isinstance(long_ev, int | float) and isinstance(short_ev, int | float):
            row["best"] = "LONG" if long_ev >= short_ev else "SHORT"
        elif isinstance(long_ev, int | float):
            row["best"] = "LONG"
        elif isinstance(short_ev, int | float):
            row["best"] = "SHORT"
        ordered.append(row)
    return ordered
