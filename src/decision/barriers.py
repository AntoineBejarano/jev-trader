from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import Settings
from src.decision.costs import cost_snapshot
from src.market.models import MarketState, Volatility


@dataclass(frozen=True, slots=True)
class Barrier:
    horizon_seconds: int
    target_bps: float
    stop_bps: float
    sigma_bps: float
    cost_floor_bps: float
    volatility_target_bps: float = 0.0
    target_bound_by_cost_floor: bool = False
    horizon_cost_constrained: bool = False


class BarrierPlanner:
    """Volatility- and cost-aware barriers. Not fitted to historical PnL."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def plan(self, state: MarketState) -> tuple[Barrier, ...]:
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
        cost_floor = cost.round_trip_cost_bps + self.settings.scalp_profit_buffer_bps
        return tuple(
            self._barrier(horizon, state.volatility, cost_floor)
            for horizon in self.settings.scalp_horizons
        )

    def _barrier(self, horizon: int, vol: Volatility, cost_floor: float) -> Barrier:
        sigma = _sigma_bps(vol, horizon)
        vol_target = max(
            self.settings.scalp_min_target_bps,
            sigma * self.settings.scalp_target_vol_mult,
        )
        bound_by_floor = cost_floor > vol_target
        target = max(cost_floor, vol_target)
        stop = max(self.settings.scalp_min_stop_bps, sigma * self.settings.scalp_stop_vol_mult)
        return Barrier(
            horizon_seconds=horizon,
            target_bps=target,
            stop_bps=stop,
            sigma_bps=sigma,
            cost_floor_bps=cost_floor,
            volatility_target_bps=vol_target,
            target_bound_by_cost_floor=bound_by_floor,
            horizon_cost_constrained=bound_by_floor,
        )


def _sigma_bps(vol: Volatility, horizon_seconds: int) -> float:
    """Scale available realized-vol windows to horizon H. RV is log-move RMS, not annualized."""
    anchors: list[tuple[int, float]] = []
    if vol.rv_30s is not None:
        anchors.append((30, vol.rv_30s * 10_000.0))
    if vol.rv_1m is not None:
        anchors.append((60, vol.rv_1m * 10_000.0))
    if vol.rv_5m is not None:
        anchors.append((300, vol.rv_5m * 10_000.0))
    if vol.rv_15m is not None:
        anchors.append((900, vol.rv_15m * 10_000.0))
    if not anchors:
        return 0.0
    nearest_seconds, nearest_sigma = min(anchors, key=lambda item: abs(item[0] - horizon_seconds))
    if nearest_seconds <= 0 or nearest_sigma <= 0:
        return 0.0
    scale = (horizon_seconds / nearest_seconds) ** 0.5
    return float(nearest_sigma * scale)
