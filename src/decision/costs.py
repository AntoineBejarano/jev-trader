from __future__ import annotations

from dataclasses import dataclass

# 1 bps = 1e-4 of price. Kept local to avoid an import cycle with execution.pricing.
BPS = 10_000.0

# Assumed VIP 0 USD-M regular user, no BNB discount.
# Official example: maker 0.02% = 2 bps, taker 0.05% = 5 bps.
# This is an ASSUMED PAPER COST MODEL, not the owner's live account rate.
ASSUMED_USDM_MAKER_FEE_BPS = 2.0
ASSUMED_USDM_TAKER_FEE_BPS = 5.0
ASSUMED_COST_SOURCE = "binance_usdm_vip0_regular_no_bnb"


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    maker_fee_bps: float
    taker_fee_bps: float
    slippage_bps: float
    spread_bps: float
    entry_half_spread_bps: float
    expected_exit_half_spread_bps: float
    entry_leg_cost_bps: float
    exit_leg_cost_bps: float
    round_trip_cost_bps: float
    assumed: bool
    source: str


def one_way_cost_bps(
    *,
    taker_fee_bps: float,
    slippage_bps: float,
    half_spread_bps: float,
) -> float:
    """Single executable leg versus mid: taker + slippage + half-spread."""
    return taker_fee_bps + slippage_bps + max(0.0, half_spread_bps)


def round_trip_cost_bps(
    *,
    taker_fee_bps: float,
    spread_bps: float,
    slippage_bps: float,
    expected_exit_spread_bps: float | None = None,
) -> float:
    """Taker round-trip versus mid, matching conservative paper fills.

    LONG entry crosses the ask; LONG exit crosses the bid (and vice versa for SHORT).
    That pays half-spread on each leg, adverse slippage on both legs, and taker twice.

    When expected_exit_spread is omitted, both legs use the current spread:

    round_trip_cost_bps =
        2 * taker_fee_bps
        + spread_bps
        + 2 * slippage_bps
    """
    entry_half = max(0.0, spread_bps) / 2.0
    exit_spread = spread_bps if expected_exit_spread_bps is None else expected_exit_spread_bps
    exit_half = max(0.0, exit_spread) / 2.0
    return (
        one_way_cost_bps(
            taker_fee_bps=taker_fee_bps, slippage_bps=slippage_bps, half_spread_bps=entry_half
        )
        + one_way_cost_bps(
            taker_fee_bps=taker_fee_bps, slippage_bps=slippage_bps, half_spread_bps=exit_half
        )
    )


def cost_snapshot(
    *,
    taker_fee_bps: float,
    maker_fee_bps: float,
    spread_bps: float | None,
    slippage_bps: float,
    assumed: bool,
    source: str,
    expected_exit_spread_bps: float | None = None,
) -> CostSnapshot:
    spread = 0.0 if spread_bps is None else spread_bps
    exit_spread = spread if expected_exit_spread_bps is None else expected_exit_spread_bps
    entry_half = max(0.0, spread) / 2.0
    exit_half = max(0.0, exit_spread) / 2.0
    entry_leg = one_way_cost_bps(
        taker_fee_bps=taker_fee_bps, slippage_bps=slippage_bps, half_spread_bps=entry_half
    )
    exit_leg = one_way_cost_bps(
        taker_fee_bps=taker_fee_bps, slippage_bps=slippage_bps, half_spread_bps=exit_half
    )
    return CostSnapshot(
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=taker_fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread,
        entry_half_spread_bps=entry_half,
        expected_exit_half_spread_bps=exit_half,
        entry_leg_cost_bps=entry_leg,
        exit_leg_cost_bps=exit_leg,
        round_trip_cost_bps=entry_leg + exit_leg,
        assumed=assumed,
        source=source,
    )


def bps_to_price_move(price: float, bps: float) -> float:
    return price * (bps / BPS)
