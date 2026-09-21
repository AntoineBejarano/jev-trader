from __future__ import annotations

from dataclasses import dataclass

from src.execution.pricing import long_gross_pnl, short_gross_pnl
from src.market.models import Side

RECONCILE_TOLERANCE_USDT = 1e-8


@dataclass(frozen=True, slots=True)
class RoundTripAttribution:
    """Explain a completed trade. Fills remain the PnL source of truth.

    Gross PnL is computed from fill prices. Spread and slippage here are
    attribution only — they must not be subtracted again from net PnL.
    """

    market_move_pnl: float
    entry_spread_cost: float
    exit_spread_cost: float
    total_spread_cost: float
    entry_slippage_cost: float
    exit_slippage_cost: float
    total_slippage_cost: float
    entry_fee: float
    exit_fee: float
    total_fees: float
    funding_cost: float
    gross_pnl: float
    net_pnl: float
    reconstructed_net_pnl: float
    residual: float

    def as_dict(self) -> dict[str, float]:
        return {
            "market_move_pnl": self.market_move_pnl,
            "entry_spread_cost": self.entry_spread_cost,
            "exit_spread_cost": self.exit_spread_cost,
            "total_spread_cost": self.total_spread_cost,
            "entry_slippage_cost": self.entry_slippage_cost,
            "exit_slippage_cost": self.exit_slippage_cost,
            "total_slippage_cost": self.total_slippage_cost,
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "total_fees": self.total_fees,
            "funding_cost": self.funding_cost,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "reconstructed_net_pnl": self.reconstructed_net_pnl,
            "residual": self.residual,
        }


def _mid(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0


def attribute_round_trip(
    *,
    side: Side,
    quantity: float,
    entry_fill: float,
    entry_bid: float,
    entry_ask: float,
    exit_fill: float,
    exit_bid: float,
    exit_ask: float,
    entry_fee: float,
    exit_fee: float,
    funding_cost: float = 0.0,
) -> RoundTripAttribution:
    """Decompose fill PnL into mid-to-mid move, spread, slippage, fees, funding."""
    entry_mid = _mid(entry_bid, entry_ask)
    exit_mid = _mid(exit_bid, exit_ask)
    if side is Side.LONG:
        gross = long_gross_pnl(quantity, entry_fill, exit_fill)
        market_move = (exit_mid - entry_mid) * quantity
        entry_spread = (entry_ask - entry_mid) * quantity
        exit_spread = (exit_mid - exit_bid) * quantity
        entry_slip = (entry_fill - entry_ask) * quantity
        exit_slip = (exit_bid - exit_fill) * quantity
    elif side is Side.SHORT:
        gross = short_gross_pnl(quantity, entry_fill, exit_fill)
        market_move = (entry_mid - exit_mid) * quantity
        entry_spread = (entry_mid - entry_bid) * quantity
        exit_spread = (exit_ask - exit_mid) * quantity
        entry_slip = (entry_bid - entry_fill) * quantity
        exit_slip = (exit_fill - exit_ask) * quantity
    else:
        raise ValueError("attribution requires LONG or SHORT")
    total_spread = entry_spread + exit_spread
    total_slip = entry_slip + exit_slip
    total_fees = entry_fee + exit_fee
    net = gross - total_fees - funding_cost
    reconstructed = market_move - total_spread - total_slip - total_fees - funding_cost
    return RoundTripAttribution(
        market_move_pnl=market_move,
        entry_spread_cost=entry_spread,
        exit_spread_cost=exit_spread,
        total_spread_cost=total_spread,
        entry_slippage_cost=entry_slip,
        exit_slippage_cost=exit_slip,
        total_slippage_cost=total_slip,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        total_fees=total_fees,
        funding_cost=funding_cost,
        gross_pnl=gross,
        net_pnl=net,
        reconstructed_net_pnl=reconstructed,
        residual=net - reconstructed,
    )


def pnl_on_notional_bps(pnl_usdt: float, notional: float) -> float:
    if notional == 0:
        return 0.0
    return pnl_usdt / notional * 10_000.0


def pnl_on_equity_bps(pnl_usdt: float, starting_equity: float) -> float:
    if starting_equity == 0:
        return 0.0
    return pnl_usdt / starting_equity * 10_000.0


def pnl_on_equity_pct(pnl_usdt: float, starting_equity: float) -> float:
    if starting_equity == 0:
        return 0.0
    return pnl_usdt / starting_equity


def profit_giveback_bps(mfe_bps: float | None, current_pnl_bps: float | None) -> float | None:
    """MFE minus current PnL when there was a favorable excursion. Context only."""
    if mfe_bps is None or current_pnl_bps is None:
        return None
    if mfe_bps <= 0:
        return 0.0
    return mfe_bps - current_pnl_bps
