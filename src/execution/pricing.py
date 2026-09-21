from __future__ import annotations

from dataclasses import dataclass

from src.decision.actions import Action

BPS = 10_000.0


@dataclass(frozen=True, slots=True)
class FillQuote:
    intent: Action
    is_buy: bool
    best_bid: float
    best_ask: float
    raw_price: float
    fill_price: float
    spread_bps: float
    slippage_bps: float
    slippage_cost_per_unit: float


def bps_to_fraction(bps: float) -> float:
    return bps / BPS


def fee_on_notional(notional: float, fee_bps: float) -> float:
    return abs(notional) * bps_to_fraction(fee_bps)


def apply_adverse_slippage(price: float, *, is_buy: bool, slippage_bps: float) -> float:
    """Slippage always worsens the fill. Buys pay more; sells receive less."""
    fraction = bps_to_fraction(slippage_bps)
    if is_buy:
        return price * (1.0 + fraction)
    return price * (1.0 - fraction)


def spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.0
    return (ask - bid) / mid * BPS


def is_buy_for(intent: Action, *, current_long: bool) -> bool:
    if intent is Action.OPEN_LONG:
        return True
    if intent is Action.OPEN_SHORT:
        return False
    if intent is Action.CLOSE:
        return not current_long
    raise ValueError(f"intent {intent} does not map to a single buy/sell")


def quote_market_fill(
    *,
    intent: Action,
    bid: float,
    ask: float,
    slippage_bps: float,
    current_long: bool = False,
) -> FillQuote:
    if bid <= 0 or ask <= 0 or bid >= ask:
        raise ValueError("valid bid < ask required for paper execution")
    buy = is_buy_for(intent, current_long=current_long)
    raw = ask if buy else bid
    fill = apply_adverse_slippage(raw, is_buy=buy, slippage_bps=slippage_bps)
    return FillQuote(
        intent=intent,
        is_buy=buy,
        best_bid=bid,
        best_ask=ask,
        raw_price=raw,
        fill_price=fill,
        spread_bps=spread_bps(bid, ask),
        slippage_bps=slippage_bps,
        slippage_cost_per_unit=abs(fill - raw),
    )


def long_gross_pnl(quantity: float, entry_price: float, exit_price: float) -> float:
    return (exit_price - entry_price) * quantity


def short_gross_pnl(quantity: float, entry_price: float, exit_price: float) -> float:
    return (entry_price - exit_price) * quantity


def excursion_bps(*, side: str, entry_price: float, mark_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    if side == "LONG":
        return (mark_price - entry_price) / entry_price * BPS
    return (entry_price - mark_price) / entry_price * BPS


def conservative_mark(*, side: str, bid: float, ask: float) -> float:
    """Executable mark: LONG at bid, SHORT at ask. Never mid."""
    if side == "LONG":
        return bid
    if side == "SHORT":
        return ask
    return (bid + ask) / 2.0
