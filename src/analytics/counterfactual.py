from __future__ import annotations

from src.execution.pricing import apply_adverse_slippage


def hypothetical_side_net_pnl_bps(
    *,
    side: str,
    entry_bid: float,
    entry_ask: float,
    exit_bid: float,
    exit_ask: float,
    taker_fee_bps: float,
    slippage_bps: float,
) -> float | None:
    """Realistic taker counterfactual: never mid. PnL in bps of entry fill notional."""
    if min(entry_bid, entry_ask, exit_bid, exit_ask) <= 0:
        return None
    if entry_bid >= entry_ask or exit_bid >= exit_ask:
        return None
    if side == "LONG":
        entry = apply_adverse_slippage(entry_ask, is_buy=True, slippage_bps=slippage_bps)
        exit_px = apply_adverse_slippage(exit_bid, is_buy=False, slippage_bps=slippage_bps)
        gross_bps = (exit_px - entry) / entry * 10_000.0
    elif side == "SHORT":
        entry = apply_adverse_slippage(entry_bid, is_buy=False, slippage_bps=slippage_bps)
        exit_px = apply_adverse_slippage(exit_ask, is_buy=True, slippage_bps=slippage_bps)
        gross_bps = (entry - exit_px) / entry * 10_000.0
    else:
        return None
    return gross_bps - 2.0 * taker_fee_bps


def book_from_market_state(payload: dict[str, object]) -> tuple[float | None, float | None]:
    bid = payload.get("best_bid")
    ask = payload.get("best_ask")
    bid_f = float(bid) if isinstance(bid, int | float) else None
    ask_f = float(ask) if isinstance(ask, int | float) else None
    return bid_f, ask_f
