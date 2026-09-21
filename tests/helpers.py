from __future__ import annotations

from src.market.models import (
    MarketHealth,
    MarketState,
    Microstructure,
    PositionState,
    Side,
    Volatility,
)


def make_market_state(
    *,
    symbol: str = "ETHUSDT",
    exchange: str = "binance_usdm",
    side: Side = Side.FLAT,
    stale: bool = False,
    trading_allowed: bool = True,
    warmup_ready: bool = True,
    price: float = 3500.0,
    bid: float | None = None,
    ask: float | None = None,
    age_seconds: float = 0.0,
    quantity: float = 0.0,
    reasons: tuple[str, ...] = (),
    spread_bps: float | None = None,
    rv_1m: float | None = None,
    rv_30s: float | None = None,
) -> MarketState:
    best_bid = bid if bid is not None else price - 0.05
    best_ask = ask if ask is not None else price + 0.05
    health_reasons = reasons
    if not warmup_ready:
        health_reasons = health_reasons + ("warmup_incomplete",)
    if stale:
        health_reasons = health_reasons + ("stale_order_book",)
    return MarketState(
        timestamp="2026-09-21T14:00:00+00:00",
        timestamp_ms=1_000_000,
        symbol=symbol,
        exchange=exchange,
        price=price,
        last_trade_price=price,
        best_bid=best_bid,
        best_ask=best_ask,
        position=PositionState(
            symbol=symbol,
            exchange=exchange,
            side=side,
            current_price=price,
            entry_price=None if side is Side.FLAT else price,
            quantity=quantity,
            age_seconds=age_seconds,
        ),
        health=MarketHealth(
            trading_allowed=trading_allowed and not stale and warmup_ready,
            stale=stale,
            warmup_ready=warmup_ready,
            ws_public_connected=not stale,
            ws_market_connected=not stale,
            reasons=health_reasons,
        ),
        microstructure=Microstructure(spread_bps=spread_bps),
        volatility=Volatility(rv_30s=rv_30s, rv_1m=rv_1m),
    )
