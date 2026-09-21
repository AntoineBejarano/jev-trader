from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.market.buffer import MarketBuffer, Trade
from src.market.models import MarketHealth
from src.market.orderbook import LocalOrderBook

logger = logging.getLogger(__name__)


@dataclass
class InstrumentRuntime:
    symbol: str
    exchange: str
    buffer: MarketBuffer
    book: LocalOrderBook = field(default_factory=LocalOrderBook)

    def mark_books_stale(self) -> None:
        self.book.clear()


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(value)


def _levels(raw: list[list[str]] | list[tuple[str, str]]) -> list[tuple[float, float]]:
    return [(_as_float(price), _as_float(qty)) for price, qty in raw]


def combined_stream_url(base: str, streams: list[str]) -> str:
    joined = "/".join(streams)
    return f"{base.rstrip('/')}/stream?streams={joined}"


def public_streams_for(symbol: str) -> list[str]:
    lower = symbol.lower()
    return [f"{lower}@depth20@100ms", f"{lower}@bookTicker"]


def market_streams_for(symbol: str) -> list[str]:
    lower = symbol.lower()
    return [f"{lower}@aggTrade", f"{lower}@markPrice@1s", f"{lower}@forceOrder"]


def unwrap_combined(payload: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if "stream" in payload and "data" in payload and isinstance(payload["data"], dict):
        stream = str(payload["stream"])
        return stream, payload["data"]
    return None, payload


def apply_depth(runtime: InstrumentRuntime, data: dict[str, Any]) -> None:
    received_at = runtime.buffer.clock()
    if "bids" in data and "asks" in data:
        last_update_id = _as_int(data.get("lastUpdateId") or data.get("u") or 0)
        runtime.book.apply_partial_snapshot(
            bids=_levels(data["bids"]),
            asks=_levels(data["asks"]),
            last_update_id=last_update_id,
            event_time_ms=received_at,
        )
    elif "b" in data and "a" in data:
        last_update_id = _as_int(data.get("u") or 0)
        runtime.book.apply_partial_snapshot(
            bids=_levels(data["b"]),
            asks=_levels(data["a"]),
            last_update_id=last_update_id,
            event_time_ms=received_at,
        )
    mid = runtime.book.mid
    if mid is not None:
        runtime.buffer.record_mid(mid, received_at)


def apply_book_ticker(runtime: InstrumentRuntime, data: dict[str, Any]) -> None:
    bid = _as_float(data["b"])
    ask = _as_float(data["a"])
    bid_qty = _as_float(data["B"])
    ask_qty = _as_float(data["A"])
    update_id = _as_int(data.get("u") or 0)
    received_at = runtime.buffer.clock()
    runtime.book.apply_book_ticker(bid, bid_qty, ask, ask_qty, update_id, received_at)
    mid = runtime.book.mid
    if mid is not None:
        runtime.buffer.record_mid(mid, received_at)


def apply_agg_trade(runtime: InstrumentRuntime, data: dict[str, Any]) -> None:
    # Binance `m` is true when the buyer is the maker, so the seller was the aggressor.
    buyer_is_maker = bool(data.get("m"))
    trade = Trade(
        ts_ms=_as_int(data.get("T") or data.get("E") or runtime.buffer.clock()),
        price=_as_float(data["p"]),
        qty=_as_float(data["q"]),
        is_aggressor_buy=not buyer_is_maker,
    )
    runtime.buffer.record_trade(trade)


def apply_mark_price(runtime: InstrumentRuntime, data: dict[str, Any]) -> None:
    runtime.buffer.record_mark(
        mark_price=_as_float(data["p"]),
        index_price=_as_float(data["i"]) if data.get("i") is not None else None,
        funding_rate=_as_float(data["r"]) if data.get("r") is not None else None,
        next_funding_time_ms=_as_int(data["T"]) if data.get("T") is not None else None,
        ts_ms=_as_int(data.get("E") or runtime.buffer.clock()),
    )


def apply_force_order(runtime: InstrumentRuntime, data: dict[str, Any]) -> None:
    order = data.get("o") or data
    side = str(order.get("S") or "")
    qty = _as_float(order.get("z") or order.get("q") or 0)
    price = _as_float(order.get("ap") or order.get("p") or 0)
    if qty <= 0 or price <= 0:
        return
    # A sell liquidation is an aggressive sell (longs liquidated).
    is_aggressor_buy = side.upper() == "BUY"
    trade = Trade(
        ts_ms=_as_int(order.get("T") or data.get("E") or runtime.buffer.clock()),
        price=price,
        qty=qty,
        is_aggressor_buy=is_aggressor_buy,
        is_liquidation=True,
    )
    runtime.buffer.record_trade(trade)


def apply_open_interest(runtime: InstrumentRuntime, payload: dict[str, Any]) -> None:
    oi = _as_float(payload["openInterest"])
    ts_ms = _as_int(payload.get("time") or runtime.buffer.clock())
    runtime.buffer.record_open_interest(oi, ts_ms)


def dispatch_stream_message(
    runtime: InstrumentRuntime, stream: str | None, data: dict[str, Any]
) -> None:
    event = str(data.get("e") or "")
    stream_name = stream or ""
    if "bookTicker" in stream_name or event == "bookTicker":
        apply_book_ticker(runtime, data)
        return
    if "aggTrade" in stream_name or event == "aggTrade":
        apply_agg_trade(runtime, data)
        return
    if "markPrice" in stream_name or event == "markPriceUpdate":
        apply_mark_price(runtime, data)
        return
    if "forceOrder" in stream_name or event == "forceOrder":
        apply_force_order(runtime, data)
        return
    if "depth" in stream_name or event == "depthUpdate" or "bids" in data:
        apply_depth(runtime, data)
        return
    logger.debug("Unhandled stream payload for %s: %s", runtime.symbol, stream_name or event)


def connection_health(
    *,
    symbol: str,
    exchange: str,
    ws_public_connected: bool,
    ws_market_connected: bool,
) -> MarketHealth:
    reasons: list[str] = []
    if not ws_public_connected:
        reasons.append("ws_public_disconnected")
    if not ws_market_connected:
        reasons.append("ws_market_disconnected")
    stale = bool(reasons)
    return MarketHealth(
        trading_allowed=not stale,
        stale=stale,
        warmup_ready=False,
        ws_public_connected=ws_public_connected,
        ws_market_connected=ws_market_connected,
        reasons=tuple(reasons),
    )
