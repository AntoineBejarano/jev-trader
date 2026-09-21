from __future__ import annotations

from dataclasses import dataclass, field


def book_imbalance(bid_qty: float, ask_qty: float) -> float | None:
    total = bid_qty + ask_qty
    if total <= 0.0:
        return None
    return (bid_qty - ask_qty) / total


def _qty_at_levels(levels: list[tuple[float, float]], depth: int) -> float:
    return sum(qty for _, qty in levels[:depth])


@dataclass
class LocalOrderBook:
    """Top-of-book plus a compact depth ladder. Instrument-agnostic."""

    last_update_id: int | None = None
    event_time_ms: int | None = None
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    best_bid: float | None = None
    best_bid_qty: float | None = None
    best_ask: float | None = None
    best_ask_qty: float | None = None
    ticker_update_id: int | None = None
    ticker_time_ms: int | None = None
    valid: bool = False

    def apply_partial_snapshot(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        last_update_id: int,
        event_time_ms: int,
    ) -> None:
        self.bids = sorted(bids, key=lambda level: level[0], reverse=True)
        self.asks = sorted(asks, key=lambda level: level[0])
        self.last_update_id = last_update_id
        self.event_time_ms = event_time_ms
        if self.bids:
            self.best_bid, self.best_bid_qty = self.bids[0]
        if self.asks:
            self.best_ask, self.best_ask_qty = self.asks[0]
        self.valid = bool(self.bids and self.asks and self.best_bid and self.best_ask)
        if self.valid and self.best_bid is not None and self.best_ask is not None:
            self.valid = self.best_bid < self.best_ask

    def apply_book_ticker(
        self,
        bid: float,
        bid_qty: float,
        ask: float,
        ask_qty: float,
        update_id: int,
        event_time_ms: int,
    ) -> None:
        if bid <= 0 or ask <= 0 or bid >= ask:
            return
        self.best_bid = bid
        self.best_bid_qty = bid_qty
        self.best_ask = ask
        self.best_ask_qty = ask_qty
        self.ticker_update_id = update_id
        self.ticker_time_ms = event_time_ms
        if self.bids:
            self.bids[0] = (bid, bid_qty)
        else:
            self.bids = [(bid, bid_qty)]
        if self.asks:
            self.asks[0] = (ask, ask_qty)
        else:
            self.asks = [(ask, ask_qty)]
        self.valid = True

    def clear(self) -> None:
        self.last_update_id = None
        self.event_time_ms = None
        self.bids = []
        self.asks = []
        self.best_bid = None
        self.best_bid_qty = None
        self.best_ask = None
        self.best_ask_qty = None
        self.ticker_update_id = None
        self.ticker_time_ms = None
        self.valid = False

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float | None:
        mid = self.mid
        spread = self.spread
        if mid is None or spread is None or mid <= 0:
            return None
        return (spread / mid) * 10_000.0

    def imbalance(self, levels: int) -> float | None:
        if levels <= 0:
            return None
        if levels == 1:
            bid_qty = self.best_bid_qty or 0.0
            ask_qty = self.best_ask_qty or 0.0
            return book_imbalance(bid_qty, ask_qty)
        return book_imbalance(_qty_at_levels(self.bids, levels), _qty_at_levels(self.asks, levels))

    def last_update_ms(self) -> int | None:
        times = [value for value in (self.event_time_ms, self.ticker_time_ms) if value is not None]
        return max(times) if times else None
