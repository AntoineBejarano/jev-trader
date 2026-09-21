from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

Clock = Callable[[], int]

# Seeded OHLC mids must outlive the live-mid window so r_1h lookback exists.
SEEDED_PRICE_CONTEXT_MS = 2 * 60 * 60 * 1000 + 10 * 60 * 1000


def wall_clock_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True, slots=True)
class MidSample:
    ts_ms: int
    mid: float
    seeded: bool = False


@dataclass(frozen=True, slots=True)
class Trade:
    ts_ms: int
    price: float
    qty: float
    is_aggressor_buy: bool
    is_liquidation: bool = False


@dataclass(frozen=True, slots=True)
class OpenInterestSample:
    ts_ms: int
    open_interest: float


@dataclass
class MarkSnapshot:
    ts_ms: int | None = None
    mark_price: float | None = None
    index_price: float | None = None
    funding_rate: float | None = None
    next_funding_time_ms: int | None = None


@dataclass
class MarketBuffer:
    """Rolling per-instrument history. One instance per symbol."""

    symbol: str
    exchange: str
    history_ms: int
    mid_sample_interval_ms: int = 100
    clock: Clock = wall_clock_ms
    mids: deque[MidSample] = field(default_factory=deque)
    trades: deque[Trade] = field(default_factory=deque)
    open_interest: deque[OpenInterestSample] = field(default_factory=deque)
    mark: MarkSnapshot = field(default_factory=MarkSnapshot)
    last_trade_price: float | None = None
    last_trade_ts_ms: int | None = None
    last_mid: float | None = None
    last_mid_ts_ms: int | None = None
    last_event_ts_ms: int | None = None
    session_cvd: float = 0.0

    def record_mid(self, mid: float, ts_ms: int | None = None) -> None:
        ts = ts_ms if ts_ms is not None else self.clock()
        self.last_mid = mid
        self.last_mid_ts_ms = ts
        self.last_event_ts_ms = ts
        if self.mids and ts - self.mids[-1].ts_ms < self.mid_sample_interval_ms:
            # Keep the bucket's original timestamp. Updating it to `ts` would
            # reset the interval on every bookTicker and collapse the series
            # to one sample during liquid BTC markets.
            self.mids[-1] = MidSample(ts_ms=self.mids[-1].ts_ms, mid=mid)
            self._evict(ts)
            return
        self.mids.append(MidSample(ts_ms=ts, mid=mid, seeded=False))
        self._evict(ts)

    def seed_historical_mids(self, samples: list[tuple[int, float]]) -> None:
        """Insert deterministic historical mids. Does not count toward live warmup."""
        if not samples:
            return
        live = [sample for sample in self.mids if not sample.seeded]
        seeded = [
            MidSample(ts_ms=ts_ms, mid=mid, seeded=True)
            for ts_ms, mid in samples
            if ts_ms > 0 and mid > 0
        ]
        merged = sorted(seeded + live, key=lambda item: item.ts_ms)
        deduped: list[MidSample] = []
        seen: set[int] = set()
        for sample in merged:
            if sample.ts_ms in seen:
                continue
            seen.add(sample.ts_ms)
            deduped.append(sample)
        self.mids = deque(deduped)
        if self.last_mid is None and deduped:
            self.last_mid = deduped[-1].mid
            self.last_mid_ts_ms = deduped[-1].ts_ms
        now = self.clock()
        self._evict(now)

    def live_mids(self) -> list[MidSample]:
        return [sample for sample in self.mids if not sample.seeded]

    def record_trade(self, trade: Trade) -> None:
        self.trades.append(trade)
        self.last_trade_price = trade.price
        self.last_trade_ts_ms = trade.ts_ms
        self.last_event_ts_ms = trade.ts_ms
        signed = trade.qty if trade.is_aggressor_buy else -trade.qty
        self.session_cvd += signed
        self._evict(trade.ts_ms)

    def record_open_interest(self, open_interest: float, ts_ms: int | None = None) -> None:
        ts = ts_ms if ts_ms is not None else self.clock()
        self.open_interest.append(OpenInterestSample(ts_ms=ts, open_interest=open_interest))
        self._evict(ts)

    def record_mark(
        self,
        *,
        mark_price: float,
        index_price: float | None,
        funding_rate: float | None,
        next_funding_time_ms: int | None,
        ts_ms: int,
    ) -> None:
        self.mark = MarkSnapshot(
            ts_ms=ts_ms,
            mark_price=mark_price,
            index_price=index_price,
            funding_rate=funding_rate,
            next_funding_time_ms=next_funding_time_ms,
        )
        self.last_event_ts_ms = ts_ms

    def mids_as_lists(self) -> tuple[list[int], list[float]]:
        timestamps = [sample.ts_ms for sample in self.mids]
        values = [sample.mid for sample in self.mids]
        return timestamps, values

    def trades_in_window(self, now_ms: int, window_ms: int) -> list[Trade]:
        cutoff = now_ms - window_ms
        return [trade for trade in self.trades if trade.ts_ms >= cutoff]

    def _evict(self, now_ms: int) -> None:
        live_cutoff = now_ms - self.history_ms
        seeded_cutoff = now_ms - max(self.history_ms, SEEDED_PRICE_CONTEXT_MS)
        kept: deque[MidSample] = deque()
        for sample in self.mids:
            cutoff = seeded_cutoff if sample.seeded else live_cutoff
            if sample.ts_ms >= cutoff:
                kept.append(sample)
        self.mids = kept
        while self.trades and self.trades[0].ts_ms < live_cutoff:
            self.trades.popleft()
        while self.open_interest and self.open_interest[0].ts_ms < live_cutoff:
            self.open_interest.popleft()
