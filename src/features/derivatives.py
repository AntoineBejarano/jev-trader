from __future__ import annotations

from src.market.buffer import MarketBuffer, OpenInterestSample
from src.market.models import Derivatives


def _oi_at_or_before(samples: list[OpenInterestSample], target_ms: int) -> float | None:
    chosen: float | None = None
    for sample in samples:
        if sample.ts_ms <= target_ms:
            chosen = sample.open_interest
        else:
            break
    return chosen


def compute_derivatives(
    buffer: MarketBuffer,
    now_ms: int,
    *,
    long_liquidations: float | None,
    short_liquidations: float | None,
    last_trade_price: float | None,
) -> Derivatives:
    mark = buffer.mark
    oi_now = buffer.open_interest[-1].open_interest if buffer.open_interest else None
    oi_1m = _oi_at_or_before(list(buffer.open_interest), now_ms - 60_000)
    oi_5m = _oi_at_or_before(list(buffer.open_interest), now_ms - 300_000)
    basis = None
    if mark.mark_price is not None:
        spot = mark.index_price if mark.index_price is not None else last_trade_price
        if spot is not None and spot != 0:
            basis = mark.mark_price / spot - 1.0
    return Derivatives(
        open_interest=oi_now,
        oi_change_1m=None if oi_now is None or oi_1m is None else oi_now - oi_1m,
        oi_change_5m=None if oi_now is None or oi_5m is None else oi_now - oi_5m,
        funding=mark.funding_rate,
        mark_price=mark.mark_price,
        index_price=mark.index_price,
        basis=basis,
        long_liquidations=long_liquidations,
        short_liquidations=short_liquidations,
    )
