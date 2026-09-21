from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.market.buffer import MarketBuffer

# r_15m / r_1h are serialized into Jev MarketState when present. They are
# intended context, not optional research gates. Bootstrap only deterministic
# USD-M kline OHLC closes so those returns exist without waiting one live hour.
# Do not fabricate order-book or aggressive-flow history.

KLINE_INTERVAL = "1m"
# 1m USD-M klines: enough deterministic OHLC to populate r_1h / 30m candles
# immediately. Do not fabricate order-book or aggressive-flow history.
KLINE_LIMIT = 130


def klines_to_mid_samples(klines: Sequence[Sequence[Any]]) -> list[tuple[int, float]]:
    """Map Binance USD-M klines to (timestamp_ms, close) samples.

    Each bar contributes the open at open-time and the close at close-time.
    Intra-bar high/low order is unknown, so those prices are not invented.
    """
    samples: list[tuple[int, float]] = []
    seen: set[int] = set()
    for row in klines:
        if len(row) < 7:
            continue
        try:
            open_time = int(row[0])
            open_price = float(row[1])
            close_price = float(row[4])
            close_time = int(row[6])
        except (TypeError, ValueError):
            continue
        for ts_ms, price in ((open_time, open_price), (close_time, close_price)):
            if ts_ms <= 0 or price <= 0 or ts_ms in seen:
                continue
            seen.add(ts_ms)
            samples.append((ts_ms, price))
    samples.sort(key=lambda item: item[0])
    return samples


def seed_buffer_from_klines(buffer: MarketBuffer, klines: Sequence[Sequence[Any]]) -> int:
    samples = klines_to_mid_samples(klines)
    buffer.seed_historical_mids(samples)
    return len(samples)
