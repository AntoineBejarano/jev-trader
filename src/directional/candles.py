from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

WINDOW_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "10m": 600_000,
    "15m": 900_000,
    "30m": 1_800_000,
}
ROLLING_HORIZONS_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "10m": 600_000,
    "30m": 1_800_000,
}
ROLLING_HORIZONS_SECONDS: tuple[int, ...] = (60, 300, 600, 1800)


def aligned_start(ts_ms: int, window_ms: int) -> int:
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    return (ts_ms // window_ms) * window_ms


def aligned_end(ts_ms: int, window_ms: int) -> int:
    return aligned_start(ts_ms, window_ms) + window_ms


@dataclass(frozen=True, slots=True)
class Candle:
    timeframe: str
    start_ms: int
    end_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    complete: bool = True

    @property
    def direction(self) -> str:
        return "UP" if self.close > self.open else "DOWN"

    @property
    def body_bps(self) -> float:
        if self.open <= 0:
            return 0.0
        return (self.close - self.open) / self.open * 10_000.0

    @property
    def range_bps(self) -> float:
        if self.open <= 0:
            return 0.0
        return (self.high - self.low) / self.open * 10_000.0

    @property
    def upper_wick_ratio(self) -> float:
        width = self.high - self.low
        if width <= 0:
            return 0.0
        top = self.high - max(self.open, self.close)
        return top / width

    @property
    def lower_wick_ratio(self) -> float:
        width = self.high - self.low
        if width <= 0:
            return 0.0
        bottom = min(self.open, self.close) - self.low
        return bottom / width

    @property
    def close_location(self) -> float:
        width = self.high - self.low
        if width <= 0:
            return 0.5
        return (self.close - self.low) / width


def candle_from_mids(
    timestamps: Sequence[int],
    values: Sequence[float],
    *,
    timeframe: str,
    start_ms: int,
    volume: float = 0.0,
    now_ms: int | None = None,
) -> Candle | None:
    window_ms = WINDOW_MS[timeframe]
    end_ms = start_ms + window_ms
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    for ts_ms, price in zip(timestamps, values, strict=True):
        if ts_ms < start_ms or ts_ms >= end_ms:
            continue
        if price <= 0:
            continue
        opens.append(price)
        highs.append(price)
        lows.append(price)
        closes.append(price)
    if not opens:
        return None
    complete = True if now_ms is None else now_ms >= end_ms
    return Candle(
        timeframe=timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
        open=opens[0],
        high=max(highs),
        low=min(lows),
        close=closes[-1],
        volume=volume,
        complete=complete,
    )


def aggregate_10m(candles_1m: Sequence[Candle]) -> list[Candle]:
    """Deterministic 10m bars from completed 1m candles. Not a Binance 10m kline."""
    grouped: dict[int, list[Candle]] = {}
    for candle in candles_1m:
        if candle.timeframe != "1m" or not candle.complete:
            continue
        start = aligned_start(candle.start_ms, WINDOW_MS["10m"])
        grouped.setdefault(start, []).append(candle)
    out: list[Candle] = []
    for start, group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: item.start_ms)
        if len(group) < 10:
            continue
        first = group[0]
        last = group[-1]
        out.append(
            Candle(
                timeframe="10m",
                start_ms=start,
                end_ms=start + WINDOW_MS["10m"],
                open=first.open,
                high=max(item.high for item in group),
                low=min(item.low for item in group),
                close=last.close,
                volume=sum(item.volume for item in group),
                complete=True,
            )
        )
    return out


def rolling_direction(*, reference_mid: float, future_mid: float) -> str:
    return "UP" if future_mid > reference_mid else "DOWN"


def rolling_return_bps(*, reference_mid: float, future_mid: float) -> float:
    if reference_mid <= 0:
        return 0.0
    return (future_mid / reference_mid - 1.0) * 10_000.0


def streak_hit(directions: Sequence[str], *, count: int, side: str) -> bool | None:
    if len(directions) < count:
        return None
    return all(item == side for item in directions[:count])


def candle_summary(
    candle: Candle, *, volume_baseline: float | None = None
) -> dict[str, float | int | str | bool | None]:
    vol_rel = None
    if volume_baseline is not None and volume_baseline > 0:
        vol_rel = candle.volume / volume_baseline
    return {
        "timeframe": candle.timeframe,
        "window_start": candle.start_ms,
        "window_end": candle.end_ms,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "direction": candle.direction,
        "complete": candle.complete,
        "body_bps": candle.body_bps,
        "range_bps": candle.range_bps,
        "upper_wick_ratio": candle.upper_wick_ratio,
        "lower_wick_ratio": candle.lower_wick_ratio,
        "close_location": candle.close_location,
        "volume_vs_baseline": vol_rel,
    }
