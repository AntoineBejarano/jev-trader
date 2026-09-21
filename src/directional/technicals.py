from __future__ import annotations

import math
from collections.abc import Sequence


def ema(values: Sequence[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    current = sum(values[:period]) / period
    for value in values[period:]:
        current = alpha * value + (1.0 - alpha) * current
    return current


def ema_series(values: Sequence[float], period: int) -> list[float]:
    if period <= 0 or len(values) < period:
        return []
    alpha = 2.0 / (period + 1.0)
    current = sum(values[:period]) / period
    out = [current]
    for value in values[period:]:
        current = alpha * value + (1.0 - alpha) * current
        out.append(current)
    return out


def rsi(values: Sequence[float], period: int = 14) -> float | None:
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for index in range(1, period + 1):
        delta = values[index] - values[index - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    for index in range(period + 1, len(values)):
        delta = values[index] - values[index - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger_z(values: Sequence[float], period: int = 20) -> float | None:
    if len(values) < period:
        return None
    window = values[-period:]
    mean = sum(window) / period
    var = sum((item - mean) ** 2 for item in window) / period
    std = math.sqrt(var)
    if std <= 0:
        return 0.0
    return (window[-1] - mean) / std


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None
    trs: list[float] = []
    for index in range(1, len(closes)):
        tr = max(
            highs[index] - lows[index],
            abs(highs[index] - closes[index - 1]),
            abs(lows[index] - closes[index - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    if len(closes) < period * 2 + 1:
        return None
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    trs: list[float] = []
    for index in range(1, len(closes)):
        up = highs[index] - highs[index - 1]
        down = lows[index - 1] - lows[index]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        trs.append(
            max(
                highs[index] - lows[index],
                abs(highs[index] - closes[index - 1]),
                abs(lows[index] - closes[index - 1]),
            )
        )
    if len(trs) < period:
        return None

    def _smooth(values: list[float]) -> list[float]:
        first = sum(values[:period])
        out = [first]
        for value in values[period:]:
            out.append(out[-1] - (out[-1] / period) + value)
        return out

    atr_s = _smooth(trs)
    plus_s = _smooth(plus_dm)
    minus_s = _smooth(minus_dm)
    dx: list[float] = []
    for atr_v, plus_v, minus_v in zip(atr_s, plus_s, minus_s, strict=False):
        if atr_v <= 0:
            continue
        plus_di = 100.0 * plus_v / atr_v
        minus_di = 100.0 * minus_v / atr_v
        denom = plus_di + minus_di
        dx.append(0.0 if denom <= 0 else 100.0 * abs(plus_di - minus_di) / denom)
    if len(dx) < period:
        return None
    return sum(dx[-period:]) / period


def slope_bps(values: Sequence[float], lookback: int) -> float | None:
    if lookback <= 0 or len(values) < lookback + 1:
        return None
    start = values[-lookback - 1]
    end = values[-1]
    if start <= 0:
        return None
    return math.log(end / start) * 10_000.0


def technical_panel(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    *,
    price: float,
) -> dict[str, float | None]:
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema9_prev = ema(closes[:-1], 9) if len(closes) > 9 else None
    ema21_prev = ema(closes[:-1], 21) if len(closes) > 21 else None
    atr_v = atr(highs, lows, closes, 14)
    return {
        "ema9": ema9,
        "ema21": ema21,
        "price_to_ema9_bps": None if ema9 is None or ema9 <= 0 else (price / ema9 - 1.0) * 10_000.0,
        "price_to_ema21_bps": None
        if ema21 is None or ema21 <= 0
        else (price / ema21 - 1.0) * 10_000.0,
        "ema9_to_ema21_bps": None
        if ema9 is None or ema21 is None or ema21 <= 0
        else (ema9 / ema21 - 1.0) * 10_000.0,
        "ema9_slope_bps": None
        if ema9 is None or ema9_prev is None or ema9_prev <= 0
        else (ema9 / ema9_prev - 1.0) * 10_000.0,
        "ema21_slope_bps": None
        if ema21 is None or ema21_prev is None or ema21_prev <= 0
        else (ema21 / ema21_prev - 1.0) * 10_000.0,
        "rsi14": rsi(closes, 14),
        "bollinger_z": bollinger_z(closes, 20),
        "atr_bps": None if atr_v is None or price <= 0 else atr_v / price * 10_000.0,
        "adx14": adx(highs, lows, closes, 14),
        "log_slope_5_bps": slope_bps(closes, 5),
        "log_slope_14_bps": slope_bps(closes, 14),
    }
