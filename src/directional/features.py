from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from src.directional.candles import (
    WINDOW_MS,
    aggregate_10m,
    aligned_end,
    aligned_start,
    candle_from_mids,
    candle_summary,
)
from src.directional.constants import FEATURE_SCHEMA_VERSION
from src.directional.technicals import technical_panel
from src.features.flow import aggressive_flow, trade_velocity, volume_velocity
from src.features.intelligence import (
    classify_direction,
    classify_liquidity,
    classify_volatility,
    vol_ratio_short_long,
    vwap_bps,
)
from src.features.returns import compute_returns, mid_at_or_before, simple_return, window_high_low
from src.features.volatility import compute_volatility, realized_variance, sampled_log_returns
from src.market.buffer import MarketBuffer, Trade
from src.market.models import Returns
from src.market.orderbook import LocalOrderBook

BPS = 10_000.0
EXTRA_RETURN_MS = {"r_10m": 600_000, "r_30m": 1_800_000}
EXTRA_RV_MS = {"rv_3m": 180_000, "rv_30m": 1_800_000}


def _qty(levels: list[tuple[float, float]], depth: int) -> float:
    return sum(qty for _, qty in levels[:depth])


def _microprice(book: LocalOrderBook) -> float | None:
    if (
        book.best_bid is None
        or book.best_ask is None
        or book.best_bid_qty is None
        or book.best_ask_qty is None
    ):
        return None
    total = book.best_bid_qty + book.best_ask_qty
    if total <= 0:
        return None
    return (book.best_ask * book.best_bid_qty + book.best_bid * book.best_ask_qty) / total


def _flow_volumes(trades: list[Trade]) -> tuple[float, float]:
    buy = sum(trade.qty for trade in trades if trade.is_aggressor_buy)
    sell = sum(trade.qty for trade in trades if not trade.is_aggressor_buy)
    return buy, sell


def _sigma_bps(rv: float | None, window_seconds: float, horizon_seconds: int) -> float | None:
    if rv is None or window_seconds <= 0:
        return None
    return rv * math.sqrt(horizon_seconds / window_seconds) * BPS


def _recent_candles(
    timestamps: list[int],
    values: list[float],
    *,
    timeframe: str,
    now_ms: int,
    count: int,
    volumes: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    window_ms = WINDOW_MS[timeframe]
    current_start = aligned_start(now_ms, window_ms)
    rows: list[dict[str, Any]] = []
    completed: list[float] = []
    for step in range(count, 0, -1):
        start = current_start - step * window_ms
        candle = candle_from_mids(
            timestamps,
            values,
            timeframe=timeframe,
            start_ms=start,
            volume=(volumes or {}).get(start, 0.0),
            now_ms=now_ms,
        )
        if candle is None or not candle.complete:
            continue
        completed.append(candle.volume)
        baseline = None if len(completed) < 2 else sum(completed[:-1]) / max(1, len(completed) - 1)
        rows.append(candle_summary(candle, volume_baseline=baseline))
    return rows[-count:]


def _active_candle(
    timestamps: list[int],
    values: list[float],
    *,
    timeframe: str,
    now_ms: int,
    mid: float | None,
) -> dict[str, Any] | None:
    window_ms = WINDOW_MS[timeframe]
    start = aligned_start(now_ms, window_ms)
    end = aligned_end(now_ms, window_ms)
    candle = candle_from_mids(
        timestamps, values, timeframe=timeframe, start_ms=start, now_ms=now_ms
    )
    if candle is None and mid is None:
        return None
    open_px = None if candle is None else candle.open
    current = mid if mid is not None else None if candle is None else candle.close
    elapsed = max(0, now_ms - start)
    remaining = max(0, end - now_ms)
    ret = None
    if open_px and current:
        ret = simple_return(current, open_px)
    return {
        "timeframe": timeframe,
        "window_start": start,
        "window_end": end,
        "candle_open": open_px,
        "current_mid": current,
        "elapsed_seconds": elapsed / 1000.0,
        "remaining_seconds": remaining / 1000.0,
        "remaining_frac": remaining / window_ms,
        "current_return_from_open": ret,
        "high": None if candle is None else candle.high,
        "low": None if candle is None else candle.low,
    }


def _trade_imbalance(trades: list[Trade]) -> float | None:
    return aggressive_flow(trades)


def build_directional_state(
    *,
    buffer: MarketBuffer,
    book: LocalOrderBook,
    now_ms: int,
    cross: dict[str, Any] | None = None,
    prev_depths: dict[str, float] | None = None,
) -> dict[str, Any]:
    timestamps, values = buffer.mids_as_lists()
    mid = book.mid if book.mid is not None else buffer.last_mid
    price = mid or 0.0
    returns = compute_returns(timestamps, values, now_ms, price)
    extra_returns: dict[str, float | None] = {}
    for name, window in EXTRA_RETURN_MS.items():
        prev = mid_at_or_before(timestamps, values, now_ms - window)
        extra_returns[name] = None if prev is None else simple_return(price, prev)
    high_1m, low_1m = window_high_low(timestamps, values, now_ms, 60_000)
    high_5m, low_5m = window_high_low(timestamps, values, now_ms, 300_000)
    high_15m, low_15m = window_high_low(timestamps, values, now_ms, 900_000)
    high_30m, low_30m = window_high_low(timestamps, values, now_ms, 1_800_000)
    vol = compute_volatility(
        timestamps,
        values,
        now_ms,
        high_5m=high_5m,
        low_5m=low_5m,
        high_15m=high_15m,
        low_15m=low_15m,
        current_price=price if price else 1.0,
    )
    extra_rv = {
        name: realized_variance(sampled_log_returns(timestamps, values, now_ms, window))
        for name, window in EXTRA_RV_MS.items()
    }
    trades = {
        "1s": buffer.trades_in_window(now_ms, 1_000),
        "5s": buffer.trades_in_window(now_ms, 5_000),
        "15s": buffer.trades_in_window(now_ms, 15_000),
        "30s": buffer.trades_in_window(now_ms, 30_000),
        "60s": buffer.trades_in_window(now_ms, 60_000),
        "5m": buffer.trades_in_window(now_ms, 300_000),
        "15m": buffer.trades_in_window(now_ms, 900_000),
        "30m": buffer.trades_in_window(now_ms, 1_800_000),
    }
    depths = {
        "bid_1": _qty(book.bids, 1),
        "ask_1": _qty(book.asks, 1),
        "bid_5": _qty(book.bids, 5),
        "ask_5": _qty(book.asks, 5),
        "bid_10": _qty(book.bids, 10),
        "ask_10": _qty(book.asks, 10),
        "bid_20": _qty(book.bids, 20),
        "ask_20": _qty(book.asks, 20),
    }
    previous_depths = prev_depths or {}
    depth_change = {
        key: None if key not in previous_depths else depths[key] - previous_depths[key]
        for key in depths
    }
    buy_1m, sell_1m = _flow_volumes(trades["60s"])
    ask_depth = depths["ask_10"] or 0.0
    bid_depth = depths["bid_10"] or 0.0
    micro = _microprice(book)
    dt = datetime.fromtimestamp(now_ms / 1000, tz=UTC)
    merged_returns = Returns(
        r_5s=returns.r_5s,
        r_15s=returns.r_15s,
        r_30s=returns.r_30s,
        r_1m=returns.r_1m,
        r_3m=returns.r_3m,
        r_5m=returns.r_5m,
        r_15m=returns.r_15m,
        r_1h=returns.r_1h,
    )
    raw_direction = classify_direction(merged_returns)
    candles_1m_raw = []
    current_1m_start = aligned_start(now_ms, WINDOW_MS["1m"])
    for step in range(40, 0, -1):
        start = current_1m_start - step * WINDOW_MS["1m"]
        candle = candle_from_mids(timestamps, values, timeframe="1m", start_ms=start, now_ms=now_ms)
        if candle is not None and candle.complete:
            candles_1m_raw.append(candle)
    candles_10m = [candle_summary(item) for item in aggregate_10m(candles_1m_raw)[-4:]]
    closes_1m = [item.close for item in candles_1m_raw]
    highs_1m = [item.high for item in candles_1m_raw]
    lows_1m = [item.low for item in candles_1m_raw]
    tech_1m = technical_panel(closes_1m, highs_1m, lows_1m, price=price)
    candles_5m_raw = []
    current_5m_start = aligned_start(now_ms, WINDOW_MS["5m"])
    for step in range(24, 0, -1):
        start = current_5m_start - step * WINDOW_MS["5m"]
        candle = candle_from_mids(timestamps, values, timeframe="5m", start_ms=start, now_ms=now_ms)
        if candle is not None and candle.complete:
            candles_5m_raw.append(candle)
    tech_5m = technical_panel(
        [item.close for item in candles_5m_raw],
        [item.high for item in candles_5m_raw],
        [item.low for item in candles_5m_raw],
        price=price,
    )
    sigma = {
        "1m": _sigma_bps(vol.rv_1m, 60.0, 60),
        "5m": _sigma_bps(vol.rv_1m, 60.0, 300),
        "10m": _sigma_bps(vol.rv_1m, 60.0, 600),
        "30m": _sigma_bps(vol.rv_5m, 300.0, 1800) or _sigma_bps(vol.rv_1m, 60.0, 1800),
    }
    missing: list[str] = []
    if extra_returns["r_10m"] is None:
        missing.append("r_10m")
    if extra_returns["r_30m"] is None:
        missing.append("r_30m")
    if merged_returns.r_15m is None:
        missing.append("r_15m")
    if merged_returns.r_1h is None:
        missing.append("r_1h")
    if vol.rv_15m is None:
        missing.append("rv_15m")
    mark = buffer.mark
    oi = list(buffer.open_interest)
    oi_now = oi[-1].open_interest if oi else None
    def _oi_before(delta_ms: int) -> float | None:
        target = now_ms - delta_ms
        chosen = None
        for sample in oi:
            if sample.ts_ms <= target:
                chosen = sample.open_interest
        return chosen
    oi_1m = _oi_before(60_000)
    oi_5m = _oi_before(300_000)
    oi_15m = _oi_before(900_000)
    if oi_now is None:
        missing.append("open_interest")
    if mark.funding_rate is None:
        missing.append("funding")
    liq_buy = sum(t.qty for t in trades["5m"] if t.is_liquidation and t.is_aggressor_buy)
    liq_sell = sum(t.qty for t in trades["5m"] if t.is_liquidation and not t.is_aggressor_buy)
    liq_total = liq_buy + liq_sell
    funding_in = None
    if mark.next_funding_time_ms is not None:
        funding_in = max(0.0, (mark.next_funding_time_ms - now_ms) / 1000.0)
    basis_bps = None
    if mark.mark_price and mark.index_price:
        basis_bps = (mark.mark_price / mark.index_price - 1.0) * BPS
    typical_spread = book.spread_bps
    fragility = None
    if typical_spread is not None and depths["bid_5"] + depths["ask_5"] > 0:
        fragility = typical_spread / max(depths["bid_5"] + depths["ask_5"], 1e-9)
    return {
        "schema": FEATURE_SCHEMA_VERSION,
        "timestamp_ms": now_ms,
        "symbol": buffer.symbol,
        "exchange": buffer.exchange,
        "utc_hour": dt.hour,
        "day_of_week": dt.weekday(),
        "weekend": dt.weekday() >= 5,
        "missing": missing,
        "sigma_bps": sigma,
        "microstructure": {
            "mid": mid,
            "last": buffer.last_trade_price,
            "best_bid": book.best_bid,
            "best_ask": book.best_ask,
            "spread_bps": book.spread_bps,
            "microprice": micro,
            "microprice_vs_mid_bps": None
            if micro is None or not mid
            else (micro / mid - 1.0) * BPS,
            "depth": depths,
            "imbalance": {
                "1": book.imbalance(1),
                "5": book.imbalance(5),
                "10": book.imbalance(10),
                "20": book.imbalance(20),
            },
            "order_flow_imbalance": {
                key: _trade_imbalance(trades[key]) for key in ("1s", "5s", "15s", "30s", "60s")
            },
            "aggressive_buy_1m": buy_1m,
            "aggressive_sell_1m": sell_1m,
            "trade_imbalance": {
                key: _trade_imbalance(trades[key]) for key in ("5s", "15s", "30s", "60s")
            },
            "trade_velocity_1m": trade_velocity(trades["60s"], 60.0),
            "volume_velocity_1m": volume_velocity(trades["60s"], 60.0),
            "depth_change": depth_change,
            "buy_pressure": None if ask_depth <= 0 else buy_1m / ask_depth,
            "sell_pressure": None if bid_depth <= 0 else sell_1m / bid_depth,
            "liquidity_fragility": fragility,
        },
        "price_action": {
            "returns": {
                **merged_returns.model_dump(),
                **extra_returns,
            },
            "realized_vol": {
                "rv_30s": vol.rv_30s,
                "rv_1m": vol.rv_1m,
                "rv_3m": extra_rv["rv_3m"],
                "rv_5m": vol.rv_5m,
                "rv_15m": vol.rv_15m,
                "rv_30m": extra_rv["rv_30m"],
            },
            "vwap_bps": {
                "1m": vwap_bps(price, trades["60s"]),
                "5m": vwap_bps(price, trades["5m"]),
                "15m": vwap_bps(price, trades["15m"]),
                "30m": vwap_bps(price, trades["30m"]),
            },
            "range": {
                "high_1m": high_1m,
                "low_1m": low_1m,
                "high_5m": high_5m,
                "low_5m": low_5m,
                "high_15m": high_15m,
                "low_15m": low_15m,
                "high_30m": high_30m,
                "low_30m": low_30m,
                "range_pos_1m": None
                if not price or high_1m is None or low_1m is None or high_1m == low_1m
                else (price - low_1m) / (high_1m - low_1m),
                "range_pos_5m": None
                if not price or high_5m is None or low_5m is None or high_5m == low_5m
                else (price - low_5m) / (high_5m - low_5m),
            },
            "recent_candles": {
                "1m": _recent_candles(timestamps, values, timeframe="1m", now_ms=now_ms, count=5),
                "5m": _recent_candles(timestamps, values, timeframe="5m", now_ms=now_ms, count=4),
                "15m": _recent_candles(timestamps, values, timeframe="15m", now_ms=now_ms, count=3),
                "30m": _recent_candles(timestamps, values, timeframe="30m", now_ms=now_ms, count=2),
                "10m": candles_10m,
            },
            "active_candles": {
                tf: _active_candle(timestamps, values, timeframe=tf, now_ms=now_ms, mid=mid)
                for tf in ("1m", "5m", "10m", "30m")
            },
        },
        "technical": {"1m": tech_1m, "5m": tech_5m},
        "derivatives": {
            "open_interest": oi_now,
            "oi_delta_1m": None if oi_now is None or oi_1m is None else oi_now - oi_1m,
            "oi_delta_5m": None if oi_now is None or oi_5m is None else oi_now - oi_5m,
            "oi_delta_15m": None if oi_now is None or oi_15m is None else oi_now - oi_15m,
            "funding": mark.funding_rate,
            "funding_in_seconds": funding_in,
            "mark_price": mark.mark_price,
            "index_price": mark.index_price,
            "basis_bps": basis_bps,
            "liquidation_imbalance": None if liq_total <= 0 else (liq_buy - liq_sell) / liq_total,
        },
        "cross_market": cross or {},
        "regime": {
            "direction": {"UPTREND": "UP", "DOWNTREND": "DOWN", "RANGE": "RANGE"}.get(
                raw_direction or "", raw_direction
            ),
            "volatility": classify_volatility(vol_ratio_short_long(vol)),
            "liquidity": classify_liquidity(
                spread_bps=book.spread_bps,
                typical_spread_bps=book.spread_bps,
                trade_velocity=trade_velocity(trades["60s"], 60.0),
            ),
            "utc_hour": dt.hour,
            "day_of_week": dt.weekday(),
            "weekend": dt.weekday() >= 5,
        },
        "health": {
            "book_valid": book.valid,
            "has_mid": mid is not None,
        },
    }


def feature_groups(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "MICROSTRUCTURE": state.get("microstructure") or {},
        "PRICE ACTION": state.get("price_action") or {},
        "TECHNICAL": state.get("technical") or {},
        "DERIVATIVES": state.get("derivatives") or {},
        "CROSS MARKET": state.get("cross_market") or {},
        "REGIME": state.get("regime") or {},
    }
