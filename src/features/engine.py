from __future__ import annotations

from collections import deque
from datetime import UTC, datetime

from src.features.derivatives import compute_derivatives
from src.features.flow import (
    aggressive_flow,
    liquidation_notional,
    trade_velocity,
    volume_velocity,
    window_cvd,
)
from src.features.intelligence import SPREAD_SAMPLE_LIMIT, build_intelligence
from src.features.returns import compute_returns, window_high_low
from src.features.volatility import compute_volatility
from src.market.buffer import MarketBuffer, wall_clock_ms
from src.market.models import (
    PRIMARY_WARMUP_MS,
    CrossAssetContext,
    MarketHealth,
    MarketState,
    Microstructure,
    Momentum,
    PositionState,
    Ranges,
    Side,
)
from src.market.orderbook import LocalOrderBook

BPS = 10_000.0


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def _distance_bps(price: float, reference: float | None) -> float | None:
    if reference is None or price <= 0:
        return None
    return ((reference - price) / price) * BPS


def _range_position(price: float, high: float | None, low: float | None) -> float | None:
    if high is None or low is None:
        return None
    width = high - low
    if width <= 0:
        return None
    return (price - low) / width


def assess_health(
    *,
    book: LocalOrderBook,
    buffer: MarketBuffer,
    now_ms: int,
    stale_timeout_ms: int,
    ws_public_connected: bool,
    ws_market_connected: bool,
    kill_switch: bool,
) -> MarketHealth:
    reasons: list[str] = []
    last_book_ms = book.last_update_ms()
    book_age = None if last_book_ms is None else now_ms - last_book_ms
    trade_age = None if buffer.last_trade_ts_ms is None else now_ms - buffer.last_trade_ts_ms
    mark_age = None if buffer.mark.ts_ms is None else now_ms - buffer.mark.ts_ms

    if kill_switch:
        reasons.append("kill_switch")
    if not ws_public_connected:
        reasons.append("ws_public_disconnected")
    if not ws_market_connected:
        reasons.append("ws_market_disconnected")
    if not book.valid:
        reasons.append("order_book_invalid")
    if book_age is None or book_age > stale_timeout_ms:
        reasons.append("stale_order_book")
    if mark_age is None or mark_age > max(stale_timeout_ms, 8_000):
        reasons.append("stale_mark_price")

    warmup_ready = False
    live_mids = buffer.live_mids()
    if live_mids and (now_ms - live_mids[0].ts_ms) >= PRIMARY_WARMUP_MS:
        warmup_ready = True
    else:
        reasons.append("warmup_incomplete")

    stale = any(
        reason.startswith("stale_")
        or reason.endswith("_disconnected")
        or reason == "order_book_invalid"
        for reason in reasons
    )
    trading_allowed = not stale and not kill_switch and warmup_ready
    return MarketHealth(
        trading_allowed=trading_allowed,
        stale=stale,
        warmup_ready=warmup_ready,
        book_age_ms=book_age,
        trade_age_ms=trade_age,
        mark_age_ms=mark_age,
        ws_public_connected=ws_public_connected,
        ws_market_connected=ws_market_connected,
        reasons=tuple(reasons),
    )


class FeatureEngine:
    """Deterministic conversion of rolling buffers into a compact MarketState."""

    def __init__(self) -> None:
        self._spread_samples: deque[float] = deque(maxlen=SPREAD_SAMPLE_LIMIT)
        self._prev_imbalance_1: float | None = None

    def snapshot(
        self,
        *,
        buffer: MarketBuffer,
        book: LocalOrderBook,
        now_ms: int | None = None,
        stale_timeout_ms: int = 5_000,
        ws_public_connected: bool = False,
        ws_market_connected: bool = False,
        kill_switch: bool = False,
        position: PositionState | None = None,
        context: CrossAssetContext | None = None,
    ) -> MarketState:
        now = now_ms if now_ms is not None else wall_clock_ms()
        timestamps, values = buffer.mids_as_lists()
        price = book.mid if book.mid is not None else buffer.last_mid
        # r_15m / r_1h are intended Jev context. Historical kline mids may be
        # seeded so these returns exist without waiting one live hour. Seeded
        # mids do not satisfy warmup; missing r_15m/r_1h is not research-invalid.
        returns = (
            compute_returns(timestamps, values, now, price)
            if price is not None
            else compute_returns(timestamps, values, now, buffer.last_mid or 0.0)
        )
        high_1m, low_1m = window_high_low(timestamps, values, now, 60_000)
        high_5m, low_5m = window_high_low(timestamps, values, now, 300_000)
        high_15m, low_15m = window_high_low(timestamps, values, now, 900_000)
        high_1h, low_1h = window_high_low(timestamps, values, now, 3_600_000)

        trades_5s = buffer.trades_in_window(now, 5_000)
        trades_30s = buffer.trades_in_window(now, 30_000)
        trades_1m = buffer.trades_in_window(now, 60_000)
        trades_5m = buffer.trades_in_window(now, 300_000)

        current_price = price if price is not None else 0.0
        volatility = compute_volatility(
            timestamps,
            values,
            now,
            high_5m=high_5m,
            low_5m=low_5m,
            high_15m=high_15m,
            low_15m=low_15m,
            current_price=current_price if current_price else 1.0,
        )
        derivatives = compute_derivatives(
            buffer,
            now,
            long_liquidations=liquidation_notional(trades_5m, aggressor_buy=True),
            short_liquidations=liquidation_notional(trades_5m, aggressor_buy=False),
            last_trade_price=buffer.last_trade_price,
        )
        health = assess_health(
            book=book,
            buffer=buffer,
            now_ms=now,
            stale_timeout_ms=stale_timeout_ms,
            ws_public_connected=ws_public_connected,
            ws_market_connected=ws_market_connected,
            kill_switch=kill_switch,
        )
        pos = position or PositionState(
            symbol=buffer.symbol,
            exchange=buffer.exchange,
            side=Side.FLAT,
            current_price=price,
        )
        imbalance_1 = book.imbalance(1)
        flow_5s = aggressive_flow(trades_5s)
        flow_30s = aggressive_flow(trades_30s)
        flow_1m = aggressive_flow(trades_1m)
        if book.spread_bps is not None:
            self._spread_samples.append(book.spread_bps)
        intelligence = build_intelligence(
            price=current_price,
            trades_1m=trades_1m,
            trades_5m=trades_5m,
            returns=returns,
            vol=volatility,
            spread_bps=book.spread_bps,
            spread_samples=self._spread_samples,
            flow_5s=flow_5s,
            flow_30s=flow_30s,
            flow_1m=flow_1m,
            imbalance_1=imbalance_1,
            prev_imbalance_1=self._prev_imbalance_1,
            trade_velocity=trade_velocity(trades_1m, 60.0),
        )
        self._prev_imbalance_1 = imbalance_1
        return MarketState(
            timestamp=_iso(now),
            timestamp_ms=now,
            symbol=buffer.symbol,
            exchange=buffer.exchange,
            price=price,
            last_trade_price=buffer.last_trade_price,
            best_bid=book.best_bid,
            best_ask=book.best_ask,
            returns=returns,
            momentum=Momentum(
                acceleration=None
                if returns.r_15s is None or returns.r_1m is None
                else returns.r_15s - returns.r_1m,
                short_term_direction=returns.r_30s,
                medium_term_direction=returns.r_5m,
            ),
            microstructure=Microstructure(
                spread_bps=book.spread_bps,
                book_imbalance_1=imbalance_1,
                book_imbalance_5=book.imbalance(5),
                book_imbalance_10=book.imbalance(10),
                aggressive_flow_5s=flow_5s,
                aggressive_flow_30s=flow_30s,
                aggressive_flow_1m=flow_1m,
                trade_velocity=trade_velocity(trades_1m, 60.0),
                volume_velocity=volume_velocity(trades_1m, 60.0),
                cvd_1m=window_cvd(trades_1m),
                cvd_5m=window_cvd(trades_5m),
            ),
            volatility=volatility,
            derivatives=derivatives,
            ranges=Ranges(
                high_1m=high_1m,
                low_1m=low_1m,
                high_5m=high_5m,
                low_5m=low_5m,
                high_15m=high_15m,
                low_15m=low_15m,
                high_1h=high_1h,
                low_1h=low_1h,
                dist_from_high_1m_bps=(
                    _distance_bps(current_price, high_1m) if current_price else None
                ),
                dist_from_low_1m_bps=(
                    None
                    if not current_price or low_1m is None
                    else ((current_price - low_1m) / current_price) * BPS
                ),
                dist_from_high_5m_bps=(
                    _distance_bps(current_price, high_5m) if current_price else None
                ),
                dist_from_low_5m_bps=(
                    None
                    if not current_price or low_5m is None
                    else ((current_price - low_5m) / current_price) * BPS
                ),
                dist_from_high_15m_bps=(
                    _distance_bps(current_price, high_15m) if current_price else None
                ),
                dist_from_low_15m_bps=(
                    None
                    if not current_price or low_15m is None
                    else ((current_price - low_15m) / current_price) * BPS
                ),
                range_position_1m=(
                    _range_position(current_price, high_1m, low_1m) if current_price else None
                ),
                range_position_5m=(
                    _range_position(current_price, high_5m, low_5m) if current_price else None
                ),
            ),
            intelligence=intelligence,
            position=pos,
            health=health,
            context=context,
            feature_schema_version="ms.v2",
        )
