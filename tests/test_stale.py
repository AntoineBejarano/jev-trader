from __future__ import annotations

from src.features.engine import FeatureEngine
from src.market.binance import InstrumentRuntime, apply_book_ticker
from src.market.buffer import MarketBuffer, Trade
from src.market.orderbook import LocalOrderBook


def _ready_book(now_ms: int) -> LocalOrderBook:
    book = LocalOrderBook()
    book.apply_partial_snapshot(
        bids=[(100.0, 2.0), (99.9, 1.0)],
        asks=[(100.2, 2.0), (100.3, 1.0)],
        last_update_id=10,
        event_time_ms=now_ms,
    )
    return book


def test_stale_book_blocks_trading() -> None:
    now = 50_000
    buffer = MarketBuffer(symbol="BTCUSDT", exchange="binance_usdm", history_ms=3_600_000)
    buffer.record_mid(100.1, now - 40_000)
    book = _ready_book(now - 20_000)
    state = FeatureEngine().snapshot(
        buffer=buffer,
        book=book,
        now_ms=now,
        stale_timeout_ms=5_000,
        ws_public_connected=True,
        ws_market_connected=True,
    )
    assert state.health.stale is True
    assert state.health.trading_allowed is False
    assert "stale_order_book" in state.health.reasons


def test_disconnect_blocks_trading_even_with_fresh_local_book() -> None:
    now = 10_000
    buffer = MarketBuffer(symbol="BTCUSDT", exchange="binance_usdm", history_ms=3_600_000)
    book = _ready_book(now)
    buffer.record_mark(
        mark_price=100.1,
        index_price=100.0,
        funding_rate=0.0001,
        next_funding_time_ms=now + 1_000,
        ts_ms=now,
    )
    state = FeatureEngine().snapshot(
        buffer=buffer,
        book=book,
        now_ms=now,
        stale_timeout_ms=5_000,
        ws_public_connected=False,
        ws_market_connected=True,
    )
    assert state.health.trading_allowed is False
    assert "ws_public_disconnected" in state.health.reasons


def test_kill_switch_blocks_trading() -> None:
    now = 10_000
    buffer = MarketBuffer(symbol="ETHUSDT", exchange="binance_usdm", history_ms=3_600_000)
    book = _ready_book(now)
    buffer.record_mark(
        mark_price=100.1,
        index_price=100.0,
        funding_rate=0.0,
        next_funding_time_ms=now,
        ts_ms=now,
    )
    state = FeatureEngine().snapshot(
        buffer=buffer,
        book=book,
        now_ms=now,
        stale_timeout_ms=5_000,
        ws_public_connected=True,
        ws_market_connected=True,
        kill_switch=True,
    )
    assert state.health.trading_allowed is False
    assert "kill_switch" in state.health.reasons


def test_fresh_connected_market_is_not_stale_during_warmup() -> None:
    now = 10_000
    buffer = MarketBuffer(symbol="BTCUSDT", exchange="binance_usdm", history_ms=3_600_000)
    book = _ready_book(now)
    buffer.record_mid(100.1, now)
    buffer.record_trade(Trade(ts_ms=now, price=100.1, qty=1.0, is_aggressor_buy=True))
    buffer.record_mark(
        mark_price=100.15,
        index_price=100.1,
        funding_rate=0.0001,
        next_funding_time_ms=now + 3_600_000,
        ts_ms=now,
    )
    state = FeatureEngine().snapshot(
        buffer=buffer,
        book=book,
        now_ms=now,
        stale_timeout_ms=5_000,
        ws_public_connected=True,
        ws_market_connected=True,
    )
    assert state.health.stale is False
    assert state.health.warmup_ready is False
    assert state.health.trading_allowed is False
    assert "warmup_incomplete" in state.health.reasons


def test_five_minute_history_marks_warmup_ready_and_allows_trading() -> None:
    now = 300_000
    buffer = MarketBuffer(symbol="BTCUSDT", exchange="binance_usdm", history_ms=3_600_000)
    book = _ready_book(now)
    for ts in range(0, now + 1, 1_000):
        buffer.record_mid(100.0, ts)
    buffer.record_mark(
        mark_price=100.15,
        index_price=100.1,
        funding_rate=0.0001,
        next_funding_time_ms=now + 3_600_000,
        ts_ms=now,
    )
    state = FeatureEngine().snapshot(
        buffer=buffer,
        book=book,
        now_ms=now,
        stale_timeout_ms=5_000,
        ws_public_connected=True,
        ws_market_connected=True,
    )
    assert state.health.warmup_ready is True
    assert state.health.trading_allowed is True


def test_book_ticker_staleness_uses_receive_time_not_exchange_clock() -> None:
    now = 50_000

    def clock() -> int:
        return now

    runtime = InstrumentRuntime(
        symbol="BTCUSDT",
        exchange="binance_usdm",
        buffer=MarketBuffer(
            symbol="BTCUSDT",
            exchange="binance_usdm",
            history_ms=3_600_000,
            clock=clock,
        ),
    )
    apply_book_ticker(
        runtime,
        {
            "s": "BTCUSDT",
            "b": "100.0",
            "B": "1.0",
            "a": "100.2",
            "A": "1.0",
            "u": 1,
            "E": now - 30_000,
            "T": now - 30_000,
        },
    )
    runtime.buffer.record_mark(
        mark_price=100.1,
        index_price=100.0,
        funding_rate=0.0001,
        next_funding_time_ms=now + 1_000,
        ts_ms=now,
    )
    state = FeatureEngine().snapshot(
        buffer=runtime.buffer,
        book=runtime.book,
        now_ms=now,
        stale_timeout_ms=10_000,
        ws_public_connected=True,
        ws_market_connected=True,
    )
    assert runtime.book.last_update_ms() == now
    assert "stale_order_book" not in state.health.reasons
    assert state.health.stale is False
