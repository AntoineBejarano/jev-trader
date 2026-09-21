from __future__ import annotations

from src.features.engine import FeatureEngine
from src.market.binance import InstrumentRuntime, apply_agg_trade, apply_depth, apply_mark_price
from src.market.buffer import MarketBuffer
from src.market.models import Side
from src.market.orderbook import LocalOrderBook


def test_market_state_is_tagged_with_symbol_and_exchange() -> None:
    buffer = MarketBuffer(symbol="SOLUSDT", exchange="binance_usdm", history_ms=3_600_000)
    now = 300_000
    for ts in range(0, now + 1, 1_000):
        buffer.record_mid(150.0 + ts / 1_000_000, ts)
    book = LocalOrderBook()
    book.apply_partial_snapshot(
        bids=[(150.1, 3.0)],
        asks=[(150.2, 1.0)],
        last_update_id=9,
        event_time_ms=now,
    )
    buffer.record_mark(
        mark_price=150.15,
        index_price=150.10,
        funding_rate=0.0002,
        next_funding_time_ms=now + 1,
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
    assert state.symbol == "SOLUSDT"
    assert state.exchange == "binance_usdm"
    assert state.position.side is Side.FLAT
    assert state.price is not None
    assert state.returns.r_5m is not None
    dumped = state.canonical_dict()
    assert dumped["symbol"] == "SOLUSDT"
    assert dumped["exchange"] == "binance_usdm"


def test_binance_payloads_update_runtime_without_global_symbol() -> None:
    runtime = InstrumentRuntime(
        symbol="ETHUSDT",
        exchange="binance_usdm",
        buffer=MarketBuffer(symbol="ETHUSDT", exchange="binance_usdm", history_ms=60_000),
    )
    apply_depth(
        runtime,
        {
            "lastUpdateId": 11,
            "E": 1_000,
            "bids": [["2000.0", "1.5"], ["1999.5", "2.0"]],
            "asks": [["2000.5", "1.0"], ["2001.0", "3.0"]],
        },
    )
    apply_agg_trade(
        runtime,
        {"T": 1_010, "p": "2000.2", "q": "0.4", "m": False},
    )
    apply_mark_price(
        runtime,
        {"E": 1_020, "p": "2000.3", "i": "2000.1", "r": "0.0001", "T": 3_600_000},
    )
    assert runtime.book.mid == 2000.25
    assert runtime.buffer.last_trade_price == 2000.2
    assert runtime.buffer.session_cvd == 0.4
    assert runtime.buffer.mark.funding_rate == 0.0001
