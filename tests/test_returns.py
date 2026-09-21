from __future__ import annotations

import pytest
from src.features.returns import compute_returns, simple_return
from src.market.buffer import MarketBuffer, MidSample


def test_simple_return() -> None:
    assert simple_return(101, 100) == pytest.approx(0.01)
    assert simple_return(100, 0) is None


def test_windowed_returns_are_deterministic() -> None:
    timestamps = [0, 5_000, 15_000, 60_000]
    values = [100.0, 101.0, 102.0, 103.0]
    returns = compute_returns(timestamps, values, now_ms=60_000, current_price=103.0)
    assert returns.r_5s == pytest.approx(103 / 102 - 1)
    assert returns.r_1m == pytest.approx(103 / 100 - 1)
    assert returns.r_1h is None


def test_buffer_downsamples_mids() -> None:
    buffer = MarketBuffer(
        symbol="ETHUSDT",
        exchange="binance_usdm",
        history_ms=60_000,
        mid_sample_interval_ms=100,
        clock=lambda: 1_000,
    )
    buffer.record_mid(100.0, 1_000)
    buffer.record_mid(100.1, 1_050)
    buffer.record_mid(100.2, 1_200)
    assert [sample.mid for sample in buffer.mids] == [100.1, 100.2]
    assert [sample.ts_ms for sample in buffer.mids] == [1_000, 1_200]
    assert isinstance(buffer.mids[0], MidSample)
    assert buffer.last_mid_ts_ms == 1_200


def test_high_frequency_ticks_open_a_bucket_per_interval() -> None:
    buffer = MarketBuffer(
        symbol="BTCUSDT",
        exchange="binance_usdm",
        history_ms=60_000,
        mid_sample_interval_ms=100,
        clock=lambda: 0,
    )
    for offset in range(0, 1_000, 20):
        buffer.record_mid(100.0 + offset / 1_000, 1_000 + offset)
    timestamps = [sample.ts_ms for sample in buffer.mids]
    assert timestamps == list(range(1_000, 2_000, 100))
    assert buffer.last_mid_ts_ms == 1_980


def test_dense_buffer_populates_short_returns() -> None:
    buffer = MarketBuffer(
        symbol="BTCUSDT",
        exchange="binance_usdm",
        history_ms=60_000,
        mid_sample_interval_ms=100,
        clock=lambda: 0,
    )
    for offset in range(0, 6_000, 20):
        buffer.record_mid(100.0 + offset / 10_000, offset)
    now_ms = 5_980
    _timestamps, values = buffer.mids_as_lists()
    returns = compute_returns(_timestamps, values, now_ms, values[-1])
    assert len(buffer.mids) == 60
    assert returns.r_5s is not None
    assert returns.r_5s > 0
    assert returns.r_1m is None


def test_lookback_allows_small_clock_skew() -> None:
    timestamps = [5_100]
    values = [101.0]
    returns = compute_returns(timestamps, values, now_ms=10_000, current_price=102.0)
    assert returns.r_5s == pytest.approx(102 / 101 - 1)
