from __future__ import annotations

import pytest
from src.features.flow import aggressive_flow, window_cvd
from src.market.buffer import MarketBuffer, Trade


def test_cvd_and_aggressive_flow() -> None:
    trades = [
        Trade(ts_ms=1, price=100, qty=10, is_aggressor_buy=True),
        Trade(ts_ms=2, price=100, qty=4, is_aggressor_buy=False),
        Trade(ts_ms=3, price=100, qty=1, is_aggressor_buy=True),
    ]
    assert window_cvd(trades) == pytest.approx(7.0)
    assert aggressive_flow(trades) == pytest.approx((11 - 4) / 15)


def test_binance_buyer_is_maker_means_sell_aggressor() -> None:
    # Covered via MarketBuffer: aggressor buy is the inverse of buyer-is-maker.
    buffer = MarketBuffer(symbol="SOLUSDT", exchange="binance_usdm", history_ms=60_000)
    buffer.record_trade(Trade(ts_ms=10, price=20, qty=3, is_aggressor_buy=False))
    buffer.record_trade(Trade(ts_ms=11, price=20, qty=5, is_aggressor_buy=True))
    assert buffer.session_cvd == pytest.approx(2.0)
    recent = buffer.trades_in_window(11, 5)
    assert window_cvd(recent) == pytest.approx(2.0)
