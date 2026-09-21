from __future__ import annotations

from src.market.binance import InstrumentRuntime, apply_force_order, unwrap_combined
from src.market.buffer import MarketBuffer


def test_combined_envelope_unwrap() -> None:
    stream, data = unwrap_combined(
        {"stream": "btcusdt@aggTrade", "data": {"e": "aggTrade", "s": "BTCUSDT"}}
    )
    assert stream == "btcusdt@aggTrade"
    assert data["e"] == "aggTrade"


def test_force_order_records_liquidation() -> None:
    runtime = InstrumentRuntime(
        symbol="BTCUSDT",
        exchange="binance_usdm",
        buffer=MarketBuffer(symbol="BTCUSDT", exchange="binance_usdm", history_ms=60_000),
    )
    apply_force_order(
        runtime,
        {
            "e": "forceOrder",
            "E": 50,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",
                "ap": "100.0",
                "z": "2.0",
                "T": 50,
            },
        },
    )
    assert runtime.buffer.trades[0].is_liquidation is True
    assert runtime.buffer.trades[0].is_aggressor_buy is False
    assert runtime.buffer.session_cvd == -2.0
