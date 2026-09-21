from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from src.config.settings import Settings
from src.market.binance import (
    InstrumentRuntime,
    apply_open_interest,
    combined_stream_url,
    connection_health,
    dispatch_stream_message,
    market_streams_for,
    public_streams_for,
    unwrap_combined,
)
from src.market.buffer import MarketBuffer, wall_clock_ms
from src.market.history import KLINE_INTERVAL, KLINE_LIMIT, seed_buffer_from_klines
from src.market.models import MarketHealth
from src.market.ws import ReconnectingWebSocket

logger = logging.getLogger(__name__)


class BinanceMarketDataAdapter:
    """Public USD-M futures market data. No authenticated trading methods."""

    exchange = "binance_usdm"

    def __init__(self, settings: Settings, symbols: tuple[str, ...]) -> None:
        self.settings = settings
        self.symbols = symbols
        self.runtimes: dict[str, InstrumentRuntime] = {
            symbol: InstrumentRuntime(
                symbol=symbol,
                exchange=settings.trading_exchange,
                buffer=MarketBuffer(
                    symbol=symbol,
                    exchange=settings.trading_exchange,
                    history_ms=settings.history_ms,
                    mid_sample_interval_ms=settings.mid_sample_interval_ms,
                    clock=wall_clock_ms,
                ),
            )
            for symbol in symbols
        }
        self._http: httpx.AsyncClient | None = None
        self._public_ws: ReconnectingWebSocket | None = None
        self._market_ws: ReconnectingWebSocket | None = None
        self._oi_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        self._http = httpx.AsyncClient(timeout=10.0)
        await self.bootstrap_price_history()
        public_streams: list[str] = []
        market_streams: list[str] = []
        for symbol in self.symbols:
            public_streams.extend(public_streams_for(symbol))
            market_streams.extend(market_streams_for(symbol))
        self._public_ws = ReconnectingWebSocket(
            name="binance-public",
            url=combined_stream_url(self.settings.binance_ws_public_base, public_streams),
            on_message=self._on_public_message,
            on_disconnect=self._on_public_disconnect,
            on_connect=self._on_public_connect,
        )
        self._market_ws = ReconnectingWebSocket(
            name="binance-market",
            url=combined_stream_url(self.settings.binance_ws_market_base, market_streams),
            on_message=self._on_market_message,
            on_disconnect=self._on_market_disconnect,
            on_connect=self._on_market_connect,
        )
        self._public_ws.start()
        self._market_ws.start()
        self._oi_task = asyncio.create_task(self._poll_open_interest(), name="oi-poll")

    async def bootstrap_price_history(self) -> None:
        """Seed 15m/1h return context from public USD-M klines. Not live warmup."""
        if self._http is None:
            return
        for symbol in self.symbols:
            try:
                response = await self._http.get(
                    f"{self.settings.binance_futures_rest_base}/fapi/v1/klines",
                    params={"symbol": symbol, "interval": KLINE_INTERVAL, "limit": KLINE_LIMIT},
                )
                response.raise_for_status()
                payload = response.json()
            except Exception:
                logger.warning("Price-history bootstrap failed for %s", symbol, exc_info=True)
                continue
            if not isinstance(payload, list):
                logger.warning("Price-history bootstrap for %s returned a non-list", symbol)
                continue
            seeded = seed_buffer_from_klines(self.runtimes[symbol].buffer, payload)
            logger.info("Bootstrapped %s historical mid samples for %s", seeded, symbol)

    async def wait_connected(self, timeout: float = 15.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if self._market_ready():
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Timed out waiting for Binance market-data sockets")
            await asyncio.sleep(0.05)

    def _market_ready(self) -> bool:
        if not (self.ws_public_connected and self.ws_market_connected):
            return False
        return all(
            runtime.book.valid and runtime.buffer.mark.ts_ms is not None
            for runtime in self.runtimes.values()
        )

    async def subscribe_market_data(self, symbols: tuple[str, ...]) -> None:
        if symbols != self.symbols:
            raise ValueError("Phase 1 adapter is constructed with a fixed symbol set")

    async def close(self) -> None:
        if self._oi_task is not None:
            self._oi_task.cancel()
            try:
                await self._oi_task
            except asyncio.CancelledError:
                pass
            self._oi_task = None
        if self._public_ws is not None:
            await self._public_ws.stop()
        if self._market_ws is not None:
            await self._market_ws.stop()
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def health(self) -> MarketHealth:
        return connection_health(
            symbol=",".join(self.symbols),
            exchange=self.settings.trading_exchange,
            ws_public_connected=self.ws_public_connected,
            ws_market_connected=self.ws_market_connected,
        )

    def runtime(self, symbol: str) -> InstrumentRuntime:
        return self.runtimes[symbol]

    @property
    def ws_public_connected(self) -> bool:
        return bool(self._public_ws and self._public_ws.connected)

    @property
    def ws_market_connected(self) -> bool:
        return bool(self._market_ws and self._market_ws.connected)

    def ws_public_state(self) -> str:
        if self._public_ws is None:
            return "stopped"
        return self._public_ws.state

    def ws_market_state(self) -> str:
        if self._market_ws is None:
            return "stopped"
        return self._market_ws.state

    def last_public_message_at_ms(self) -> int | None:
        if self._public_ws is None:
            return None
        return self._public_ws.last_message_at_ms

    def last_market_message_at_ms(self) -> int | None:
        if self._market_ws is None:
            return None
        return self._market_ws.last_message_at_ms

    async def _on_public_connect(self) -> None:
        logger.info("Public market-data socket connected")

    async def _on_market_connect(self) -> None:
        logger.info("Market-data socket connected")

    async def _on_public_message(self, payload: dict[str, Any]) -> None:
        stream, data = unwrap_combined(payload)
        runtime = self._runtime_from(stream, data)
        if runtime is None:
            return
        dispatch_stream_message(runtime, stream, data)

    async def _on_market_message(self, payload: dict[str, Any]) -> None:
        stream, data = unwrap_combined(payload)
        runtime = self._runtime_from(stream, data)
        if runtime is None:
            return
        dispatch_stream_message(runtime, stream, data)

    async def _on_public_disconnect(self) -> None:
        logger.warning("Public market-data socket dropped; marking books stale")
        for runtime in self.runtimes.values():
            runtime.mark_books_stale()

    async def _on_market_disconnect(self) -> None:
        logger.warning("Market-data socket dropped; mark/trade streams are stale")

    def _runtime_from(self, stream: str | None, data: dict[str, Any]) -> InstrumentRuntime | None:
        symbol = data.get("s")
        nested = data.get("o")
        if not symbol and isinstance(nested, dict):
            symbol = nested.get("s")
        if not symbol and stream:
            symbol = stream.split("@", 1)[0].upper()
        if not isinstance(symbol, str):
            return None
        return self.runtimes.get(symbol.upper())

    async def _poll_open_interest(self) -> None:
        assert self._http is not None
        while True:
            for symbol in self.symbols:
                try:
                    response = await self._http.get(
                        f"{self.settings.binance_futures_rest_base}/fapi/v1/openInterest",
                        params={"symbol": symbol},
                    )
                    response.raise_for_status()
                    apply_open_interest(self.runtimes[symbol], response.json())
                except Exception:
                    logger.warning("Open interest poll failed for %s", symbol, exc_info=True)
            await asyncio.sleep(self.settings.open_interest_poll_seconds)
