from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

OnMessage = Callable[[dict[str, Any]], Awaitable[None]]
OnDisconnect = Callable[[], Awaitable[None]]
OnConnect = Callable[[], Awaitable[None]]
SocketState = Literal["connecting", "connected", "reconnecting", "stopped"]


class ReconnectingWebSocket:
    """Combined-stream client. Does not send client pings: Binance USD-M often
    ignores them and the library then dies with keepalive ping timeout.

    Liveness is: incoming frames + auto-pong to server pings + silence watchdog.
    """

    def __init__(
        self,
        *,
        name: str,
        url: str,
        on_message: OnMessage,
        on_disconnect: OnDisconnect,
        on_connect: OnConnect | None = None,
        ping_interval: float | None = None,
        ping_timeout: float | None = None,
        silence_timeout_seconds: float = 20.0,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        self.name = name
        self.url = url
        self.on_message = on_message
        self.on_disconnect = on_disconnect
        self.on_connect = on_connect
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.silence_timeout_seconds = silence_timeout_seconds
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.connected = False
        self.state: SocketState = "stopped"
        self.last_message_at_ms: int | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self.state = "connecting"
            self._task = asyncio.create_task(self._run(), name=f"ws:{self.name}")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.connected = False
        self.state = "stopped"

    async def _run(self) -> None:
        backoff = self.initial_backoff
        while not self._stop.is_set():
            watchdog: asyncio.Task[None] | None = None
            try:
                self.state = "connecting"
                logger.info("Connecting %s", self.name)
                async with websockets.connect(
                    self.url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    close_timeout=5,
                    max_size=2**22,
                    open_timeout=15,
                ) as ws:
                    self.connected = True
                    self.state = "connected"
                    self.last_message_at_ms = int(time.time() * 1000)
                    backoff = self.initial_backoff
                    logger.info("Connected %s", self.name)
                    if self.on_connect is not None:
                        await self.on_connect()
                    watchdog = asyncio.create_task(
                        self._silence_watchdog(ws), name=f"ws-watch:{self.name}"
                    )
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        try:
                            payload = json.loads(raw)
                        except ValueError:
                            logger.warning("%s received non-JSON payload", self.name)
                            continue
                        if not isinstance(payload, dict):
                            continue
                        self.last_message_at_ms = int(time.time() * 1000)
                        await self.on_message(payload)
            except asyncio.CancelledError:
                self.connected = False
                self.state = "stopped"
                raise
            except (ConnectionClosed, OSError, TimeoutError) as exc:
                logger.warning("%s disconnected: %s", self.name, exc)
            except Exception:
                logger.exception("%s failed", self.name)
            finally:
                if watchdog is not None:
                    watchdog.cancel()
                    try:
                        await watchdog
                    except asyncio.CancelledError:
                        pass
            was_connected = self.connected
            self.connected = False
            if was_connected:
                self.state = "reconnecting"
                await self.on_disconnect()
            if self._stop.is_set():
                self.state = "stopped"
                break
            jitter = random.uniform(0, backoff * 0.2)
            sleep_for = min(self.max_backoff, backoff) + jitter
            logger.info("Reconnecting %s in %.1fs", self.name, sleep_for)
            self.state = "reconnecting"
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=sleep_for)
            except TimeoutError:
                pass
            backoff = min(self.max_backoff, backoff * 2)

    async def _silence_watchdog(self, ws: Any) -> None:
        timeout_ms = int(self.silence_timeout_seconds * 1000)
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            if self.last_message_at_ms is None:
                continue
            age = int(time.time() * 1000) - self.last_message_at_ms
            if age > timeout_ms:
                logger.warning(
                    "%s silent for %sms; closing socket to reconnect", self.name, age
                )
                await ws.close()
                return
