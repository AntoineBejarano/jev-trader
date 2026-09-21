from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from src.market.ws import ReconnectingWebSocket


async def _on_message(_payload: dict[str, Any]) -> None:
    return None


async def _on_disconnect() -> None:
    return None


def test_public_socket_does_not_send_client_pings() -> None:
    socket = ReconnectingWebSocket(
        name="binance-public",
        url="wss://fstream.binance.com/public/stream",
        on_message=_on_message,
        on_disconnect=_on_disconnect,
    )
    assert socket.ping_interval is None
    assert socket.ping_timeout is None
    assert socket.state == "stopped"
    assert socket.silence_timeout_seconds == 20.0


def test_reconnect_callbacks_are_optional() -> None:
    socket = ReconnectingWebSocket(
        name="binance-market",
        url="wss://fstream.binance.com/market/stream",
        on_message=_on_message,
        on_disconnect=_on_disconnect,
        on_connect=_typed_connect,
    )
    assert socket.on_connect is _typed_connect


async def _typed_connect() -> None:
    return None


def test_on_connect_signature_matches() -> None:
    cb: Callable[[], Awaitable[None]] = _typed_connect
    socket = ReconnectingWebSocket(
        name="t",
        url="ws://localhost",
        on_message=_on_message,
        on_disconnect=_on_disconnect,
        on_connect=cb,
    )
    assert socket.on_connect is not None
