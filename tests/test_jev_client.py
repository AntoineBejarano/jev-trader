from __future__ import annotations

import httpx
import pytest
from src.config.settings import Settings
from src.jev.client import JevClient
from src.jev.models import JevCallStatus, JevSafetyAction
from src.market.models import Side
from tests.helpers import make_market_state
from tests.test_jev_parse import ENTRY_PAYLOAD


def _settings(**overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "openrouter_api_key": "test-key",
        "jev_timeout_seconds": 0.4,
        "jev_failure_timeout_seconds": 30,
    }
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


@pytest.mark.asyncio
async def test_successful_entry_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.url.path.endswith("/api/alpha/decisions")
        return httpx.Response(200, json=ENTRY_PAYLOAD)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = JevClient(_settings(), http=http)
    result = await client.decide(make_market_state())
    await client.aclose()
    await http.aclose()
    assert result.status is JevCallStatus.OK
    assert result.decision is not None
    assert result.decision.exposure_choice == "FLAT"
    assert calls["n"] == 1
    assert result.safety_action is JevSafetyAction.NONE


@pytest.mark.asyncio
async def test_stale_market_skips_jev() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("stale data must not query Jev")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = JevClient(_settings(), http=http)
    result = await client.decide(make_market_state(stale=True))
    await http.aclose()
    assert result.status is JevCallStatus.SKIPPED_STALE
    assert result.safety_action is JevSafetyAction.BLOCK_OPEN


@pytest.mark.asyncio
async def test_timeout_blocks_open() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = JevClient(_settings(jev_timeout_seconds=0.05), http=http)
    result = await client.decide(make_market_state())
    await http.aclose()
    assert result.status is JevCallStatus.TIMEOUT
    assert result.safety_action is JevSafetyAction.BLOCK_OPEN


@pytest.mark.asyncio
async def test_invalid_response_does_not_open() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "typesafe/jev-1.13", "answers": {}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = JevClient(_settings(), http=http)
    result = await client.decide(make_market_state())
    await http.aclose()
    assert result.status is JevCallStatus.INVALID_RESPONSE
    assert result.safety_action is JevSafetyAction.BLOCK_OPEN


@pytest.mark.asyncio
async def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=ENTRY_PAYLOAD)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = JevClient(_settings(), http=http)
    result = await client.decide(make_market_state())
    await http.aclose()
    assert result.status is JevCallStatus.OK
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_missing_key_flattens_if_positioned() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    client = JevClient(_settings(openrouter_api_key=""), http=http)
    result = await client.decide(make_market_state(side=Side.LONG))
    await http.aclose()
    assert result.status is JevCallStatus.SKIPPED_NO_API_KEY
    assert result.safety_action is JevSafetyAction.FLATTEN
