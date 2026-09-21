from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from pydantic import ValidationError

from src.config.settings import Settings, TradingStrategy
from src.decision.barriers import Barrier
from src.directional.parse import parse_directional_decision, validate_directional_answers
from src.directional.questions import QUESTION_SET_VERSION as DIRECTIONAL_QUESTION_SET
from src.directional.questions import directional_questions
from src.jev.models import (
    JevCallResult,
    JevCallStatus,
    JevSafetyAction,
    QuestionMode,
)
from src.jev.parse import decode_raw, parse_decision, validate_required_answers
from src.jev.questions import mode_for_side, question_set_version, questions_for
from src.jev.safety import safety_action_on_jev_failure
from src.jev.scalp_parse import parse_scalp_decision, validate_scalp_answers
from src.jev.scalp_questions import scalp_question_set_version, scalp_questions
from src.jev.state_serializer import serialize_market_state
from src.market.models import MarketState, Side

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class JevClient:
    """OpenRouter Decisions API client. Not a chatbot wrapper."""

    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_http = http is None
        timeout = httpx.Timeout(settings.jev_timeout_seconds, connect=5.0)
        self._http = http or httpx.AsyncClient(timeout=timeout)
        self._last_success_ms: dict[str, int] = {}
        self._unavailable_since_ms: dict[str, int] = {}

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    def skip_result(
        self,
        state: MarketState,
        status: JevCallStatus,
        *,
        error: str | None = None,
        safety_action: JevSafetyAction | None = None,
        barriers: tuple[Barrier, ...] | None = None,
    ) -> JevCallResult:
        mode = mode_for_side(state.position.side)
        version = self._question_set_version(barriers)
        return JevCallResult(
            status=status,
            symbol=state.symbol,
            exchange=state.exchange,
            mode=mode,
            question_set_version=version,
            error=error,
            safety_action=safety_action or self._safety(state.position.side, state.symbol),
        )

    async def decide(
        self,
        state: MarketState,
        *,
        barriers: tuple[Barrier, ...] | None = None,
        forecast_state: dict[str, Any] | None = None,
    ) -> JevCallResult:
        mode = mode_for_side(state.position.side)
        version = self._question_set_version(barriers)
        if not self.settings.openrouter_api_key:
            return self.skip_result(
                state,
                JevCallStatus.SKIPPED_NO_API_KEY,
                error="missing api key",
                safety_action=(
                    JevSafetyAction.BLOCK_OPEN
                    if state.position.side is Side.FLAT
                    else JevSafetyAction.FLATTEN
                ),
                barriers=barriers,
            )
        if self.settings.kill_switch:
            return self.skip_result(
                state, JevCallStatus.SKIPPED_KILL_SWITCH, barriers=barriers
            )
        if state.health.stale:
            return self.skip_result(state, JevCallStatus.SKIPPED_STALE, barriers=barriers)

        questions = self._questions(mode, barriers)
        if forecast_state is not None:
            serialized_state = forecast_state
        else:
            serialized_state = serialize_market_state(state)
        payload = {
            "model": self.settings.jev_model,
            "state": serialized_state,
            "questions": questions,
            "session_id": f"{state.exchange}:{state.symbol}",
        }
        started = time.perf_counter()
        try:
            raw = await self._post_with_retry(payload)
        except TimeoutError as exc:
            self._mark_failure(state.symbol)
            return JevCallResult(
                status=JevCallStatus.TIMEOUT,
                symbol=state.symbol,
                exchange=state.exchange,
                mode=mode,
                question_set_version=version,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=str(exc),
                safety_action=self._safety(state.position.side, state.symbol),
            )
        except httpx.HTTPError as exc:
            self._mark_failure(state.symbol)
            return JevCallResult(
                status=JevCallStatus.UNAVAILABLE,
                symbol=state.symbol,
                exchange=state.exchange,
                mode=mode,
                question_set_version=version,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=str(exc),
                safety_action=self._safety(state.position.side, state.symbol),
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.debug("Jev raw response %s:%s %s", state.exchange, state.symbol, raw)
        try:
            parsed = decode_raw(raw)
            if self.settings.trading_strategy is TradingStrategy.SCALP:
                if barriers is None:
                    raise ValueError("SCALP Jev calls require barriers")
                validate_scalp_answers(parsed, [item.horizon_seconds for item in barriers])
                decision = parse_scalp_decision(
                    parsed,
                    mode=mode,
                    barriers=barriers,
                    latency_ms=latency_ms,
                )
            elif self.settings.trading_strategy is TradingStrategy.DIRECTIONAL:
                validate_directional_answers(parsed)
                decision = parse_directional_decision(
                    parsed,
                    mode=mode,
                    latency_ms=latency_ms,
                )
            else:
                validate_required_answers(parsed, mode)
                decision = parse_decision(
                    parsed,
                    mode=mode,
                    question_set_version=version,
                    latency_ms=latency_ms,
                )
        except (ValidationError, ValueError, TypeError) as exc:
            self._mark_failure(state.symbol)
            return JevCallResult(
                status=JevCallStatus.INVALID_RESPONSE,
                symbol=state.symbol,
                exchange=state.exchange,
                mode=mode,
                question_set_version=version,
                latency_ms=latency_ms,
                error=str(exc),
                raw_response=raw,
                safety_action=self._safety(state.position.side, state.symbol),
            )

        self._mark_success(state.symbol)
        return JevCallResult(
            status=JevCallStatus.OK,
            symbol=state.symbol,
            exchange=state.exchange,
            mode=mode,
            question_set_version=version,
            latency_ms=latency_ms,
            decision=decision,
            raw_response=raw,
            safety_action=JevSafetyAction.NONE,
        )

    async def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        attempts = 3
        for attempt in range(attempts):
            try:
                response = await self._http.post(
                    self.settings.jev_decisions_url,
                    headers={
                        "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/local/jev-trader",
                        "X-OpenRouter-Title": "Jev Trader",
                    },
                    json=payload,
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise TimeoutError("Jev request timed out") from exc
                await asyncio.sleep(0.4 * (2**attempt))
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < attempts - 1:
                await asyncio.sleep(0.4 * (2**attempt))
                continue
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"Jev HTTP {response.status_code}: {response.text[:300]}",
                    request=response.request,
                    response=response,
                )
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Jev response is not a JSON object")
            return data
        raise TimeoutError(str(last_error) if last_error else "Jev request failed")

    def _mark_success(self, symbol: str) -> None:
        self._last_success_ms[symbol] = int(time.time() * 1000)
        self._unavailable_since_ms.pop(symbol, None)

    def _mark_failure(self, symbol: str) -> None:
        self._unavailable_since_ms.setdefault(symbol, int(time.time() * 1000))

    def _safety(self, side: Side, symbol: str) -> JevSafetyAction:
        started = self._unavailable_since_ms.get(symbol)
        unavailable_for = 0.0 if started is None else (time.time() * 1000 - started) / 1000.0
        return safety_action_on_jev_failure(
            side=side,
            unavailable_for_seconds=unavailable_for,
            failure_timeout_seconds=self.settings.jev_failure_timeout_seconds,
        )

    def _question_set_version(self, barriers: tuple[Barrier, ...] | None) -> str:
        if self.settings.trading_strategy is TradingStrategy.SCALP:
            horizons = (
                self.settings.scalp_horizons
                if barriers is None
                else tuple(item.horizon_seconds for item in barriers)
            )
            return scalp_question_set_version(horizons)
        if self.settings.trading_strategy is TradingStrategy.DIRECTIONAL:
            return DIRECTIONAL_QUESTION_SET
        return question_set_version(self.settings.primary_horizon_seconds)

    def _questions(
        self, mode: QuestionMode, barriers: tuple[Barrier, ...] | None
    ) -> dict[str, Any]:
        if self.settings.trading_strategy is TradingStrategy.SCALP:
            if barriers is None:
                raise ValueError("SCALP Jev calls require barriers")
            return scalp_questions(barriers)
        if self.settings.trading_strategy is TradingStrategy.DIRECTIONAL:
            return directional_questions()
        return questions_for(mode, self.settings.primary_horizon_seconds)
