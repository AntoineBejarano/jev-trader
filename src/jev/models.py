from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuestionMode(StrEnum):
    ENTRY = "ENTRY"
    POSITION = "POSITION"


class QuestionType(StrEnum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class NoulQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: dict[str, str]


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


class NoulAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: Literal["noul"]
    noul: float


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: Literal["choice"]
    choice: str
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: Literal["score"]
    score: float


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None


class DecisionsResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    model: str
    id: str | None = None
    provider: str | None = None
    usage: Usage | None = None


class ScalpHorizonForecast(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon_seconds: int
    p_up: float = 0.0
    p_down: float = 0.0
    p_neither: float = 0.0
    p_long_target: float = 0.0
    p_long_stop: float = 0.0
    p_long_neither: float = 0.0
    p_short_target: float = 0.0
    p_short_stop: float = 0.0
    p_short_neither: float = 0.0
    expected_return_bps: float = 0.0
    target_bps: float
    stop_bps: float
    neither_return_bps: float = 0.0
    long_neither_return_bps: float = 0.0
    short_neither_return_bps: float = 0.0
    neither_payoff_method: str = "JEV_CONDITIONAL"
    score_raw: float | None = None
    long_neither_score_raw: float | None = None
    short_neither_score_raw: float | None = None
    score_mapping_version: str = "jev.score.return.v1"
    question_set_version: str | None = None
    volatility_target_bps: float = 0.0
    cost_floor_bps: float = 0.0
    target_bound_by_cost_floor: bool = False
    horizon_cost_constrained: bool = False

    @model_validator(mode="before")
    @classmethod
    def _fill_side_races(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        p_up = float(data.get("p_up") or 0.0)
        p_down = float(data.get("p_down") or 0.0)
        p_neither = float(data.get("p_neither") or 0.0)
        shared = p_up + p_down + p_neither
        long_keys = {"p_long_target", "p_long_stop", "p_long_neither"}
        short_keys = {"p_short_target", "p_short_stop", "p_short_neither"}
        if long_keys.isdisjoint(data.keys()) and shared > 0.0:
            data["p_long_target"] = p_up
            data["p_long_stop"] = p_down
            data["p_long_neither"] = p_neither
        if short_keys.isdisjoint(data.keys()) and shared > 0.0:
            data["p_short_target"] = p_down
            data["p_short_stop"] = p_up
            data["p_short_neither"] = p_neither
        neither = float(data.get("neither_return_bps") or data.get("expected_return_bps") or 0.0)
        if "long_neither_return_bps" not in data and neither != 0.0:
            data["long_neither_return_bps"] = neither
        if "short_neither_return_bps" not in data and neither != 0.0:
            data["short_neither_return_bps"] = neither
        if data.get("long_neither_score_raw") is None and data.get("score_raw") is not None:
            data["long_neither_score_raw"] = data.get("score_raw")
        if data.get("short_neither_score_raw") is None and data.get("score_raw") is not None:
            data["short_neither_score_raw"] = data.get("score_raw")
        return data


class ParsedJevDecision(BaseModel):
    """Normalized probabilities used later by DecisionPolicy. Not an order."""

    model_config = ConfigDict(frozen=True)

    mode: QuestionMode
    question_set_version: str
    model: str
    request_id: str | None = None
    latency_ms: float
    long_probability: float | None = None
    short_probability: float | None = None
    continuation_probability: float | None = None
    reversal_probability: float | None = None
    exhaustion_probability: float | None = None
    noise_probability: float | None = None
    evidence_probability: float | None = None
    exposure_choice: str | None = None
    exposure_choice_probabilities: dict[str, float] = Field(default_factory=dict)
    position_action: str | None = None
    position_action_probabilities: dict[str, float] = Field(default_factory=dict)
    scalp_forecasts: tuple[ScalpHorizonForecast, ...] = ()
    directional_forecast: dict[str, Any] | None = None
    raw_answers: dict[str, Any] = Field(default_factory=dict)


class JevCallStatus(StrEnum):
    OK = "OK"
    SKIPPED_STALE = "SKIPPED_STALE"
    SKIPPED_KILL_SWITCH = "SKIPPED_KILL_SWITCH"
    SKIPPED_NO_API_KEY = "SKIPPED_NO_API_KEY"
    TIMEOUT = "TIMEOUT"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    UNAVAILABLE = "UNAVAILABLE"


class JevSafetyAction(StrEnum):
    NONE = "NONE"
    BLOCK_OPEN = "BLOCK_OPEN"
    FLATTEN = "FLATTEN"


class JevCallResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: JevCallStatus
    symbol: str
    exchange: str
    mode: QuestionMode
    question_set_version: str
    latency_ms: float | None = None
    decision: ParsedJevDecision | None = None
    safety_action: JevSafetyAction = JevSafetyAction.NONE
    error: str | None = None
    raw_response: dict[str, Any] | None = None
