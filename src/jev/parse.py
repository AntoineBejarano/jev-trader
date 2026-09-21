from __future__ import annotations

from typing import Any

from src.jev.models import (
    ChoiceAnswer,
    DecisionsResponse,
    NoulAnswer,
    ParsedJevDecision,
    QuestionMode,
    ScoreAnswer,
)


def _noul(
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer], key: str
) -> float | None:
    answer = answers.get(key)
    if isinstance(answer, NoulAnswer):
        return answer.noul
    return None


def _choice(
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer], key: str
) -> ChoiceAnswer | None:
    answer = answers.get(key)
    if isinstance(answer, ChoiceAnswer):
        return answer
    return None


def parse_decision(
    response: DecisionsResponse,
    *,
    mode: QuestionMode,
    question_set_version: str,
    latency_ms: float,
) -> ParsedJevDecision:
    answers = response.answers
    exposure = _choice(answers, "desired_exposure")
    action = _choice(answers, "position_action")
    return ParsedJevDecision(
        mode=mode,
        question_set_version=question_set_version,
        model=response.model,
        request_id=response.id,
        latency_ms=latency_ms,
        long_probability=_noul(answers, "long_ev"),
        short_probability=_noul(answers, "short_ev"),
        continuation_probability=_noul(answers, "bullish_continuation")
        if mode is QuestionMode.ENTRY
        else _noul(answers, "continuation"),
        reversal_probability=_noul(answers, "reversal_starting")
        if mode is QuestionMode.ENTRY
        else _noul(answers, "reversal_against"),
        exhaustion_probability=_noul(answers, "move_exhausted"),
        noise_probability=_noul(answers, "is_noise"),
        evidence_probability=_noul(answers, "sufficient_evidence"),
        exposure_choice=None if exposure is None else exposure.choice,
        exposure_choice_probabilities={} if exposure is None else exposure.probabilities,
        position_action=None if action is None else action.choice,
        position_action_probabilities={} if action is None else action.probabilities,
        raw_answers={key: answer.model_dump() for key, answer in answers.items()},
    )


def required_keys(mode: QuestionMode) -> tuple[str, ...]:
    if mode is QuestionMode.ENTRY:
        return (
            "long_ev",
            "short_ev",
            "sufficient_evidence",
            "desired_exposure",
        )
    return (
        "maintain_ev",
        "sufficient_evidence",
        "position_action",
    )


def validate_required_answers(response: DecisionsResponse, mode: QuestionMode) -> None:
    missing = [key for key in required_keys(mode) if key not in response.answers]
    if missing:
        raise ValueError(f"Jev response missing required answers: {missing}")


def decode_raw(payload: dict[str, Any]) -> DecisionsResponse:
    return DecisionsResponse.model_validate(payload)
