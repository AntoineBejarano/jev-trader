from __future__ import annotations

from typing import Any

from src.directional.constants import MAGNITUDE_BUCKETS
from src.directional.questions import QUESTION_SET_VERSION, directional_required_keys
from src.jev.models import (
    ChoiceAnswer,
    DecisionsResponse,
    NoulAnswer,
    ParsedJevDecision,
    QuestionMode,
    ScoreAnswer,
)
from src.jev.parse import _choice, _noul


def _p(answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer], key: str) -> float:
    value = _noul(answers, key)
    if value is None:
        return 0.5
    return min(1.0, max(0.0, float(value)))


def _bucket(
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer], key: str
) -> dict[str, Any]:
    choice = _choice(answers, key)
    if choice is None:
        return {"choice": "NORMAL", "probabilities": {name: 0.25 for name in MAGNITUDE_BUCKETS}}
    probs = {name: float(choice.probabilities.get(name, 0.0)) for name in MAGNITUDE_BUCKETS}
    total = sum(probs.values())
    if total <= 0:
        probs = {name: 1.0 if name == choice.choice else 0.0 for name in MAGNITUDE_BUCKETS}
        if choice.choice not in MAGNITUDE_BUCKETS:
            probs = {name: 0.25 for name in MAGNITUDE_BUCKETS}
    else:
        probs = {name: value / total for name, value in probs.items()}
    selected = (
        choice.choice
        if choice.choice in MAGNITUDE_BUCKETS
        else max(probs, key=lambda name: probs[name])
    )
    return {"choice": selected, "probabilities": probs}


def parse_directional_forecast(response: DecisionsResponse) -> dict[str, Any]:
    answers = response.answers
    p_up = {
        "1m": _p(answers, "up_60s"),
        "5m": _p(answers, "up_300s"),
        "10m": _p(answers, "up_600s"),
        "30m": _p(answers, "up_1800s"),
    }
    warnings: list[str] = []
    persist = {
        "1m_2_up": _p(answers, "next_2_1m_up"),
        "1m_3_up": _p(answers, "next_3_1m_up"),
        "1m_5_up": _p(answers, "next_5_1m_up"),
        "1m_2_down": _p(answers, "next_2_1m_down"),
        "1m_3_down": _p(answers, "next_3_1m_down"),
        "1m_5_down": _p(answers, "next_5_1m_down"),
        "5m_2_up": _p(answers, "next_2_5m_up"),
        "5m_3_up": _p(answers, "next_3_5m_up"),
        "5m_2_down": _p(answers, "next_2_5m_down"),
        "5m_3_down": _p(answers, "next_3_5m_down"),
    }
    if persist["1m_3_up"] > persist["1m_2_up"] + 1e-9:
        warnings.append("probability_consistency_warning:1m_up_3>2")
    if persist["1m_5_up"] > persist["1m_3_up"] + 1e-9:
        warnings.append("probability_consistency_warning:1m_up_5>3")
    if persist["1m_3_down"] > persist["1m_2_down"] + 1e-9:
        warnings.append("probability_consistency_warning:1m_down_3>2")
    if persist["1m_5_down"] > persist["1m_3_down"] + 1e-9:
        warnings.append("probability_consistency_warning:1m_down_5>3")
    if persist["5m_3_up"] > persist["5m_2_up"] + 1e-9:
        warnings.append("probability_consistency_warning:5m_up_3>2")
    if persist["5m_3_down"] > persist["5m_2_down"] + 1e-9:
        warnings.append("probability_consistency_warning:5m_down_3>2")
    sides = {label: "UP" if prob >= 0.5 else "DOWN" for label, prob in p_up.items()}
    unique = set(sides.values())
    consensus = next(iter(unique)) if len(unique) == 1 else "MIXED"
    character = _choice(answers, "market_character")
    return {
        "p_up": p_up,
        "p_down": {label: 1.0 - prob for label, prob in p_up.items()},
        "term_structure": [
            {
                "horizon": label,
                "horizon_seconds": seconds,
                "p_up": p_up[label],
                "p_down": 1.0 - p_up[label],
                "side": sides[label],
            }
            for label, seconds in (("1m", 60), ("5m", 300), ("10m", 600), ("30m", 1800))
        ],
        "magnitude": {
            "1m": _bucket(answers, "move_60s"),
            "5m": _bucket(answers, "move_300s"),
            "10m": _bucket(answers, "move_600s"),
            "30m": _bucket(answers, "move_1800s"),
        },
        "candle_up": {
            "1m": _p(answers, "candle_up_1m"),
            "5m": _p(answers, "candle_up_5m"),
            "10m": _p(answers, "candle_up_10m"),
            "30m": _p(answers, "candle_up_30m"),
        },
        "persistence": persist,
        "diagnostics": {
            "momentum_strengthening": _p(answers, "momentum_strengthening"),
            "momentum_weakening": _p(answers, "momentum_weakening"),
            "reversal_60s": _p(answers, "reversal_60s"),
            "market_character": None if character is None else character.choice,
        },
        "direction_consensus": consensus,
        "direction_conflict": consensus == "MIXED",
        "probability_consistency_warnings": warnings,
    }


def parse_directional_decision(
    response: DecisionsResponse,
    *,
    mode: QuestionMode,
    latency_ms: float,
) -> ParsedJevDecision:
    forecast = parse_directional_forecast(response)
    p_up_1m = float(forecast["p_up"]["1m"])
    p_up_5m = float(forecast["p_up"]["5m"])
    return ParsedJevDecision(
        mode=mode,
        question_set_version=QUESTION_SET_VERSION,
        model=response.model,
        request_id=response.id,
        latency_ms=latency_ms,
        long_probability=p_up_1m,
        short_probability=1.0 - p_up_1m,
        continuation_probability=p_up_5m,
        raw_answers={key: answer.model_dump() for key, answer in response.answers.items()},
        directional_forecast=forecast,
    )


def validate_directional_answers(response: DecisionsResponse) -> None:
    missing = [key for key in directional_required_keys() if key not in response.answers]
    if missing:
        raise ValueError(f"Jev response missing required directional answers: {missing}")
