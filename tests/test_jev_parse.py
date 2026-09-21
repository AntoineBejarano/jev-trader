from __future__ import annotations

from src.jev.models import QuestionMode
from src.jev.parse import decode_raw, parse_decision, validate_required_answers
from src.jev.questions import question_set_version

ENTRY_PAYLOAD = {
    "id": "gen-dec-test-1",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {
        "long_ev": {"type": "noul", "noul": 0.61},
        "short_ev": {"type": "noul", "noul": 0.22},
        "bullish_continuation": {"type": "noul", "noul": 0.58},
        "bearish_continuation": {"type": "noul", "noul": 0.19},
        "reversal_starting": {"type": "noul", "noul": 0.31},
        "is_noise": {"type": "noul", "noul": 0.44},
        "move_exhausted": {"type": "noul", "noul": 0.27},
        "sufficient_evidence": {"type": "noul", "noul": 0.52},
        "desired_exposure": {
            "type": "choice",
            "choice": "FLAT",
            "confidence": 0.71,
            "probabilities": {"LONG": 0.18, "FLAT": 0.71, "SHORT": 0.11},
        },
    },
    "usage": {"input_tokens": 400, "output_tokens": 40, "cost": 0.00002},
}


def test_parse_entry_decision_probabilities() -> None:
    parsed = decode_raw(ENTRY_PAYLOAD)
    validate_required_answers(parsed, QuestionMode.ENTRY)
    decision = parse_decision(
        parsed,
        mode=QuestionMode.ENTRY,
        question_set_version=question_set_version(300),
        latency_ms=260.0,
    )
    assert decision.long_probability == 0.61
    assert decision.short_probability == 0.22
    assert decision.exposure_choice == "FLAT"
    assert decision.exposure_choice_probabilities["FLAT"] == 0.71
    assert decision.question_set_version == "q.v1.h300"
    assert decision.model.startswith("typesafe/jev-1.13")


def test_missing_required_answer_is_invalid() -> None:
    payload = {
        "model": "typesafe/jev-1.13",
        "answers": {"long_ev": {"type": "noul", "noul": 0.4}},
    }
    parsed = decode_raw(payload)
    try:
        validate_required_answers(parsed, QuestionMode.ENTRY)
    except ValueError as exc:
        assert "desired_exposure" in str(exc)
    else:
        raise AssertionError("expected missing-answer error")
