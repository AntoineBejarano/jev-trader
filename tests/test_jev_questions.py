from __future__ import annotations

from src.jev.questions import entry_questions, position_questions, question_set_version
from src.jev.state_serializer import serialize_market_state
from tests.helpers import make_market_state


def test_question_wording_is_instrument_agnostic() -> None:
    blob = str(entry_questions(300)) + str(position_questions(300))
    lowered = blob.lower()
    assert "btc" not in lowered
    assert "eth" not in lowered
    assert "bitcoin" not in lowered


def test_question_set_version_includes_horizon() -> None:
    assert question_set_version(300) == "q.v1.h300"
    assert question_set_version(180) == "q.v1.h180"


def test_serializer_is_compact_and_tagged() -> None:
    state = make_market_state(symbol="SOLUSDT")
    payload = serialize_market_state(state)
    assert payload["symbol"] == "SOLUSDT"
    assert payload["exchange"] == "binance_usdm"
    assert "bids" not in str(payload)
    assert "asks" not in str(payload)
    assert len(str(payload)) < 4000
