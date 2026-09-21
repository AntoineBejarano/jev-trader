from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.jev.models import ChoiceQuestion, NoulQuestion, QuestionMode
from src.market.models import Side

QUESTION_SET_FAMILY = "q.v1"


def question_set_version(primary_horizon_seconds: int) -> str:
    return f"{QUESTION_SET_FAMILY}.h{primary_horizon_seconds}"


def _noul(instructions: str, true: str, false: str) -> dict[str, Any]:
    return NoulQuestion(
        instructions=instructions,
        criteria={"true": true, "false": false},
    ).model_dump()


def _choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
    return ChoiceQuestion(instructions=instructions, criteria=dict(criteria)).model_dump()


def _horizon_phrase(primary_horizon_seconds: int) -> str:
    minutes = primary_horizon_seconds / 60.0
    if minutes == int(minutes):
        return f"the next approximately {int(minutes)} minutes"
    return f"the next approximately {primary_horizon_seconds} seconds"


def entry_questions(primary_horizon_seconds: int) -> dict[str, Any]:
    horizon = _horizon_phrase(primary_horizon_seconds)
    return {
        "long_ev": _noul(
            (
                "Is taking LONG exposure now likely to have positive expected "
                f"net value over {horizon}?"
            ),
            "Long exposure now is more likely than not to be +EV after fees, spread and slippage.",
            "Long exposure now is not supported as +EV after costs.",
        ),
        "short_ev": _noul(
            (
                "Is taking SHORT exposure now likely to have positive expected "
                f"net value over {horizon}?"
            ),
            "Short exposure now is more likely than not to be +EV after fees, spread and slippage.",
            "Short exposure now is not supported as +EV after costs.",
        ),
        "bullish_continuation": _noul(
            "Is bullish continuation likely from the current state?",
            "Upward continuation is likely.",
            "Upward continuation is not likely.",
        ),
        "bearish_continuation": _noul(
            "Is bearish continuation likely from the current state?",
            "Downward continuation is likely.",
            "Downward continuation is not likely.",
        ),
        "reversal_starting": _noul(
            "Is a reversal likely to be starting now?",
            "A directional reversal is likely beginning.",
            "A reversal is not likely beginning.",
        ),
        "is_noise": _noul(
            "Is the current movement likely to be noise rather than a usable directional move?",
            "The current movement is likely noise.",
            "The current movement is likely more than noise.",
        ),
        "move_exhausted": _noul(
            "Is the current move likely exhausted?",
            "The current directional move is likely exhausted.",
            "The current directional move is not likely exhausted.",
        ),
        "sufficient_evidence": _noul(
            "Is there sufficient evidence to take directional risk now?",
            "There is sufficient evidence to take directional risk.",
            "There is not sufficient evidence to take directional risk.",
        ),
        "desired_exposure": _choice(
            f"Which exposure is best now for {horizon}: LONG, FLAT, or SHORT?",
            {
                "LONG": "Long exposure is the best of the three options now.",
                "FLAT": "Standing aside is the best of the three options now.",
                "SHORT": "Short exposure is the best of the three options now.",
            },
        ),
    }


def position_questions(primary_horizon_seconds: int) -> dict[str, Any]:
    horizon = _horizon_phrase(primary_horizon_seconds)
    return {
        "maintain_ev": _noul(
            (
                "Does maintaining the current position likely have positive "
                f"expected net value from now over {horizon}?"
            ),
            "Staying in the current position is more likely than not +EV after remaining costs.",
            "Staying in the current position is not supported as +EV after costs.",
        ),
        "continuation": _noul(
            "Is continuation in the current position direction likely?",
            "Continuation in the position direction is likely.",
            "Continuation in the position direction is not likely.",
        ),
        "reversal_against": _noul(
            "Is reversal against the current position likely?",
            "A reversal against the position is likely.",
            "A reversal against the position is not likely.",
        ),
        "move_exhausted": _noul(
            "Is the current move likely exhausted?",
            "The current directional move is likely exhausted.",
            "The current directional move is not likely exhausted.",
        ),
        "evidence_weakened": _noul(
            "Has the original directional evidence weakened materially?",
            "The original evidence for this exposure has weakened materially.",
            "The original evidence has not weakened materially.",
        ),
        "sufficient_evidence": _noul(
            "Is there sufficient evidence to continue taking this directional risk?",
            "There is sufficient evidence to continue taking this risk.",
            "There is not sufficient evidence to continue taking this risk.",
        ),
        "position_action": _choice(
            "What should happen to the current position: HOLD, REDUCE, CLOSE, or REVERSE?",
            {
                "HOLD": "Keep the current exposure unchanged.",
                "REDUCE": "Cut size but remain in the same direction.",
                "CLOSE": "Exit to flat.",
                "REVERSE": "Close and flip to the opposite direction.",
            },
        ),
    }


def questions_for(mode: QuestionMode, primary_horizon_seconds: int) -> dict[str, Any]:
    if mode is QuestionMode.POSITION:
        return position_questions(primary_horizon_seconds)
    return entry_questions(primary_horizon_seconds)


def mode_for_side(side: Side) -> QuestionMode:
    return QuestionMode.ENTRY if side is Side.FLAT else QuestionMode.POSITION
