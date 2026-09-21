from __future__ import annotations

from typing import Any

from src.jev.models import ChoiceQuestion, NoulQuestion

QUESTION_SET_VERSION = "q.directional.v1"


def _noul(instructions: str, true: str, false: str) -> dict[str, Any]:
    return NoulQuestion(
        instructions=instructions, criteria={"true": true, "false": false}
    ).model_dump()


def _choice(instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    return ChoiceQuestion(instructions=instructions, criteria=criteria).model_dump()


def directional_questions() -> dict[str, Any]:
    """Market-forecast questions only. No fees, PnL, or required EV."""
    questions: dict[str, Any] = {}
    rolling = (
        ("up_60s", 60, "1 minute"),
        ("up_300s", 300, "5 minutes"),
        ("up_600s", 600, "10 minutes"),
        ("up_1800s", 1800, "30 minutes"),
    )
    for key, seconds, label in rolling:
        questions[key] = _noul(
            (
                "Will the BTC USD-M perpetual mid be ABOVE the current reference mid "
                f"in {seconds} seconds ({label}) from now? Answer only the mid-price "
                "direction. Ignore trading fees, slippage, and whether a trade would "
                "be profitable."
            ),
            f"The mid is more likely to be above the current mid after {label}.",
            f"The mid is not more likely to be above the current mid after {label}.",
        )
        questions[f"move_{seconds}s"] = _choice(
            (
                f"Classify the absolute mid move over the next {label} relative to the "
                "horizon sigma supplied in state.sigma_bps. NOISE < 0.5 sigma, "
                "NORMAL 0.5-1 sigma, LARGE 1-2 sigma, EXTREME > 2 sigma. Do not "
                "predict an exact bps target."
            ),
            {
                "NOISE": "Absolute move likely smaller than 0.5 sigma.",
                "NORMAL": "Absolute move likely between 0.5 and 1 sigma.",
                "LARGE": "Absolute move likely between 1 and 2 sigma.",
                "EXTREME": "Absolute move likely larger than 2 sigma.",
            },
        )
    for tf, label in (
        ("1m", "1-minute"),
        ("5m", "5-minute"),
        ("10m", "10-minute"),
        ("30m", "30-minute"),
    ):
        questions[f"candle_up_{tf}"] = _noul(
            (
                f"Will the current UTC-aligned {label} window finish with close above "
                "its open? Use the active-candle remaining time in state. This is the "
                "candle direction, not a rolling horizon from now."
            ),
            f"The current {label} window is more likely to close up.",
            f"The current {label} window is not more likely to close up.",
        )
    for count in (2, 3, 5):
        questions[f"next_{count}_1m_up"] = _noul(
            (
                f"Will the next {count} completed UTC-aligned 1-minute candles ALL "
                "close up? This is consecutive-candle persistence, not whether price "
                "is higher at the end of the streak."
            ),
            f"The next {count} completed 1-minute candles are all likely to close up.",
            f"The next {count} completed 1-minute candles are not all likely to close up.",
        )
        questions[f"next_{count}_1m_down"] = _noul(
            (
                f"Will the next {count} completed UTC-aligned 1-minute candles ALL "
                "close down?"
            ),
            f"The next {count} completed 1-minute candles are all likely to close down.",
            f"The next {count} completed 1-minute candles are not all likely to close down.",
        )
    for count in (2, 3):
        questions[f"next_{count}_5m_up"] = _noul(
            (
                f"Will the next {count} completed UTC-aligned 5-minute candles ALL "
                "close up?"
            ),
            f"The next {count} completed 5-minute candles are all likely to close up.",
            f"The next {count} completed 5-minute candles are not all likely to close up.",
        )
        questions[f"next_{count}_5m_down"] = _noul(
            (
                f"Will the next {count} completed UTC-aligned 5-minute candles ALL "
                "close down?"
            ),
            f"The next {count} completed 5-minute candles are all likely to close down.",
            f"The next {count} completed 5-minute candles are not all likely to close down.",
        )
    questions["momentum_strengthening"] = _noul(
        "Is current directional momentum strengthening?",
        "Directional momentum is strengthening.",
        "Directional momentum is not strengthening.",
    )
    questions["momentum_weakening"] = _noul(
        "Is current directional momentum weakening?",
        "Directional momentum is weakening.",
        "Directional momentum is not weakening.",
    )
    questions["reversal_60s"] = _noul(
        "Is a directional reversal likely within the next 60 seconds?",
        "A directional reversal is likely within 60 seconds.",
        "A directional reversal is not likely within 60 seconds.",
    )
    questions["market_character"] = _choice(
        "Is the current market better characterized as DIRECTIONAL, TRANSITION, or CHOP?",
        {
            "DIRECTIONAL": "A usable directional regime.",
            "TRANSITION": "A regime change or transition.",
            "CHOP": "Two-sided chop without a durable direction.",
        },
    )
    return questions


def directional_required_keys() -> tuple[str, ...]:
    keys = [
        "up_60s",
        "up_300s",
        "up_600s",
        "up_1800s",
        "move_60s",
        "move_300s",
        "move_600s",
        "move_1800s",
        "candle_up_1m",
        "candle_up_5m",
        "candle_up_10m",
        "candle_up_30m",
        "next_2_1m_up",
        "next_3_1m_up",
        "next_5_1m_up",
        "next_2_1m_down",
        "next_3_1m_down",
        "next_5_1m_down",
        "next_2_5m_up",
        "next_3_5m_up",
        "next_2_5m_down",
        "next_3_5m_down",
        "momentum_strengthening",
        "momentum_weakening",
        "reversal_60s",
        "market_character",
    ]
    return tuple(keys)
