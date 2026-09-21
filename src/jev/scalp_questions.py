from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.decision.barriers import Barrier
from src.jev.models import ChoiceQuestion, NoulQuestion, ScoreQuestion

SCALP_QUESTION_SET_FAMILY = "q.scalp.v3"
SCORE_MAPPING_VERSION = "jev.score.return.v1"

LONG_TARGET_FIRST = "LONG_TARGET_FIRST"
LONG_STOP_FIRST = "LONG_STOP_FIRST"
LONG_NEITHER = "LONG_NEITHER"
SHORT_TARGET_FIRST = "SHORT_TARGET_FIRST"
SHORT_STOP_FIRST = "SHORT_STOP_FIRST"
SHORT_NEITHER = "SHORT_NEITHER"

# Ordered Jev score levels, low to high. The returned score is a (possibly
# fractional) index into this list, which we interpolate to signed bps.
# These buckets are a versioned interpretation, not unconstrained bps.
NEITHER_RETURN_CRITERIA: tuple[str, ...] = (
    "Conditional expected signed return given neither barrier is around -80 bps or worse.",
    "Conditional expected signed return given neither barrier is around -40 bps.",
    "Conditional expected signed return given neither barrier is around -15 bps.",
    "Conditional expected signed return given neither barrier is around -5 bps.",
    "Conditional expected signed return given neither barrier is approximately zero.",
    "Conditional expected signed return given neither barrier is around +5 bps.",
    "Conditional expected signed return given neither barrier is around +15 bps.",
    "Conditional expected signed return given neither barrier is around +40 bps.",
    "Conditional expected signed return given neither barrier is around +80 bps or more.",
)
NEITHER_RETURN_ANCHORS_BPS: tuple[float, ...] = (
    -80.0,
    -40.0,
    -15.0,
    -5.0,
    0.0,
    5.0,
    15.0,
    40.0,
    80.0,
)
SCORE_LEVELS: tuple[int, ...] = (-80, -40, -15, -5, 0, 5, 15, 40, 80)

# Back-compat aliases used by tests and older names.
EXPECTED_RETURN_CRITERIA = NEITHER_RETURN_CRITERIA
EXPECTED_RETURN_ANCHORS_BPS = NEITHER_RETURN_ANCHORS_BPS


def score_to_expected_return_bps(score: float) -> float:
    """Map Jev's ordered score index onto signed expected-return bps."""
    anchors = NEITHER_RETURN_ANCHORS_BPS
    if score <= 0:
        return anchors[0]
    last = float(len(anchors) - 1)
    if score >= last:
        return anchors[-1]
    lo = int(score)
    hi = lo + 1
    frac = score - lo
    return anchors[lo] * (1.0 - frac) + anchors[hi] * frac


def scalp_question_set_version(horizons: Sequence[int]) -> str:
    joined = "-".join(str(horizon) for horizon in horizons)
    return f"{SCALP_QUESTION_SET_FAMILY}.h{joined}"


def long_barrier_key(horizon: int) -> str:
    return f"h{horizon}_long_barrier"


def short_barrier_key(horizon: int) -> str:
    return f"h{horizon}_short_barrier"


def long_neither_return_key(horizon: int) -> str:
    return f"h{horizon}_long_neither_return_bps"


def short_neither_return_key(horizon: int) -> str:
    return f"h{horizon}_short_neither_return_bps"


def barrier_key(horizon: int) -> str:
    """Deprecated q.scalp.v2 key. Not used by q.scalp.v3."""
    return f"h{horizon}_barrier"


def neither_return_key(horizon: int) -> str:
    """Deprecated q.scalp.v2 key. Not used by q.scalp.v3."""
    return f"h{horizon}_neither_return_bps"


def expected_return_key(horizon: int) -> str:
    """Deprecated alias of neither_return_key; kept for parse fallback."""
    return neither_return_key(horizon)


def score_mapping_payload() -> dict[str, object]:
    return {
        "version": SCORE_MAPPING_VERSION,
        "levels_bps": list(SCORE_LEVELS),
        "note": "Jev score is an interpolated index into these levels, not raw bps.",
    }


def _neither_score_question(
    *, horizon: int, race: str, up_bps: float, down_bps: float
) -> dict[str, Any]:
    return ScoreQuestion(
        instructions=(
            f"Expected signed mid return in bps at {horizon} seconds FROM NOW, "
            f"GIVEN the {race} barrier race: neither {up_bps:+.2f} bps nor "
            f"{down_bps:+.2f} bps is hit before the horizon. This is "
            "E[R_H | neither barrier of this race], not the unconditional expected "
            "return. Positive means the mid went up. Negative means the mid went down. "
            "Choose the scored level that best matches that conditional return."
        ),
        criteria=list(NEITHER_RETURN_CRITERIA),
    ).model_dump()


def scalp_questions(barriers: Sequence[Barrier]) -> dict[str, Any]:
    questions: dict[str, Any] = {
        "is_noise": NoulQuestion(
            instructions=(
                "Is the current movement likely noise rather than a usable "
                "short-horizon directional opportunity after costs?"
            ),
            criteria={
                "true": "The current movement is likely noise.",
                "false": "The current movement is likely more than noise.",
            },
        ).model_dump()
    }
    for barrier in barriers:
        horizon = barrier.horizon_seconds
        target = barrier.target_bps
        stop = barrier.stop_bps
        questions[long_barrier_key(horizon)] = ChoiceQuestion(
            instructions=(
                f"LONG barrier race from the current mid, before {horizon} seconds. "
                f"Which outcome is most likely: +{target:.2f} bps is reached before "
                f"-{stop:.2f} bps, -{stop:.2f} bps is reached before +{target:.2f} bps, "
                "or neither of those two levels is reached? Answer only this LONG race. "
                "Do not infer it from a SHORT race."
            ),
            criteria={
                LONG_TARGET_FIRST: (
                    f"The mid is most likely to reach +{target:.2f} bps "
                    f"before -{stop:.2f} bps within {horizon}s."
                ),
                LONG_STOP_FIRST: (
                    f"The mid is most likely to reach -{stop:.2f} bps "
                    f"before +{target:.2f} bps within {horizon}s."
                ),
                LONG_NEITHER: (
                    f"Neither +{target:.2f} bps nor -{stop:.2f} bps is likely "
                    f"to be reached within {horizon}s."
                ),
            },
        ).model_dump()
        questions[short_barrier_key(horizon)] = ChoiceQuestion(
            instructions=(
                f"SHORT barrier race from the current mid, before {horizon} seconds. "
                f"Which outcome is most likely: -{target:.2f} bps is reached before "
                f"+{stop:.2f} bps, +{stop:.2f} bps is reached before -{target:.2f} bps, "
                "or neither of those two levels is reached? Answer only this SHORT race. "
                "Do not infer it from a LONG race."
            ),
            criteria={
                SHORT_TARGET_FIRST: (
                    f"The mid is most likely to reach -{target:.2f} bps "
                    f"before +{stop:.2f} bps within {horizon}s."
                ),
                SHORT_STOP_FIRST: (
                    f"The mid is most likely to reach +{stop:.2f} bps "
                    f"before -{target:.2f} bps within {horizon}s."
                ),
                SHORT_NEITHER: (
                    f"Neither -{target:.2f} bps nor +{stop:.2f} bps is likely "
                    f"to be reached within {horizon}s."
                ),
            },
        ).model_dump()
        questions[long_neither_return_key(horizon)] = _neither_score_question(
            horizon=horizon,
            race="LONG",
            up_bps=target,
            down_bps=-stop,
        )
        questions[short_neither_return_key(horizon)] = _neither_score_question(
            horizon=horizon,
            race="SHORT",
            up_bps=stop,
            down_bps=-target,
        )
    return questions


def scalp_required_keys(horizons: Sequence[int]) -> tuple[str, ...]:
    keys = ["is_noise"]
    for horizon in horizons:
        keys.append(long_barrier_key(horizon))
        keys.append(short_barrier_key(horizon))
        keys.append(long_neither_return_key(horizon))
        keys.append(short_neither_return_key(horizon))
    return tuple(keys)
