from __future__ import annotations

from collections.abc import Sequence

from src.decision.barriers import Barrier
from src.decision.ev import (
    NEITHER_JEV_CONDITIONAL,
    NEITHER_ZERO_CONSERVATIVE,
    normalize_barrier_probs,
)
from src.jev.models import (
    ChoiceAnswer,
    DecisionsResponse,
    NoulAnswer,
    ParsedJevDecision,
    QuestionMode,
    ScalpHorizonForecast,
    ScoreAnswer,
)
from src.jev.parse import _choice, _noul
from src.jev.scalp_questions import (
    LONG_NEITHER,
    LONG_STOP_FIRST,
    LONG_TARGET_FIRST,
    SCORE_MAPPING_VERSION,
    SHORT_NEITHER,
    SHORT_STOP_FIRST,
    SHORT_TARGET_FIRST,
    long_barrier_key,
    long_neither_return_key,
    scalp_question_set_version,
    scalp_required_keys,
    score_to_expected_return_bps,
    short_barrier_key,
    short_neither_return_key,
)


def _score(
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer], key: str
) -> float | None:
    answer = answers.get(key)
    if isinstance(answer, ScoreAnswer):
        return float(answer.score)
    return None


def _triplet_from_choice(
    choice: ChoiceAnswer,
    *,
    target_label: str,
    stop_label: str,
    neither_label: str,
) -> tuple[float, float, float]:
    probs = choice.probabilities
    p_target = float(probs.get(target_label, 0.0))
    p_stop = float(probs.get(stop_label, 0.0))
    p_neither = float(probs.get(neither_label, 0.0))
    if p_target + p_stop + p_neither <= 0:
        if choice.choice == target_label:
            p_target = 1.0
        elif choice.choice == stop_label:
            p_stop = 1.0
        else:
            p_neither = 1.0
    return normalize_barrier_probs(p_target, p_stop, p_neither)


def _neither_payoff(
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer], key: str
) -> tuple[float, str, float | None]:
    raw = _score(answers, key)
    if raw is None:
        return 0.0, NEITHER_ZERO_CONSERVATIVE, None
    return score_to_expected_return_bps(raw), NEITHER_JEV_CONDITIONAL, raw


def parse_scalp_decision(
    response: DecisionsResponse,
    *,
    mode: QuestionMode,
    barriers: Sequence[Barrier],
    latency_ms: float,
) -> ParsedJevDecision:
    answers = response.answers
    question_version = scalp_question_set_version(
        [barrier.horizon_seconds for barrier in barriers]
    )
    forecasts: list[ScalpHorizonForecast] = []
    for barrier in barriers:
        long_choice = _choice(answers, long_barrier_key(barrier.horizon_seconds))
        short_choice = _choice(answers, short_barrier_key(barrier.horizon_seconds))
        if long_choice is None:
            raise ValueError(f"missing LONG barrier choice for {barrier.horizon_seconds}s")
        if short_choice is None:
            raise ValueError(f"missing SHORT barrier choice for {barrier.horizon_seconds}s")
        p_long_target, p_long_stop, p_long_neither = _triplet_from_choice(
            long_choice,
            target_label=LONG_TARGET_FIRST,
            stop_label=LONG_STOP_FIRST,
            neither_label=LONG_NEITHER,
        )
        p_short_target, p_short_stop, p_short_neither = _triplet_from_choice(
            short_choice,
            target_label=SHORT_TARGET_FIRST,
            stop_label=SHORT_STOP_FIRST,
            neither_label=SHORT_NEITHER,
        )
        long_neither_bps, long_method, long_raw = _neither_payoff(
            answers, long_neither_return_key(barrier.horizon_seconds)
        )
        short_neither_bps, short_method, short_raw = _neither_payoff(
            answers, short_neither_return_key(barrier.horizon_seconds)
        )
        method = (
            NEITHER_JEV_CONDITIONAL
            if NEITHER_JEV_CONDITIONAL in {long_method, short_method}
            else NEITHER_ZERO_CONSERVATIVE
        )
        forecasts.append(
            ScalpHorizonForecast(
                horizon_seconds=barrier.horizon_seconds,
                p_up=p_long_target,
                p_down=p_long_stop,
                p_neither=p_long_neither,
                p_long_target=p_long_target,
                p_long_stop=p_long_stop,
                p_long_neither=p_long_neither,
                p_short_target=p_short_target,
                p_short_stop=p_short_stop,
                p_short_neither=p_short_neither,
                expected_return_bps=long_neither_bps,
                neither_return_bps=long_neither_bps,
                long_neither_return_bps=long_neither_bps,
                short_neither_return_bps=short_neither_bps,
                neither_payoff_method=method,
                score_raw=long_raw,
                long_neither_score_raw=long_raw,
                short_neither_score_raw=short_raw,
                score_mapping_version=SCORE_MAPPING_VERSION,
                question_set_version=question_version,
                target_bps=barrier.target_bps,
                stop_bps=barrier.stop_bps,
                volatility_target_bps=barrier.volatility_target_bps,
                cost_floor_bps=barrier.cost_floor_bps,
                target_bound_by_cost_floor=barrier.target_bound_by_cost_floor,
                horizon_cost_constrained=barrier.horizon_cost_constrained,
            )
        )
    return ParsedJevDecision(
        mode=mode,
        question_set_version=question_version,
        model=response.model,
        request_id=response.id,
        latency_ms=latency_ms,
        noise_probability=_noul(answers, "is_noise"),
        scalp_forecasts=tuple(forecasts),
        raw_answers={key: answer.model_dump() for key, answer in answers.items()},
    )


def validate_scalp_answers(response: DecisionsResponse, horizons: Sequence[int]) -> None:
    missing = [key for key in scalp_required_keys(horizons) if key not in response.answers]
    if missing:
        raise ValueError(f"Jev response missing required scalp answers: {missing}")
