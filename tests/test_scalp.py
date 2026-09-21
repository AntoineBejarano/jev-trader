from __future__ import annotations

import pytest
from src.analytics.counterfactual import hypothetical_side_net_pnl_bps
from src.analytics.latency import LatencyTracker
from src.decision.actions import Action, ActionSource
from src.decision.barriers import Barrier, BarrierPlanner
from src.decision.costs import round_trip_cost_bps
from src.decision.ev import long_gross_ev_bps, rank_opportunities, short_gross_ev_bps
from src.decision.policy import DecisionPolicy
from src.decision.scalp import ScalpPolicy
from src.jev.models import (
    JevCallResult,
    JevCallStatus,
    JevSafetyAction,
    ParsedJevDecision,
    QuestionMode,
    ScalpHorizonForecast,
)
from src.jev.scalp_parse import parse_scalp_decision, validate_scalp_answers
from src.jev.scalp_questions import scalp_questions, score_to_expected_return_bps
from src.market.models import Side
from src.risk.engine import RiskEngine
from tests.helpers import make_market_state
from tests.paper_helpers import execute_action, fresh_account, jev_result, paper_settings


def scalp_settings(**overrides: object):
    payload = {
        "trading_strategy": "SCALP",
        "trading_mode": "PAPER",
        "taker_fee_bps": 0.0,
        "maker_fee_bps": 0.0,
        "scalp_taker_fee_bps": 0.0,
        "scalp_maker_fee_bps": 2.0,
        "slippage_bps": 0.0,
        "scalp_min_net_ev_bps": 1.0,
        "scalp_min_probability_edge": 0.05,
        "scalp_max_noise": 0.85,
        "scalp_min_position_age_seconds": 8.0,
        "scalp_reverse_advantage_bps": 1.5,
        "scalp_hold_min_net_ev_bps": 0.0,
        "scalp_close_ev_bps": 0.0,
        "scalp_horizons_seconds": "15,30,60",
    }
    payload.update(overrides)
    return paper_settings(**payload)


def _forecast(
    horizon: int,
    *,
    p_up: float,
    p_down: float,
    p_neither: float,
    expected: float,
    target: float,
    stop: float,
) -> ScalpHorizonForecast:
    return ScalpHorizonForecast(
        horizon_seconds=horizon,
        p_up=p_up,
        p_down=p_down,
        p_neither=p_neither,
        expected_return_bps=expected,
        target_bps=target,
        stop_bps=stop,
    )


def _scalp_result(
    state,
    forecasts: tuple[ScalpHorizonForecast, ...],
    *,
    noise: float = 0.1,
    status: JevCallStatus = JevCallStatus.OK,
) -> JevCallResult:
    decision = None
    if status is JevCallStatus.OK:
        decision = ParsedJevDecision(
            mode=QuestionMode.ENTRY,
            question_set_version="q.scalp.v1.h15-30-60",
            model="typesafe/jev-1.13",
            latency_ms=40.0,
            noise_probability=noise,
            scalp_forecasts=forecasts,
        )
    return JevCallResult(
        status=status,
        symbol=state.symbol,
        exchange=state.exchange,
        mode=QuestionMode.ENTRY,
        question_set_version="q.scalp.v1.h15-30-60",
        latency_ms=40.0,
        decision=decision,
        safety_action=JevSafetyAction.NONE,
    )


def _barriers_for(forecasts: tuple[ScalpHorizonForecast, ...]) -> tuple[Barrier, ...]:
    return tuple(
        Barrier(
            horizon_seconds=item.horizon_seconds,
            target_bps=item.target_bps,
            stop_bps=item.stop_bps,
            sigma_bps=0.0,
            cost_floor_bps=item.target_bps,
        )
        for item in forecasts
    )


def test_baseline_policy_is_unchanged() -> None:
    policy = DecisionPolicy(paper_settings())
    state = make_market_state()
    weak = jev_result(state, exposure="LONG", long_p=0.90, evidence=0.20)
    strong = jev_result(state, exposure="LONG", long_p=0.90, evidence=0.80)
    assert policy.propose(state, weak) is Action.NO_ACTION
    assert policy.propose(state, strong) is Action.OPEN_LONG


def test_scalp_ignores_baseline_thresholds() -> None:
    settings = scalp_settings()
    assert settings.taker_fee_bps == 0.0
    assert settings.entry_threshold == 0.80
    assert settings.evidence_threshold == 0.75
    policy = ScalpPolicy(settings)
    state = make_market_state(spread_bps=1.0)
    forecasts = (
        _forecast(
            30,
            p_up=0.70,
            p_down=0.15,
            p_neither=0.15,
            expected=4.0,
            target=8.0,
            stop=4.0,
        ),
    )
    result = _scalp_result(state, forecasts)
    plan = policy.evaluate(state, result, _barriers_for(forecasts))
    assert plan.candidate is Action.OPEN_LONG
    baseline = DecisionPolicy(settings)
    weak_baseline = jev_result(state, exposure="LONG", long_p=0.2, evidence=0.2)
    assert baseline.propose(state, weak_baseline) is Action.NO_ACTION


def test_barrier_planner_cost_floor() -> None:
    settings = scalp_settings(
        scalp_min_target_bps=1.5,
        scalp_profit_buffer_bps=0.5,
        scalp_taker_fee_bps=5.0,
        slippage_bps=1.0,
    )
    state = make_market_state(spread_bps=2.0, rv_1m=0.0)
    planned = BarrierPlanner(settings).plan(state)
    cost = round_trip_cost_bps(taker_fee_bps=5.0, spread_bps=2.0, slippage_bps=1.0)
    floor = cost + 0.5
    assert planned
    for barrier in planned:
        assert barrier.target_bps >= floor
        assert barrier.cost_floor_bps == pytest.approx(floor)


def test_barrier_planner_uses_vol_when_larger_than_floor() -> None:
    settings = scalp_settings(
        scalp_min_target_bps=1.5,
        scalp_profit_buffer_bps=0.5,
        scalp_taker_fee_bps=5.0,
        slippage_bps=1.0,
    )
    state = make_market_state(spread_bps=1.0, rv_1m=0.01)
    planned = {item.horizon_seconds: item for item in BarrierPlanner(settings).plan(state)}
    sixty = planned[60]
    assert sixty.sigma_bps == pytest.approx(100.0)
    assert sixty.target_bps == pytest.approx(100.0)


def test_long_and_short_ev_and_neither() -> None:
    long_gross = long_gross_ev_bps(
        p_up=0.50,
        p_down=0.20,
        p_neither=0.30,
        target_bps=10.0,
        stop_bps=5.0,
        expected_return_bps=2.0,
    )
    short_gross = short_gross_ev_bps(
        p_up=0.50,
        p_down=0.20,
        p_neither=0.30,
        target_bps=10.0,
        stop_bps=5.0,
        expected_return_bps=2.0,
    )
    assert long_gross == pytest.approx(0.50 * 10 - 0.20 * 5 + 0.30 * 2)
    assert short_gross == pytest.approx(0.20 * 10 - 0.50 * 5 - 0.30 * 2)


def test_best_horizon_and_side_selection() -> None:
    settings = scalp_settings(scalp_min_net_ev_bps=0.5, scalp_min_probability_edge=0.0)
    policy = ScalpPolicy(settings)
    state = make_market_state(spread_bps=0.5)
    forecasts = (
        _forecast(15, p_up=0.55, p_down=0.25, p_neither=0.20, expected=1.0, target=4.0, stop=3.0),
        _forecast(30, p_up=0.20, p_down=0.70, p_neither=0.10, expected=-5.0, target=6.0, stop=3.0),
        _forecast(60, p_up=0.40, p_down=0.40, p_neither=0.20, expected=0.0, target=5.0, stop=5.0),
    )
    plan = policy.evaluate(state, _scalp_result(state, forecasts), _barriers_for(forecasts))
    ranked = rank_opportunities(plan.opportunities)
    assert ranked[0] is plan.best
    assert plan.best is not None
    assert plan.best.side == "SHORT"
    assert plan.best.horizon_seconds == 30
    assert plan.candidate is Action.OPEN_SHORT


def test_negative_net_ev_is_no_trade() -> None:
    settings = scalp_settings(
        scalp_min_net_ev_bps=1.0, scalp_taker_fee_bps=5.0, slippage_bps=1.0
    )
    policy = ScalpPolicy(settings)
    state = make_market_state(spread_bps=2.0)
    forecasts = (
        _forecast(30, p_up=0.34, p_down=0.33, p_neither=0.33, expected=0.0, target=3.0, stop=3.0),
    )
    plan = policy.evaluate(state, _scalp_result(state, forecasts), _barriers_for(forecasts))
    assert plan.best is not None
    assert plan.best.net_ev_bps < 1.0
    assert plan.candidate is Action.NO_ACTION
    assert plan.policy_reason == "net_ev_below_edge"


def test_positive_long_and_short_candidates() -> None:
    settings = scalp_settings(scalp_min_net_ev_bps=0.2, scalp_min_probability_edge=0.0)
    policy = ScalpPolicy(settings)
    state = make_market_state(spread_bps=0.2)
    long_f = (
        _forecast(15, p_up=0.80, p_down=0.10, p_neither=0.10, expected=3.0, target=6.0, stop=2.0),
    )
    short_f = (
        _forecast(15, p_up=0.10, p_down=0.80, p_neither=0.10, expected=-3.0, target=6.0, stop=2.0),
    )
    long_plan = policy.evaluate(state, _scalp_result(state, long_f), _barriers_for(long_f))
    short_plan = policy.evaluate(state, _scalp_result(state, short_f), _barriers_for(short_f))
    assert long_plan.candidate is Action.OPEN_LONG
    assert short_plan.candidate is Action.OPEN_SHORT


def test_hold_close_and_reverse() -> None:
    settings = scalp_settings(scalp_min_net_ev_bps=1.0, scalp_reverse_advantage_bps=1.5)
    policy = ScalpPolicy(settings)
    hold_state = make_market_state(side=Side.LONG, age_seconds=30, spread_bps=0.2)
    hold_f = (
        _forecast(30, p_up=0.70, p_down=0.15, p_neither=0.15, expected=3.0, target=8.0, stop=3.0),
    )
    hold_plan = policy.evaluate(
        hold_state, _scalp_result(hold_state, hold_f), _barriers_for(hold_f)
    )
    assert hold_plan.candidate is Action.HOLD

    close_f = (
        _forecast(30, p_up=0.30, p_down=0.40, p_neither=0.30, expected=-0.5, target=4.0, stop=4.0),
    )
    close_plan = policy.evaluate(
        hold_state, _scalp_result(hold_state, close_f), _barriers_for(close_f)
    )
    assert close_plan.candidate is Action.CLOSE

    reverse_f = (
        _forecast(30, p_up=0.15, p_down=0.75, p_neither=0.10, expected=-5.0, target=8.0, stop=3.0),
    )
    reverse_plan = policy.evaluate(
        hold_state, _scalp_result(hold_state, reverse_f), _barriers_for(reverse_f)
    )
    assert reverse_plan.candidate is Action.REVERSE

    short_state = make_market_state(side=Side.SHORT, age_seconds=30, spread_bps=0.2)
    long_rev = (
        _forecast(30, p_up=0.75, p_down=0.15, p_neither=0.10, expected=5.0, target=8.0, stop=3.0),
    )
    short_to_long = policy.evaluate(
        short_state, _scalp_result(short_state, long_rev), _barriers_for(long_rev)
    )
    assert short_to_long.candidate is Action.REVERSE


def test_reversal_hysteresis() -> None:
    settings = scalp_settings(scalp_min_net_ev_bps=1.0, scalp_reverse_advantage_bps=1.5)
    policy = ScalpPolicy(settings)
    state = make_market_state(side=Side.LONG, age_seconds=30, spread_bps=0.0)
    mild = (
        _forecast(30, p_up=0.40, p_down=0.45, p_neither=0.15, expected=0.0, target=20.0, stop=5.0),
    )
    plan = policy.evaluate(state, _scalp_result(state, mild), _barriers_for(mild))
    assert plan.current_side_ev_bps is not None
    assert plan.opposite_ev_bps is not None
    assert plan.opposite_ev_bps > plan.current_side_ev_bps
    assert plan.opposite_ev_bps < plan.current_side_ev_bps + 1.5
    assert plan.candidate is Action.HOLD


def test_round_trip_fees_spread_slippage() -> None:
    cost = round_trip_cost_bps(taker_fee_bps=5.0, spread_bps=1.2, slippage_bps=1.0)
    assert cost == pytest.approx(2 * 5.0 + 1.2 + 2 * 1.0)


def test_counterfactual_long_and_short_use_book_and_fees() -> None:
    long_pnl = hypothetical_side_net_pnl_bps(
        side="LONG",
        entry_bid=100.0,
        entry_ask=100.10,
        exit_bid=100.20,
        exit_ask=100.30,
        taker_fee_bps=5.0,
        slippage_bps=1.0,
    )
    short_pnl = hypothetical_side_net_pnl_bps(
        side="SHORT",
        entry_bid=100.0,
        entry_ask=100.10,
        exit_bid=100.20,
        exit_ask=100.30,
        taker_fee_bps=5.0,
        slippage_bps=1.0,
    )
    assert long_pnl is not None
    assert short_pnl is not None
    assert long_pnl < 0
    assert short_pnl < 0


def test_latency_tracker_percentiles() -> None:
    tracker = LatencyTracker()
    for value in range(1, 101):
        tracker.record(float(value))
    stats = tracker.stats()
    assert stats.count == 100
    assert stats.p50_ms == pytest.approx(50.0)
    assert stats.p95_ms == pytest.approx(95.0)
    assert stats.p99_ms == pytest.approx(99.0)


def test_risk_engine_overrides_scalp_open() -> None:
    settings = scalp_settings(kill_switch=True)
    risk = RiskEngine(settings)
    account = fresh_account()
    state = make_market_state()
    result = _scalp_result(
        state,
        (_forecast(30, p_up=0.9, p_down=0.05, p_neither=0.05, expected=5.0, target=8.0, stop=3.0),),
    )
    decided = risk.evaluate(state, Action.OPEN_LONG, account, result, now_ms=1_000_000)
    assert decided.final_action is Action.NO_ACTION
    assert decided.blocked_reason == "kill_switch"
    assert decided.source is ActionSource.RISK_ENGINE


def test_jev_timeout_is_no_action() -> None:
    policy = ScalpPolicy(scalp_settings())
    state = make_market_state()
    result = _scalp_result(state, (), status=JevCallStatus.TIMEOUT)
    plan = policy.evaluate(state, result, ())
    assert plan.candidate is Action.NO_ACTION


def test_stale_websocket_still_forced_by_risk() -> None:
    risk = RiskEngine(scalp_settings())
    account = fresh_account()
    state = make_market_state(side=Side.LONG, age_seconds=40, stale=True)
    result = _scalp_result(state, (), status=JevCallStatus.SKIPPED_STALE)
    decided = risk.evaluate(state, Action.HOLD, account, result, now_ms=1_000_000)
    assert decided.final_action is Action.CLOSE
    assert decided.blocked_reason == "stale_market_data"


def test_paper_scalp_fees_are_not_zero() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    from src.execution.paper import PaperExecutionEngine
    from src.risk.engine import RiskDecision

    engine = PaperExecutionEngine(scalp_settings(scalp_taker_fee_bps=5.0, slippage_bps=1.0))
    risk = RiskDecision(Action.OPEN_LONG, Action.OPEN_LONG, None, ActionSource.JEV)
    report = engine.execute(
        account=account,
        risk=risk,
        symbol=state.symbol,
        exchange=state.exchange,
        bid=100.0,
        ask=100.10,
        now_ms=1,
        decision_id="fee",
        already_processed=False,
    )
    assert report is not None
    assert report.fills[0].fee == pytest.approx(report.fills[0].notional * 0.0005)


def test_score_index_maps_to_signed_bps() -> None:
    assert score_to_expected_return_bps(0) == pytest.approx(-80.0)
    assert score_to_expected_return_bps(4) == pytest.approx(0.0)
    assert score_to_expected_return_bps(8) == pytest.approx(80.0)
    assert score_to_expected_return_bps(4.5) == pytest.approx(2.5)
    barriers = (
        Barrier(
            horizon_seconds=15,
            target_bps=4.2,
            stop_bps=3.1,
            sigma_bps=2.0,
            cost_floor_bps=3.5,
        ),
    )
    questions = scalp_questions(barriers)
    blob = str(questions)
    assert "LONG_TARGET_FIRST" in blob
    assert "LONG_STOP_FIRST" in blob
    assert "LONG_NEITHER" in blob
    assert "SHORT_TARGET_FIRST" in blob
    assert "SHORT_STOP_FIRST" in blob
    assert "SHORT_NEITHER" in blob
    assert "UP_TARGET_FIRST" not in blob
    assert "btc" not in blob.lower()
    assert questions["h15_long_neither_return_bps"]["type"] == "score"
    assert questions["h15_short_neither_return_bps"]["type"] == "score"
    assert isinstance(questions["h15_long_neither_return_bps"]["criteria"], list)


def test_parse_scalp_answers() -> None:
    barriers = (
        Barrier(
            horizon_seconds=15,
            target_bps=5.0,
            stop_bps=3.0,
            sigma_bps=1.0,
            cost_floor_bps=4.0,
        ),
    )
    payload = {
        "model": "typesafe/jev-1.13",
        "answers": {
            "is_noise": {"type": "noul", "noul": 0.2},
            "h15_long_barrier": {
                "type": "choice",
                "choice": "LONG_STOP_FIRST",
                "probabilities": {
                    "LONG_TARGET_FIRST": 0.19,
                    "LONG_STOP_FIRST": 0.67,
                    "LONG_NEITHER": 0.14,
                },
            },
            "h15_short_barrier": {
                "type": "choice",
                "choice": "SHORT_TARGET_FIRST",
                "probabilities": {
                    "SHORT_TARGET_FIRST": 0.55,
                    "SHORT_STOP_FIRST": 0.25,
                    "SHORT_NEITHER": 0.20,
                },
            },
            "h15_long_neither_return_bps": {"type": "score", "score": 3.0},
            "h15_short_neither_return_bps": {"type": "score", "score": 5.0},
        },
    }
    from src.jev.parse import decode_raw

    parsed = decode_raw(payload)
    validate_scalp_answers(parsed, [15])
    decision = parse_scalp_decision(
        parsed, mode=QuestionMode.ENTRY, barriers=barriers, latency_ms=12.0
    )
    assert decision.scalp_forecasts[0].p_long_stop == pytest.approx(0.67)
    assert decision.scalp_forecasts[0].p_short_target == pytest.approx(0.55)
    assert decision.scalp_forecasts[0].long_neither_return_bps == pytest.approx(-5.0)
    assert decision.scalp_forecasts[0].short_neither_return_bps == pytest.approx(5.0)
    assert decision.question_set_version.startswith("q.scalp.v3")


def test_reverse_is_two_fills() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    execute_action(Action.OPEN_LONG, account, state, now_ms=1, decision_id="o")
    later = make_market_state(price=100.0, bid=100.0, ask=100.10, side=Side.LONG)
    report = execute_action(Action.REVERSE, account, later, now_ms=20_000, decision_id="r")
    assert report is not None
    assert len(report.fills) == 2
    assert report.fills[0].intent is Action.CLOSE
    assert report.fills[1].intent is Action.OPEN_SHORT
