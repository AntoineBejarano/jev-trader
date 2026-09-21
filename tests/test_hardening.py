from __future__ import annotations

from types import SimpleNamespace

import pytest
from src.analytics.session import scalp_session_metrics
from src.api.snapshot import _jev_view
from src.config.settings import TradingStrategy
from src.decision.actions import Action, ActionSource
from src.decision.barriers import BarrierPlanner
from src.decision.costs import cost_snapshot, one_way_cost_bps, round_trip_cost_bps
from src.decision.ev import (
    COST_KIND_ENTRY,
    COST_KIND_HOLD,
    NEITHER_JEV_CONDITIONAL,
    NEITHER_ZERO_CONSERVATIVE,
    long_gross_ev_bps,
    opportunities_for_forecast,
    short_gross_ev_bps,
)
from src.decision.scalp import ScalpPolicy, plan_payload
from src.execution.attribution import (
    RECONCILE_TOLERANCE_USDT,
    attribute_round_trip,
    pnl_on_equity_bps,
    pnl_on_equity_pct,
    pnl_on_notional_bps,
    profit_giveback_bps,
)
from src.execution.paper import PaperExecutionEngine
from src.jev.models import (
    JevCallResult,
    JevCallStatus,
    JevSafetyAction,
    ParsedJevDecision,
    QuestionMode,
)
from src.jev.scalp_questions import SCORE_MAPPING_VERSION
from src.market.models import Side
from src.risk.engine import RiskDecision
from tests.helpers import make_market_state
from tests.paper_helpers import fresh_account
from tests.test_scalp import _barriers_for, _forecast, _scalp_result, scalp_settings


def test_entry_ev_uses_full_lifecycle_costs() -> None:
    cost = cost_snapshot(
        taker_fee_bps=5.0,
        maker_fee_bps=2.0,
        spread_bps=2.0,
        slippage_bps=1.0,
        assumed=True,
        source="test",
    )
    assert cost.round_trip_cost_bps == pytest.approx(2 * 5.0 + 2.0 + 2 * 1.0)
    forecast = _forecast(
        30, p_up=0.70, p_down=0.10, p_neither=0.20, expected=0.0, target=12.0, stop=4.0
    )
    long_opp, _short = opportunities_for_forecast(
        forecast, cost_bps=cost.round_trip_cost_bps, cost_kind=COST_KIND_ENTRY
    )
    gross = long_gross_ev_bps(
        p_up=0.70,
        p_down=0.10,
        p_neither=0.20,
        target_bps=12.0,
        stop_bps=4.0,
        expected_return_bps=0.0,
    )
    assert long_opp.cost_kind == COST_KIND_ENTRY
    assert long_opp.net_ev_bps == pytest.approx(gross - cost.round_trip_cost_bps)


def test_hold_ev_does_not_recharge_sunk_entry_costs() -> None:
    settings = scalp_settings(
        scalp_taker_fee_bps=5.0,
        slippage_bps=1.0,
        scalp_min_net_ev_bps=1.0,
        scalp_min_probability_edge=0.0,
    )
    policy = ScalpPolicy(settings)
    state = make_market_state(side=Side.LONG, age_seconds=30, spread_bps=0.0)
    forecasts = (
        _forecast(30, p_up=1.0, p_down=0.0, p_neither=0.0, expected=0.0, target=10.0, stop=4.0),
    )
    plan = policy.evaluate(state, _scalp_result(state, forecasts), _barriers_for(forecasts))
    entry_cost = 2 * 5.0 + 2 * 1.0
    exit_cost = 5.0 + 1.0
    assert plan.best is not None
    assert plan.best.net_ev_bps == pytest.approx(10.0 - entry_cost)
    assert plan.hold_net_ev_bps == pytest.approx(10.0 - exit_cost)
    assert plan.close_now_value_bps == pytest.approx(-exit_cost)
    assert plan.hold_advantage_bps == pytest.approx(10.0)
    assert plan.candidate is Action.HOLD


def test_close_now_calculation() -> None:
    cost = cost_snapshot(
        taker_fee_bps=5.0,
        maker_fee_bps=2.0,
        spread_bps=1.0,
        slippage_bps=1.0,
        assumed=True,
        source="test",
    )
    assert cost.exit_leg_cost_bps == pytest.approx(
        one_way_cost_bps(taker_fee_bps=5.0, slippage_bps=1.0, half_spread_bps=0.5)
    )
    settings = scalp_settings(scalp_taker_fee_bps=5.0, slippage_bps=1.0)
    policy = ScalpPolicy(settings)
    state = make_market_state(side=Side.SHORT, age_seconds=30, spread_bps=1.0)
    forecasts = (
        _forecast(15, p_up=0.5, p_down=0.5, p_neither=0.0, expected=0.0, target=2.0, stop=2.0),
    )
    plan = policy.evaluate(state, _scalp_result(state, forecasts), _barriers_for(forecasts))
    assert plan.close_now_value_bps == pytest.approx(-cost.exit_leg_cost_bps)


def test_reverse_ev_prices_three_legs() -> None:
    settings = scalp_settings(
        scalp_taker_fee_bps=5.0,
        slippage_bps=1.0,
        scalp_min_net_ev_bps=1.0,
        scalp_reverse_advantage_bps=1.5,
        scalp_min_probability_edge=0.0,
    )
    policy = ScalpPolicy(settings)
    state = make_market_state(side=Side.LONG, age_seconds=30, spread_bps=0.0)
    forecasts = (
        _forecast(30, p_up=0.05, p_down=0.90, p_neither=0.05, expected=0.0, target=20.0, stop=2.0),
    )
    plan = policy.evaluate(state, _scalp_result(state, forecasts), _barriers_for(forecasts))
    exit_leg = 5.0 + 1.0
    short_gross = short_gross_ev_bps(
        p_up=0.05,
        p_down=0.90,
        p_neither=0.05,
        target_bps=20.0,
        stop_bps=2.0,
        expected_return_bps=0.0,
    )
    assert plan.reverse_net_ev_bps == pytest.approx(short_gross - 3 * exit_leg)
    zero_settings = scalp_settings(
        scalp_taker_fee_bps=0.0,
        slippage_bps=0.0,
        scalp_min_net_ev_bps=1.0,
        scalp_reverse_advantage_bps=1.5,
        scalp_min_probability_edge=0.0,
    )
    zero_plan = ScalpPolicy(zero_settings).evaluate(
        state, _scalp_result(state, forecasts), _barriers_for(forecasts)
    )
    assert zero_plan.candidate is Action.REVERSE


def test_conditional_neither_and_zero_fallback() -> None:
    kwargs = dict(
        p_up=0.40,
        p_down=0.20,
        p_neither=0.40,
        target_bps=10.0,
        stop_bps=5.0,
        expected_return_bps=8.0,
    )
    conditional = long_gross_ev_bps(**kwargs, neither_payoff_method=NEITHER_JEV_CONDITIONAL)
    zeroed = long_gross_ev_bps(**kwargs, neither_payoff_method=NEITHER_ZERO_CONSERVATIVE)
    assert conditional == pytest.approx(0.40 * 10 - 0.20 * 5 + 0.40 * 8)
    assert zeroed == pytest.approx(0.40 * 10 - 0.20 * 5)
    short_c = short_gross_ev_bps(**kwargs, neither_payoff_method=NEITHER_JEV_CONDITIONAL)
    short_z = short_gross_ev_bps(**kwargs, neither_payoff_method=NEITHER_ZERO_CONSERVATIVE)
    assert short_c == pytest.approx(0.20 * 10 - 0.40 * 5 - 0.40 * 8)
    assert short_z == pytest.approx(0.20 * 10 - 0.40 * 5)


def test_long_short_ev_unit() -> None:
    long_ev = long_gross_ev_bps(
        p_up=0.6, p_down=0.3, p_neither=0.1, target_bps=12.0, stop_bps=6.0, expected_return_bps=2.0
    )
    short_ev = short_gross_ev_bps(
        p_up=0.6, p_down=0.3, p_neither=0.1, target_bps=12.0, stop_bps=6.0, expected_return_bps=2.0
    )
    assert long_ev == pytest.approx(0.6 * 12 - 0.3 * 6 + 0.1 * 2)
    assert short_ev == pytest.approx(0.3 * 12 - 0.6 * 6 - 0.1 * 2)


def test_pnl_bps_on_notional_and_equity() -> None:
    assert pnl_on_notional_bps(-1.20, 1000.0) == pytest.approx(-12.0)
    assert pnl_on_equity_bps(-1.20, 10000.0) == pytest.approx(-1.2)
    assert pnl_on_equity_pct(-1.20, 10000.0) == pytest.approx(-0.00012)
    metrics = scalp_session_metrics(
        [],
        runtime_seconds=10,
        max_drawdown=0.0,
        starting_balance=10_000.0,
        position_notional=1_000.0,
    )
    assert metrics["session_return_bps_on_starting_equity"] == pytest.approx(0.0)
    assert metrics["session_net_pnl_usdt"] == pytest.approx(0.0)


def test_slippage_spread_attribution_reconciles() -> None:
    qty = 1000.0 / 100.05
    attr = attribute_round_trip(
        side=Side.LONG,
        quantity=qty,
        entry_fill=100.05 * 1.0001,
        entry_bid=100.0,
        entry_ask=100.05,
        exit_fill=100.20 * 0.9999,
        exit_bid=100.20,
        exit_ask=100.25,
        entry_fee=0.5,
        exit_fee=0.5,
    )
    assert attr.entry_slippage_cost == pytest.approx(qty * (100.05 * 1.0001 - 100.05))
    assert attr.exit_slippage_cost == pytest.approx(qty * (100.20 - 100.20 * 0.9999))
    assert abs(attr.residual) <= RECONCILE_TOLERANCE_USDT
    assert attr.net_pnl == pytest.approx(attr.reconstructed_net_pnl, abs=1e-8)


def test_paper_round_trip_two_sided_slippage() -> None:
    account = fresh_account()
    engine = PaperExecutionEngine(scalp_settings(scalp_taker_fee_bps=5.0, slippage_bps=1.0))

    opened = engine.execute(
        account=account,
        risk=RiskDecision(Action.OPEN_SHORT, Action.OPEN_SHORT, None, ActionSource.JEV),
        symbol="ETHUSDT",
        exchange="binance_usdm",
        bid=100.0,
        ask=100.10,
        now_ms=1,
        decision_id="in",
        already_processed=False,
    )
    assert opened is not None
    closed = engine.execute(
        account=account,
        risk=RiskDecision(Action.CLOSE, Action.CLOSE, None, ActionSource.JEV),
        symbol="ETHUSDT",
        exchange="binance_usdm",
        bid=100.0,
        ask=100.10,
        now_ms=10_000,
        decision_id="out",
        already_processed=False,
    )
    assert closed is not None
    trade = closed.closed
    assert trade is not None
    assert trade.entry_slippage_cost == pytest.approx(opened.fills[0].slippage_cost)
    assert trade.exit_slippage_cost == pytest.approx(closed.fills[0].slippage_cost)
    assert trade.total_slippage_cost == pytest.approx(
        trade.entry_slippage_cost + trade.exit_slippage_cost
    )
    assert abs(trade.attribution_residual) < 1e-8
    recon = (
        trade.market_move_pnl
        - trade.total_spread_cost
        - trade.total_slippage_cost
        - trade.total_fees
        - trade.funding_cost
    )
    assert trade.net_pnl == pytest.approx(recon, abs=1e-8)


def test_profit_giveback_feature() -> None:
    assert profit_giveback_bps(12.0, 2.0) == pytest.approx(10.0)
    assert profit_giveback_bps(12.0, 11.0) == pytest.approx(1.0)
    assert profit_giveback_bps(-3.0, -4.0) == pytest.approx(0.0)


def test_cost_constrained_horizon_flag() -> None:
    settings = scalp_settings(
        scalp_min_target_bps=1.5,
        scalp_profit_buffer_bps=0.5,
        scalp_taker_fee_bps=5.0,
        slippage_bps=1.0,
    )
    state = make_market_state(spread_bps=2.0, rv_1m=0.0)
    planned = BarrierPlanner(settings).plan(state)
    assert planned
    for barrier in planned:
        assert barrier.target_bound_by_cost_floor is True
        assert barrier.horizon_cost_constrained is True
        assert barrier.volatility_target_bps <= barrier.cost_floor_bps


def test_hold_not_forced_by_negative_pnl() -> None:
    settings = scalp_settings(
        scalp_taker_fee_bps=0.0,
        slippage_bps=0.0,
        scalp_min_probability_edge=0.0,
    )
    policy = ScalpPolicy(settings)
    state = make_market_state(side=Side.SHORT, age_seconds=40, spread_bps=0.0)
    state = state.model_copy(
        update={
            "position": state.position.model_copy(
                update={"unrealized_pnl_bps": -8.0, "max_adverse_excursion_bps": -8.0}
            )
        }
    )
    forecasts = (
        _forecast(30, p_up=0.05, p_down=0.85, p_neither=0.10, expected=0.0, target=10.0, stop=3.0),
    )
    plan = policy.evaluate(state, _scalp_result(state, forecasts), _barriers_for(forecasts))
    assert plan.candidate is Action.HOLD


def test_configured_vs_resolved_jev_model_and_scalp_dashboard() -> None:
    settings = SimpleNamespace(
        jev_model="typesafe/jev-1.13",
        trading_strategy=TradingStrategy.SCALP,
    )
    state = make_market_state()
    forecasts = (
        _forecast(15, p_up=0.2, p_down=0.7, p_neither=0.1, expected=-5.0, target=8.0, stop=3.0),
    )
    result = JevCallResult(
        status=JevCallStatus.OK,
        symbol=state.symbol,
        exchange=state.exchange,
        mode=QuestionMode.ENTRY,
        question_set_version="q.scalp.v2.h15",
        latency_ms=12.0,
        decision=ParsedJevDecision(
            mode=QuestionMode.ENTRY,
            question_set_version="q.scalp.v2.h15",
            model="typesafe/jev-1.13-20260917",
            latency_ms=12.0,
            noise_probability=0.1,
            scalp_forecasts=forecasts,
        ),
        safety_action=JevSafetyAction.NONE,
    )
    plan = ScalpPolicy(scalp_settings()).evaluate(state, result, _barriers_for(forecasts))
    payload = plan_payload(plan, latency={"jev_latency_ms": 12.0})
    engine = SimpleNamespace(
        settings=settings,
        last_result=result,
        last_risk=SimpleNamespace(
            candidate_action=Action.OPEN_SHORT,
            final_action=Action.OPEN_SHORT,
            blocked_reason=None,
        ),
        last_plan=plan,
    )
    view = _jev_view(engine, state)  # type: ignore[arg-type]
    assert view is not None
    assert view["kind"] == "SCALP"
    assert "long_probability" not in view
    assert view["configured_model"] == "typesafe/jev-1.13"
    assert view["resolved_response_model"] == "typesafe/jev-1.13-20260917"
    assert view["horizons"]
    assert payload["score_mapping_version"] == SCORE_MAPPING_VERSION
    assert payload["hold_opportunities"]
    hold_kinds = {item["cost_kind"] for item in payload["hold_opportunities"]}
    assert hold_kinds == {COST_KIND_HOLD}


def test_round_trip_still_matches_two_one_way_legs() -> None:
    rt = round_trip_cost_bps(taker_fee_bps=5.0, spread_bps=1.2, slippage_bps=1.0)
    assert rt == pytest.approx(2 * 5.0 + 1.2 + 2 * 1.0)
