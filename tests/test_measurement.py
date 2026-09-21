from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from src.analytics.barriers import (
    AMBIGUOUS,
    DOWN_FIRST,
    LONG_STOP_FIRST,
    NEITHER,
    SHORT_TARGET_FIRST,
    UP_FIRST,
    barrier_first_touch,
)
from src.analytics.calibration import (
    brier_multiclass,
    ev_realized_bins,
    multiclass_log_loss,
    nonoverlapping_indices,
    reliability_bins,
    summarize_horizon,
)
from src.analytics.report import build_report_payload, render_report
from src.config.settings import Settings
from src.decision.action_values import (
    ACTION_VALUE_REFERENCE,
    close_value_from_mid,
    compute_position_action_values,
    hold_value_from_mid,
    reverse_value_from_mid,
)
from src.decision.actions import Action
from src.decision.costs import cost_snapshot
from src.decision.scalp import ScalpPolicy, plan_payload
from src.experiment.fingerprint import config_hash, excluded_secret_fields, research_fingerprint
from src.jev.models import (
    JevCallResult,
    JevCallStatus,
    JevSafetyAction,
    ParsedJevDecision,
    QuestionMode,
)
from src.market.models import Side
from src.storage.database import Database
from src.storage.models import ExperimentRunRow, JevDecisionRow
from src.storage.repository import Repository
from tests.helpers import make_market_state
from tests.test_scalp import _barriers_for, _forecast, _scalp_result, scalp_settings
from tests.test_storage import POSTGRES_SMOKE_URL, _ok_result


def test_action_value_reference_is_mid_incremental() -> None:
    assert ACTION_VALUE_REFERENCE == "MID_INCREMENTAL_BPS_FROM_NOW"


def test_close_does_not_double_count_spread() -> None:
    cost = cost_snapshot(
        taker_fee_bps=5.0,
        maker_fee_bps=2.0,
        spread_bps=2.0,
        slippage_bps=1.0,
        assumed=True,
        source="test",
    )
    close = close_value_from_mid(exit_now_bps=cost.exit_leg_cost_bps)
    one_half = 5.0 + 1.0 + 1.0
    assert close == pytest.approx(-one_half)
    assert close != pytest.approx(-(5.0 + 1.0 + 2.0))
    assert close != pytest.approx(-(5.0 + 1.0 + 1.0 + 1.0))


def test_hold_excludes_sunk_entry_costs() -> None:
    hold = hold_value_from_mid(expected_payoff_bps=8.0, expected_future_exit_bps=6.5)
    assert hold == pytest.approx(1.5)
    assert hold != pytest.approx(8.0 - 13.0)


def test_reverse_prices_three_incremental_legs() -> None:
    value = reverse_value_from_mid(
        opposite_payoff_bps=20.0,
        exit_now_bps=6.5,
        opposite_entry_bps=6.5,
        opposite_future_exit_bps=6.5,
    )
    assert value == pytest.approx(20.0 - 19.5)


def test_hold_not_defined_as_reentry() -> None:
    settings = scalp_settings(
        scalp_taker_fee_bps=5.0,
        slippage_bps=1.0,
        scalp_min_probability_edge=0.0,
        scalp_min_net_ev_bps=1.0,
    )
    policy = ScalpPolicy(settings)
    state = make_market_state(side=Side.SHORT, age_seconds=40, spread_bps=0.0)
    forecasts = (
        _forecast(30, p_up=0.10, p_down=0.80, p_neither=0.10, expected=0.0, target=10.0, stop=3.0),
    )
    plan = policy.evaluate(state, _scalp_result(state, forecasts), _barriers_for(forecasts))
    assert plan.entry_net_ev_if_flat_bps is not None
    assert plan.entry_net_ev_if_flat_bps < 0
    assert plan.hold_net_ev_bps is not None
    assert plan.hold_net_ev_bps > plan.close_now_value_bps
    assert plan.candidate is Action.HOLD
    payload = plan_payload(plan, latency={}, reference_price=state.price, timestamp=state.timestamp)
    assert payload["hold_value_bps"] == plan.hold_net_ev_bps
    assert payload["close_value_bps"] == plan.close_now_value_bps
    assert payload["reverse_value_bps"] == plan.reverse_net_ev_bps
    assert payload["chosen_action"] == "HOLD"
    assert payload["predictions"]


def test_compute_position_action_values_common_unit() -> None:
    cost = cost_snapshot(
        taker_fee_bps=5.0,
        maker_fee_bps=2.0,
        spread_bps=0.0,
        slippage_bps=1.0,
        assumed=True,
        source="test",
    )
    settings = scalp_settings(
        scalp_taker_fee_bps=5.0, slippage_bps=1.0, scalp_min_probability_edge=0.0
    )
    policy = ScalpPolicy(settings)
    state = make_market_state(side=Side.LONG, age_seconds=40, spread_bps=0.0)
    forecasts = (
        _forecast(30, p_up=0.70, p_down=0.10, p_neither=0.20, expected=0.0, target=12.0, stop=4.0),
    )
    plan = policy.evaluate(state, _scalp_result(state, forecasts), _barriers_for(forecasts))
    hold, close, reverse, entry, _g, _r = compute_position_action_values(
        current_side="LONG",
        cost=cost,
        hold_opportunities=plan.hold_opportunities,
        entry_opportunities=plan.opportunities,
    )
    assert hold == pytest.approx(plan.hold_net_ev_bps)
    assert close == pytest.approx(plan.close_now_value_bps)
    assert reverse == pytest.approx(plan.reverse_net_ev_bps)
    assert entry == pytest.approx(plan.entry_net_ev_if_flat_bps)


def test_barrier_up_first() -> None:
    outcome = barrier_first_touch(
        reference_mid=100.0,
        path=((1_000, 100.0), (2_000, 100.20)),
        target_bps=10.0,
        start_ms=1_000,
        horizon_ms=5_000,
    )
    assert outcome == UP_FIRST


def test_barrier_down_first() -> None:
    outcome = barrier_first_touch(
        reference_mid=100.0,
        path=((1_000, 100.0), (2_000, 99.80)),
        target_bps=10.0,
        start_ms=1_000,
        horizon_ms=5_000,
    )
    assert outcome == DOWN_FIRST


def test_barrier_neither() -> None:
    outcome = barrier_first_touch(
        reference_mid=100.0,
        path=((1_000, 100.0), (2_000, 100.01), (6_000, 100.02)),
        target_bps=10.0,
        start_ms=1_000,
        horizon_ms=5_000,
    )
    assert outcome == NEITHER


def test_barrier_ambiguous_same_interval() -> None:
    outcome = barrier_first_touch(
        reference_mid=100.0,
        path=((1_000, 100.12), (2_000, 99.88)),
        target_bps=10.0,
        start_ms=1_000,
        horizon_ms=5_000,
    )
    assert outcome == AMBIGUOUS


def test_brier_and_log_loss() -> None:
    predicted = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    outcomes = [UP_FIRST, DOWN_FIRST, NEITHER]
    assert brier_multiclass(predicted, outcomes) == pytest.approx(0.0)
    assert multiclass_log_loss(predicted, outcomes) == pytest.approx(0.0)
    messy = ((0.6, 0.3, 0.1),)
    assert brier_multiclass(messy, [DOWN_FIRST]) == pytest.approx(0.36 + 0.49 + 0.01)


def test_calibration_and_ev_bins() -> None:
    bins = reliability_bins([0.65, 0.66, 0.20], [True, False, False])
    mid = next(item for item in bins if item["lo"] == pytest.approx(0.6))
    assert mid["n"] == 2
    ev = ev_realized_bins([0.5, 1.5, 3.0, 5.0, -1.0], [0.2, 1.1, 2.0, 4.0, -2.0])
    labels = [row["bin"] for row in ev]
    assert labels == ["lt_0", "0_1", "1_2", "2_4", "4_plus"]
    assert ev[2]["n"] == 1
    summary = summarize_horizon(
        [
            {
                "p_up_first": 0.2,
                "p_down_first": 0.7,
                "p_neither": 0.1,
                "outcome": DOWN_FIRST,
                "predicted_entry_net_ev": 2.0,
                "realized_selected_net_pnl": 1.5,
            }
        ]
        * 5,
        min_n=20,
    )
    assert summary["insufficient"] is True
    spaced = nonoverlapping_indices([0, 10_000, 70_000], min_gap_ms=60_000)
    assert spaced == [0, 2]


def test_fingerprint_stable_and_excludes_secrets() -> None:
    settings = Settings(
        openrouter_api_key="sk-secret",
        binance_api_secret="secret",
        database_url="postgresql://user:password@localhost/db",
        experiment_id="scalp-v2-calibration",
    )
    first = research_fingerprint(settings, resolved_model=None)
    second = research_fingerprint(settings, resolved_model="typesafe/jev-1.13-20260917")
    assert config_hash(first) == config_hash(second)
    blob = json.dumps(first)
    assert settings.openrouter_api_key not in blob
    assert "postgresql://user:password" not in blob
    assert "sk-secret" not in blob
    excluded = excluded_secret_fields(settings)
    assert "openrouter_api_key" in excluded
    assert "binance_api_secret" in excluded
    assert "database_url" in excluded


@pytest.mark.asyncio
async def test_run_and_experiment_attribution(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'run.db'}")
    await db.create_all()
    repo = Repository(db)
    repo.bind_run(run_id="run-1", experiment_id="scalp-v2-calibration", config_hash="abc")
    await repo.persist_experiment_run(
        run_id="run-1",
        experiment_id="scalp-v2-calibration",
        config_hash="abc",
        frozen=True,
        started_at_ms=1_700_000_000_000,
        fingerprint={"experiment_id": "scalp-v2-calibration"},
    )
    state = make_market_state()
    decision_id = await repo.persist_tick(state, _ok_result())
    await repo.record_system_event(event_type="ENGINE_START")
    async with db.session() as session:
        decision = await session.get(JevDecisionRow, decision_id)
        run = await session.get(ExperimentRunRow, "run-1")
    assert decision is not None
    assert decision.run_id == "run-1"
    assert decision.experiment_id == "scalp-v2-calibration"
    assert decision.config_hash == "abc"
    assert run is not None
    assert run.frozen is True
    latest = await repo.latest_experiment_run()
    assert latest is not None
    assert latest.run_id == "run-1"
    await db.aclose()


@pytest.mark.asyncio
async def test_barrier_outcome_enrichment_from_path(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'path.db'}")
    await db.create_all()
    repo = Repository(db)
    t0 = 1_700_000_000_000
    stamp = datetime.fromtimestamp(t0 / 1000, tz=UTC).isoformat()
    state = make_market_state(symbol="BTCUSDT", price=100.0).model_copy(
        update={"timestamp_ms": t0, "timestamp": stamp}
    )
    result = JevCallResult(
        status=JevCallStatus.OK,
        symbol="BTCUSDT",
        exchange="binance_usdm",
        mode=QuestionMode.ENTRY,
        question_set_version="q.scalp.v2",
        latency_ms=10.0,
        decision=ParsedJevDecision(
            mode=QuestionMode.ENTRY,
            question_set_version="q.scalp.v2",
            model="typesafe/jev-1.13-20260917",
            latency_ms=10.0,
            scalp_forecasts=(),
        ),
        safety_action=JevSafetyAction.NONE,
    )
    payload = {
        "predictions": [
            {
                "horizon_seconds": 15,
                "reference_price": 100.0,
                "target_bps": 10.0,
                "stop_bps": 3.0,
                "p_up_first": 0.2,
                "p_down_first": 0.7,
                "p_neither": 0.1,
                "p_long_target": 0.2,
                "p_long_stop": 0.7,
                "p_long_neither": 0.1,
                "p_short_target": 0.7,
                "p_short_stop": 0.2,
                "p_short_neither": 0.1,
                "predicted_entry_net_ev": 1.5,
                "predicted_entry_net_side": "SHORT",
            }
        ],
        "cost": {"taker_fee_bps": 5.0, "slippage_bps": 1.0},
    }
    decision_id = await repo.persist_tick(state, result, scalp_payload=payload)
    later = state.model_copy(
        update={
            "price": 99.80,
            "best_bid": 99.79,
            "best_ask": 99.81,
            "timestamp_ms": t0 + 5_000,
            "timestamp": datetime.fromtimestamp((t0 + 5_000) / 1000, tz=UTC).isoformat(),
        }
    )
    await repo.persist_tick(later, result, scalp_payload=payload)
    updated = await repo.enrich_outcomes(now_ms=t0 + 20_000)
    assert updated >= 1
    async with db.session() as session:
        row = await session.get(JevDecisionRow, decision_id)
    assert row is not None
    outcomes = (row.scalp_payload or {}).get("barrier_outcomes")
    assert outcomes["15s"]["long"]["outcome"] == LONG_STOP_FIRST
    assert outcomes["15s"]["short"]["outcome"] == SHORT_TARGET_FIRST
    await db.aclose()


@pytest.mark.asyncio
async def test_session_report(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'report.db'}")
    await db.create_all()
    repo = Repository(db)
    await repo.persist_experiment_run(
        run_id="run-report",
        experiment_id="scalp-v2-calibration",
        config_hash="hash",
        frozen=True,
        started_at_ms=1_700_000_000_000,
        fingerprint={"paper_equity": 10_000, "paper_position_notional": 1_000},
    )
    run = await repo.get_experiment_run("run-report")
    assert run is not None
    payload = build_report_payload(run=run, decisions=[], trades=[], events=[])
    text = render_report(payload)
    assert "run-report" in text
    assert "scalp-v2-calibration" in text
    assert "Brier" in text or "brier" in text.lower() or "CALIBRATION" in text
    await db.aclose()


@pytest.mark.asyncio
async def test_postgres_run_attribution() -> None:
    db = Database(POSTGRES_SMOKE_URL)
    try:
        await db.drop_all()
        await db.create_all()
    except Exception as exc:
        await db.aclose()
        pytest.skip(f"Postgres in OrbStack is not reachable: {exc}")
    repo = Repository(db)
    repo.bind_run(run_id="pg-run", experiment_id="scalp-v2-calibration", config_hash="pghash")
    state = make_market_state(symbol="BTCUSDT", price=86_000.0)
    decision_id = await repo.persist_tick(state, _ok_result("BTCUSDT"))
    async with db.session() as session:
        decision = await session.get(JevDecisionRow, decision_id)
    assert decision is not None
    assert decision.run_id == "pg-run"
    await db.drop_all()
    await db.aclose()
