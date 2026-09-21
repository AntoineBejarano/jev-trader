from __future__ import annotations

from types import SimpleNamespace

from src.analytics.barriers import (
    LONG_AMBIGUOUS,
    LONG_STOP_FIRST,
    LONG_TARGET_FIRST,
    SHORT_NEITHER,
    SHORT_STOP_FIRST,
    SHORT_TARGET_FIRST,
    long_barrier_outcome,
    short_barrier_outcome,
)
from src.analytics.calibration import summarize_long_race, summarize_short_race
from src.analytics.research_sample import research_sample_status
from src.analytics.session import scalp_session_metrics
from src.api.snapshot import _account_view
from src.decision.ev import (
    long_gross_ev_bps,
    normalize_barrier_probs,
    opportunities_for_forecast,
    short_gross_ev_bps,
)
from src.experiment.fingerprint import QUESTION_VERSION, STRATEGY_VERSION
from src.features.engine import assess_health
from src.jev.models import JevCallResult, JevCallStatus, QuestionMode, ScalpHorizonForecast
from src.jev.scalp_questions import SCALP_QUESTION_SET_FAMILY
from src.market.buffer import MarketBuffer
from src.market.history import klines_to_mid_samples, seed_buffer_from_klines
from src.market.models import PRIMARY_WARMUP_MS
from src.market.orderbook import LocalOrderBook
from src.portfolio.account import PaperAccount
from src.storage.repository import summarize_run_calibration
from tests.helpers import make_market_state


def test_question_family_is_v3() -> None:
    assert SCALP_QUESTION_SET_FAMILY == "q.scalp.v3"
    assert QUESTION_VERSION == "q.scalp.v3"
    assert STRATEGY_VERSION == "scalp.action_values.v2"


def test_asymmetric_long_barrier_outcome() -> None:
    path = ((1_000, 100.0), (1_400, 100.12), (1_800, 99.90))
    outcome = long_barrier_outcome(
        reference_mid=100.0,
        path=path,
        target_bps=10.0,
        stop_bps=3.0,
        start_ms=1_000,
        horizon_ms=5_000,
    )
    assert outcome == LONG_TARGET_FIRST


def test_asymmetric_short_barrier_outcome() -> None:
    path = ((1_000, 100.0), (1_400, 99.88), (1_800, 100.10))
    outcome = short_barrier_outcome(
        reference_mid=100.0,
        path=path,
        target_bps=10.0,
        stop_bps=3.0,
        start_ms=1_000,
        horizon_ms=5_000,
    )
    assert outcome == SHORT_TARGET_FIRST


def test_long_stop_is_not_short_target_when_barriers_differ() -> None:
    path = ((1_000, 100.0), (1_500, 99.96))
    long_out = long_barrier_outcome(
        reference_mid=100.0,
        path=path,
        target_bps=12.0,
        stop_bps=4.0,
        start_ms=1_000,
        horizon_ms=5_000,
    )
    short_out = short_barrier_outcome(
        reference_mid=100.0,
        path=path,
        target_bps=12.0,
        stop_bps=4.0,
        start_ms=1_000,
        horizon_ms=5_000,
    )
    assert long_out == LONG_STOP_FIRST
    assert short_out == SHORT_NEITHER


def test_long_probability_normalization() -> None:
    target, stop, neither = normalize_barrier_probs(0.2, 0.2, 0.1)
    assert target + stop + neither == 1.0
    assert target == 0.4
    assert stop == 0.4
    assert neither == 0.2


def test_short_probability_normalization() -> None:
    target, stop, neither = normalize_barrier_probs(0.0, 0.0, 0.0)
    assert (target, stop, neither) == (0.0, 0.0, 1.0)


def test_long_ev_formula() -> None:
    value = long_gross_ev_bps(
        p_long_target=0.50,
        p_long_stop=0.20,
        p_long_neither=0.30,
        target_bps=10.0,
        stop_bps=4.0,
        expected_return_bps=2.0,
    )
    assert value == 0.50 * 10.0 - 0.20 * 4.0 + 0.30 * 2.0


def test_short_ev_formula() -> None:
    value = short_gross_ev_bps(
        p_short_target=0.40,
        p_short_stop=0.25,
        p_short_neither=0.35,
        target_bps=10.0,
        stop_bps=4.0,
        expected_return_bps=2.0,
    )
    assert value == 0.40 * 10.0 - 0.25 * 4.0 - 0.35 * 2.0


def test_asymmetric_races_are_not_inferred() -> None:
    forecast = ScalpHorizonForecast(
        horizon_seconds=30,
        p_long_target=0.70,
        p_long_stop=0.10,
        p_long_neither=0.20,
        p_short_target=0.15,
        p_short_stop=0.55,
        p_short_neither=0.30,
        long_neither_return_bps=1.0,
        short_neither_return_bps=-3.0,
        target_bps=12.0,
        stop_bps=4.0,
    )
    long_opp, short_opp = opportunities_for_forecast(forecast, cost_bps=0.0)
    expected_long = 0.70 * 12.0 - 0.10 * 4.0 + 0.20 * 1.0
    expected_short = 0.15 * 12.0 - 0.55 * 4.0 - 0.30 * (-3.0)
    assert long_opp.gross_ev_bps == expected_long
    assert short_opp.gross_ev_bps == expected_short
    assert long_opp.p_target == 0.70
    assert short_opp.p_target == 0.15


def test_side_specific_calibration() -> None:
    long_rows = [
        {
            "p_long_target": 0.8,
            "p_long_stop": 0.1,
            "p_long_neither": 0.1,
            "outcome": LONG_TARGET_FIRST,
            "predicted_entry_net_long_bps": 2.0,
            "realized_long_net_pnl": 1.5,
        }
    ] * 5
    short_rows = [
        {
            "p_short_target": 0.2,
            "p_short_stop": 0.7,
            "p_short_neither": 0.1,
            "outcome": SHORT_STOP_FIRST,
            "predicted_entry_net_short_bps": -1.0,
            "realized_short_net_pnl": -0.5,
        }
    ] * 5
    long_summary = summarize_long_race(long_rows, min_n=20)
    short_summary = summarize_short_race(short_rows, min_n=20)
    assert long_summary["target"]["observed_frequency"] == 1.0
    assert short_summary["stop"]["observed_frequency"] == 1.0
    assert long_summary["insufficient"] is True
    assert "up" in long_summary


def test_warmup_sample_exclusion() -> None:
    state = make_market_state(warmup_ready=False, trading_allowed=False)
    result = JevCallResult(
        status=JevCallStatus.OK,
        symbol=state.symbol,
        exchange=state.exchange,
        mode=QuestionMode.ENTRY,
        question_set_version="q.scalp.v3",
        latency_ms=1.0,
    )
    valid, reason = research_sample_status(state, result)
    assert valid is False
    assert reason == "warmup_incomplete"
    row = SimpleNamespace(
        timestamp_ms=1,
        scalp_payload={
            "research_sample_valid": False,
            "invalid_reason": "warmup_incomplete",
            "predictions": [
                {
                    "horizon_seconds": 15,
                    "p_long_target": 0.5,
                    "p_long_stop": 0.3,
                    "p_long_neither": 0.2,
                    "p_short_target": 0.4,
                    "p_short_stop": 0.4,
                    "p_short_neither": 0.2,
                    "predicted_entry_net_long_bps": 1.0,
                    "predicted_entry_net_short_bps": 0.2,
                }
            ],
            "barrier_outcomes": {
                "15s": {
                    "long": {"outcome": LONG_TARGET_FIRST},
                    "short": {"outcome": SHORT_NEITHER},
                }
            },
        },
    )
    summary = summarize_run_calibration([row])  # type: ignore[arg-type]
    assert summary["n_raw_predictions"] == 1
    assert summary["n_valid_research"] == 0
    assert summary["horizons"] == {}


def test_valid_research_sample_inclusion() -> None:
    state = make_market_state(warmup_ready=True)
    result = JevCallResult(
        status=JevCallStatus.OK,
        symbol=state.symbol,
        exchange=state.exchange,
        mode=QuestionMode.ENTRY,
        question_set_version="q.scalp.v3",
        latency_ms=1.0,
    )
    valid, reason = research_sample_status(state, result)
    assert valid is True
    assert reason is None
    row = SimpleNamespace(
        timestamp_ms=1,
        scalp_payload={
            "research_sample_valid": True,
            "invalid_reason": None,
            "predictions": [
                {
                    "horizon_seconds": 15,
                    "p_long_target": 0.5,
                    "p_long_stop": 0.3,
                    "p_long_neither": 0.2,
                    "p_short_target": 0.4,
                    "p_short_stop": 0.4,
                    "p_short_neither": 0.2,
                    "predicted_entry_net_long_bps": 1.0,
                    "predicted_entry_net_short_bps": 0.2,
                }
            ],
            "barrier_outcomes": {
                "15s": {
                    "long": {"outcome": LONG_TARGET_FIRST},
                    "short": {"outcome": SHORT_NEITHER},
                }
            },
        },
    )
    summary = summarize_run_calibration([row])  # type: ignore[arg-type]
    assert summary["n_raw_predictions"] == 1
    assert summary["n_valid_research"] == 1
    assert summary["horizons"]["15s"]["long"]["n"] == 1
    assert summary["horizons"]["15s"]["short"]["n"] == 1


def test_realized_vs_mtm_metric_labeling() -> None:
    realized = scalp_session_metrics(
        [],
        runtime_seconds=60.0,
        max_drawdown=0.0,
        starting_balance=10_000.0,
        position_notional=1_000.0,
    )
    assert realized["metrics_kind"] == "REALIZED_TRADE_METRICS"
    assert realized["includes_open_position_mtm"] is False
    engine = SimpleNamespace(
        account=PaperAccount.new(starting_balance=10_000.0, currency="USDT", now_ms=1),
        settings=SimpleNamespace(execution_notional=1_000.0),
    )
    account = _account_view(engine, now_ms=1)  # type: ignore[arg-type]
    assert account["metrics_kind"] == "MARK_TO_MARKET_ACCOUNT_METRICS"
    assert account["includes_open_position_mtm"] is True


def test_seeded_klines_do_not_satisfy_warmup() -> None:
    buffer = MarketBuffer(symbol="BTCUSDT", exchange="binance_usdm", history_ms=3_600_000)
    klines = [
        [1_000, "100", "101", "99", "100.5", "1", 61_000],
        [61_000, "100.5", "102", "100", "101", "1", 121_000],
    ]
    samples = klines_to_mid_samples(klines)
    assert samples[0] == (1_000, 100.0)
    assert samples[-1] == (121_000, 101.0)
    seeded = seed_buffer_from_klines(buffer, klines)
    assert seeded == 3
    assert all(sample.seeded for sample in buffer.mids)
    book = LocalOrderBook()
    book.apply_partial_snapshot(
        bids=[(99.5, 1.0)],
        asks=[(100.5, 1.0)],
        last_update_id=1,
        event_time_ms=121_000,
    )
    health = assess_health(
        book=book,
        buffer=buffer,
        now_ms=121_000 + PRIMARY_WARMUP_MS,
        stale_timeout_ms=5_000,
        ws_public_connected=True,
        ws_market_connected=True,
        kill_switch=False,
    )
    assert health.warmup_ready is False
    assert "warmup_incomplete" in health.reasons
    buffer.record_mid(101.2, ts_ms=200_000)
    still = assess_health(
        book=book,
        buffer=buffer,
        now_ms=200_000 + 1_000,
        stale_timeout_ms=5_000,
        ws_public_connected=True,
        ws_market_connected=True,
        kill_switch=False,
    )
    assert still.warmup_ready is False


def test_long_ambiguous_same_interval() -> None:
    path = ((1_000, 100.12), (1_500, 99.96))
    outcome = long_barrier_outcome(
        reference_mid=100.0,
        path=path,
        target_bps=10.0,
        stop_bps=4.0,
        start_ms=1_000,
        horizon_ms=5_000,
    )
    assert outcome == LONG_AMBIGUOUS
