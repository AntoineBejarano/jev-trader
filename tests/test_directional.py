from __future__ import annotations

from pathlib import Path

import pytest
from src.analytics.counterfactual import hypothetical_side_net_pnl_bps
from src.config.settings import Settings, TradingStrategy
from src.decision.actions import Action
from src.directional.calibration import brier_binary, log_loss_binary, up_probability_bins
from src.directional.candles import (
    aggregate_10m,
    aligned_end,
    aligned_start,
    candle_from_mids,
    rolling_direction,
    rolling_return_bps,
    streak_hit,
)
from src.directional.constants import FEATURE_SCHEMA_VERSION, QUESTION_VERSION, STRATEGY_VERSION
from src.directional.coverage import feature_coverage, missing_required
from src.directional.features import build_directional_state
from src.directional.outcomes import counterfactual_horizon, magnitude_bucket, rolling_outcome
from src.directional.parse import parse_directional_forecast
from src.directional.policy import DirectionalPolicy
from src.directional.questions import (
    CANDLE_DIRECTION_QUESTIONS,
    ROLLING_DIRECTION_QUESTIONS,
    directional_questions,
    directional_required_keys,
)
from src.directional.serialize import serialize_directional_state
from src.experiment.fingerprint import (
    QUESTION_VERSION as SCALP_QUESTION_VERSION,
)
from src.experiment.fingerprint import (
    STRATEGY_VERSION as SCALP_STRATEGY_VERSION,
)
from src.experiment.fingerprint import (
    config_hash,
    research_fingerprint,
)
from src.jev.models import JevCallResult, JevCallStatus, ParsedJevDecision, QuestionMode
from src.jev.parse import decode_raw
from src.market.buffer import MarketBuffer
from src.market.models import Side
from src.market.orderbook import LocalOrderBook
from src.storage.database import Database
from src.storage.models import DirectionalForecastRow
from src.storage.repository import Repository, summarize_directional_calibration
from src.trader.engine import TraderEngine
from tests.helpers import make_market_state


def _settings(**overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "trading_strategy": "DIRECTIONAL",
        "trading_mode": "PAPER",
        "dir_entry_confidence": 0.70,
        "dir_exit_opposite_confidence": 0.65,
        "dir_reverse_confidence": 0.75,
        "dir_confirmations_required": 2,
    }
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def _mids(
    start: int = 0, *, n: int = 40, step: int = 60_000, open_px: float = 100.0
) -> tuple[list[int], list[float]]:
    stamps = [start + i * step for i in range(n)]
    values = [open_px + (0.2 if i % 2 == 0 else -0.1) for i in range(n)]
    return stamps, values


def _directional_payload(
    *,
    p_1m: float = 0.84,
    p_5m: float = 0.77,
    p_10m: float = 0.63,
    p_30m: float = 0.49,
    strengthening: float = 0.61,
    weakening: float = 0.29,
    reversal: float = 0.22,
    persist_2_up: float = 0.55,
    persist_2_down: float = 0.18,
) -> dict:
    answers: dict[str, object] = {
        "up_60s": {"type": "noul", "noul": p_1m},
        "up_300s": {"type": "noul", "noul": p_5m},
        "up_600s": {"type": "noul", "noul": p_10m},
        "up_1800s": {"type": "noul", "noul": p_30m},
        "move_60s": {
            "type": "choice",
            "choice": "NORMAL",
            "probabilities": {"NOISE": 0.1, "NORMAL": 0.6, "LARGE": 0.2, "EXTREME": 0.1},
        },
        "move_300s": {
            "type": "choice",
            "choice": "NORMAL",
            "probabilities": {"NOISE": 0.1, "NORMAL": 0.5, "LARGE": 0.3, "EXTREME": 0.1},
        },
        "move_600s": {
            "type": "choice",
            "choice": "LARGE",
            "probabilities": {"NOISE": 0.05, "NORMAL": 0.25, "LARGE": 0.5, "EXTREME": 0.2},
        },
        "move_1800s": {
            "type": "choice",
            "choice": "NOISE",
            "probabilities": {"NOISE": 0.55, "NORMAL": 0.3, "LARGE": 0.1, "EXTREME": 0.05},
        },
        "candle_up_1m": {"type": "noul", "noul": 0.72},
        "candle_up_5m": {"type": "noul", "noul": 0.66},
        "candle_up_10m": {"type": "noul", "noul": 0.58},
        "candle_up_30m": {"type": "noul", "noul": 0.51},
        "next_2_1m_up": {"type": "noul", "noul": persist_2_up},
        "next_3_1m_up": {"type": "noul", "noul": 0.40},
        "next_5_1m_up": {"type": "noul", "noul": 0.22},
        "next_2_1m_down": {"type": "noul", "noul": persist_2_down},
        "next_3_1m_down": {"type": "noul", "noul": 0.10},
        "next_5_1m_down": {"type": "noul", "noul": 0.04},
        "next_2_5m_up": {"type": "noul", "noul": 0.48},
        "next_3_5m_up": {"type": "noul", "noul": 0.31},
        "next_2_5m_down": {"type": "noul", "noul": 0.20},
        "next_3_5m_down": {"type": "noul", "noul": 0.11},
        "momentum_strengthening": {"type": "noul", "noul": strengthening},
        "momentum_weakening": {"type": "noul", "noul": weakening},
        "reversal_60s": {"type": "noul", "noul": reversal},
        "market_character": {
            "type": "choice",
            "choice": "DIRECTIONAL",
            "probabilities": {"DIRECTIONAL": 0.7, "TRANSITION": 0.2, "CHOP": 0.1},
        },
    }
    return {"id": "dir-1", "model": "typesafe/jev-1.13-20260917", "answers": answers}


def test_versions_are_isolated() -> None:
    assert QUESTION_VERSION == "q.directional.v1"
    assert STRATEGY_VERSION == "directional.v1"
    assert FEATURE_SCHEMA_VERSION == "directional.state.v1"
    assert SCALP_QUESTION_VERSION == "q.scalp.v3"
    assert SCALP_STRATEGY_VERSION == "scalp.action_values.v2"


def test_aligned_boundaries_and_10m_aggregation() -> None:
    assert aligned_start(1_000, 60_000) == 0
    assert aligned_end(1_000, 60_000) == 60_000
    stamps = list(range(0, 600_000, 1_000))
    values = [100.0 + (i / 1000) * 0.01 for i in range(len(stamps))]
    candles = []
    for start in range(0, 600_000, 60_000):
        candle = candle_from_mids(stamps, values, timeframe="1m", start_ms=start, now_ms=600_000)
        assert candle is not None
        candles.append(candle)
    bars = aggregate_10m(candles)
    assert len(bars) == 1
    assert bars[0].timeframe == "10m"
    assert bars[0].open == candles[0].open
    assert bars[0].close == candles[-1].close
    assert bars[0].direction in {"UP", "DOWN"}


def test_rolling_direction_and_candle_outcome() -> None:
    assert rolling_direction(reference_mid=100.0, future_mid=100.2) == "UP"
    assert rolling_direction(reference_mid=100.0, future_mid=99.9) == "DOWN"
    assert rolling_return_bps(reference_mid=100.0, future_mid=100.1) == pytest.approx(10.0)
    stamps = [0, 10_000, 50_000]
    values = [100.0, 100.4, 100.2]
    candle = candle_from_mids(stamps, values, timeframe="1m", start_ms=0, now_ms=60_000)
    assert candle is not None
    assert candle.direction == "UP"
    assert candle.close == 100.2


def test_streak_outcomes() -> None:
    assert streak_hit(["UP", "UP", "DOWN"], count=2, side="UP") is True
    assert streak_hit(["UP", "UP", "DOWN"], count=3, side="UP") is False
    assert streak_hit(["DOWN", "DOWN", "DOWN", "DOWN", "DOWN"], count=5, side="DOWN") is True
    assert streak_hit(["UP"], count=2, side="UP") is None


def test_term_structure_and_consistency_warning() -> None:
    parsed = parse_directional_forecast(decode_raw(_directional_payload()))
    assert [row["horizon"] for row in parsed["term_structure"]] == ["1m", "5m", "10m", "30m"]
    assert parsed["p_up"]["1m"] == 0.84
    assert parsed["p_down"]["1m"] == pytest.approx(0.16)
    assert parsed["direction_consensus"] == "MIXED"
    assert parsed["direction_conflict"] is True
    broken = _directional_payload()
    broken["answers"]["next_3_1m_up"] = {"type": "noul", "noul": 0.90}
    warned = parse_directional_forecast(decode_raw(broken))
    warnings = warned["probability_consistency_warnings"]
    assert any("probability_consistency_warning" in item for item in warnings)


def test_questions_exclude_economics() -> None:
    blob = str(directional_questions())
    assert "required EV" not in blob
    assert "SCALP_MIN_NET_EV" not in blob
    keys = set(directional_required_keys())
    assert "up_60s" in keys
    assert "next_5_1m_up" in keys
    assert "move_1800s" in keys


def test_magnitude_buckets() -> None:
    assert magnitude_bucket(4.0, 10.0) == "NOISE"
    assert magnitude_bucket(7.0, 10.0) == "NORMAL"
    assert magnitude_bucket(15.0, 10.0) == "LARGE"
    assert magnitude_bucket(25.0, 10.0) == "EXTREME"
    outcome = rolling_outcome(
        reference_mid=100.0, future_mid=100.2, sigma_bps=10.0, path=[100.3, 99.9, 100.2]
    )
    assert outcome["direction"] == "UP"
    assert outcome["mfe_bps"] == pytest.approx(30.0)
    assert outcome["mae_bps"] == pytest.approx(-10.0)


def test_brier_logloss_and_bins() -> None:
    preds = [0.55, 0.65, 0.75, 0.92]
    hits = [True, False, True, True]
    assert brier_binary(preds, hits) == pytest.approx(
        ((0.55 - 1) ** 2 + (0.65 - 0) ** 2 + (0.75 - 1) ** 2 + (0.92 - 1) ** 2) / 4
    )
    assert log_loss_binary([0.8], [True]) == pytest.approx(-__import__("math").log(0.8))
    bins = up_probability_bins(preds, hits)
    labels = [row["bin"] for row in bins]
    assert labels[0] == "0.50_0.60"
    assert labels[-1] == "0.90_plus"


def test_counterfactual_gross_vs_net() -> None:
    raw = counterfactual_horizon(
        predicted_side="UP",
        entry_bid=100.0,
        entry_ask=100.1,
        exit_bid=100.3,
        exit_ask=100.4,
        reference_mid=100.05,
        future_mid=100.35,
        taker_fee_bps=5.0,
        slippage_bps=1.0,
    )
    assert raw["direction_correct"] is True
    assert raw["raw_market_return_bps"] > 0
    net = hypothetical_side_net_pnl_bps(
        side="LONG",
        entry_bid=100.0,
        entry_ask=100.1,
        exit_bid=100.11,
        exit_ask=100.12,
        taker_fee_bps=5.0,
        slippage_bps=1.0,
    )
    tiny = counterfactual_horizon(
        predicted_side="UP",
        entry_bid=100.0,
        entry_ask=100.1,
        exit_bid=100.11,
        exit_ask=100.12,
        reference_mid=100.05,
        future_mid=100.08,
        taker_fee_bps=5.0,
        slippage_bps=1.0,
    )
    assert tiny["correct_but_unprofitable"] is True
    assert net is not None and net < 0


def test_hysteresis_needs_confirmations() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    state = make_market_state(symbol="BTCUSDT", price=100.0)
    forecast = parse_directional_forecast(decode_raw(_directional_payload()))
    result = JevCallResult(
        status=JevCallStatus.OK,
        symbol="BTCUSDT",
        exchange="binance_usdm",
        mode=QuestionMode.ENTRY,
        question_set_version="q.directional.v1",
        decision=ParsedJevDecision(
            mode=QuestionMode.ENTRY,
            question_set_version="q.directional.v1",
            model="typesafe/jev-1.13",
            latency_ms=10.0,
            directional_forecast=forecast,
        ),
    )
    first = policy.evaluate(state, result)
    assert first.candidate is Action.NO_ACTION
    second = policy.evaluate(state, result)
    assert second.candidate is Action.OPEN_LONG
    long_state = make_market_state(symbol="BTCUSDT", price=100.0, side=Side.LONG, quantity=0.01)
    hold = policy.evaluate(long_state, result)
    assert hold.candidate is Action.HOLD
    down = parse_directional_forecast(
        decode_raw(_directional_payload(p_1m=0.20, p_5m=0.22, p_10m=0.30, p_30m=0.40))
    )
    down_result = result.model_copy(
        update={
            "decision": result.decision.model_copy(update={"directional_forecast": down})
            if result.decision
            else None
        }
    )
    first_down = policy.evaluate(long_state, down_result)
    assert first_down.candidate is Action.HOLD
    close = policy.evaluate(long_state, down_result)
    assert close.candidate in {Action.CLOSE, Action.REVERSE}


def test_feature_state_omits_economics() -> None:
    buffer = MarketBuffer(symbol="BTCUSDT", exchange="binance_usdm", history_ms=3_600_000)
    now = 1_800_000
    for i in range(0, now, 1_000):
        buffer.record_mid(100.0 + i / 100_000, ts_ms=i)
    book = LocalOrderBook()
    book.apply_partial_snapshot(
        bids=[(99.9, 1.0), (99.8, 1.0), (99.7, 1.0), (99.6, 1.0), (99.5, 1.0)],
        asks=[(100.1, 1.0), (100.2, 1.0), (100.3, 1.0), (100.4, 1.0), (100.5, 1.0)],
        last_update_id=1,
        event_time_ms=now,
    )
    state = build_directional_state(buffer=buffer, book=book, now_ms=now)
    state["fees"] = 5.0
    state["unrealized_pnl"] = 12.0
    payload = serialize_directional_state(state)
    blob = str(payload)
    assert "fees" not in payload
    assert "unrealized_pnl" not in payload
    assert "SCALP_MIN_NET_EV" not in blob
    assert payload["schema"] == FEATURE_SCHEMA_VERSION
    assert "microstructure" in payload
    assert "technical" in payload


def test_scalp_fingerprint_unchanged_for_scalp() -> None:
    scalp = Settings(trading_strategy="SCALP", experiment_id="scalp-v3-manual-001", _env_file=None)
    fp = research_fingerprint(scalp, resolved_model=None)
    assert fp["question_version"] == "q.scalp.v3"
    assert fp["strategy_version"] == "scalp.action_values.v2"
    directional = _settings(experiment_id="directional-v1-manual-001", experiment_frozen=True)
    dfp = research_fingerprint(directional, resolved_model=None)
    assert dfp["question_version"] == "q.directional.v1"
    assert dfp["strategy_version"] == "directional.v1"
    assert config_hash(fp) != config_hash(dfp)


def test_directional_uses_vip0_fees() -> None:
    settings = _settings(taker_fee_bps=0.0, maker_fee_bps=0.0)
    assert settings.trading_strategy is TradingStrategy.DIRECTIONAL
    assert settings.execution_taker_fee_bps == 5.0
    assert settings.execution_maker_fee_bps == 2.0


@pytest.mark.asyncio
async def test_directional_postgres_style_persistence(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'dir.db'}")
    await db.create_all()
    repo = Repository(db)
    repo.bind_run(run_id="run-dir", experiment_id="directional-v1-test", config_hash="abc")
    state = make_market_state(symbol="BTCUSDT", price=100.0)
    result = JevCallResult(
        status=JevCallStatus.OK,
        symbol="BTCUSDT",
        exchange="binance_usdm",
        mode=QuestionMode.ENTRY,
        question_set_version="q.directional.v1",
        decision=ParsedJevDecision(
            mode=QuestionMode.ENTRY,
            question_set_version="q.directional.v1",
            model="typesafe/jev-1.13",
            latency_ms=12.0,
            directional_forecast=parse_directional_forecast(decode_raw(_directional_payload())),
        ),
        raw_response=_directional_payload(),
    )
    await repo.persist_tick(
        state,
        result,
        candidate_action="NO_ACTION",
        final_action="NO_ACTION",
        directional_payload={
            "question_version": "q.directional.v1",
            "strategy_version": "directional.v1",
            "feature_schema_version": "directional.state.v1",
            "feature_state": {"schema": "directional.state.v1", "mid": 100.0},
            "forecast": {
                "p_up": {"1m": 0.84, "5m": 0.77, "10m": 0.63, "30m": 0.49},
                "reference_mid": 100.0,
                "reference_ts": state.timestamp_ms,
                "research_sample_valid": True,
            },
            "policy": {"candidate": "NO_ACTION"},
            "research_sample_valid": True,
        },
    )
    rows = await repo.list_directional_forecasts("run-dir")
    assert len(rows) == 1
    assert rows[0].question_version == "q.directional.v1"
    summary = summarize_directional_calibration(rows)
    assert summary["n_forecasts"] == 1
    counts = await repo.count_directional("directional-v1-test")
    assert counts["forecasts"] == 1
    wiped = await repo.wipe_directional_research("directional-v1-test")
    assert wiped["directional_forecasts"] == 1
    leftover = await repo.list_directional_forecasts("run-dir")
    assert leftover == []
    await db.aclose()


@pytest.mark.asyncio
async def test_dashboard_directional_serialization(tmp_path: Path) -> None:
    engine = TraderEngine(
        _settings(database_url=f"sqlite:///{tmp_path / 'op.db'}", openrouter_api_key="x")
    )
    await engine.db.create_all()
    payload = await engine.operator_snapshot()
    assert payload["safety"]["live"] == "BLOCKED"
    assert "DIRECTIONAL" in payload["safety"]["banner"]
    assert payload["experiment"]["question_version"] == "q.directional.v1"
    assert payload["experiment"]["strategy_version"] == "directional.v1"
    assert payload["directional"] is not None
    await engine.db.aclose()


@pytest.mark.asyncio
async def test_restart_idempotent_directional_row(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'idemp.db'}")
    await db.create_all()
    repo = Repository(db)
    repo.bind_run(run_id="r", experiment_id="directional-v1-id", config_hash="h")
    state = make_market_state(symbol="BTCUSDT", price=101.0)
    result = JevCallResult(
        status=JevCallStatus.OK,
        symbol="BTCUSDT",
        exchange="binance_usdm",
        mode=QuestionMode.ENTRY,
        question_set_version="q.directional.v1",
        decision=ParsedJevDecision(
            mode=QuestionMode.ENTRY,
            question_set_version="q.directional.v1",
            model="typesafe/jev-1.13",
            latency_ms=8.0,
            directional_forecast=parse_directional_forecast(decode_raw(_directional_payload())),
        ),
    )
    payload = {"forecast": {"p_up": {"1m": 0.8}}, "policy": {}}
    first = await repo.persist_tick(state, result, directional_payload=payload)
    second = await repo.persist_tick(state, result, directional_payload=payload)
    assert first != second
    rows = await repo.list_directional_forecasts("r")
    assert len(rows) == 2
    assert all(isinstance(row, DirectionalForecastRow) for row in rows)
    await db.aclose()


def test_rolling_and_candle_questions_are_independent() -> None:
    questions = directional_questions()
    rolling_keys = [key for key, _label, _seconds in ROLLING_DIRECTION_QUESTIONS]
    candle_keys = [key for key, _label in CANDLE_DIRECTION_QUESTIONS]
    assert rolling_keys == ["up_60s", "up_300s", "up_600s", "up_1800s"]
    assert candle_keys == ["candle_up_1m", "candle_up_5m", "candle_up_10m", "candle_up_30m"]
    assert len(set(rolling_keys)) == 4
    assert len(set(candle_keys)) == 4
    texts = [questions[key]["instructions"] for key in rolling_keys + candle_keys]
    assert len(set(texts)) == 8
    parsed = parse_directional_forecast(
        decode_raw(_directional_payload(p_1m=0.21, p_5m=0.33, p_10m=0.44, p_30m=0.55))
    )
    assert parsed["p_up"] == {"1m": 0.21, "5m": 0.33, "10m": 0.44, "30m": 0.55}
    audit = {row["question_id"]: row for row in parsed["answer_audit"]}
    assert audit["up_60s"]["raw_jev_value"] == 0.21
    assert audit["up_300s"]["parsed_probability"] == 0.33
    assert audit["up_600s"]["horizon"] == "10m"
    assert audit["up_1800s"]["parsed_probability"] == 0.55
    assert audit["candle_up_1m"]["parsed_probability"] == 0.72
    flipped = _directional_payload(p_1m=0.21, p_5m=0.33, p_10m=0.44, p_30m=0.55)
    flipped["answers"]["up_600s"] = {"type": "noul", "noul": 0.91}
    flipped["answers"]["candle_up_5m"] = {"type": "noul", "noul": 0.11}
    changed = parse_directional_forecast(decode_raw(flipped))
    assert changed["p_up"]["1m"] == 0.21
    assert changed["p_up"]["5m"] == 0.33
    assert changed["p_up"]["10m"] == 0.91
    assert changed["p_up"]["30m"] == 0.55
    assert changed["candle_up"]["1m"] == 0.72
    assert changed["candle_up"]["5m"] == 0.11
    identical = parse_directional_forecast(
        decode_raw(_directional_payload(p_1m=0.22, p_5m=0.22, p_10m=0.22, p_30m=0.22))
    )
    assert identical["p_up"] == {"1m": 0.22, "5m": 0.22, "10m": 0.22, "30m": 0.22}


def test_warmup_rows_excluded_from_official_calibration() -> None:
    valid = DirectionalForecastRow(
        id="v",
        timestamp=__import__("datetime").datetime.now(__import__("datetime").UTC),
        timestamp_ms=1,
        symbol="BTCUSDT",
        exchange="binance_usdm",
        question_version="q.directional.v1",
        strategy_version="directional.v1",
        feature_schema_version="directional.state.v1",
        feature_state={},
        forecast={"p_up": {"1m": 0.8}, "research_sample_valid": True},
        policy={},
        rolling_outcomes={"1m": {"direction": "UP", "baselines": {}}},
        research_sample_valid=True,
        run_id="run-a",
    )
    warmup = DirectionalForecastRow(
        id="w",
        timestamp=__import__("datetime").datetime.now(__import__("datetime").UTC),
        timestamp_ms=2,
        symbol="BTCUSDT",
        exchange="binance_usdm",
        question_version="q.directional.v1",
        strategy_version="directional.v1",
        feature_schema_version="directional.state.v1",
        feature_state={},
        forecast={
            "p_up": {"1m": 0.9},
            "research_sample_valid": False,
            "invalid_reason": "warmup_incomplete",
        },
        policy={},
        rolling_outcomes={"1m": {"direction": "DOWN", "baselines": {}}},
        research_sample_valid=False,
        invalid_reason="warmup_incomplete",
        run_id="run-a",
    )
    summary = summarize_directional_calibration([valid, warmup])
    assert summary["n_raw_predictions"] == 2
    assert summary["n_valid_research"] == 1
    assert summary["n_nonoverlapping_valid"] == 1
    assert summary["horizons"]["1m"]["n"] == 1
    assert summary["horizons"]["1m"]["accuracy"] == 1.0


def test_r_1h_is_required_and_seeded_history_survives_eviction() -> None:
    from src.market.buffer import MarketBuffer
    from src.market.history import seed_buffer_from_klines

    now = 3_600_000 * 3
    buffer = MarketBuffer(
        symbol="BTCUSDT", exchange="binance_usdm", history_ms=3_600_000, clock=lambda: now
    )
    klines = []
    start = now - 120 * 60_000
    for i in range(120):
        open_time = start + i * 60_000
        close_time = open_time + 59_999
        px = 100.0 + i * 0.01
        klines.append([open_time, px, px, px, px + 0.01, 1.0, close_time])
    seeded = seed_buffer_from_klines(buffer, klines)
    assert seeded > 100
    buffer.record_mid(102.0, now)
    stamps, values = buffer.mids_as_lists()
    assert stamps[0] <= now - 3_600_000
    from src.features.returns import compute_returns

    returns = compute_returns(stamps, values, now, 102.0)
    assert returns.r_1h is not None
    coverage = feature_coverage(
        {
            "microstructure": {
                "mid": 102.0,
                "best_bid": 101.9,
                "best_ask": 102.1,
                "spread_bps": 1.0,
            },
            "price_action": {
                "returns": returns.model_dump() | {"r_10m": 0.01, "r_30m": 0.02},
                "active_candles": {"1m": {"open": 1}, "5m": {"open": 1}},
            },
            "health": {"book_valid": True, "has_mid": True},
        }
    )
    assert "price_action.returns.r_1h" not in coverage["required_missing"]
    assert missing_required({"microstructure": {}})


def test_live_overwrite_keeps_seeded_flag_for_r_1h() -> None:
    from src.features.returns import compute_returns
    from src.market.buffer import MarketBuffer
    from src.market.history import seed_buffer_from_klines

    now = 3_600_000 * 3
    buffer = MarketBuffer(
        symbol="BTCUSDT", exchange="binance_usdm", history_ms=3_600_000, clock=lambda: now
    )
    klines = []
    start = now - 120 * 60_000
    for i in range(120):
        open_time = start + i * 60_000
        close_time = open_time + 59_999
        px = 100.0 + i * 0.01
        klines.append([open_time, px, px, px, px + 0.01, 1.0, close_time])
    seed_buffer_from_klines(buffer, klines)
    last_ts = buffer.mids[-1].ts_ms
    buffer.record_mid(102.0, last_ts + 50)
    assert buffer.mids[-1].seeded is True
    stamps, values = buffer.mids_as_lists()
    assert stamps[0] <= now - 3_600_000
    returns = compute_returns(stamps, values, now, 102.0)
    assert returns.r_1h is not None


@pytest.mark.asyncio
async def test_current_run_trades_are_not_mixed(tmp_path: Path) -> None:
    db = Database(f"sqlite:///{tmp_path / 'scope.db'}")
    await db.create_all()
    repo = Repository(db)
    repo.bind_run(run_id="run-new", experiment_id="directional-v1-scope", config_hash="h")
    from datetime import UTC, datetime

    from src.storage.models import TradeRow

    async with db.session() as session:
        session.add(
            TradeRow(
                opened_at=datetime.now(UTC),
                symbol="BTCUSDT",
                exchange="binance_usdm",
                side="LONG",
                quantity=0.01,
                entry_price=100.0,
                exit_price=101.0,
                entry_notional=1000.0,
                holding_seconds=10.0,
                gross_pnl=1.0,
                fees=0.1,
                net_pnl=0.9,
                payload={},
                run_id="run-old",
                experiment_id="scalp-v3-manual-001",
            )
        )
        await session.commit()
    recent = await repo.list_recent_trades(20, run_id="run-new")
    assert recent == []
    mixed = await repo.list_recent_trades(20)
    assert len(mixed) == 1
    from src.portfolio.account import PaperAccount

    await repo.save_paper_account(
        PaperAccount.new(starting_balance=10_000, currency="USDT", now_ms=1)
    )
    restored = await repo.load_paper_account()
    assert restored is not None
    assert restored.closed_trades == []
    await db.aclose()
