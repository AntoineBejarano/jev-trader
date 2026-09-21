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
from src.directional.features import build_directional_state
from src.directional.outcomes import counterfactual_horizon, magnitude_bucket, rolling_outcome
from src.directional.parse import parse_directional_forecast
from src.directional.policy import DirectionalPolicy
from src.directional.questions import directional_questions, directional_required_keys
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
    *, p_1m: float = 0.84, p_5m: float = 0.77, p_10m: float = 0.63, p_30m: float = 0.49
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
        "next_2_1m_up": {"type": "noul", "noul": 0.55},
        "next_3_1m_up": {"type": "noul", "noul": 0.40},
        "next_5_1m_up": {"type": "noul", "noul": 0.22},
        "next_2_1m_down": {"type": "noul", "noul": 0.18},
        "next_3_1m_down": {"type": "noul", "noul": 0.10},
        "next_5_1m_down": {"type": "noul", "noul": 0.04},
        "next_2_5m_up": {"type": "noul", "noul": 0.48},
        "next_3_5m_up": {"type": "noul", "noul": 0.31},
        "next_2_5m_down": {"type": "noul", "noul": 0.20},
        "next_3_5m_down": {"type": "noul", "noul": 0.11},
        "momentum_strengthening": {"type": "noul", "noul": 0.61},
        "momentum_weakening": {"type": "noul", "noul": 0.29},
        "reversal_60s": {"type": "noul", "noul": 0.22},
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
            },
            "policy": {"candidate": "NO_ACTION"},
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
