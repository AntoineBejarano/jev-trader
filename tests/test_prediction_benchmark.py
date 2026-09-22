# ruff: noqa: E501
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from src.analytics.calibration import nonoverlapping_indices
from src.analytics.prediction_benchmark import extract_samples
from src.analytics.prediction_metrics import (
    balanced_accuracy,
    brier_skill_score,
    expanding_climatology,
    predicted_side_and_confidence,
    research_status,
    roc_auc,
    segment_episodes,
    signed_future_return,
)
from src.directional.calibration import brier_binary
from src.jev.models import JevCallResult, JevCallStatus, QuestionMode
from src.storage.database import Database
from src.storage.models import DirectionalForecastRow
from src.storage.repository import Repository
from tests.helpers import make_market_state


def _ts(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def test_expanding_climatology_is_point_in_time() -> None:
    labels = [True, True, False, False]
    clim = expanding_climatology(labels)
    assert clim[0] == 0.5
    assert clim[1] == pytest.approx(1.0)
    assert clim[2] == pytest.approx(1.0)
    assert clim[3] == pytest.approx(2.0 / 3.0)
    assert clim[3] != sum(labels) / len(labels)


def test_balanced_accuracy_handles_imbalance() -> None:
    predicted = [False, False, True, True]
    actual = [False, True, False, True]
    assert balanced_accuracy(predicted, actual) == pytest.approx(0.5)


def test_roc_auc_perfect_and_reversed() -> None:
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [False, False, True, True]
    assert roc_auc(scores, labels) == pytest.approx(1.0)
    assert roc_auc(scores, [True, True, False, False]) == pytest.approx(0.0)
    assert roc_auc([0.5, 0.5, 0.5], [True, False, True]) is None or roc_auc(
        [0.2, 0.8], [True, True]
    ) is None


def test_roc_auc_undefined_when_one_class() -> None:
    assert roc_auc([0.2, 0.8], [True, True]) is None


def test_brier_skill_score_vs_climatology() -> None:
    predicted = [0.9, 0.8, 0.2, 0.1]
    actual = [True, True, False, False]
    model = brier_binary(predicted, actual)
    clim = expanding_climatology(actual)
    skill = brier_skill_score(model, brier_binary(clim, actual))
    assert model is not None
    assert skill is not None
    assert skill > 0


def test_signed_return_up_down_symmetry() -> None:
    up = signed_future_return(p_up=0.7, reference_mid=100.0, future_mid=101.0)
    down = signed_future_return(p_up=0.3, reference_mid=100.0, future_mid=101.0)
    assert up == pytest.approx(0.01)
    assert down < 0
    assert predicted_side_and_confidence(0.28) == ("DOWN", 0.72)


def test_nonoverlapping_sampling() -> None:
    stamps = [0, 10_000, 59_000, 60_000, 120_000]
    assert nonoverlapping_indices(stamps, min_gap_ms=60_000) == [0, 3, 4]


def test_episode_segmentation() -> None:
    samples = [
        {"timestamp_ms": 0, "p_up": 0.20, "horizon_outcomes": {}},
        {"timestamp_ms": 5_000, "p_up": 0.22, "horizon_outcomes": {}},
        {"timestamp_ms": 10_000, "p_up": 0.40, "horizon_outcomes": {}},
        {"timestamp_ms": 15_000, "p_up": 0.80, "horizon_outcomes": {}},
    ]
    episodes = segment_episodes(samples, threshold=0.70)
    down = [item for item in episodes if item["side"] == "DOWN"]
    assert len(down) == 1
    assert down[0]["n_ticks"] == 2
    assert down[0]["duration_ms"] == 5_000
    up = [item for item in segment_episodes(samples, threshold=0.75) if item["side"] == "UP"]
    assert len(up) == 1


def test_research_status_is_conservative() -> None:
    assert research_status(
        n_nonoverlap=10, auc=0.8, auc_lo=0.7, brier_skill=0.1, delta_vs_momentum=0.05
    ) == "INSUFFICIENT_SAMPLE"
    assert research_status(
        n_nonoverlap=40, auc=0.49, auc_lo=0.4, brier_skill=-0.1, delta_vs_momentum=-0.02
    ) == "NO_MEASURED_LIFT"
    assert research_status(
        n_nonoverlap=40, auc=0.56, auc_lo=0.48, brier_skill=0.02, delta_vs_momentum=-0.01
    ) == "WEAK_MEASURED_LIFT"
    assert research_status(
        n_nonoverlap=40, auc=0.58, auc_lo=0.52, brier_skill=0.04, delta_vs_momentum=0.03
    ) == "MEASURED_LIFT_PENDING_REPLICATION"


def _forecast(
    *,
    ts: int,
    p_1m: float,
    direction: str,
    valid: bool,
    reason: str | None = None,
    candidate: str = "NO_ACTION",
    decision_id: str = "d1",
) -> DirectionalForecastRow:
    return DirectionalForecastRow(
        id=f"f-{ts}",
        timestamp=_ts(ts),
        timestamp_ms=ts,
        symbol="BTCUSDT",
        exchange="binance_usdm",
        decision_id=decision_id,
        question_version="q.directional.v1",
        strategy_version="directional.v1",
        feature_schema_version="directional.state.v1",
        feature_state={"microstructure": {"mid": 100.0}, "regime": {"direction": "RANGE"}},
        forecast={
            "p_up": {"1m": p_1m, "5m": p_1m, "10m": 0.4, "30m": 0.4},
            "reference_mid": 100.0,
            "research_sample_valid": valid,
        },
        policy={"candidate": candidate},
        rolling_outcomes={
            "1m": {
                "direction": direction,
                "future_mid": 101.0 if direction == "UP" else 99.0,
                "return_bps": 100.0 if direction == "UP" else -100.0,
                "baselines": {"momentum": "DOWN", "recent_direction": "DOWN"},
            }
        },
        research_sample_valid=valid,
        invalid_reason=reason,
    )


def test_extract_includes_risk_blocked_and_in_position_rows() -> None:
    rows = [
        _forecast(ts=0, p_1m=0.3, direction="DOWN", valid=True, candidate="OPEN_SHORT", decision_id="a"),
        _forecast(ts=5_000, p_1m=0.28, direction="DOWN", valid=True, candidate="HOLD", decision_id="b"),
        _forecast(ts=10_000, p_1m=0.25, direction="DOWN", valid=True, candidate="NO_ACTION", decision_id="c"),
    ]
    rows[2].forecast = {
        **rows[2].forecast,
        "execution": {"position_open": False, "risk_blocked": True, "risk_block_reason": "max_consecutive_losses"},
    }
    extracted = extract_samples(rows)
    samples = extracted["by_horizon"]["1m"]
    assert len(samples) == 3
    assert extracted["execution_counts"]["position_open"] == 1
    assert extracted["execution_counts"]["risk_blocked"] == 1
    assert all(item["research_sample_valid"] for item in samples)


@pytest.mark.asyncio
async def test_persist_keeps_research_row_when_risk_blocks(tmp_path: Path) -> None:
    db_path = tmp_path / "pred.db"
    db = Database(f"sqlite:///{db_path}")
    await db.create_all()
    repo = Repository(db)
    repo.bind_run(run_id="run-pred", experiment_id="directional-v1-manual-002", config_hash="h")
    state = make_market_state(symbol="BTCUSDT", price=100.0)
    result = JevCallResult(
        status=JevCallStatus.OK,
        symbol="BTCUSDT",
        exchange="binance_usdm",
        mode=QuestionMode.ENTRY,
        question_set_version="q.directional.v1",
    )
    await repo.persist_tick(
        state,
        result,
        candidate_action="OPEN_SHORT",
        final_action="NO_ACTION",
        blocked_reason="max_consecutive_losses",
        directional_payload={
            "forecast": {"p_up": {"1m": 0.28}, "research_sample_valid": True, "reference_mid": 100.0},
            "policy": {"candidate": "OPEN_SHORT"},
            "research_sample_valid": True,
            "execution": {
                "position_open": False,
                "risk_blocked": True,
                "risk_block_reason": "max_consecutive_losses",
            },
        },
    )
    rows = await repo.list_directional_forecasts("run-pred")
    assert len(rows) == 1
    assert rows[0].research_sample_valid is True
    assert rows[0].forecast["p_up"]["1m"] == 0.28
    await db.aclose()
