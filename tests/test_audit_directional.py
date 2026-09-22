from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.analytics.audit_directional import entry_funnel
from src.analytics.audit_math import (
    check_candle_label,
    check_rolling_label,
    decompose_round_trip,
    ece_binary,
    in_bin,
    predicted_side_and_confidence,
    recover_reference_mids,
    signed_future_return,
    vote_from_p_up,
)
from src.analytics.calibration import nonoverlapping_indices
from src.directional.candles import WINDOW_MS, aligned_start
from src.storage.models import DirectionalForecastRow


def test_signed_return_up_and_down_are_not_the_same_formula() -> None:
    up = signed_future_return(p_up=0.7, reference_mid=100.0, future_mid=101.0)
    down = signed_future_return(p_up=0.3, reference_mid=100.0, future_mid=101.0)
    assert up == pytest.approx(0.01)
    assert abs(down - (100.0 / 101.0 - 1.0)) < 1e-12
    assert down < 0
    assert up != -down


def test_bucket_symmetry_uses_predicted_side_confidence() -> None:
    assert predicted_side_and_confidence(0.72) == ("UP", 0.72)
    assert predicted_side_and_confidence(0.28) == ("DOWN", 0.72)
    assert in_bin(0.72, 0.70, 0.75)
    assert in_bin(0.72, 0.70, 0.75) == in_bin(0.72, 0.70, 0.75)


def test_nonoverlapping_sampling_keeps_horizon_gap() -> None:
    stamps = [0, 10_000, 59_000, 60_000, 120_000]
    chosen = nonoverlapping_indices(stamps, min_gap_ms=60_000)
    assert chosen == [0, 3, 4]


def test_rolling_label_is_future_mid_gt_reference() -> None:
    ok = check_rolling_label(
        reference_mid=100.0,
        future_mid=100.5,
        stored_direction="UP",
        stored_return_bps=50.0,
    )
    assert ok["ok"] is True
    bad = check_rolling_label(
        reference_mid=100.0,
        future_mid=99.0,
        stored_direction="UP",
        stored_return_bps=-100.0,
    )
    assert bad["ok"] is False
    assert "direction_mismatch" in bad["issues"]


def test_utc_candle_alignment() -> None:
    start = aligned_start(1_800_000 + 123, WINDOW_MS["30m"])
    assert start == 1_800_000
    good = check_candle_label(
        {
            "open": 100.0,
            "close": 101.0,
            "direction": "UP",
            "window_start": start,
            "window_end": start + WINDOW_MS["30m"],
        },
        timeframe="30m",
    )
    assert good["ok"] is True
    bad = check_candle_label(
        {
            "open": 100.0,
            "close": 99.0,
            "direction": "UP",
            "window_start": start + 1,
            "window_end": start + WINDOW_MS["30m"],
        },
        timeframe="30m",
    )
    assert "direction_mismatch" in bad["issues"]
    assert "window_not_utc_aligned" in bad["issues"]


def test_pnl_decomposition_invariants_long() -> None:
    qty = 1.0
    entry_bid, entry_ask = 100.0, 100.10
    exit_bid, exit_ask = 101.0, 101.10
    entry_fill = 100.11
    exit_fill = 100.99
    decomp = decompose_round_trip(
        side="LONG",
        quantity=qty,
        entry_fill=entry_fill,
        exit_fill=exit_fill,
        entry_bid=entry_bid,
        entry_ask=entry_ask,
        exit_bid=exit_bid,
        exit_ask=exit_ask,
        entry_fee=0.05,
        exit_fee=0.05,
        stored_market_move=None,
        stored_entry_slip=0.01,
        stored_exit_slip=0.01,
        stored_entry_spread=0.05,
        stored_exit_spread=0.05,
        stored_gross=(exit_fill - entry_fill) * qty,
        stored_fees=0.10,
        stored_net=(exit_fill - entry_fill) * qty - 0.10,
    )
    assert decomp["identity_ideal_minus_execution_eq_gross"] is True
    assert decomp["identity_gross_minus_fees_eq_net"] is True
    assert decomp["fill_to_fill_gross_pnl_usdt"] == (exit_fill - entry_fill) * qty


def test_recovered_mids_match_short_identity() -> None:
    qty = 0.01
    entry_mid, exit_mid = recover_reference_mids(
        side="SHORT",
        quantity=qty,
        entry_fill=99.99,
        exit_fill=100.02,
        entry_slippage=0.0001,
        exit_slippage=0.0001,
        entry_spread=0.00005,
        exit_spread=0.00005,
    )
    assert entry_mid is not None and exit_mid is not None
    assert entry_mid > 99.99
    assert exit_mid < 100.02


def test_funnel_counters_are_side_symmetric() -> None:
    now = datetime(2026, 9, 21, 22, 0, tzinfo=UTC)
    rows = []
    for index, p_up in enumerate([0.80, 0.81, 0.20, 0.19]):
        rows.append(
            DirectionalForecastRow(
                timestamp=now,
                timestamp_ms=1_000_000 + index * 5_000,
                symbol="BTCUSDT",
                exchange="binance_usdm",
                question_version="q.directional.v1",
                strategy_version="directional.v1",
                feature_schema_version="directional.state.v1",
                feature_state={},
                forecast={"p_up": {"1m": p_up, "5m": p_up}},
                policy={
                    "candidate": "NO_ACTION" if index % 2 == 0 else (
                        "OPEN_LONG" if p_up >= 0.7 else "OPEN_SHORT"
                    ),
                    "reason": (
                        "insufficient_confirmation" if index % 2 == 0 else "directional_entry"
                    ),
                },
                rolling_outcomes={},
                candle_outcomes={},
                persistence_outcomes={},
                counterfactuals={},
            )
        )
    funnel = entry_funnel(rows, [], [])
    assert funnel["same_side_confidence_ge_0_70"]["UP"] == 2
    assert funnel["same_side_confidence_ge_0_70"]["DOWN"] == 2
    assert vote_from_p_up(0.70, 0.70) == "UP"
    assert vote_from_p_up(0.30, 0.70) == "DOWN"


def test_ece_zero_when_perfectly_calibrated() -> None:
    predicted = [0.0] * 10 + [1.0] * 10
    hits = [False] * 10 + [True] * 10
    value = ece_binary(predicted, hits, (("a", 0.0, 0.5), ("b", 0.5, 1.0)))
    assert value is not None
    assert value < 1e-9
