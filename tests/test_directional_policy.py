from __future__ import annotations

from src.decision.actions import Action
from src.directional.parse import parse_directional_forecast
from src.directional.policy import DirectionalPolicy
from src.directional.thesis import classify_horizons, elapsed_multiple
from src.jev.models import JevCallResult, JevCallStatus, ParsedJevDecision, QuestionMode
from src.jev.parse import decode_raw
from src.market.models import Side
from src.portfolio.account import OpenPosition
from tests.helpers import make_market_state
from tests.test_directional import _directional_payload, _settings


def _result(**payload_kw: float) -> JevCallResult:
    forecast = parse_directional_forecast(decode_raw(_directional_payload(**payload_kw)))
    return JevCallResult(
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
            directional_forecast=forecast,
        ),
    )


def _open(policy: DirectionalPolicy, *, side: str) -> dict:
    if side == "UP":
        result = _result(p_1m=0.84, p_5m=0.77, p_10m=0.68, p_30m=0.55)
        action = Action.OPEN_LONG
    else:
        result = _result(p_1m=0.16, p_5m=0.23, p_10m=0.32, p_30m=0.45)
        action = Action.OPEN_SHORT
    flat = make_market_state(symbol="BTCUSDT", price=100.0)
    first = policy.evaluate(flat, result)
    assert first.candidate is Action.NO_ACTION
    second = policy.evaluate(flat, result)
    assert second.candidate is action
    assert second.entry_signal is True
    assert second.entry_thesis is not None
    return second.entry_thesis


def test_entry_and_hold_thresholds_are_separated() -> None:
    settings = _settings()
    assert settings.dir_close_thesis_floor < settings.dir_hold_floor
    assert settings.dir_hold_floor < settings.dir_entry_confidence
    assert settings.dir_entry_confidence < settings.dir_reverse_confidence
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="UP")
    long_state = make_market_state(symbol="BTCUSDT", price=100.0, side=Side.LONG, quantity=0.01)
    mild = _result(p_1m=0.62, p_5m=0.64, p_10m=0.60, p_30m=0.58)
    hold = policy.evaluate(long_state, mild, entry_thesis=thesis)
    assert hold.candidate is Action.HOLD
    assert hold.hold_signal is True
    assert hold.entry_signal is False
    assert settings.dir_entry_confidence > 0.62


def test_close_without_opposite_entry_signal() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="DOWN")
    short = make_market_state(
        symbol="BTCUSDT",
        price=100.0,
        side=Side.SHORT,
        quantity=0.01,
        mfe_bps=18.0,
        unrealized_pnl_bps=4.0,
    )
    weak = _result(
        p_1m=0.58,
        p_5m=0.38,
        p_10m=0.39,
        p_30m=0.33,
        weakening=0.72,
        strengthening=0.22,
        reversal=0.64,
    )
    first = policy.evaluate(short, weak, entry_thesis=thesis)
    assert first.candidate is Action.HOLD
    assert first.reverse_signal is False
    second = policy.evaluate(short, weak, entry_thesis=thesis)
    assert second.candidate is Action.CLOSE
    assert second.close_signal is True
    assert second.reverse_signal is False
    assert second.close_reason in {
        "short_horizon_flipped",
        "term_structure_conflict",
        "thesis_decayed",
        "momentum_weakening_with_giveback",
        "opposite_signal_not_strong_enough_to_reverse",
    }
    assert float(weak.decision.directional_forecast["p_up"]["1m"]) < settings.dir_reverse_confidence


def test_reverse_is_stricter_than_close() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="DOWN")
    short = make_market_state(symbol="BTCUSDT", price=100.0, side=Side.SHORT, quantity=0.01)
    opposite = _result(
        p_1m=0.82,
        p_5m=0.76,
        p_10m=0.69,
        p_30m=0.60,
        persist_2_up=0.72,
        persist_2_down=0.12,
        strengthening=0.70,
        weakening=0.20,
        reversal=0.25,
    )
    first = policy.evaluate(short, opposite, entry_thesis=thesis)
    assert first.candidate is Action.HOLD
    second = policy.evaluate(short, opposite, entry_thesis=thesis)
    assert second.candidate is Action.REVERSE
    assert second.reverse_signal is True
    assert second.reverse_reason == "reverse_confirmed"


def test_primary_and_supporting_horizons_persisted() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="UP")
    assert thesis["entry_primary_horizon"] == "5m"
    assert thesis["entry_supporting_horizons"] == ["1m", "5m", "10m"]
    assert "30m" in thesis["entry_context_horizons"]
    assert thesis["entry_p_up_1m"] == 0.84
    primary, supporting, context = classify_horizons(
        {"1m": 0.84, "5m": 0.77, "10m": 0.68, "30m": 0.55},
        "UP",
        entry_confidence=0.70,
        support_confidence=0.62,
    )
    assert primary == "5m"
    assert supporting == ["1m", "5m", "10m"]
    assert context == ["30m"]


def test_distant_horizon_cannot_indefinitely_dominate() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="DOWN")
    assert thesis["entry_primary_horizon"] == "5m"
    short = make_market_state(symbol="BTCUSDT", price=100.0, side=Side.SHORT, quantity=0.01)
    distant = _result(p_1m=0.58, p_5m=0.49, p_10m=0.45, p_30m=0.20)
    policy.evaluate(short, distant, entry_thesis=thesis)
    closed = policy.evaluate(short, distant, entry_thesis=thesis)
    assert closed.candidate is Action.CLOSE
    assert closed.close_reason in {
        "term_structure_conflict",
        "short_horizon_flipped",
        "thesis_decayed",
        "opposite_signal_not_strong_enough_to_reverse",
    }
    same = closed.position_thesis["components"]["same_side"]
    assert same["30m"] >= 0.75
    assert same["5m"] < settings.dir_hold_floor


def test_thesis_decay_and_conflict() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="UP")
    long_state = make_market_state(symbol="BTCUSDT", price=100.0, side=Side.LONG, quantity=0.01)
    decayed = _result(p_1m=0.51, p_5m=0.52, p_10m=0.54, p_30m=0.56)
    policy.evaluate(long_state, decayed, entry_thesis=thesis)
    closed = policy.evaluate(long_state, decayed, entry_thesis=thesis)
    assert closed.position_thesis is not None
    assert closed.position_thesis["thesis_decay"] > 0
    assert closed.candidate in {Action.CLOSE, Action.HOLD}


def test_profit_giveback_strengthens_close_only() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="DOWN")
    weak = _result(
        p_1m=0.46,
        p_5m=0.42,
        p_10m=0.40,
        p_30m=0.35,
        weakening=0.72,
        strengthening=0.22,
        reversal=0.55,
    )
    intact = make_market_state(
        symbol="BTCUSDT",
        price=100.0,
        side=Side.SHORT,
        quantity=0.01,
        mfe_bps=0.0,
        unrealized_pnl_bps=2.0,
    )
    policy.evaluate(intact, weak, entry_thesis=thesis)
    hold = policy.evaluate(intact, weak, entry_thesis=thesis)
    giveback_state = make_market_state(
        symbol="BTCUSDT",
        price=100.0,
        side=Side.SHORT,
        quantity=0.01,
        mfe_bps=20.0,
        unrealized_pnl_bps=8.0,
    )
    policy2 = DirectionalPolicy(settings)
    thesis2 = _open(policy2, side="DOWN")
    policy2.evaluate(giveback_state, weak, entry_thesis=thesis2)
    closed = policy2.evaluate(giveback_state, weak, entry_thesis=thesis2)
    assert hold.candidate is Action.HOLD
    assert closed.candidate is Action.CLOSE
    assert closed.close_reason == "momentum_weakening_with_giveback"


def test_position_age_requires_revalidation_not_autoclose() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="UP")
    strong = _result(p_1m=0.84, p_5m=0.77, p_10m=0.68, p_30m=0.55)
    stale_strong = make_market_state(
        symbol="BTCUSDT",
        price=100.0,
        side=Side.LONG,
        quantity=0.01,
        age_seconds=2_000.0,
    )
    hold = policy.evaluate(stale_strong, strong, entry_thesis=thesis)
    assert hold.candidate is Action.HOLD
    assert hold.position_thesis["primary_horizon_elapsed_multiple"] >= 3
    weak_policy = DirectionalPolicy(settings)
    weak_thesis = _open(weak_policy, side="UP")
    weak = _result(
        p_1m=0.53,
        p_5m=0.52,
        p_10m=0.51,
        p_30m=0.50,
        strengthening=0.20,
        weakening=0.75,
        reversal=0.40,
    )
    stale_weak = make_market_state(
        symbol="BTCUSDT",
        price=100.0,
        side=Side.LONG,
        quantity=0.01,
        age_seconds=2_000.0,
    )
    weak_policy.evaluate(stale_weak, weak, entry_thesis=weak_thesis)
    closed = weak_policy.evaluate(stale_weak, weak, entry_thesis=weak_thesis)
    assert closed.candidate is Action.CLOSE
    assert elapsed_multiple(2_000.0, "5m") > 3


def test_hysteresis_avoids_single_tick_flip() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="UP")
    long_state = make_market_state(symbol="BTCUSDT", price=100.0, side=Side.LONG, quantity=0.01)
    down = _result(p_1m=0.22, p_5m=0.28, p_10m=0.34, p_30m=0.40)
    first = policy.evaluate(long_state, down, entry_thesis=thesis)
    assert first.candidate is Action.HOLD
    second = policy.evaluate(long_state, down, entry_thesis=thesis)
    assert second.candidate in {Action.CLOSE, Action.REVERSE}


def test_long_and_short_symmetry() -> None:
    settings = _settings()
    long_policy = DirectionalPolicy(settings)
    short_policy = DirectionalPolicy(settings)
    long_thesis = _open(long_policy, side="UP")
    short_thesis = _open(short_policy, side="DOWN")
    assert long_thesis["entry_primary_horizon"] == short_thesis["entry_primary_horizon"]
    long_state = make_market_state(symbol="BTCUSDT", price=100.0, side=Side.LONG, quantity=0.01)
    short_state = make_market_state(symbol="BTCUSDT", price=100.0, side=Side.SHORT, quantity=0.01)
    long_hold = long_policy.evaluate(
        long_state, _result(p_1m=0.66, p_5m=0.64, p_10m=0.61, p_30m=0.58), entry_thesis=long_thesis
    )
    short_hold = short_policy.evaluate(
        short_state,
        _result(p_1m=0.34, p_5m=0.36, p_10m=0.39, p_30m=0.42),
        entry_thesis=short_thesis,
    )
    assert long_hold.candidate is Action.HOLD
    assert short_hold.candidate is Action.HOLD


def test_entry_thesis_survives_restart_payload() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="UP")
    position = OpenPosition(
        symbol="BTCUSDT",
        exchange="binance_usdm",
        side=Side.LONG,
        quantity=0.01,
        entry_price=100.0,
        entry_timestamp_ms=1_000,
        entry_notional=1000.0,
        entry_fee=0.5,
        fees_paid=0.5,
        directional_thesis=thesis,
    )
    restored = OpenPosition.from_payload(position.to_payload())
    assert restored.directional_thesis is not None
    assert restored.directional_thesis["entry_primary_horizon"] == "5m"
    assert restored.directional_thesis["entry_p_up_1m"] == 0.84


def test_dashboard_thesis_serialization() -> None:
    settings = _settings()
    policy = DirectionalPolicy(settings)
    thesis = _open(policy, side="DOWN")
    short = make_market_state(symbol="BTCUSDT", price=100.0, side=Side.SHORT, quantity=0.01)
    hold = policy.evaluate(
        short, _result(p_1m=0.28, p_5m=0.30, p_10m=0.34, p_30m=0.40), entry_thesis=thesis
    )
    blob = hold.position_thesis
    assert blob is not None
    for key in (
        "primary_horizon",
        "entry_p_up",
        "current_p_up",
        "entry_consensus",
        "current_consensus",
        "entry_thesis_score",
        "current_thesis_score",
        "thesis_decay",
        "primary_horizon_elapsed_multiple",
        "momentum_strengthening",
        "momentum_weakening",
        "reversal_risk",
        "mfe_bps",
        "mae_bps",
        "profit_giveback_bps",
        "recommended",
        "reason",
    ):
        assert key in blob
    assert blob["recommended"] == "HOLD"
    assert hold.entry_thesis["entry_question_version"] == "q.directional.v1"
