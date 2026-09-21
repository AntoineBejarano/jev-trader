from __future__ import annotations

from src.decision.actions import Action, ActionSource
from src.jev.models import JevCallStatus, JevSafetyAction
from src.market.models import Side
from src.risk.engine import RiskEngine
from tests.helpers import make_market_state
from tests.paper_helpers import execute_action, fresh_account, jev_result, paper_settings


def test_warmup_blocks_open() -> None:
    risk = RiskEngine(paper_settings())
    account = fresh_account()
    state = make_market_state(warmup_ready=False, trading_allowed=False)
    result = jev_result(state, exposure="LONG", long_p=0.9, evidence=0.9)
    decided = risk.evaluate(state, Action.OPEN_LONG, account, result, now_ms=1_000_000)
    assert decided.final_action is Action.NO_ACTION
    assert decided.blocked_reason == "warmup_incomplete"
    assert decided.source is ActionSource.RISK_ENGINE


def test_reduce_disabled() -> None:
    risk = RiskEngine(paper_settings())
    account = fresh_account()
    state = make_market_state(side=Side.LONG, age_seconds=30)
    result = jev_result(state, position_action="REDUCE")
    decided = risk.evaluate(state, Action.REDUCE, account, result, now_ms=1_000_000)
    assert decided.final_action is Action.HOLD
    assert decided.blocked_reason == "reduce_disabled"


def test_min_age_blocks_close() -> None:
    risk = RiskEngine(paper_settings())
    account = fresh_account()
    state = make_market_state(side=Side.LONG, age_seconds=5)
    result = jev_result(state, position_action="CLOSE")
    decided = risk.evaluate(state, Action.CLOSE, account, result, now_ms=1_000_000)
    assert decided.final_action is Action.HOLD
    assert decided.blocked_reason == "min_position_age"


def test_stale_while_positioned_forces_close() -> None:
    risk = RiskEngine(paper_settings())
    account = fresh_account()
    state = make_market_state(side=Side.LONG, age_seconds=40, stale=True)
    result = jev_result(state, status=JevCallStatus.SKIPPED_STALE, safety=JevSafetyAction.NONE)
    decided = risk.evaluate(state, Action.NO_ACTION, account, result, now_ms=1_000_000)
    assert decided.final_action is Action.CLOSE
    assert decided.blocked_reason == "stale_market_data"
    assert decided.source is ActionSource.RISK_ENGINE


def test_jev_failure_flatten() -> None:
    risk = RiskEngine(paper_settings())
    account = fresh_account()
    state = make_market_state(side=Side.LONG, age_seconds=40)
    result = jev_result(
        state, status=JevCallStatus.TIMEOUT, safety=JevSafetyAction.FLATTEN
    )
    decided = risk.evaluate(state, Action.NO_ACTION, account, result, now_ms=1_000_000)
    assert decided.final_action is Action.CLOSE
    assert decided.blocked_reason == "jev_unavailable"


def test_max_loss_per_trade_closes() -> None:
    risk = RiskEngine(paper_settings(max_loss_per_trade_pct=0.01))
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    execute_action(Action.OPEN_LONG, account, state, now_ms=1, decision_id="o")
    account.mark(symbol="ETHUSDT", exchange="binance_usdm", bid=50.0, ask=50.10, now_ms=20_000)
    positioned = make_market_state(side=Side.LONG, age_seconds=40, price=50.0, bid=50.0, ask=50.10)
    positioned = positioned.model_copy(
        update={
            "position": account.position_state(
                "ETHUSDT", "binance_usdm", now_ms=20_000, current_price=50.0
            )
        }
    )
    result = jev_result(positioned, position_action="HOLD")
    decided = risk.evaluate(positioned, Action.HOLD, account, result, now_ms=20_000)
    assert decided.final_action is Action.CLOSE
    assert decided.blocked_reason == "max_loss_per_trade"


def test_cooldown_blocks_reentry() -> None:
    risk = RiskEngine(
        paper_settings(min_seconds_between_orders=1.0, min_flat_cooldown_seconds=10.0)
    )
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    execute_action(Action.OPEN_LONG, account, state, now_ms=1_000_000, decision_id="o")
    later = make_market_state(price=100.0, bid=100.0, ask=100.10, side=Side.LONG)
    execute_action(Action.CLOSE, account, later, now_ms=1_020_000, decision_id="c")
    flat = make_market_state(price=100.0, bid=100.0, ask=100.10)
    result = jev_result(flat, exposure="LONG", long_p=0.9, evidence=0.9)
    decided = risk.evaluate(flat, Action.OPEN_LONG, account, result, now_ms=1_025_000)
    assert decided.final_action is Action.NO_ACTION
    assert decided.blocked_reason == "cooldown"
