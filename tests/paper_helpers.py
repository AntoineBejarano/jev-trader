from __future__ import annotations

from src.config.settings import Settings
from src.decision.actions import Action
from src.jev.models import (
    JevCallResult,
    JevCallStatus,
    JevSafetyAction,
    ParsedJevDecision,
    QuestionMode,
)
from src.market.models import MarketState, Side
from src.portfolio.account import PaperAccount


def paper_settings(**overrides: object) -> Settings:
    payload: dict[str, object] = {
        "openrouter_api_key": "test-key",
        "trading_mode": "PAPER",
        "paper_initial_balance": 10_000.0,
        "paper_currency": "USDT",
        "taker_fee_bps": 4.0,
        "maker_fee_bps": 2.0,
        "slippage_bps": 1.0,
        "max_position_notional": 1_000.0,
        "max_leverage": 1.0,
        "entry_threshold": 0.80,
        "evidence_threshold": 0.75,
        "hold_threshold": 0.55,
        "close_threshold": 0.70,
        "reverse_threshold": 0.88,
        "min_position_age_seconds": 15.0,
        "min_flat_cooldown_seconds": 10.0,
        "min_seconds_between_orders": 10.0,
        "max_loss_per_trade_pct": 1.0,
        "max_daily_loss_pct": 3.0,
        "max_daily_drawdown_pct": 3.0,
        "max_consecutive_losses": 5,
        "max_orders_per_minute": 6,
        "max_simultaneous_positions": 1,
    }
    payload.update(overrides)
    return Settings(_env_file=None, **payload)  # type: ignore[arg-type]


def fresh_account(now_ms: int = 1_000_000) -> PaperAccount:
    return PaperAccount.new(starting_balance=10_000.0, currency="USDT", now_ms=now_ms)


def jev_result(
    state: MarketState,
    *,
    status: JevCallStatus = JevCallStatus.OK,
    exposure: str | None = "FLAT",
    position_action: str | None = None,
    long_p: float = 0.2,
    short_p: float = 0.2,
    evidence: float = 0.2,
    reversal: float = 0.2,
    continuation: float = 0.2,
    safety: JevSafetyAction = JevSafetyAction.NONE,
) -> JevCallResult:
    mode = QuestionMode.ENTRY if state.position.side is Side.FLAT else QuestionMode.POSITION
    decision = None
    if status is JevCallStatus.OK:
        decision = ParsedJevDecision(
            mode=mode,
            question_set_version="q.v1.h300",
            model="typesafe/jev-1.13-20260917",
            latency_ms=10.0,
            long_probability=long_p,
            short_probability=short_p,
            continuation_probability=continuation,
            reversal_probability=reversal,
            evidence_probability=evidence,
            exposure_choice=exposure,
            position_action=position_action,
        )
    return JevCallResult(
        status=status,
        symbol=state.symbol,
        exchange=state.exchange,
        mode=mode,
        question_set_version="q.v1.h300",
        latency_ms=10.0,
        decision=decision,
        safety_action=safety,
    )


def execute_action(
    action: Action,
    account: PaperAccount,
    state: MarketState,
    *,
    now_ms: int,
    decision_id: str,
    already: bool = False,
    source: str = "JEV",
):
    from src.decision.actions import ActionSource
    from src.execution.paper import PaperExecutionEngine
    from src.risk.engine import RiskDecision

    engine = PaperExecutionEngine(paper_settings())
    risk = RiskDecision(action, action, None, ActionSource(source))
    bid = state.best_bid or 0.0
    ask = state.best_ask or 0.0
    return engine.execute(
        account=account,
        risk=risk,
        symbol=state.symbol,
        exchange=state.exchange,
        bid=bid,
        ask=ask,
        now_ms=now_ms,
        decision_id=decision_id,
        already_processed=already,
    )
