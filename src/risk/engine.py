from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import Settings
from src.decision.actions import (
    EXECUTABLE_OPENS,
    Action,
    ActionSource,
)
from src.jev.models import JevCallResult, JevSafetyAction
from src.market.models import MarketState, Side
from src.portfolio.account import PaperAccount


@dataclass(frozen=True, slots=True)
class RiskDecision:
    candidate_action: Action
    final_action: Action
    blocked_reason: str | None
    source: ActionSource


class RiskEngine:
    """Hard limits. Jev cannot override these."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self,
        state: MarketState,
        candidate: Action,
        account: PaperAccount,
        result: JevCallResult,
        *,
        now_ms: int,
    ) -> RiskDecision:
        positioned = state.position.side is not Side.FLAT
        force_close = self._forced_close_reason(state, account, result)
        if force_close is not None:
            if positioned:
                return RiskDecision(candidate, Action.CLOSE, force_close, ActionSource.RISK_ENGINE)
            return RiskDecision(candidate, Action.NO_ACTION, force_close, ActionSource.RISK_ENGINE)

        if candidate is Action.HOLD:
            return RiskDecision(candidate, Action.HOLD, None, ActionSource.JEV)
        if candidate is Action.NO_ACTION:
            return RiskDecision(candidate, Action.NO_ACTION, None, ActionSource.JEV)
        if candidate is Action.REDUCE:
            return RiskDecision(
                candidate, Action.HOLD, "reduce_disabled", ActionSource.RISK_ENGINE
            )

        if candidate is Action.CLOSE:
            age_block = self._min_age_block(state)
            if age_block is not None:
                return RiskDecision(candidate, Action.HOLD, age_block, ActionSource.RISK_ENGINE)
            return RiskDecision(candidate, Action.CLOSE, None, ActionSource.JEV)

        if candidate is Action.REVERSE:
            age_block = self._min_age_block(state)
            if age_block is not None:
                return RiskDecision(candidate, Action.HOLD, age_block, ActionSource.RISK_ENGINE)
            open_block = self._open_block(state, account, now_ms)
            if open_block is not None:
                return RiskDecision(candidate, Action.CLOSE, open_block, ActionSource.RISK_ENGINE)
            return RiskDecision(candidate, Action.REVERSE, None, ActionSource.JEV)

        if candidate in EXECUTABLE_OPENS:
            open_block = self._open_block(state, account, now_ms)
            if open_block is not None:
                return RiskDecision(
                    candidate, Action.NO_ACTION, open_block, ActionSource.RISK_ENGINE
                )
            return RiskDecision(candidate, candidate, None, ActionSource.JEV)

        return RiskDecision(candidate, Action.NO_ACTION, "unknown_action", ActionSource.RISK_ENGINE)

    def _forced_close_reason(
        self,
        state: MarketState,
        account: PaperAccount,
        result: JevCallResult,
    ) -> str | None:
        if self.settings.kill_switch:
            return "kill_switch"
        if state.health.stale:
            return "stale_market_data"
        if result.safety_action is JevSafetyAction.FLATTEN:
            return "jev_unavailable"
        if state.position.side is Side.FLAT:
            return None
        if self._trade_loss_breached(state, account):
            return "max_loss_per_trade"
        if self._daily_loss_breached(account):
            return "max_daily_loss"
        if self._daily_drawdown_breached(account):
            return "max_daily_drawdown"
        duration = self.settings.max_position_duration_or_none
        if duration is not None and state.position.age_seconds >= duration:
            return "max_position_duration"
        return None

    def _open_block(self, state: MarketState, account: PaperAccount, now_ms: int) -> str | None:
        if self.settings.kill_switch:
            return "kill_switch"
        if not state.health.trading_allowed:
            reasons = state.health.reasons
            if "warmup_incomplete" in reasons:
                return "warmup_incomplete"
            if reasons:
                return reasons[0]
            return "trading_not_allowed"
        if state.health.stale:
            return "stale_market_data"
        if (
            account.open_position_count >= self.settings.max_simultaneous_positions
            and state.position.side is Side.FLAT
        ):
            return "max_simultaneous_positions"
        if account.consecutive_losses >= self.settings.max_consecutive_losses:
            return "max_consecutive_losses"
        if self._daily_loss_breached(account):
            return "max_daily_loss"
        if self._daily_drawdown_breached(account):
            return "max_daily_drawdown"
        last_order = account.last_order_ms
        min_gap_ms = int(self.settings.active_min_seconds_between_orders * 1000)
        if last_order is not None and now_ms - last_order < min_gap_ms:
            return "min_seconds_between_orders"
        last_flat = account.last_flat_ms(state.symbol, state.exchange)
        cooldown_ms = int(self.settings.min_flat_cooldown_seconds * 1000)
        if last_flat is not None and now_ms - last_flat < cooldown_ms:
            return "cooldown"
        if account.orders_in_last_minute(now_ms) >= self.settings.max_orders_per_minute:
            return "max_orders_per_minute"
        if account.cash <= 0:
            return "insufficient_cash"
        max_notional = self.settings.execution_notional
        if max_notional <= 0:
            return "max_position_notional"
        if max_notional > account.cash * self.settings.max_leverage:
            return "max_leverage"
        return None

    def _min_age_block(self, state: MarketState) -> str | None:
        if state.position.age_seconds < self.settings.active_min_position_age_seconds:
            return "min_position_age"
        return None

    def _trade_loss_breached(self, state: MarketState, account: PaperAccount) -> bool:
        position = account.get_position(state.symbol, state.exchange)
        if position is None or position.entry_notional <= 0:
            return False
        limit = self.settings.max_loss_per_trade_pct / 100.0
        return position.unrealized_pnl <= -limit * position.entry_notional

    def _daily_loss_breached(self, account: PaperAccount) -> bool:
        limit = self.settings.max_daily_loss_pct / 100.0 * account.starting_balance
        return account.day_net_pnl <= -limit

    def _daily_drawdown_breached(self, account: PaperAccount) -> bool:
        if account.day_peak_equity <= 0:
            return False
        dd = (account.day_peak_equity - account.equity) / account.day_peak_equity
        return dd >= self.settings.max_daily_drawdown_pct / 100.0
