from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import Settings
from src.decision.actions import Action, ActionSource
from src.execution.attribution import attribute_round_trip
from src.execution.funding import funding_cost_usdt
from src.execution.pricing import fee_on_notional, quote_market_fill
from src.market.models import Side
from src.portfolio.account import ClosedTrade, OpenPosition, PaperAccount
from src.risk.engine import RiskDecision


@dataclass(frozen=True, slots=True)
class PaperFill:
    intent: Action
    is_buy: bool
    quantity: float
    best_bid: float
    best_ask: float
    raw_price: float
    fill_price: float
    spread_bps: float
    slippage_bps: float
    slippage_cost: float
    notional: float
    fee: float
    side: str


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    decision_id: str
    source: ActionSource
    fills: tuple[PaperFill, ...]
    opened: OpenPosition | None
    closed: ClosedTrade | None
    remained_flat_after_reverse: bool = False
    skipped_duplicate: bool = False


class PaperExecutionEngine:
    """Simulated USD-M fills from the live book. Never uses mid. Never sends exchange orders."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def execute(
        self,
        *,
        account: PaperAccount,
        risk: RiskDecision,
        symbol: str,
        exchange: str,
        bid: float,
        ask: float,
        now_ms: int,
        decision_id: str,
        already_processed: bool,
    ) -> ExecutionReport | None:
        if already_processed:
            return ExecutionReport(
                decision_id=decision_id,
                source=risk.source,
                fills=(),
                opened=None,
                closed=None,
                skipped_duplicate=True,
            )
        action = risk.final_action
        if action is Action.HOLD or action is Action.NO_ACTION or action is Action.REDUCE:
            return None
        if action is Action.OPEN_LONG:
            opened = self._open(
                account, Side.LONG, symbol, exchange, bid, ask, now_ms, decision_id, risk.source
            )
            return ExecutionReport(decision_id, risk.source, (opened[0],), opened[1], None)
        if action is Action.OPEN_SHORT:
            opened = self._open(
                account, Side.SHORT, symbol, exchange, bid, ask, now_ms, decision_id, risk.source
            )
            return ExecutionReport(decision_id, risk.source, (opened[0],), opened[1], None)
        if action is Action.CLOSE:
            closed_fill, trade = self._close(
                account, symbol, exchange, bid, ask, now_ms, decision_id, risk.source
            )
            return ExecutionReport(decision_id, risk.source, (closed_fill,), None, trade)
        if action is Action.REVERSE:
            return self._reverse(account, risk, symbol, exchange, bid, ask, now_ms, decision_id)
        return None

    def _open(
        self,
        account: PaperAccount,
        side: Side,
        symbol: str,
        exchange: str,
        bid: float,
        ask: float,
        now_ms: int,
        decision_id: str,
        source: ActionSource,
    ) -> tuple[PaperFill, OpenPosition]:
        intent = Action.OPEN_LONG if side is Side.LONG else Action.OPEN_SHORT
        fill = self._build_fill(intent, bid=bid, ask=ask, current_long=False)
        touch = ask if side is Side.LONG else bid
        entry_spread = abs(((bid + ask) / 2.0) - touch) * fill.quantity
        position = account.open_position(
            symbol=symbol,
            exchange=exchange,
            side=side,
            quantity=fill.quantity,
            fill_price=fill.fill_price,
            fee=fill.fee,
            now_ms=now_ms,
            decision_id=decision_id,
            source=source.value,
            bid=bid,
            ask=ask,
            entry_slippage_cost=fill.slippage_cost,
            entry_spread_cost=entry_spread,
        )
        return fill, position

    def _close(
        self,
        account: PaperAccount,
        symbol: str,
        exchange: str,
        bid: float,
        ask: float,
        now_ms: int,
        decision_id: str,
        source: ActionSource,
    ) -> tuple[PaperFill, ClosedTrade]:
        position = account.get_position(symbol, exchange)
        if position is None:
            raise RuntimeError("CLOSE without an open paper position")
        fill = self._build_fill(
            Action.CLOSE,
            bid=bid,
            ask=ask,
            current_long=position.side is Side.LONG,
            quantity=position.quantity,
        )
        funding = funding_cost_usdt(
            enabled=self.settings.paper_funding_enabled,
            side=position.side.value,
            notional=position.entry_notional,
            funding_rate=None,
            entry_timestamp_ms=position.entry_timestamp_ms,
            exit_timestamp_ms=now_ms,
            next_funding_time_ms=None,
        )
        attribution = None
        if position.entry_bid is not None and position.entry_ask is not None:
            breakdown = attribute_round_trip(
                side=position.side,
                quantity=position.quantity,
                entry_fill=position.entry_price,
                entry_bid=position.entry_bid,
                entry_ask=position.entry_ask,
                exit_fill=fill.fill_price,
                exit_bid=bid,
                exit_ask=ask,
                entry_fee=position.entry_fee,
                exit_fee=fill.fee,
                funding_cost=funding,
            )
            attribution = breakdown.as_dict()
        trade = account.close_position(
            symbol=symbol,
            exchange=exchange,
            fill_price=fill.fill_price,
            fee=fill.fee,
            now_ms=now_ms,
            decision_id=decision_id,
            source=source.value,
            slippage_cost=fill.slippage_cost,
            exit_bid=bid,
            exit_ask=ask,
            attribution=attribution,
            funding_cost=funding,
            funding_accounting=(
                "accrual_enabled" if self.settings.paper_funding_enabled else "disabled_zero"
            ),
        )
        return fill, trade

    def _reverse(
        self,
        account: PaperAccount,
        risk: RiskDecision,
        symbol: str,
        exchange: str,
        bid: float,
        ask: float,
        now_ms: int,
        decision_id: str,
    ) -> ExecutionReport:
        position = account.get_position(symbol, exchange)
        if position is None:
            raise RuntimeError("REVERSE without an open paper position")
        next_side = Side.SHORT if position.side is Side.LONG else Side.LONG
        close_fill, trade = self._close(
            account, symbol, exchange, bid, ask, now_ms, decision_id, risk.source
        )
        # Risk already approved the reverse. Re-check cash/leverage after realizing the close.
        if account.cash * self.settings.max_leverage < self.settings.execution_notional:
            return ExecutionReport(
                decision_id,
                ActionSource.RISK_ENGINE,
                (close_fill,),
                None,
                trade,
                remained_flat_after_reverse=True,
            )
        open_fill, opened = self._open(
            account, next_side, symbol, exchange, bid, ask, now_ms, decision_id, risk.source
        )
        return ExecutionReport(
            decision_id, risk.source, (close_fill, open_fill), opened, trade
        )

    def _build_fill(
        self,
        intent: Action,
        *,
        bid: float,
        ask: float,
        current_long: bool,
        quantity: float | None = None,
    ) -> PaperFill:
        quote = quote_market_fill(
            intent=intent,
            bid=bid,
            ask=ask,
            slippage_bps=self.settings.slippage_bps,
            current_long=current_long,
        )
        qty = (
            self.settings.execution_notional / quote.fill_price
            if quantity is None
            else quantity
        )
        notional = qty * quote.fill_price
        fee = fee_on_notional(notional, self.settings.execution_taker_fee_bps)
        liquidity_side = "BUY" if quote.is_buy else "SELL"
        return PaperFill(
            intent=intent,
            is_buy=quote.is_buy,
            quantity=qty,
            best_bid=quote.best_bid,
            best_ask=quote.best_ask,
            raw_price=quote.raw_price,
            fill_price=quote.fill_price,
            spread_bps=quote.spread_bps,
            slippage_bps=quote.slippage_bps,
            slippage_cost=quote.slippage_cost_per_unit * qty,
            notional=notional,
            fee=fee,
            side=liquidity_side,
        )
