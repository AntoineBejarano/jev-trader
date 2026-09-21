from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.execution.attribution import profit_giveback_bps
from src.execution.pricing import (
    BPS,
    conservative_mark,
    excursion_bps,
    long_gross_pnl,
    short_gross_pnl,
)
from src.market.models import PositionState, Side

FLAT_KEY_SEP = ":"


def _key(symbol: str, exchange: str) -> tuple[str, str]:
    return symbol.upper(), exchange


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def utc_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).date().isoformat()


@dataclass
class OpenPosition:
    symbol: str
    exchange: str
    side: Side
    quantity: float
    entry_price: float
    entry_timestamp_ms: int
    entry_notional: float
    entry_fee: float
    fees_paid: float
    entry_decision_id: str | None = None
    entry_source: str = "JEV"
    mfe_bps: float = 0.0
    mae_bps: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_bps: float = 0.0
    entry_horizon_seconds: int | None = None
    entry_net_ev_bps: float | None = None
    entry_bid: float | None = None
    entry_ask: float | None = None
    entry_slippage_cost: float = 0.0
    entry_spread_cost: float = 0.0

    def mark(self, *, bid: float, ask: float, now_ms: int | None = None) -> None:
        del now_ms
        px = conservative_mark(side=self.side.value, bid=bid, ask=ask)
        self.current_price = px
        if self.side is Side.LONG:
            self.unrealized_pnl = long_gross_pnl(self.quantity, self.entry_price, px)
        else:
            self.unrealized_pnl = short_gross_pnl(self.quantity, self.entry_price, px)
        self.unrealized_pnl_bps = excursion_bps(
            side=self.side.value, entry_price=self.entry_price, mark_price=px
        )
        self.mfe_bps = max(self.mfe_bps, self.unrealized_pnl_bps)
        self.mae_bps = min(self.mae_bps, self.unrealized_pnl_bps)

    def age_seconds(self, now_ms: int) -> float:
        return max(0.0, (now_ms - self.entry_timestamp_ms) / 1000.0)

    def to_state(self, *, now_ms: int, realized_pnl: float) -> PositionState:
        net = realized_pnl + self.unrealized_pnl - self.fees_paid
        return PositionState(
            symbol=self.symbol,
            exchange=self.exchange,
            side=self.side,
            quantity=self.quantity,
            entry_price=self.entry_price,
            current_price=self.current_price,
            entry_timestamp=_iso(self.entry_timestamp_ms),
            entry_notional=self.entry_notional,
            age_seconds=self.age_seconds(now_ms),
            unrealized_pnl=self.unrealized_pnl,
            unrealized_pnl_bps=self.unrealized_pnl_bps,
            realized_pnl=realized_pnl,
            gross_pnl=self.unrealized_pnl,
            fees_paid=self.fees_paid,
            net_pnl=net,
            max_favorable_excursion_bps=self.mfe_bps,
            max_adverse_excursion_bps=self.mae_bps,
            max_favorable_excursion=self.mfe_bps,
            max_adverse_excursion=self.mae_bps,
            profit_giveback_bps=profit_giveback_bps(self.mfe_bps, self.unrealized_pnl_bps),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_timestamp_ms": self.entry_timestamp_ms,
            "entry_notional": self.entry_notional,
            "entry_fee": self.entry_fee,
            "fees_paid": self.fees_paid,
            "entry_decision_id": self.entry_decision_id,
            "entry_source": self.entry_source,
            "mfe_bps": self.mfe_bps,
            "mae_bps": self.mae_bps,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_bps": self.unrealized_pnl_bps,
            "entry_horizon_seconds": self.entry_horizon_seconds,
            "entry_net_ev_bps": self.entry_net_ev_bps,
            "entry_bid": self.entry_bid,
            "entry_ask": self.entry_ask,
            "entry_slippage_cost": self.entry_slippage_cost,
            "entry_spread_cost": self.entry_spread_cost,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> OpenPosition:
        return cls(
            symbol=str(payload["symbol"]),
            exchange=str(payload["exchange"]),
            side=Side(str(payload["side"])),
            quantity=float(payload["quantity"]),
            entry_price=float(payload["entry_price"]),
            entry_timestamp_ms=int(payload["entry_timestamp_ms"]),
            entry_notional=float(payload["entry_notional"]),
            entry_fee=float(payload["entry_fee"]),
            fees_paid=float(payload["fees_paid"]),
            entry_decision_id=payload.get("entry_decision_id"),
            entry_source=str(payload.get("entry_source") or "JEV"),
            mfe_bps=float(payload.get("mfe_bps") or 0.0),
            mae_bps=float(payload.get("mae_bps") or 0.0),
            current_price=float(payload.get("current_price") or payload["entry_price"]),
            unrealized_pnl=float(payload.get("unrealized_pnl") or 0.0),
            unrealized_pnl_bps=float(payload.get("unrealized_pnl_bps") or 0.0),
            entry_horizon_seconds=(
                None
                if payload.get("entry_horizon_seconds") is None
                else int(payload["entry_horizon_seconds"])
            ),
            entry_net_ev_bps=(
                None
                if payload.get("entry_net_ev_bps") is None
                else float(payload["entry_net_ev_bps"])
            ),
            entry_bid=(
                None if payload.get("entry_bid") is None else float(payload["entry_bid"])
            ),
            entry_ask=(
                None if payload.get("entry_ask") is None else float(payload["entry_ask"])
            ),
            entry_slippage_cost=float(payload.get("entry_slippage_cost") or 0.0),
            entry_spread_cost=float(payload.get("entry_spread_cost") or 0.0),
        )


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    symbol: str
    exchange: str
    side: Side
    quantity: float
    entry_price: float
    exit_price: float
    entry_notional: float
    entry_timestamp_ms: int
    exit_timestamp_ms: int
    holding_seconds: float
    gross_pnl: float
    entry_fee: float
    exit_fee: float
    total_fees: float
    slippage_cost: float
    funding_cost: float
    net_pnl: float
    gross_pnl_bps: float
    net_pnl_bps: float
    mfe_bps: float
    mae_bps: float
    entry_jev_decision_id: str | None
    exit_jev_decision_id: str | None
    entry_source: str
    exit_source: str
    entry_horizon_seconds: int | None = None
    entry_net_ev_bps: float | None = None
    exit_reason: str | None = None
    entry_slippage_cost: float = 0.0
    exit_slippage_cost: float = 0.0
    total_slippage_cost: float = 0.0
    entry_spread_cost: float = 0.0
    exit_spread_cost: float = 0.0
    total_spread_cost: float = 0.0
    market_move_pnl: float = 0.0
    attribution_residual: float = 0.0
    funding_accounting: str = "disabled_zero"


@dataclass
class PaperAccount:
    """Futures-style paper book: cash = start - fees + realized gross; equity includes uPnL."""

    currency: str
    starting_balance: float
    cash: float
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    last_order_ms: int | None = None
    day: str = ""
    day_start_equity: float = 0.0
    day_peak_equity: float = 0.0
    day_realized_pnl: float = 0.0
    day_fees: float = 0.0
    positions: dict[tuple[str, str], OpenPosition] = field(default_factory=dict)
    last_flat_at: dict[str, int] = field(default_factory=dict)
    order_timestamps_ms: deque[int] = field(default_factory=deque)
    closed_trades: list[ClosedTrade] = field(default_factory=list)

    @classmethod
    def new(cls, *, starting_balance: float, currency: str, now_ms: int) -> PaperAccount:
        day = utc_day(now_ms)
        return cls(
            currency=currency,
            starting_balance=starting_balance,
            cash=starting_balance,
            peak_equity=starting_balance,
            day=day,
            day_start_equity=starting_balance,
            day_peak_equity=starting_balance,
        )

    @property
    def unrealized_pnl(self) -> float:
        return sum(position.unrealized_pnl for position in self.positions.values())

    @property
    def equity(self) -> float:
        return self.starting_balance + self.realized_pnl + self.unrealized_pnl - self.fees_paid

    @property
    def net_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid

    @property
    def drawdown(self) -> float:
        return max(0.0, self.peak_equity - self.equity)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return self.drawdown / self.peak_equity

    @property
    def day_net_pnl(self) -> float:
        return self.equity - self.day_start_equity

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    def last_flat_ms(self, symbol: str, exchange: str) -> int | None:
        return self.last_flat_at.get(f"{exchange}{FLAT_KEY_SEP}{symbol}")

    def get_position(self, symbol: str, exchange: str) -> OpenPosition | None:
        return self.positions.get(_key(symbol, exchange))

    def orders_in_last_minute(self, now_ms: int) -> int:
        cutoff = now_ms - 60_000
        while self.order_timestamps_ms and self.order_timestamps_ms[0] < cutoff:
            self.order_timestamps_ms.popleft()
        return len(self.order_timestamps_ms)

    def roll_day(self, now_ms: int) -> None:
        today = utc_day(now_ms)
        if self.day == today:
            return
        self.day = today
        self.day_start_equity = self.equity
        self.day_peak_equity = self.equity
        self.day_realized_pnl = 0.0
        self.day_fees = 0.0

    def mark(self, *, symbol: str, exchange: str, bid: float, ask: float, now_ms: int) -> None:
        self.roll_day(now_ms)
        position = self.get_position(symbol, exchange)
        if position is not None:
            position.mark(bid=bid, ask=ask, now_ms=now_ms)
        self._refresh_peaks()

    def position_state(
        self, symbol: str, exchange: str, *, now_ms: int, current_price: float | None
    ) -> PositionState:
        position = self.get_position(symbol, exchange)
        if position is None:
            return PositionState(
                symbol=symbol,
                exchange=exchange,
                side=Side.FLAT,
                current_price=current_price,
                realized_pnl=self.realized_pnl,
                fees_paid=self.fees_paid,
                net_pnl=self.net_pnl,
            )
        return position.to_state(now_ms=now_ms, realized_pnl=self.realized_pnl)

    def open_position(
        self,
        *,
        symbol: str,
        exchange: str,
        side: Side,
        quantity: float,
        fill_price: float,
        fee: float,
        now_ms: int,
        decision_id: str | None,
        source: str,
        bid: float,
        ask: float,
        entry_slippage_cost: float = 0.0,
        entry_spread_cost: float = 0.0,
    ) -> OpenPosition:
        if self.get_position(symbol, exchange) is not None:
            raise RuntimeError(f"already positioned in {exchange}:{symbol}")
        if side is Side.FLAT or quantity <= 0 or fill_price <= 0:
            raise ValueError("invalid paper open")
        position = OpenPosition(
            symbol=symbol,
            exchange=exchange,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            entry_timestamp_ms=now_ms,
            entry_notional=quantity * fill_price,
            entry_fee=fee,
            fees_paid=fee,
            entry_decision_id=decision_id,
            entry_source=source,
            entry_bid=bid,
            entry_ask=ask,
            entry_slippage_cost=entry_slippage_cost,
            entry_spread_cost=entry_spread_cost,
        )
        position.mark(bid=bid, ask=ask, now_ms=now_ms)
        self.positions[_key(symbol, exchange)] = position
        self.fees_paid += fee
        self.cash -= fee
        self.day_fees += fee
        self._note_order(now_ms)
        self._refresh_peaks()
        return position

    def close_position(
        self,
        *,
        symbol: str,
        exchange: str,
        fill_price: float,
        fee: float,
        now_ms: int,
        decision_id: str | None,
        source: str,
        slippage_cost: float,
        exit_bid: float | None = None,
        exit_ask: float | None = None,
        attribution: dict[str, float] | None = None,
        funding_cost: float = 0.0,
        funding_accounting: str = "disabled_zero",
    ) -> ClosedTrade:
        del exit_bid, exit_ask
        position = self.positions.pop(_key(symbol, exchange), None)
        if position is None:
            raise RuntimeError(f"no open position for {exchange}:{symbol}")
        if position.side is Side.LONG:
            gross = long_gross_pnl(position.quantity, position.entry_price, fill_price)
        else:
            gross = short_gross_pnl(position.quantity, position.entry_price, fill_price)
        total_fees = position.fees_paid + fee
        net = gross - total_fees - funding_cost
        notional = position.entry_notional
        attr = attribution or {}
        entry_slip = float(attr.get("entry_slippage_cost") or position.entry_slippage_cost)
        exit_slip = float(attr.get("exit_slippage_cost") or slippage_cost)
        total_slip = float(attr.get("total_slippage_cost") or (entry_slip + exit_slip))
        entry_spread = float(attr.get("entry_spread_cost") or position.entry_spread_cost)
        exit_spread = float(attr.get("exit_spread_cost") or 0.0)
        total_spread = float(attr.get("total_spread_cost") or (entry_spread + exit_spread))
        market_move = float(attr.get("market_move_pnl") or 0.0)
        residual = float(attr.get("residual") or 0.0)
        trade = ClosedTrade(
            symbol=symbol,
            exchange=exchange,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=fill_price,
            entry_notional=notional,
            entry_timestamp_ms=position.entry_timestamp_ms,
            exit_timestamp_ms=now_ms,
            holding_seconds=position.age_seconds(now_ms),
            gross_pnl=gross,
            entry_fee=position.entry_fee,
            exit_fee=fee,
            total_fees=total_fees,
            slippage_cost=total_slip,
            funding_cost=funding_cost,
            net_pnl=net,
            gross_pnl_bps=0.0 if notional == 0 else gross / notional * BPS,
            net_pnl_bps=0.0 if notional == 0 else net / notional * BPS,
            mfe_bps=position.mfe_bps,
            mae_bps=position.mae_bps,
            entry_jev_decision_id=position.entry_decision_id,
            exit_jev_decision_id=decision_id,
            entry_source=position.entry_source,
            exit_source=source,
            entry_horizon_seconds=position.entry_horizon_seconds,
            entry_net_ev_bps=position.entry_net_ev_bps,
            entry_slippage_cost=entry_slip,
            exit_slippage_cost=exit_slip,
            total_slippage_cost=total_slip,
            entry_spread_cost=entry_spread,
            exit_spread_cost=exit_spread,
            total_spread_cost=total_spread,
            market_move_pnl=market_move,
            attribution_residual=residual,
            funding_accounting=funding_accounting,
        )
        self.realized_pnl += gross
        self.fees_paid += fee
        self.cash += gross - fee
        self.day_realized_pnl += gross
        self.day_fees += fee
        self.closed_trades.append(trade)
        self.last_flat_at[f"{exchange}{FLAT_KEY_SEP}{symbol}"] = now_ms
        if net < 0:
            self.consecutive_losses += 1
        elif net > 0:
            self.consecutive_losses = 0
        self._note_order(now_ms)
        self._refresh_peaks()
        return trade

    def _note_order(self, now_ms: int) -> None:
        self.last_order_ms = now_ms
        self.order_timestamps_ms.append(now_ms)
        self.orders_in_last_minute(now_ms)

    def _refresh_peaks(self) -> None:
        equity = self.equity
        self.peak_equity = max(self.peak_equity, equity)
        self.day_peak_equity = max(self.day_peak_equity, equity)

    def equity_snapshot(self, now_ms: int) -> dict[str, Any]:
        return {
            "timestamp": _iso(now_ms),
            "timestamp_ms": now_ms,
            "currency": self.currency,
            "starting_balance": self.starting_balance,
            "balance": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees": self.fees_paid,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "drawdown": self.drawdown,
            "drawdown_pct": self.drawdown_pct,
            "net_pnl": self.net_pnl,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "starting_balance": self.starting_balance,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "peak_equity": self.peak_equity,
            "consecutive_losses": self.consecutive_losses,
            "last_order_ms": self.last_order_ms,
            "day": self.day,
            "day_start_equity": self.day_start_equity,
            "day_peak_equity": self.day_peak_equity,
            "day_realized_pnl": self.day_realized_pnl,
            "day_fees": self.day_fees,
            "last_flat_at": self.last_flat_at,
            "order_timestamps_ms": list(self.order_timestamps_ms),
            "positions": [position.to_payload() for position in self.positions.values()],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PaperAccount:
        account = cls(
            currency=str(payload["currency"]),
            starting_balance=float(payload["starting_balance"]),
            cash=float(payload["cash"]),
            realized_pnl=float(payload.get("realized_pnl") or 0.0),
            fees_paid=float(payload.get("fees_paid") or 0.0),
            peak_equity=float(payload.get("peak_equity") or payload["starting_balance"]),
            consecutive_losses=int(payload.get("consecutive_losses") or 0),
            last_order_ms=payload.get("last_order_ms"),
            day=str(payload.get("day") or ""),
            day_start_equity=float(payload.get("day_start_equity") or payload["starting_balance"]),
            day_peak_equity=float(payload.get("day_peak_equity") or payload["starting_balance"]),
            day_realized_pnl=float(payload.get("day_realized_pnl") or 0.0),
            day_fees=float(payload.get("day_fees") or 0.0),
            last_flat_at={
                str(key): int(value)
                for key, value in dict(payload.get("last_flat_at") or {}).items()
            },
            order_timestamps_ms=deque(
                int(value) for value in list(payload.get("order_timestamps_ms") or [])
            ),
        )
        for raw in payload.get("positions") or []:
            if not isinstance(raw, dict):
                continue
            position = OpenPosition.from_payload(raw)
            account.positions[_key(position.symbol, position.exchange)] = position
        return account
