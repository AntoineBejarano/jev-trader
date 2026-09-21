from __future__ import annotations

import pytest
from src.decision.actions import Action
from src.execution.paper import PaperExecutionEngine
from src.execution.pricing import apply_adverse_slippage, quote_market_fill
from src.market.models import Side
from src.portfolio.analytics import compute_paper_metrics
from tests.helpers import make_market_state
from tests.paper_helpers import execute_action, fresh_account, paper_settings


def test_open_long_executes_from_ask_with_adverse_slippage() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    report = execute_action(Action.OPEN_LONG, account, state, now_ms=1_000_000, decision_id="d1")
    assert report is not None
    fill = report.fills[0]
    expected = apply_adverse_slippage(100.10, is_buy=True, slippage_bps=1.0)
    assert fill.is_buy is True
    assert fill.fill_price == pytest.approx(expected)
    assert fill.fill_price > fill.best_ask or fill.fill_price == pytest.approx(expected)
    assert fill.fill_price > 100.10
    assert account.get_position("ETHUSDT", "binance_usdm") is not None
    assert account.get_position("ETHUSDT", "binance_usdm").side is Side.LONG


def test_open_short_executes_from_bid_with_adverse_slippage() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    report = execute_action(Action.OPEN_SHORT, account, state, now_ms=1_000_000, decision_id="d1")
    assert report is not None
    fill = report.fills[0]
    assert fill.is_buy is False
    assert fill.fill_price < 100.0
    assert report.opened is not None
    assert report.opened.side is Side.SHORT


def test_close_long_executes_from_bid() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    execute_action(Action.OPEN_LONG, account, state, now_ms=1_000_000, decision_id="open")
    later = make_market_state(price=101.0, bid=101.0, ask=101.10, side=Side.LONG, quantity=1)
    report = execute_action(Action.CLOSE, account, later, now_ms=1_030_000, decision_id="close")
    assert report is not None
    assert report.fills[0].is_buy is False
    assert report.fills[0].fill_price < 101.0
    assert report.closed is not None
    assert account.get_position("ETHUSDT", "binance_usdm") is None


def test_close_short_executes_from_ask() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    execute_action(Action.OPEN_SHORT, account, state, now_ms=1_000_000, decision_id="open")
    later = make_market_state(price=99.0, bid=99.0, ask=99.10, side=Side.SHORT, quantity=1)
    report = execute_action(Action.CLOSE, account, later, now_ms=1_030_000, decision_id="close")
    assert report is not None
    assert report.fills[0].is_buy is True
    assert report.fills[0].fill_price > 99.10
    assert report.closed is not None
    assert report.closed.side is Side.SHORT


def test_hold_does_not_execute() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    assert execute_action(Action.HOLD, account, state, now_ms=1, decision_id="h") is None
    assert execute_action(Action.NO_ACTION, account, state, now_ms=1, decision_id="n") is None
    assert account.open_position_count == 0


def test_long_to_short_reversal_is_close_then_open() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    execute_action(Action.OPEN_LONG, account, state, now_ms=1_000_000, decision_id="open")
    later = make_market_state(price=100.0, bid=100.0, ask=100.10, side=Side.LONG)
    report = execute_action(Action.REVERSE, account, later, now_ms=1_030_000, decision_id="rev")
    assert report is not None
    assert len(report.fills) == 2
    assert report.fills[0].intent is Action.CLOSE
    assert report.fills[1].intent is Action.OPEN_SHORT
    assert report.closed is not None
    assert report.opened is not None
    assert report.opened.side is Side.SHORT
    assert report.closed.total_fees == pytest.approx(
        report.closed.entry_fee + report.closed.exit_fee
    )
    assert account.fees_paid > report.closed.total_fees  # close fees + new open fee


def test_short_to_long_reversal() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    execute_action(Action.OPEN_SHORT, account, state, now_ms=1_000_000, decision_id="open")
    later = make_market_state(price=100.0, bid=100.0, ask=100.10, side=Side.SHORT)
    report = execute_action(Action.REVERSE, account, later, now_ms=1_030_000, decision_id="rev")
    assert report is not None
    assert report.opened is not None
    assert report.opened.side is Side.LONG


def test_entry_and_exit_each_charge_taker_fee() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    opened = execute_action(Action.OPEN_LONG, account, state, now_ms=1_000_000, decision_id="open")
    assert opened is not None
    entry_fee = opened.fills[0].fee
    assert entry_fee == pytest.approx(opened.fills[0].notional * 0.0004)
    later = make_market_state(price=100.0, bid=100.0, ask=100.10, side=Side.LONG)
    closed = execute_action(Action.CLOSE, account, later, now_ms=1_040_000, decision_id="close")
    assert closed is not None
    trade = closed.closed
    assert trade is not None
    assert trade.entry_fee == pytest.approx(entry_fee)
    assert trade.exit_fee > 0
    assert trade.total_fees == pytest.approx(trade.entry_fee + trade.exit_fee)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.total_fees)


def test_long_and_short_pnl_signs() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    execute_action(Action.OPEN_LONG, account, state, now_ms=1, decision_id="l")
    up = make_market_state(price=110.0, bid=110.0, ask=110.10, side=Side.LONG)
    long_close = execute_action(Action.CLOSE, account, up, now_ms=20_000, decision_id="lc")
    assert long_close is not None
    assert long_close.closed is not None
    assert long_close.closed.gross_pnl > 0

    execute_action(Action.OPEN_SHORT, account, state, now_ms=30_000, decision_id="s")
    down = make_market_state(price=90.0, bid=90.0, ask=90.10, side=Side.SHORT)
    short_close = execute_action(Action.CLOSE, account, down, now_ms=50_000, decision_id="sc")
    assert short_close is not None
    assert short_close.closed is not None
    assert short_close.closed.gross_pnl > 0


def test_mfe_mae_and_age() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    execute_action(Action.OPEN_LONG, account, state, now_ms=1_000_000, decision_id="open")
    pos = account.get_position("ETHUSDT", "binance_usdm")
    assert pos is not None
    account.mark(symbol="ETHUSDT", exchange="binance_usdm", bid=101.0, ask=101.10, now_ms=1_010_000)
    assert pos.mfe_bps > 0
    account.mark(symbol="ETHUSDT", exchange="binance_usdm", bid=99.0, ask=99.10, now_ms=1_020_000)
    assert pos.mae_bps < 0
    later = make_market_state(price=99.0, bid=99.0, ask=99.10, side=Side.LONG)
    closed = execute_action(Action.CLOSE, account, later, now_ms=1_030_000, decision_id="c")
    assert closed is not None
    assert closed.closed is not None
    assert closed.closed.mfe_bps > 0
    assert closed.closed.mae_bps < 0
    assert closed.closed.holding_seconds == pytest.approx(30.0)


def test_equity_peak_and_drawdown() -> None:
    account = fresh_account()
    assert account.equity == 10_000
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    execute_action(Action.OPEN_LONG, account, state, now_ms=1, decision_id="o")
    account.mark(symbol="ETHUSDT", exchange="binance_usdm", bid=110.0, ask=110.10, now_ms=2)
    assert account.equity > account.starting_balance - account.fees_paid
    peak = account.peak_equity
    account.mark(symbol="ETHUSDT", exchange="binance_usdm", bid=90.0, ask=90.10, now_ms=3)
    assert account.drawdown > 0
    assert account.peak_equity == peak
    metrics = compute_paper_metrics(account.closed_trades, max_drawdown=account.drawdown)
    assert metrics.total_trades == 0


def test_duplicate_decision_is_idempotent() -> None:
    account = fresh_account()
    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    first = execute_action(Action.OPEN_LONG, account, state, now_ms=1, decision_id="same")
    second = execute_action(
        Action.OPEN_LONG, account, state, now_ms=2, decision_id="same", already=True
    )
    assert first is not None
    assert second is not None
    assert second.skipped_duplicate is True
    assert account.open_position_count == 1


def test_quote_never_uses_mid() -> None:
    quote = quote_market_fill(
        intent=Action.OPEN_LONG, bid=100.0, ask=100.2, slippage_bps=1.0
    )
    mid = 100.1
    assert quote.raw_price == 100.2
    assert quote.fill_price != mid
    assert quote.fill_price > quote.raw_price


def test_paper_engine_sizes_to_max_notional() -> None:
    engine = PaperExecutionEngine(paper_settings())
    account = fresh_account()
    from src.decision.actions import ActionSource
    from src.risk.engine import RiskDecision

    state = make_market_state(price=100.0, bid=100.0, ask=100.10)
    report = engine.execute(
        account=account,
        risk=RiskDecision(Action.OPEN_LONG, Action.OPEN_LONG, None, ActionSource.JEV),
        symbol=state.symbol,
        exchange=state.exchange,
        bid=100.0,
        ask=100.10,
        now_ms=1,
        decision_id="n",
        already_processed=False,
    )
    assert report is not None
    assert report.fills[0].notional == pytest.approx(1_000.0)
