from __future__ import annotations

from statistics import median
from typing import Any

from src.execution.attribution import pnl_on_equity_bps, pnl_on_equity_pct, pnl_on_notional_bps
from src.market.models import Side
from src.portfolio.account import ClosedTrade
from src.portfolio.analytics import compute_paper_metrics


def scalp_session_metrics(
    trades: list[ClosedTrade],
    *,
    runtime_seconds: float,
    max_drawdown: float,
    starting_balance: float,
    position_notional: float,
) -> dict[str, Any]:
    base = compute_paper_metrics(trades, max_drawdown=max_drawdown)
    hours = runtime_seconds / 3600.0 if runtime_seconds > 0 else 0.0
    pnls = [trade.net_pnl for trade in trades]
    holds = [trade.holding_seconds for trade in trades]
    longs = [trade for trade in trades if trade.side is Side.LONG]
    shorts = [trade for trade in trades if trade.side is Side.SHORT]
    reversals = sum(1 for trade in trades if trade.exit_reason == "JEV_REVERSE")
    turnover = sum(trade.entry_notional for trade in trades)
    best = None if not trades else max(trades, key=lambda item: item.net_pnl)
    worst = None if not trades else min(trades, key=lambda item: item.net_pnl)

    def _rate(value: float) -> float | None:
        if hours <= 0:
            return None
        return value / hours

    return {
        "runtime_seconds": runtime_seconds,
        "trades": base.total_trades,
        "wins": base.winning_trades,
        "losses": base.losing_trades,
        "win_rate": base.win_rate,
        "gross_pnl": base.gross_pnl,
        "fees": base.total_fees,
        "slippage": base.total_slippage,
        "net_pnl": base.net_pnl,
        "session_net_pnl_usdt": base.net_pnl,
        "session_return_pct_on_starting_equity": pnl_on_equity_pct(
            base.net_pnl, starting_balance
        ),
        "session_return_bps_on_starting_equity": pnl_on_equity_bps(
            base.net_pnl, starting_balance
        ),
        "net_pnl_pct": pnl_on_equity_pct(base.net_pnl, starting_balance),
        "average_trade_return_bps_on_notional": (
            None
            if not trades
            else sum(pnl_on_notional_bps(t.net_pnl, t.entry_notional) for t in trades)
            / len(trades)
        ),
        "max_drawdown": base.max_drawdown,
        "turnover": turnover,
        "trades_per_hour": _rate(float(base.total_trades)),
        "turnover_per_hour": _rate(turnover),
        "gross_pnl_per_hour": _rate(base.gross_pnl),
        "fees_per_hour": _rate(base.total_fees),
        "net_pnl_per_hour": _rate(base.net_pnl),
        "average_trade_pnl": base.expectancy,
        "median_trade_pnl": None if not pnls else float(median(pnls)),
        "average_holding_seconds": base.average_holding_seconds,
        "median_holding_seconds": None if not holds else float(median(holds)),
        "profit_factor": base.profit_factor,
        "average_winner": base.average_winner,
        "average_loser": base.average_loser,
        "expectancy_per_trade": base.expectancy,
        "long_expectancy": None if not longs else sum(t.net_pnl for t in longs) / len(longs),
        "short_expectancy": None if not shorts else sum(t.net_pnl for t in shorts) / len(shorts),
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "reversals": reversals,
        "position_notional": position_notional,
        "metrics_kind": "REALIZED_TRADE_METRICS",
        "includes_open_position_mtm": False,
        "exposure_pct": None if starting_balance == 0 else position_notional / starting_balance,
        "best_trade_net": None if best is None else best.net_pnl,
        "worst_trade_net": None if worst is None else worst.net_pnl,
    }
