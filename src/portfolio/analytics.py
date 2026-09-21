from __future__ import annotations

from dataclasses import dataclass

from src.market.models import Side
from src.portfolio.account import ClosedTrade


@dataclass(frozen=True, slots=True)
class PaperMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float | None
    gross_pnl: float
    net_pnl: float
    total_fees: float
    total_slippage: float
    average_winner: float | None
    average_loser: float | None
    expectancy: float | None
    profit_factor: float | None
    max_drawdown: float
    average_holding_seconds: float | None
    long_pnl: float
    short_pnl: float


def compute_paper_metrics(
    trades: list[ClosedTrade], *, max_drawdown: float
) -> PaperMetrics:
    total = len(trades)
    winners = [trade for trade in trades if trade.net_pnl > 0]
    losers = [trade for trade in trades if trade.net_pnl < 0]
    gross = sum(trade.gross_pnl for trade in trades)
    net = sum(trade.net_pnl for trade in trades)
    fees = sum(trade.total_fees for trade in trades)
    slip = sum(trade.slippage_cost for trade in trades)
    win_sum = sum(trade.net_pnl for trade in winners)
    lose_sum = sum(trade.net_pnl for trade in losers)
    long_pnl = sum(trade.net_pnl for trade in trades if trade.side is Side.LONG)
    short_pnl = sum(trade.net_pnl for trade in trades if trade.side is Side.SHORT)
    hold = [trade.holding_seconds for trade in trades]
    profit_factor = None
    if lose_sum < 0:
        profit_factor = win_sum / abs(lose_sum) if lose_sum != 0 else None
    elif win_sum > 0:
        profit_factor = float("inf")
    return PaperMetrics(
        total_trades=total,
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=None if total == 0 else len(winners) / total,
        gross_pnl=gross,
        net_pnl=net,
        total_fees=fees,
        total_slippage=slip,
        average_winner=None if not winners else win_sum / len(winners),
        average_loser=None if not losers else lose_sum / len(losers),
        expectancy=None if total == 0 else net / total,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        average_holding_seconds=None if not hold else sum(hold) / len(hold),
        long_pnl=long_pnl,
        short_pnl=short_pnl,
    )
