from __future__ import annotations

from src.market.buffer import Trade


def signed_qty(trade: Trade) -> float:
    return trade.qty if trade.is_aggressor_buy else -trade.qty


def window_cvd(trades: list[Trade]) -> float:
    return sum(signed_qty(trade) for trade in trades)


def aggressive_flow(trades: list[Trade]) -> float | None:
    buy = sum(trade.qty for trade in trades if trade.is_aggressor_buy)
    sell = sum(trade.qty for trade in trades if not trade.is_aggressor_buy)
    total = buy + sell
    if total <= 0:
        return None
    return (buy - sell) / total


def trade_velocity(trades: list[Trade], window_seconds: float) -> float | None:
    if window_seconds <= 0:
        return None
    return len(trades) / window_seconds


def volume_velocity(trades: list[Trade], window_seconds: float) -> float | None:
    if window_seconds <= 0:
        return None
    return sum(trade.qty for trade in trades) / window_seconds


def liquidation_notional(trades: list[Trade], *, aggressor_buy: bool) -> float:
    return sum(
        trade.qty * trade.price
        for trade in trades
        if trade.is_liquidation and trade.is_aggressor_buy is aggressor_buy
    )
