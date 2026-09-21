from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON, TypeDecorator


class JSONType(TypeDecorator[Any]):
    """JSON on SQLite, JSONB on PostgreSQL."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class RunAttributionMixin:
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    experiment_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class MarketSnapshotRow(RunAttributionMixin, Base):
    __tablename__ = "market_snapshots"
    __table_args__ = (
        Index("ix_snapshots_symbol_exchange_ts", "symbol", "exchange", "timestamp_ms"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_state: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    market_state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class JevDecisionRow(RunAttributionMixin, Base):
    __tablename__ = "jev_decisions"
    __table_args__ = (
        Index("ix_decisions_symbol_exchange_ts", "symbol", "exchange", "timestamp_ms"),
        Index("ix_decisions_outcomes_pending", "timestamp_ms"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    market_state: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    market_state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    question_set_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_jev_response: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    parsed_probabilities: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    candidate_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    final_action: Mapped[str] = mapped_column(String(32), nullable=False)
    blocked_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    scalp_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    jev_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    safety_action: Mapped[str] = mapped_column(String(32), nullable=False, default="NONE")
    current_position: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    price_after_10s: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_after_30s: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_after_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_after_3m: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_after_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_after_10m: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_after_10s: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_after_30s: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_after_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_after_3m: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_after_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_after_10m: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class OrderRow(RunAttributionMixin, Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_decision_id", "decision_id"),
        Index("ix_orders_symbol_exchange_ts", "symbol", "exchange", "timestamp_ms"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timestamp_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    decision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    intent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    fee: Mapped[float | None] = mapped_column(Float, nullable=True)
    notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FillRow(RunAttributionMixin, Base):
    __tablename__ = "fills"
    __table_args__ = (Index("ix_fills_decision_id", "decision_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timestamp_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    order_id: Mapped[str] = mapped_column(String(36), nullable=False)
    decision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PositionRow(RunAttributionMixin, Base):
    __tablename__ = "positions"
    __table_args__ = (
        Index("ix_positions_symbol_exchange", "symbol", "exchange", unique=True),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timestamp_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TradeRow(RunAttributionMixin, Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_symbol_exchange_ts", "symbol", "exchange", "opened_at"),
        Index("ix_trades_entry_decision", "entry_jev_decision_id"),
        Index("ix_trades_exit_decision", "exit_jev_decision_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    gross_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fees: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    net_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    entry_jev_decision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    exit_jev_decision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SystemEventRow(RunAttributionMixin, Base):
    __tablename__ = "system_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    exchange: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DailyMetricRow(Base):
    __tablename__ = "daily_metrics"
    __table_args__ = (
        Index("ix_daily_metrics_day_scope", "day", "symbol", "exchange", unique=True),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    day: Mapped[str] = mapped_column(String(10), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, default="PORTFOLIO")
    exchange: Mapped[str] = mapped_column(String(32), nullable=False, default="ALL")
    gross_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    net_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fees: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


PAPER_ACCOUNT_ID = "default"


class PaperAccountRow(Base):
    __tablename__ = "paper_accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    currency: Mapped[str] = mapped_column(String(16), nullable=False)
    starting_balance: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fees_paid: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    peak_equity: Mapped[float] = mapped_column(Float, nullable=False)
    drawdown: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class EquitySnapshotRow(RunAttributionMixin, Base):
    __tablename__ = "equity_snapshots"
    __table_args__ = (Index("ix_equity_snapshots_ts", "timestamp_ms"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, default="PORTFOLIO")
    exchange: Mapped[str] = mapped_column(String(32), nullable=False, default="ALL")
    balance: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fees: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    peak_equity: Mapped[float] = mapped_column(Float, nullable=False)
    drawdown: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ExperimentRunRow(Base):
    __tablename__ = "experiment_runs"
    __table_args__ = (
        Index("ix_experiment_runs_experiment_started", "experiment_id", "started_at_ms"),
    )

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    frozen: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    fingerprint: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DirectionalForecastRow(RunAttributionMixin, Base):
    __tablename__ = "directional_forecasts"
    __table_args__ = (
        Index("ix_directional_symbol_exchange_ts", "symbol", "exchange", "timestamp_ms"),
        Index("ix_directional_decision_id", "decision_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    decision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    question_version: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_state: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    forecast: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    policy: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    rolling_outcomes: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    candle_outcomes: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    persistence_outcomes: Mapped[dict[str, Any]] = mapped_column(
        JSONType, nullable=False, default=dict
    )
    counterfactuals: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    mature: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    research_sample_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    invalid_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
