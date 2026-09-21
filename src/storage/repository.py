from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.analytics.barriers import (
    LONG_AMBIGUOUS,
    LONG_NEITHER,
    LONG_STOP_FIRST,
    LONG_TARGET_FIRST,
    SHORT_AMBIGUOUS,
    SHORT_NEITHER,
    SHORT_STOP_FIRST,
    SHORT_TARGET_FIRST,
    long_barrier_outcome,
    short_barrier_outcome,
)
from src.analytics.calibration import (
    nonoverlapping_indices,
    summarize_long_race,
    summarize_short_race,
)
from src.analytics.counterfactual import book_from_market_state, hypothetical_side_net_pnl_bps
from src.decision.actions import Action
from src.directional.calibration import baseline_accuracy, summarize_binary
from src.directional.constants import HORIZON_LABELS, HORIZON_SECONDS
from src.directional.outcomes import enrich_forecast_row
from src.execution.paper import ExecutionReport, PaperFill
from src.jev.models import JevCallResult, JevCallStatus
from src.market.buffer import wall_clock_ms
from src.market.models import MarketState, Side
from src.portfolio.account import ClosedTrade, OpenPosition, PaperAccount
from src.storage.database import Database
from src.storage.hashing import market_state_hash
from src.storage.models import (
    PAPER_ACCOUNT_ID,
    Base,
    DirectionalForecastRow,
    EquitySnapshotRow,
    ExperimentRunRow,
    FillRow,
    JevDecisionRow,
    MarketSnapshotRow,
    OrderRow,
    PaperAccountRow,
    PositionRow,
    SystemEventRow,
    TradeRow,
)

OUTCOME_WINDOWS: dict[str, int] = {
    "10s": 10_000,
    "30s": 30_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "10m": 600_000,
}

SCALP_CF_WINDOWS: dict[str, int] = {
    "15s": 15_000,
    "30s": 30_000,
    "60s": 60_000,
    "180s": 180_000,
    "300s": 300_000,
}


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.run_id: str | None = None
        self.experiment_id: str | None = None
        self.config_hash: str | None = None

    def bind_run(
        self,
        *,
        run_id: str,
        experiment_id: str,
        config_hash: str,
    ) -> None:
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.config_hash = config_hash

    def _run_fields(self) -> dict[str, str | None]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
        }

    async def persist_experiment_run(
        self,
        *,
        run_id: str,
        experiment_id: str,
        config_hash: str,
        frozen: bool,
        started_at_ms: int,
        fingerprint: dict[str, Any],
    ) -> None:
        ts = datetime.fromtimestamp(started_at_ms / 1000, tz=UTC)
        async with self.db.session() as session:
            session.add(
                ExperimentRunRow(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    config_hash=config_hash,
                    frozen=frozen,
                    started_at=ts,
                    started_at_ms=started_at_ms,
                    fingerprint=fingerprint,
                )
            )
            await session.commit()

    async def latest_experiment_run(self) -> ExperimentRunRow | None:
        async with self.db.session() as session:
            return (
                await session.execute(
                    select(ExperimentRunRow).order_by(ExperimentRunRow.started_at_ms.desc()).limit(1)
                )
            ).scalar_one_or_none()

    async def get_experiment_run(self, run_id: str) -> ExperimentRunRow | None:
        async with self.db.session() as session:
            return await session.get(ExperimentRunRow, run_id)

    async def list_experiment_runs(self, experiment_id: str) -> list[ExperimentRunRow]:
        async with self.db.session() as session:
            rows = (
                (
                    await session.execute(
                        select(ExperimentRunRow)
                        .where(ExperimentRunRow.experiment_id == experiment_id)
                        .order_by(ExperimentRunRow.started_at_ms.desc())
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    async def record_system_event(
        self,
        *,
        event_type: str,
        level: str = "INFO",
        symbol: str | None = None,
        exchange: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        async with self.db.session() as session:
            session.add(
                SystemEventRow(
                    timestamp=datetime.now(UTC),
                    level=level,
                    event_type=event_type,
                    symbol=symbol,
                    exchange=exchange,
                    payload=payload or {},
                    **self._run_fields(),
                )
            )
            await session.commit()

    async def persist_tick(
        self,
        state: MarketState,
        result: JevCallResult,
        *,
        candidate_action: str | None = None,
        final_action: str | None = None,
        blocked_reason: str | None = None,
        scalp_payload: dict[str, Any] | None = None,
        directional_payload: dict[str, Any] | None = None,
    ) -> str:
        state_payload = state.canonical_dict()
        state_hash = market_state_hash(state_payload)
        ts = datetime.fromisoformat(state.timestamp)
        decision_id = ""
        resolved_final = final_action or _final_action(result)
        resolved_blocked = blocked_reason if blocked_reason is not None else _blocked_reason(result)
        async with self.db.session() as session:
            snapshot = MarketSnapshotRow(
                timestamp=ts,
                timestamp_ms=state.timestamp_ms,
                symbol=state.symbol,
                exchange=state.exchange,
                price=state.price,
                market_state=state_payload,
                market_state_hash=state_hash,
                **self._run_fields(),
            )
            session.add(snapshot)
            decision = JevDecisionRow(
                timestamp=ts,
                timestamp_ms=state.timestamp_ms,
                symbol=state.symbol,
                exchange=state.exchange,
                market_state=state_payload,
                market_state_hash=state_hash,
                question_set_version=result.question_set_version,
                model=None if result.decision is None else result.decision.model,
                mode=result.mode.value,
                status=result.status.value,
                raw_jev_response=result.raw_response,
                parsed_probabilities=(
                    None if result.decision is None else result.decision.model_dump(mode="json")
                ),
                candidate_action=candidate_action,
                final_action=resolved_final,
                blocked_reason=resolved_blocked,
                scalp_payload=scalp_payload,
                jev_latency_ms=result.latency_ms,
                safety_action=result.safety_action.value,
                current_position=state.position.model_dump(mode="json"),
                **self._run_fields(),
            )
            session.add(decision)
            await session.flush()
            decision_id = decision.id
            if directional_payload is not None:
                session.add(
                    DirectionalForecastRow(
                        timestamp=ts,
                        timestamp_ms=state.timestamp_ms,
                        symbol=state.symbol,
                        exchange=state.exchange,
                        decision_id=decision_id,
                        question_version=str(
                            directional_payload.get("question_version") or "q.directional.v1"
                        ),
                        strategy_version=str(
                            directional_payload.get("strategy_version") or "directional.v1"
                        ),
                        feature_schema_version=str(
                            directional_payload.get("feature_schema_version")
                            or "directional.state.v1"
                        ),
                        feature_state=dict(directional_payload.get("feature_state") or {}),
                        forecast=dict(directional_payload.get("forecast") or {}),
                        policy=dict(directional_payload.get("policy") or {}),
                        rolling_outcomes=dict(directional_payload.get("rolling_outcomes") or {}),
                        candle_outcomes=dict(directional_payload.get("candle_outcomes") or {}),
                        persistence_outcomes=dict(
                            directional_payload.get("persistence_outcomes") or {}
                        ),
                        counterfactuals=dict(directional_payload.get("counterfactuals") or {}),
                        mature=False,
                        research_sample_valid=bool(
                            directional_payload.get("research_sample_valid")
                        ),
                        invalid_reason=(
                            None
                            if directional_payload.get("invalid_reason") is None
                            else str(directional_payload.get("invalid_reason"))
                        ),
                        **self._run_fields(),
                    )
                )
            await session.commit()
        return decision_id

    async def price_at_or_after(
        self, *, symbol: str, exchange: str, timestamp_ms: int
    ) -> float | None:
        async with self.db.session() as session:
            stmt = (
                select(MarketSnapshotRow.price)
                .where(
                    MarketSnapshotRow.symbol == symbol,
                    MarketSnapshotRow.exchange == exchange,
                    MarketSnapshotRow.timestamp_ms >= timestamp_ms,
                    MarketSnapshotRow.price.is_not(None),
                )
                .order_by(MarketSnapshotRow.timestamp_ms.asc())
                .limit(1)
            )
            return (await session.execute(stmt)).scalar_one_or_none()

    async def enrich_outcomes(self, now_ms: int) -> int:
        updated = 0
        async with self.db.session() as session:
            pending = (
                (
                    await session.execute(
                        select(JevDecisionRow).where(JevDecisionRow.price_after_10m.is_(None))
                    )
                )
                .scalars()
                .all()
            )
            for row in pending:
                changes: dict[str, object] = {}
                entry_price = None
                if isinstance(row.market_state, dict):
                    raw_price = row.market_state.get("price")
                    if isinstance(raw_price, int | float):
                        entry_price = float(raw_price)
                for label, window_ms in OUTCOME_WINDOWS.items():
                    price_col = f"price_after_{label}"
                    if getattr(row, price_col) is not None:
                        continue
                    target = row.timestamp_ms + window_ms
                    if now_ms < target:
                        continue
                    price = await self._price_in_session(
                        session, row.symbol, row.exchange, target
                    )
                    if price is None:
                        continue
                    changes[price_col] = price
                    if entry_price and entry_price != 0:
                        changes[f"return_after_{label}"] = price / entry_price - 1.0
                scalp_update = await self._counterfactual_payload(session, row, now_ms)
                if scalp_update is not None:
                    changes["scalp_payload"] = scalp_update
                if not changes:
                    continue
                await session.execute(
                    update(JevDecisionRow).where(JevDecisionRow.id == row.id).values(**changes)
                )
                updated += 1
            await session.commit()
        return updated

    async def _price_in_session(
        self,
        session: AsyncSession,
        symbol: str,
        exchange: str,
        timestamp_ms: int,
    ) -> float | None:
        stmt = (
            select(MarketSnapshotRow.price)
            .where(
                MarketSnapshotRow.symbol == symbol,
                MarketSnapshotRow.exchange == exchange,
                MarketSnapshotRow.timestamp_ms >= timestamp_ms,
                MarketSnapshotRow.price.is_not(None),
            )
            .order_by(MarketSnapshotRow.timestamp_ms.asc())
            .limit(1)
        )
        return (await session.execute(stmt)).scalar_one_or_none()

    async def _state_in_session(
        self,
        session: AsyncSession,
        symbol: str,
        exchange: str,
        timestamp_ms: int,
    ) -> dict[str, Any] | None:
        stmt = (
            select(MarketSnapshotRow.market_state)
            .where(
                MarketSnapshotRow.symbol == symbol,
                MarketSnapshotRow.exchange == exchange,
                MarketSnapshotRow.timestamp_ms >= timestamp_ms,
            )
            .order_by(MarketSnapshotRow.timestamp_ms.asc())
            .limit(1)
        )
        payload = (await session.execute(stmt)).scalar_one_or_none()
        return payload if isinstance(payload, dict) else None

    async def _counterfactual_payload(
        self,
        session: AsyncSession,
        row: JevDecisionRow,
        now_ms: int,
    ) -> dict[str, Any] | None:
        if not isinstance(row.scalp_payload, dict) or not isinstance(row.market_state, dict):
            return None
        payload = dict(row.scalp_payload)
        existing = dict(payload.get("counterfactual") or {})
        raw_cost = payload.get("cost")
        cost: dict[str, Any] = raw_cost if isinstance(raw_cost, dict) else {}
        taker = float(cost.get("taker_fee_bps") or 0.0)
        slip = float(cost.get("slippage_bps") or 0.0)
        entry_bid, entry_ask = book_from_market_state(row.market_state)
        changed = False
        for label, window_ms in SCALP_CF_WINDOWS.items():
            if label in existing:
                continue
            target = row.timestamp_ms + window_ms
            if now_ms < target:
                continue
            later = await self._state_in_session(session, row.symbol, row.exchange, target)
            if later is None:
                continue
            exit_bid, exit_ask = book_from_market_state(later)
            if (
                entry_bid is None
                or entry_ask is None
                or exit_bid is None
                or exit_ask is None
            ):
                continue
            existing[label] = {
                "long_net_pnl": hypothetical_side_net_pnl_bps(
                    side="LONG",
                    entry_bid=entry_bid,
                    entry_ask=entry_ask,
                    exit_bid=exit_bid,
                    exit_ask=exit_ask,
                    taker_fee_bps=taker,
                    slippage_bps=slip,
                ),
                "short_net_pnl": hypothetical_side_net_pnl_bps(
                    side="SHORT",
                    entry_bid=entry_bid,
                    entry_ask=entry_ask,
                    exit_bid=exit_bid,
                    exit_ask=exit_ask,
                    taker_fee_bps=taker,
                    slippage_bps=slip,
                ),
            }
            changed = True
        barrier_changed = await self._apply_barrier_outcomes(session, row, payload, now_ms)
        if not changed and not barrier_changed:
            return None
        if changed:
            payload["counterfactual"] = existing
        return payload

    async def _apply_barrier_outcomes(
        self,
        session: AsyncSession,
        row: JevDecisionRow,
        payload: dict[str, Any],
        now_ms: int,
    ) -> bool:
        predictions = payload.get("predictions")
        if not isinstance(predictions, list):
            return False
        existing = dict(payload.get("barrier_outcomes") or {})
        changed = False
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            try:
                horizon = int(prediction["horizon_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            label = f"{horizon}s"
            if label in existing:
                continue
            window_ms = horizon * 1000
            deadline = row.timestamp_ms + window_ms
            if now_ms < deadline:
                continue
            reference = _reference_price(prediction, row.market_state)
            target_bps = float(prediction.get("target_bps") or 0.0)
            stop_bps = float(prediction.get("stop_bps") or 0.0)
            if reference is None or reference <= 0 or target_bps <= 0 or stop_bps <= 0:
                continue
            path = await self._mid_path(
                session,
                row.symbol,
                row.exchange,
                row.timestamp_ms,
                deadline + 5_000,
            )
            if not path:
                continue
            long_outcome = long_barrier_outcome(
                reference_mid=reference,
                path=path,
                target_bps=target_bps,
                stop_bps=stop_bps,
                start_ms=row.timestamp_ms,
                horizon_ms=window_ms,
            )
            short_outcome = short_barrier_outcome(
                reference_mid=reference,
                path=path,
                target_bps=target_bps,
                stop_bps=stop_bps,
                start_ms=row.timestamp_ms,
                horizon_ms=window_ms,
            )
            existing[label] = {
                "long": {
                    "outcome": long_outcome,
                    "ambiguous": long_outcome == LONG_AMBIGUOUS,
                },
                "short": {
                    "outcome": short_outcome,
                    "ambiguous": short_outcome == SHORT_AMBIGUOUS,
                },
                "path_samples": len(path),
                "reference_price": reference,
                "target_bps": target_bps,
                "stop_bps": stop_bps,
                "horizon_seconds": horizon,
            }
            changed = True
        if changed:
            payload["barrier_outcomes"] = existing
        return changed

    async def _mid_path(
        self,
        session: AsyncSession,
        symbol: str,
        exchange: str,
        start_ms: int,
        end_ms: int,
    ) -> tuple[tuple[int, float], ...]:
        stmt = (
            select(
                MarketSnapshotRow.timestamp_ms,
                MarketSnapshotRow.price,
                MarketSnapshotRow.market_state,
            )
            .where(
                MarketSnapshotRow.symbol == symbol,
                MarketSnapshotRow.exchange == exchange,
                MarketSnapshotRow.timestamp_ms >= start_ms,
                MarketSnapshotRow.timestamp_ms <= end_ms,
            )
            .order_by(MarketSnapshotRow.timestamp_ms.asc())
        )
        points: list[tuple[int, float]] = []
        for ts_ms, price, state in (await session.execute(stmt)).all():
            mid = price if isinstance(price, int | float) else _mid_from_state(state)
            if mid is None:
                continue
            points.append((int(ts_ms), float(mid)))
        return tuple(points)

    async def has_orders_for_decision(self, decision_id: str) -> bool:
        async with self.db.session() as session:
            stmt = select(OrderRow.id).where(OrderRow.decision_id == decision_id).limit(1)
            return (await session.execute(stmt)).scalar_one_or_none() is not None

    async def persist_execution(
        self,
        *,
        report: ExecutionReport,
        now_ms: int,
        symbol: str,
        exchange: str,
    ) -> None:
        ts = datetime.fromtimestamp(now_ms / 1000, tz=UTC)
        async with self.db.session() as session:
            for index, fill in enumerate(report.fills):
                client_order_id = f"{report.decision_id}:{fill.intent.value}:{index}"
                order = OrderRow(
                    timestamp=ts,
                    timestamp_ms=now_ms,
                    symbol=symbol,
                    exchange=exchange,
                    client_order_id=client_order_id,
                    decision_id=report.decision_id,
                    intent=fill.intent.value,
                    side=fill.side,
                    status="FILLED",
                    quantity=fill.quantity,
                    price=fill.fill_price,
                    fee=fill.fee,
                    notional=fill.notional,
                    source=report.source.value,
                    payload=_fill_payload(fill, report),
                    **self._run_fields(),
                )
                session.add(order)
                await session.flush()
                session.add(
                    FillRow(
                        timestamp=ts,
                        timestamp_ms=now_ms,
                        symbol=symbol,
                        exchange=exchange,
                        order_id=order.id,
                        decision_id=report.decision_id,
                        quantity=fill.quantity,
                        price=fill.fill_price,
                        fee=fill.fee,
                        payload=_fill_payload(fill, report),
                        **self._run_fields(),
                    )
                )
            if report.closed is not None:
                session.add(_trade_row(report.closed, **self._run_fields()))
            await session.commit()

    async def upsert_open_position(
        self, position: OpenPosition | None, *, symbol: str, exchange: str, now_ms: int
    ) -> None:
        ts = datetime.fromtimestamp(now_ms / 1000, tz=UTC)
        async with self.db.session() as session:
            existing = (
                await session.execute(
                    select(PositionRow).where(
                        PositionRow.symbol == symbol, PositionRow.exchange == exchange
                    )
                )
            ).scalar_one_or_none()
            if position is None:
                if existing is not None:
                    existing.timestamp = ts
                    existing.timestamp_ms = now_ms
                    existing.side = Side.FLAT.value
                    existing.quantity = 0.0
                    existing.payload = {"side": "FLAT"}
                await session.commit()
                return
            payload = position.to_payload()
            if existing is None:
                session.add(
                    PositionRow(
                        timestamp=ts,
                        timestamp_ms=now_ms,
                        symbol=symbol,
                        exchange=exchange,
                        side=position.side.value,
                        quantity=position.quantity,
                        payload=payload,
                        **self._run_fields(),
                    )
                )
            else:
                existing.timestamp = ts
                existing.timestamp_ms = now_ms
                existing.side = position.side.value
                existing.quantity = position.quantity
                existing.payload = payload
                existing.run_id = self.run_id
                existing.experiment_id = self.experiment_id
                existing.config_hash = self.config_hash
            await session.commit()

    async def save_paper_account(self, account: PaperAccount) -> None:
        payload = account.to_payload()
        async with self.db.session() as session:
            row = await session.get(PaperAccountRow, PAPER_ACCOUNT_ID)
            if row is None:
                session.add(
                    PaperAccountRow(
                        id=PAPER_ACCOUNT_ID,
                        currency=account.currency,
                        starting_balance=account.starting_balance,
                        cash=account.cash,
                        realized_pnl=account.realized_pnl,
                        unrealized_pnl=account.unrealized_pnl,
                        fees_paid=account.fees_paid,
                        equity=account.equity,
                        peak_equity=account.peak_equity,
                        drawdown=account.drawdown,
                        payload=payload,
                    )
                )
            else:
                row.currency = account.currency
                row.starting_balance = account.starting_balance
                row.cash = account.cash
                row.realized_pnl = account.realized_pnl
                row.unrealized_pnl = account.unrealized_pnl
                row.fees_paid = account.fees_paid
                row.equity = account.equity
                row.peak_equity = account.peak_equity
                row.drawdown = account.drawdown
                row.payload = payload
            await session.commit()

    async def load_paper_account(self) -> PaperAccount | None:
        async with self.db.session() as session:
            row = await session.get(PaperAccountRow, PAPER_ACCOUNT_ID)
            if row is None:
                return None
            account = PaperAccount.from_payload(row.payload)
            stmt = select(TradeRow).order_by(TradeRow.opened_at)
            if self.experiment_id:
                stmt = stmt.where(TradeRow.experiment_id == self.experiment_id)
            trades = (await session.execute(stmt)).scalars()
            account.closed_trades = [_closed_from_row(item) for item in trades]
            return account

    async def reset_paper_account(
        self,
        *,
        starting_balance: float,
        currency: str = "USDT",
        now_ms: int | None = None,
    ) -> PaperAccount:
        """Replace the singleton PAPER account. Does not delete SCALP trade rows."""
        account = PaperAccount.new(
            starting_balance=starting_balance,
            currency=currency,
            now_ms=now_ms if now_ms is not None else wall_clock_ms(),
        )
        await self.save_paper_account(account)
        return account

    async def persist_equity_snapshot(self, account: PaperAccount, now_ms: int) -> None:
        snap = account.equity_snapshot(now_ms)
        ts = datetime.fromtimestamp(now_ms / 1000, tz=UTC)
        async with self.db.session() as session:
            session.add(
                EquitySnapshotRow(
                    timestamp=ts,
                    timestamp_ms=now_ms,
                    symbol="PORTFOLIO",
                    exchange="ALL",
                    balance=float(snap["balance"]),
                    realized_pnl=float(snap["realized_pnl"]),
                    unrealized_pnl=float(snap["unrealized_pnl"]),
                    fees=float(snap["fees"]),
                    equity=float(snap["equity"]),
                    peak_equity=float(snap["peak_equity"]),
                    drawdown=float(snap["drawdown"]),
                    drawdown_pct=float(snap["drawdown_pct"]),
                    payload=snap,
                    **self._run_fields(),
                )
            )
            await session.commit()

    async def list_recent_decisions(
        self, limit: int = 20, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        async with self.db.session() as session:
            stmt = select(JevDecisionRow).order_by(JevDecisionRow.timestamp_ms.desc()).limit(limit)
            if run_id:
                stmt = stmt.where(JevDecisionRow.run_id == run_id)
            rows = (await session.execute(stmt)).scalars().all()
        return [_decision_view(row) for row in rows]

    async def list_recent_trades(
        self, limit: int = 20, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        async with self.db.session() as session:
            stmt = select(TradeRow).order_by(TradeRow.opened_at.desc()).limit(limit)
            if run_id:
                stmt = stmt.where(TradeRow.run_id == run_id)
            rows = (await session.execute(stmt)).scalars().all()
        return [_trade_view(row) for row in rows]

    async def list_recent_events(
        self, limit: int = 20, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        async with self.db.session() as session:
            stmt = (
                select(SystemEventRow)
                .order_by(SystemEventRow.timestamp.desc())
                .limit(limit)
            )
            if run_id:
                stmt = stmt.where(SystemEventRow.run_id == run_id)
            rows = (await session.execute(stmt)).scalars().all()
        return [
            {
                "timestamp": row.timestamp.isoformat(),
                "level": row.level,
                "event_type": row.event_type,
                "symbol": row.symbol,
                "exchange": row.exchange,
            }
            for row in rows
        ]

    async def ping(self) -> bool:
        async with self.db.session() as session:
            await session.execute(text("SELECT 1"))
        return True

    async def list_decisions_for_run(
        self, run_id: str, *, limit: int = 5000
    ) -> list[JevDecisionRow]:
        async with self.db.session() as session:
            rows = (
                (
                    await session.execute(
                        select(JevDecisionRow)
                        .where(JevDecisionRow.run_id == run_id)
                        .order_by(JevDecisionRow.timestamp_ms.asc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    async def list_trades_for_run(self, run_id: str) -> list[TradeRow]:
        async with self.db.session() as session:
            rows = (
                (
                    await session.execute(
                        select(TradeRow)
                        .where(TradeRow.run_id == run_id)
                        .order_by(TradeRow.opened_at.asc())
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    async def list_trades_for_experiment(self, experiment_id: str) -> list[TradeRow]:
        async with self.db.session() as session:
            rows = (
                (
                    await session.execute(
                        select(TradeRow)
                        .where(TradeRow.experiment_id == experiment_id)
                        .order_by(TradeRow.opened_at.asc())
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    async def list_events_for_run(self, run_id: str) -> list[SystemEventRow]:
        async with self.db.session() as session:
            rows = (
                (
                    await session.execute(
                        select(SystemEventRow)
                        .where(SystemEventRow.run_id == run_id)
                        .order_by(SystemEventRow.timestamp.asc())
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    async def calibration_summary(self, run_id: str | None) -> dict[str, Any]:
        if not run_id:
            return {"horizons": {}, "min_n": 20}
        rows = await self.list_decisions_for_run(run_id)
        return summarize_run_calibration(rows)

    async def list_directional_forecasts(
        self, run_id: str | None = None
    ) -> list[DirectionalForecastRow]:
        async with self.db.session() as session:
            stmt = select(DirectionalForecastRow).order_by(
                DirectionalForecastRow.timestamp_ms.asc()
            )
            if run_id:
                stmt = stmt.where(DirectionalForecastRow.run_id == run_id)
            return list((await session.execute(stmt)).scalars().all())

    async def count_directional(self, experiment_id: str | None = None) -> dict[str, int]:
        async with self.db.session() as session:
            stmt = select(DirectionalForecastRow)
            if experiment_id:
                stmt = stmt.where(DirectionalForecastRow.experiment_id == experiment_id)
            rows = list((await session.execute(stmt)).scalars().all())
        mature_rolling = 0
        trades = 0
        for row in rows:
            if isinstance(row.rolling_outcomes, dict) and row.rolling_outcomes:
                mature_rolling += 1
            policy = row.policy if isinstance(row.policy, dict) else {}
            if policy.get("candidate") in {"OPEN_LONG", "OPEN_SHORT", "CLOSE", "REVERSE"}:
                trades += 1
        return {
            "forecasts": len(rows),
            "valid_forecasts": sum(1 for row in rows if _directional_row_valid(row)),
            "mature_rolling": mature_rolling,
            "directional_actions": trades,
        }

    async def enrich_directional_outcomes(self, now_ms: int) -> int:
        updated = 0
        async with self.db.session() as session:
            pending = list(
                (
                    await session.execute(
                        select(DirectionalForecastRow).where(DirectionalForecastRow.mature.is_(False))
                    )
                )
                .scalars()
                .all()
            )
            for row in pending:
                stamps, prices = await self._mids_in_session(
                    session, row.symbol, row.exchange, row.timestamp_ms, now_ms
                )
                forecast = dict(row.forecast or {})
                ref_mid = forecast.get("reference_mid")
                if not isinstance(ref_mid, int | float) or ref_mid <= 0:
                    continue
                raw_sigma = forecast.get("sigma_bps")
                sigma = raw_sigma if isinstance(raw_sigma, dict) else {}
                raw_returns = forecast.get("returns")
                returns = raw_returns if isinstance(raw_returns, dict) else {}
                entry_state = await self._state_in_session(
                    session, row.symbol, row.exchange, row.timestamp_ms
                )
                entry_bid, entry_ask = book_from_market_state(entry_state or {})
                exit_books: dict[str, tuple[float | None, float | None]] = {}
                for label, seconds in zip(HORIZON_LABELS, HORIZON_SECONDS, strict=True):
                    target = row.timestamp_ms + seconds * 1000
                    if now_ms < target:
                        continue
                    exit_state = await self._state_in_session(
                        session, row.symbol, row.exchange, target
                    )
                    exit_books[label] = book_from_market_state(exit_state or {})
                enriched = enrich_forecast_row(
                    forecast,
                    timestamps=stamps,
                    values=prices,
                    now_ms=now_ms,
                    reference_mid=float(ref_mid),
                    reference_ts=row.timestamp_ms,
                    sigma_bps={
                        key: (None if value is None else float(value))
                        for key, value in (sigma or {}).items()
                    },
                    entry_bid=entry_bid,
                    entry_ask=entry_ask,
                    exit_books=exit_books,
                    taker_fee_bps=5.0,
                    slippage_bps=1.0,
                    returns_at_forecast={
                        key: (None if value is None else float(value))
                        for key, value in returns.items()
                        if isinstance(value, int | float) or value is None
                    },
                )
                rolling = dict(row.rolling_outcomes or {})
                rolling.update(enriched["rolling"])
                candles = dict(row.candle_outcomes or {})
                candles.update(enriched["candles"])
                persist = dict(row.persistence_outcomes or {})
                persist.update(enriched["persistence"])
                cf = dict(row.counterfactuals or {})
                cf.update(enriched["counterfactuals"])
                mature = all(label in rolling for label in HORIZON_LABELS)
                await session.execute(
                    update(DirectionalForecastRow)
                    .where(DirectionalForecastRow.id == row.id)
                    .values(
                        rolling_outcomes=rolling,
                        candle_outcomes=candles,
                        persistence_outcomes=persist,
                        counterfactuals=cf,
                        mature=mature,
                    )
                )
                updated += 1
            await session.commit()
        return updated

    async def _mids_in_session(
        self,
        session: AsyncSession,
        symbol: str,
        exchange: str,
        start_ms: int,
        end_ms: int,
    ) -> tuple[list[int], list[float]]:
        stmt = (
            select(MarketSnapshotRow.timestamp_ms, MarketSnapshotRow.price)
            .where(
                MarketSnapshotRow.symbol == symbol,
                MarketSnapshotRow.exchange == exchange,
                MarketSnapshotRow.timestamp_ms >= start_ms,
                MarketSnapshotRow.timestamp_ms <= end_ms,
                MarketSnapshotRow.price.is_not(None),
            )
            .order_by(MarketSnapshotRow.timestamp_ms.asc())
        )
        rows = (await session.execute(stmt)).all()
        stamps: list[int] = []
        prices: list[float] = []
        for stamp, price in rows:
            if price is None:
                continue
            stamps.append(int(stamp))
            prices.append(float(price))
        return stamps, prices

    async def directional_calibration_summary(self, run_id: str | None) -> dict[str, Any]:
        rows = await self.list_directional_forecasts(run_id)
        return summarize_directional_calibration(rows)

    async def wipe_directional_research(
        self, experiment_prefix: str = "directional"
    ) -> dict[str, int]:
        """Delete DIRECTIONAL research rows only. Leaves SCALP history intact."""
        like = f"{experiment_prefix}%"
        counts: dict[str, int] = {}
        async with self.db.session() as session:
            for model, key in (
                (DirectionalForecastRow, "directional_forecasts"),
                (JevDecisionRow, "jev_decisions"),
                (MarketSnapshotRow, "market_snapshots"),
                (OrderRow, "orders"),
                (FillRow, "fills"),
                (TradeRow, "trades"),
                (EquitySnapshotRow, "equity_snapshots"),
                (SystemEventRow, "system_events"),
                (ExperimentRunRow, "experiment_runs"),
            ):
                result = await session.execute(
                    delete(model).where(model.experiment_id.like(like))
                )
                counts[key] = int(getattr(result, "rowcount", 0) or 0)
            await session.commit()
        return counts

    async def wipe_all_operator_data(self) -> dict[str, int]:
        """Delete every operator table. Leaves schema in place."""
        counts: dict[str, int] = {}
        async with self.db.session() as session:
            for table in reversed(Base.metadata.sorted_tables):
                result = await session.execute(delete(table))
                counts[table.name] = int(getattr(result, "rowcount", 0) or 0)
            await session.commit()
        return counts


def _mid_from_state(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("price")
    if isinstance(raw, int | float):
        return float(raw)
    bid = payload.get("best_bid")
    ask = payload.get("best_ask")
    if isinstance(bid, int | float) and isinstance(ask, int | float) and bid > 0 and ask > 0:
        return (float(bid) + float(ask)) / 2.0
    return None


def _reference_price(prediction: dict[str, Any], market_state: object) -> float | None:
    raw = prediction.get("reference_price")
    if isinstance(raw, int | float) and raw > 0:
        return float(raw)
    return _mid_from_state(market_state)


def _realized_side(cf_row: object, key: str) -> float | None:
    if not isinstance(cf_row, dict):
        return None
    raw = cf_row.get(key)
    if isinstance(raw, int | float):
        return float(raw)
    return None


def summarize_run_calibration(rows: list[JevDecisionRow]) -> dict[str, Any]:
    by_horizon: dict[int, list[dict[str, Any]]] = {}
    n_raw_predictions = 0
    n_valid_research = 0
    for row in rows:
        payload = row.scalp_payload if isinstance(row.scalp_payload, dict) else {}
        predictions = payload.get("predictions")
        outcomes = payload.get("barrier_outcomes")
        counterfactual = payload.get("counterfactual")
        sample_valid = payload.get("research_sample_valid")
        if not isinstance(predictions, list):
            continue
        cf = counterfactual if isinstance(counterfactual, dict) else {}
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            n_raw_predictions += 1
            if sample_valid is False:
                continue
            n_valid_research += 1
            if not isinstance(outcomes, dict):
                continue
            try:
                horizon = int(prediction["horizon_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            label = f"{horizon}s"
            outcome_row = outcomes.get(label)
            if not isinstance(outcome_row, dict):
                continue
            long_row = outcome_row.get("long") if isinstance(outcome_row.get("long"), dict) else {}
            short_row = (
                outcome_row.get("short") if isinstance(outcome_row.get("short"), dict) else {}
            )
            cf_row = cf.get(label)
            by_horizon.setdefault(horizon, []).append(
                {
                    "timestamp_ms": row.timestamp_ms,
                    "horizon_seconds": horizon,
                    "p_long_target": prediction.get("p_long_target"),
                    "p_long_stop": prediction.get("p_long_stop"),
                    "p_long_neither": prediction.get("p_long_neither"),
                    "p_short_target": prediction.get("p_short_target"),
                    "p_short_stop": prediction.get("p_short_stop"),
                    "p_short_neither": prediction.get("p_short_neither"),
                    "long_outcome": long_row.get("outcome") if isinstance(long_row, dict) else None,
                    "short_outcome": (
                        short_row.get("outcome") if isinstance(short_row, dict) else None
                    ),
                    "predicted_entry_net_long_bps": prediction.get("predicted_entry_net_long_bps"),
                    "predicted_entry_net_short_bps": prediction.get(
                        "predicted_entry_net_short_bps"
                    ),
                    "realized_long_net_pnl": _realized_side(cf_row, "long_net_pnl"),
                    "realized_short_net_pnl": _realized_side(cf_row, "short_net_pnl"),
                }
            )
    horizons: dict[str, Any] = {}
    for horizon, items in sorted(by_horizon.items()):
        long_items = [
            {
                **item,
                "outcome": item.get("long_outcome"),
            }
            for item in items
        ]
        short_items = [
            {
                **item,
                "outcome": item.get("short_outcome"),
            }
            for item in items
        ]
        long_summary = summarize_long_race(long_items)
        short_summary = summarize_short_race(short_items)
        long_valid = [
            item
            for item in long_items
            if item.get("outcome") in {LONG_TARGET_FIRST, LONG_STOP_FIRST, LONG_NEITHER}
        ]
        short_valid = [
            item
            for item in short_items
            if item.get("outcome") in {SHORT_TARGET_FIRST, SHORT_STOP_FIRST, SHORT_NEITHER}
        ]
        long_stamps = [int(item["timestamp_ms"]) for item in long_valid]
        short_stamps = [int(item["timestamp_ms"]) for item in short_valid]
        long_spaced = [
            long_valid[index]
            for index in nonoverlapping_indices(long_stamps, min_gap_ms=horizon * 1000)
        ]
        short_spaced = [
            short_valid[index]
            for index in nonoverlapping_indices(short_stamps, min_gap_ms=horizon * 1000)
        ]
        long_summary["n_nonoverlapping"] = len(long_spaced)
        short_summary["n_nonoverlapping"] = len(short_spaced)
        if long_spaced:
            long_summary["nonoverlapping"] = summarize_long_race(long_spaced)
        if short_spaced:
            short_summary["nonoverlapping"] = summarize_short_race(short_spaced)
        horizons[f"{horizon}s"] = {
            "long": long_summary,
            "short": short_summary,
            "n_raw": len(items),
            "n_valid": len(items),
        }
    return {
        "horizons": horizons,
        "min_n": 20,
        "n_raw_predictions": n_raw_predictions,
        "n_valid_research": n_valid_research,
    }


def _final_action(result: JevCallResult) -> str:
    if result.status is JevCallStatus.OK:
        return Action.NO_ACTION.value
    return result.status.value


def _blocked_reason(result: JevCallResult) -> str | None:
    if result.status is JevCallStatus.OK:
        return None
    if result.error:
        return result.error
    return result.status.value


def _fill_payload(fill: PaperFill, report: ExecutionReport) -> dict[str, Any]:
    return {
        "intent": fill.intent.value,
        "is_buy": fill.is_buy,
        "best_bid": fill.best_bid,
        "best_ask": fill.best_ask,
        "requested_price": fill.raw_price,
        "fill_price": fill.fill_price,
        "spread_bps": fill.spread_bps,
        "slippage_bps": fill.slippage_bps,
        "slippage_cost": fill.slippage_cost,
        "quantity": fill.quantity,
        "notional": fill.notional,
        "fee": fill.fee,
        "source": report.source.value,
        "decision_id": report.decision_id,
        "remained_flat_after_reverse": report.remained_flat_after_reverse,
    }


def _trade_row(trade: ClosedTrade, **run_fields: str | None) -> TradeRow:
    opened = datetime.fromtimestamp(trade.entry_timestamp_ms / 1000, tz=UTC)
    closed = datetime.fromtimestamp(trade.exit_timestamp_ms / 1000, tz=UTC)
    return TradeRow(
        opened_at=opened,
        closed_at=closed,
        symbol=trade.symbol,
        exchange=trade.exchange,
        side=trade.side.value,
        quantity=trade.quantity,
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        entry_notional=trade.entry_notional,
        holding_seconds=trade.holding_seconds,
        gross_pnl=trade.gross_pnl,
        fees=trade.total_fees,
        net_pnl=trade.net_pnl,
        entry_jev_decision_id=trade.entry_jev_decision_id,
        exit_jev_decision_id=trade.exit_jev_decision_id,
        payload={
            "entry_timestamp_ms": trade.entry_timestamp_ms,
            "exit_timestamp_ms": trade.exit_timestamp_ms,
            "gross_pnl_bps": trade.gross_pnl_bps,
            "net_pnl_bps": trade.net_pnl_bps,
            "entry_fee": trade.entry_fee,
            "exit_fee": trade.exit_fee,
            "total_fees": trade.total_fees,
            "slippage_cost": trade.slippage_cost,
            "funding_cost": trade.funding_cost,
            "mfe_bps": trade.mfe_bps,
            "mae_bps": trade.mae_bps,
            "entry_source": trade.entry_source,
            "exit_source": trade.exit_source,
            "entry_horizon_seconds": trade.entry_horizon_seconds,
            "entry_net_ev_bps": trade.entry_net_ev_bps,
            "exit_reason": trade.exit_reason,
            "entry_slippage_cost": trade.entry_slippage_cost,
            "exit_slippage_cost": trade.exit_slippage_cost,
            "total_slippage_cost": trade.total_slippage_cost,
            "entry_spread_cost": trade.entry_spread_cost,
            "exit_spread_cost": trade.exit_spread_cost,
            "total_spread_cost": trade.total_spread_cost,
            "market_move_pnl": trade.market_move_pnl,
            "attribution_residual": trade.attribution_residual,
            "funding_accounting": trade.funding_accounting,
            "trade_net_pnl_usdt": trade.net_pnl,
            "trade_net_return_bps_on_notional": trade.net_pnl_bps,
            "research": trade.research,
            "capture_ratio": None if not trade.research else trade.research.get("capture_ratio"),
            "thesis_score_entry": None
            if not trade.research
            else trade.research.get("thesis_score_entry"),
            "thesis_score_close": None
            if not trade.research
            else trade.research.get("thesis_score_close"),
            "primary_horizon": None
            if not trade.research
            else trade.research.get("primary_horizon"),
        },
        **run_fields,
    )


def _closed_from_row(row: TradeRow) -> ClosedTrade:
    payload = row.payload if isinstance(row.payload, dict) else {}
    entry_ms = int(payload.get("entry_timestamp_ms") or row.opened_at.timestamp() * 1000)
    exit_ms = int(
        payload.get("exit_timestamp_ms")
        or ((row.closed_at or row.opened_at).timestamp() * 1000)
    )
    return ClosedTrade(
        symbol=row.symbol,
        exchange=row.exchange,
        side=Side(row.side),
        quantity=float(row.quantity or 0.0),
        entry_price=float(row.entry_price or 0.0),
        exit_price=float(row.exit_price or 0.0),
        entry_notional=float(row.entry_notional or 0.0),
        entry_timestamp_ms=entry_ms,
        exit_timestamp_ms=exit_ms,
        holding_seconds=float(row.holding_seconds or 0.0),
        gross_pnl=row.gross_pnl,
        entry_fee=float(payload.get("entry_fee") or 0.0),
        exit_fee=float(payload.get("exit_fee") or 0.0),
        total_fees=row.fees,
        slippage_cost=float(payload.get("slippage_cost") or 0.0),
        funding_cost=float(payload.get("funding_cost") or 0.0),
        net_pnl=row.net_pnl,
        gross_pnl_bps=float(payload.get("gross_pnl_bps") or 0.0),
        net_pnl_bps=float(payload.get("net_pnl_bps") or 0.0),
        mfe_bps=float(payload.get("mfe_bps") or 0.0),
        mae_bps=float(payload.get("mae_bps") or 0.0),
        entry_jev_decision_id=row.entry_jev_decision_id,
        exit_jev_decision_id=row.exit_jev_decision_id,
        entry_source=str(payload.get("entry_source") or "JEV"),
        exit_source=str(payload.get("exit_source") or "JEV"),
        entry_horizon_seconds=(
            None
            if payload.get("entry_horizon_seconds") is None
            else int(payload["entry_horizon_seconds"])
        ),
        entry_net_ev_bps=(
            None if payload.get("entry_net_ev_bps") is None else float(payload["entry_net_ev_bps"])
        ),
        exit_reason=None if payload.get("exit_reason") is None else str(payload["exit_reason"]),
        entry_slippage_cost=float(payload.get("entry_slippage_cost") or 0.0),
        exit_slippage_cost=float(payload.get("exit_slippage_cost") or 0.0),
        total_slippage_cost=float(
            payload.get("total_slippage_cost") or payload.get("slippage_cost") or 0.0
        ),
        entry_spread_cost=float(payload.get("entry_spread_cost") or 0.0),
        exit_spread_cost=float(payload.get("exit_spread_cost") or 0.0),
        total_spread_cost=float(payload.get("total_spread_cost") or 0.0),
        market_move_pnl=float(payload.get("market_move_pnl") or 0.0),
        attribution_residual=float(payload.get("attribution_residual") or 0.0),
        funding_accounting=str(payload.get("funding_accounting") or "disabled_zero"),
        research=payload.get("research") if isinstance(payload.get("research"), dict) else None,
    )


def _decision_view(row: JevDecisionRow) -> dict[str, Any]:
    parsed = row.parsed_probabilities if isinstance(row.parsed_probabilities, dict) else {}
    scalp = row.scalp_payload if isinstance(row.scalp_payload, dict) else {}
    best = scalp.get("best")
    raw_forecast = parsed.get("directional_forecast")
    forecast = raw_forecast if isinstance(raw_forecast, dict) else {}
    raw_p = forecast.get("p_up")
    p_up = raw_p if isinstance(raw_p, dict) else {}
    return {
        "timestamp": row.timestamp.isoformat(),
        "timestamp_ms": row.timestamp_ms,
        "status": row.status,
        "mode": row.mode,
        "model": row.model,
        "desired_exposure": parsed.get("exposure_choice"),
        "long_probability": parsed.get("long_probability"),
        "short_probability": parsed.get("short_probability"),
        "p_up_1m": p_up.get("1m"),
        "p_up_5m": p_up.get("5m"),
        "p_up_10m": p_up.get("10m"),
        "p_up_30m": p_up.get("30m"),
        "research_sample_valid": scalp.get("research_sample_valid"),
        "evidence": parsed.get("evidence_probability"),
        "noise": parsed.get("noise_probability"),
        "continuation": parsed.get("continuation_probability"),
        "reversal": parsed.get("reversal_probability"),
        "exhaustion": parsed.get("exhaustion_probability"),
        "candidate_action": row.candidate_action,
        "final_action": row.final_action,
        "blocked_reason": row.blocked_reason,
        "jev_latency_ms": row.jev_latency_ms,
        "price_after_10s": row.price_after_10s,
        "best_side": None if not isinstance(best, dict) else best.get("side"),
        "best_horizon_seconds": None if not isinstance(best, dict) else best.get("horizon_seconds"),
        "net_ev_bps": None if not isinstance(best, dict) else best.get("net_ev_bps"),
        "scalp_candidate": None if not isinstance(scalp, dict) else scalp.get("candidate"),
    }


def _trade_view(row: TradeRow) -> dict[str, Any]:
    payload = row.payload if isinstance(row.payload, dict) else {}
    return {
        "entry_time": row.opened_at.isoformat(),
        "exit_time": None if row.closed_at is None else row.closed_at.isoformat(),
        "side": row.side,
        "entry": row.entry_price,
        "exit": row.exit_price,
        "holding_seconds": row.holding_seconds,
        "notional": row.entry_notional,
        "gross_pnl": row.gross_pnl,
        "fees": row.fees,
        "slippage": payload.get("total_slippage_cost") or payload.get("slippage_cost"),
        "net_pnl": row.net_pnl,
        "trade_net_pnl_usdt": row.net_pnl,
        "net_bps": payload.get("net_pnl_bps"),
        "trade_net_return_bps_on_notional": payload.get("trade_net_return_bps_on_notional")
        or payload.get("net_pnl_bps"),
        "entry_slippage": payload.get("entry_slippage_cost"),
        "exit_slippage": payload.get("exit_slippage_cost"),
        "entry_spread": payload.get("entry_spread_cost"),
        "exit_spread": payload.get("exit_spread_cost"),
        "total_spread": payload.get("total_spread_cost"),
        "market_move_pnl": payload.get("market_move_pnl"),
        "mfe_bps": payload.get("mfe_bps"),
        "mae_bps": payload.get("mae_bps"),
        "entry_horizon_seconds": payload.get("entry_horizon_seconds"),
        "entry_net_ev_bps": payload.get("entry_net_ev_bps"),
        "exit_reason": payload.get("exit_reason"),
        "exit_source": payload.get("exit_source"),
        "capture_ratio": payload.get("capture_ratio"),
        "thesis_score_entry": payload.get("thesis_score_entry"),
        "thesis_score_close": payload.get("thesis_score_close"),
        "primary_horizon": payload.get("primary_horizon"),
        "research": payload.get("research"),
    }


def _directional_row_valid(row: DirectionalForecastRow) -> bool:
    if row.research_sample_valid is True:
        return True
    forecast = row.forecast if isinstance(row.forecast, dict) else {}
    return forecast.get("research_sample_valid") is True


def summarize_directional_calibration(rows: list[DirectionalForecastRow]) -> dict[str, Any]:
    n_raw = len(rows)
    official = [row for row in rows if _directional_row_valid(row)]
    n_valid = len(official)
    horizons: dict[str, list[dict[str, Any]]] = {label: [] for label in HORIZON_LABELS}
    candles: dict[str, list[dict[str, Any]]] = {label: [] for label in HORIZON_LABELS}
    persistence: dict[str, list[dict[str, Any]]] = {}
    magnitude: dict[str, list[dict[str, Any]]] = {label: [] for label in HORIZON_LABELS}
    for row in official:
        forecast = row.forecast if isinstance(row.forecast, dict) else {}
        raw_p = forecast.get("p_up")
        p_up: dict[str, Any] = raw_p if isinstance(raw_p, dict) else {}
        raw_mag = forecast.get("magnitude")
        mag: dict[str, Any] = raw_mag if isinstance(raw_mag, dict) else {}
        rolling = row.rolling_outcomes if isinstance(row.rolling_outcomes, dict) else {}
        candle_rows = row.candle_outcomes if isinstance(row.candle_outcomes, dict) else {}
        persist = row.persistence_outcomes if isinstance(row.persistence_outcomes, dict) else {}
        for label in HORIZON_LABELS:
            outcome = rolling.get(label)
            if isinstance(outcome, dict) and "direction" in outcome:
                actual_up = outcome["direction"] == "UP"
                predicted = float(p_up.get(label) or 0.5)
                raw_base = outcome.get("baselines")
                baselines = raw_base if isinstance(raw_base, dict) else {}
                horizons[label].append(
                    {
                        "timestamp_ms": row.timestamp_ms,
                        "p_up": predicted,
                        "actual_up": actual_up,
                        "recent": baselines.get("recent_direction"),
                        "momentum": baselines.get("momentum"),
                    }
                )
                chosen_raw = mag.get(label)
                chosen = chosen_raw if isinstance(chosen_raw, dict) else {}
                if chosen.get("choice") and outcome.get("magnitude_bucket"):
                    magnitude[label].append(
                        {
                            "predicted": chosen.get("choice"),
                            "actual": outcome.get("magnitude_bucket"),
                        }
                    )
            candle = candle_rows.get(label)
            if isinstance(candle, dict) and candle.get("direction") in {"UP", "DOWN"}:
                candles[label].append(
                    {
                        "timestamp_ms": row.timestamp_ms,
                        "p_up": float(candle.get("p_up") or 0.5),
                        "actual_up": candle.get("direction") == "UP",
                        "remaining_segment": candle.get("remaining_segment"),
                    }
                )
        for key, item in persist.items():
            if not isinstance(item, dict) or "hit" not in item:
                continue
            persistence.setdefault(key, []).append(
                {
                    "timestamp_ms": row.timestamp_ms,
                    "p_up": float(item.get("p") or 0.0),
                    "actual_up": bool(item.get("hit")),
                }
            )
    horizon_gap = {"1m": 60_000, "5m": 300_000, "10m": 600_000, "30m": 1_800_000}
    summary_h = {}
    for label, items in horizons.items():
        block = summarize_binary(
            items,
            p_key="p_up",
            hit_key="actual_up",
            min_gap_ms=horizon_gap[label],
        )
        actual = [bool(item["actual_up"]) for item in items]
        block["baseline_50_50"] = 0.5 if items else None
        block["baseline_recent"] = None
        if items:
            block["baseline_recent"] = baseline_accuracy(
                actual, [str(item.get("recent") or "DOWN") for item in items]
            )
            block["baseline_momentum"] = baseline_accuracy(
                actual, [str(item.get("momentum") or "DOWN") for item in items]
            )
        summary_h[label] = block
    candle_summary = {}
    for label, items in candles.items():
        by_seg: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            seg = str(item.get("remaining_segment") or "unknown")
            by_seg.setdefault(seg, []).append(item)
        candle_summary[label] = {
            **summarize_binary(
                items,
                p_key="p_up",
                hit_key="actual_up",
                min_gap_ms=horizon_gap[label],
            ),
            "by_remaining": {
                seg: summarize_binary(
                    group,
                    p_key="p_up",
                    hit_key="actual_up",
                    min_gap_ms=horizon_gap[label],
                )
                for seg, group in by_seg.items()
            },
        }
    mag_summary = {}
    for label, items in magnitude.items():
        n = len(items)
        correct = sum(1 for item in items if item.get("predicted") == item.get("actual"))
        mag_summary[label] = {"n": n, "accuracy": None if n == 0 else correct / n}
    persist_summary = {
        key: summarize_binary(items, p_key="p_up", hit_key="actual_up")
        for key, items in persistence.items()
    }
    spaced_1m = summary_h.get("1m") or {}
    return {
        "kind": "DIRECTIONAL",
        "n_raw_predictions": n_raw,
        "n_valid_research": n_valid,
        "n_nonoverlapping_valid": spaced_1m.get("n_nonoverlapping", 0),
        "n_forecasts": n_valid,
        "horizons": summary_h,
        "candles": candle_summary,
        "persistence": persist_summary,
        "magnitude": mag_summary,
        "min_n": 20,
    }
