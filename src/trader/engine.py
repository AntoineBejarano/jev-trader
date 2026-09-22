from __future__ import annotations

import asyncio
import json
import logging
import signal
import uuid
from dataclasses import replace
from typing import Any

from src.analytics.latency import LatencyTracker
from src.analytics.outcomes import enrich_decision_outcomes
from src.analytics.prediction_benchmark import dashboard_prediction_research
from src.analytics.research_sample import (
    INVALID_CONTEXT,
    annotate_research_sample,
    research_sample_status,
)
from src.api.snapshot import build_operator_snapshot
from src.config.settings import Settings, TradingMode, TradingStrategy
from src.decision.actions import EXECUTABLE_ACTIONS, Action, ActionSource
from src.decision.barriers import BarrierPlanner
from src.decision.policy import DecisionPolicy
from src.decision.scalp import ScalpPlan, ScalpPolicy, plan_payload
from src.directional.constants import (
    FEATURE_SCHEMA_VERSION as DIRECTIONAL_FEATURE_SCHEMA,
)
from src.directional.constants import HORIZON_SECONDS_BY_LABEL
from src.directional.constants import (
    QUESTION_VERSION as DIRECTIONAL_QUESTION_VERSION,
)
from src.directional.constants import (
    STRATEGY_VERSION as DIRECTIONAL_STRATEGY_VERSION,
)
from src.directional.coverage import missing_required
from src.directional.cross import CrossMarketCache
from src.directional.features import build_directional_state
from src.directional.policy import DirectionalPlan, DirectionalPolicy
from src.directional.serialize import compact_inspector, serialize_directional_state
from src.directional.thesis import capture_ratio
from src.execution.paper import ExecutionReport, PaperExecutionEngine
from src.experiment.fingerprint import (
    active_question_version,
    active_strategy_version,
    config_hash,
    research_fingerprint,
)
from src.features.engine import FeatureEngine
from src.jev.client import JevClient
from src.jev.models import JevCallResult, JevCallStatus
from src.market.buffer import wall_clock_ms
from src.market.client import BinanceMarketDataAdapter
from src.market.models import MarketState, Side
from src.portfolio.account import ClosedTrade, PaperAccount
from src.risk.engine import RiskDecision, RiskEngine
from src.storage.database import Database, database_kind
from src.storage.repository import Repository, _closed_from_row, summarize_directional_calibration

logger = logging.getLogger(__name__)

LIVE_PHASE4_MESSAGE = "LIVE trading is not implemented/enabled in Phase 4"


def _bps(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 10_000:+.2f}bps"


def _p(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_state_line(state: MarketState) -> str:
    health = "STALE" if state.health.stale else "LIVE-DATA"
    if not state.health.trading_allowed:
        health += "/NO-TRADE"
    position = state.position
    pos = position.side.value
    if position.side.value != "FLAT" and position.unrealized_pnl_bps is not None:
        pos = (
            f"{position.side.value} qty={position.quantity} "
            f"uPnL={position.unrealized_pnl_bps:+.2f}bps "
            f"age={position.age_seconds:.0f}s"
        )
    return (
        f"{state.timestamp} {state.exchange}:{state.symbol} "
        f"mid={state.price} last={state.last_trade_price} "
        f"spread={state.microstructure.spread_bps}bps "
        f"r5s={_bps(state.returns.r_5s)} r1m={_bps(state.returns.r_1m)} "
        f"r5m={_bps(state.returns.r_5m)} "
        f"imb1={state.microstructure.book_imbalance_1} "
        f"flow1m={state.microstructure.aggressive_flow_1m} "
        f"oi={state.derivatives.open_interest} "
        f"fund={state.derivatives.funding} "
        f"pos={pos} {health} "
        f"reasons={list(state.health.reasons)}"
    )


def format_jev_line(result: JevCallResult) -> str:
    decision = result.decision
    if result.status is not JevCallStatus.OK or decision is None:
        return (
            f"JEV {result.question_set_version} {result.mode} "
            f"{result.exchange}:{result.symbol} status={result.status} "
            f"safety={result.safety_action} err={result.error} "
            f"latency={result.latency_ms}"
        )
    exposure = decision.exposure_choice or "n/a"
    action = decision.position_action or "n/a"
    return (
        f"JEV {decision.question_set_version} {decision.mode} "
        f"{result.exchange}:{result.symbol} "
        f"long={_p(decision.long_probability)} short={_p(decision.short_probability)} "
        f"cont={_p(decision.continuation_probability)} rev={_p(decision.reversal_probability)} "
        f"exh={_p(decision.exhaustion_probability)} noise={_p(decision.noise_probability)} "
        f"evd={_p(decision.evidence_probability)} "
        f"exposure={exposure} action={action} "
        f"latency={decision.latency_ms:.0f}ms model={decision.model}"
    )


def _socket_event(state: str, *, prefix: str = "binance") -> str:
    mapping = {
        "connected": f"{prefix}_connected",
        "connecting": f"{prefix}_reconnecting",
        "reconnecting": f"{prefix}_reconnecting",
        "stopped": f"{prefix}_disconnected",
    }
    return mapping.get(state, f"{prefix}_{state}")


class TraderEngine:
    """Phase 4: observe, query Jev, persist, optional paper execution. No live orders."""

    def __init__(
        self,
        settings: Settings,
        jev_client: JevClient | None = None,
        database: Database | None = None,
        account: PaperAccount | None = None,
    ) -> None:
        if settings.trading_mode is TradingMode.LIVE:
            raise RuntimeError(LIVE_PHASE4_MESSAGE)
        self.settings = settings
        self.adapter = BinanceMarketDataAdapter(settings, settings.symbols)
        self.features = FeatureEngine()
        self.jev = jev_client or JevClient(settings)
        self._owns_jev = jev_client is None
        self.db = database or Database(settings.database_url)
        self._owns_db = database is None
        self.repo = Repository(self.db)
        self.policy = DecisionPolicy(settings)
        self.scalp_policy = ScalpPolicy(settings)
        self.directional_policy = DirectionalPolicy(settings)
        self.barriers = BarrierPlanner(settings)
        self.risk = RiskEngine(settings)
        self.cross_market = CrossMarketCache(settings)
        self._dir_prev_depths: dict[str, dict[str, float]] = {}
        self.paper = (
            PaperExecutionEngine(settings)
            if settings.trading_mode is TradingMode.PAPER
            else None
        )
        self.account = account or PaperAccount.new(
            starting_balance=settings.paper_initial_balance,
            currency=settings.paper_currency,
            now_ms=wall_clock_ms(),
        )
        self._stop = asyncio.Event()
        self._stale_symbols: set[str] = set()
        self.lifecycle = "STOPPED"
        self.strategy_error: str | None = None
        self._strategy_task: asyncio.Task[None] | None = None
        self._control_lock = asyncio.Lock()
        self._bootstrapped = False
        self.started_at_ms: int | None = None
        self.last_state: MarketState | None = None
        self.last_result: JevCallResult | None = None
        self.last_risk: RiskDecision | None = None
        self.last_plan: ScalpPlan | None = None
        self.last_directional: DirectionalPlan | None = None
        self.last_directional_state: dict[str, Any] | None = None
        self.last_jev_at_ms: int | None = None
        self.session_started_ms: int | None = None
        self.jev_latencies = LatencyTracker()
        self.fill_latencies = LatencyTracker()
        self.ws_disconnects = 0
        self.ws_recoveries = 0
        self.positive_ev_count = 0
        self.rejected_cost_count = 0
        self.jev_failures = 0
        self._warmup_ready = False
        self._ws_public_state = "stopped"
        self._ws_market_state = "stopped"
        self._last_blocked: str | None = None
        self._jev_was_error = False
        self.run_id = str(uuid.uuid4())
        self.config_fingerprint = research_fingerprint(settings, resolved_model=None)
        self.config_hash = config_hash(self.config_fingerprint)
        self.calibration_view: dict[str, Any] = {"horizons": {}, "min_n": 20}
        self.prediction_research_view: dict[str, Any] = {"scorecard": [], "n_raw": 0}

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def strategy_running(self) -> bool:
        return self.lifecycle in {"STARTING", "RUNNING"}

    async def start_strategy(self) -> dict[str, Any]:
        async with self._control_lock:
            if self.lifecycle in {"STARTING", "RUNNING"}:
                return {
                    "ok": True,
                    "changed": False,
                    "engine": self.lifecycle,
                    "detail": "already_running",
                }
            if self.lifecycle == "STOPPING":
                return {
                    "ok": False,
                    "changed": False,
                    "engine": self.lifecycle,
                    "error": "still_stopping",
                }
            self._stop = asyncio.Event()
            self.strategy_error = None
            self.lifecycle = "STARTING"
            self._strategy_task = asyncio.create_task(
                self._strategy_session(), name="strategy-session"
            )
            return {"ok": True, "changed": True, "engine": "STARTING"}

    async def stop_strategy(self) -> dict[str, Any]:
        async with self._control_lock:
            if self.lifecycle == "STOPPED":
                return {
                    "ok": True,
                    "changed": False,
                    "engine": "STOPPED",
                    "detail": "already_stopped",
                }
            self.lifecycle = "STOPPING"
            self.request_stop()
            task = self._strategy_task
            if task is not None and not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
                except TimeoutError:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                except asyncio.CancelledError:
                    pass
            self.lifecycle = "STOPPED"
            self._strategy_task = None
            return {"ok": True, "changed": True, "engine": "STOPPED"}

    async def reset_operator_data(self) -> dict[str, Any]:
        """Stop the strategy, wipe the DB, and restore a FLAT PAPER account."""
        stopped = await self.stop_strategy()
        async with self._control_lock:
            if self.lifecycle != "STOPPED":
                return {
                    "ok": False,
                    "changed": False,
                    "engine": self.lifecycle,
                    "error": "still_running",
                }
            await self.db.create_all()
            deleted = await self.repo.wipe_all_operator_data()
            self._reset_runtime_state()
            if self.settings.trading_mode is TradingMode.PAPER:
                self.account = await self.repo.reset_paper_account(
                    starting_balance=self.settings.paper_initial_balance,
                    currency=self.settings.paper_currency,
                )
            logger.warning(
                "Operator reset deleted=%s equity=%s run_id=%s",
                deleted,
                self.account.equity,
                self.run_id,
            )
            return {
                "ok": True,
                "changed": True,
                "engine": "STOPPED",
                "stopped_strategy": bool(stopped.get("changed")),
                "deleted": deleted,
                "paper_equity": self.account.equity,
                "run_id": self.run_id,
            }

    def _reset_runtime_state(self) -> None:
        self.account = PaperAccount.new(
            starting_balance=self.settings.paper_initial_balance,
            currency=self.settings.paper_currency,
            now_ms=wall_clock_ms(),
        )
        self.policy = DecisionPolicy(self.settings)
        self.scalp_policy = ScalpPolicy(self.settings)
        self.directional_policy = DirectionalPolicy(self.settings)
        self.risk = RiskEngine(self.settings)
        self._dir_prev_depths = {}
        self._stale_symbols = set()
        self.strategy_error = None
        self._bootstrapped = False
        self.started_at_ms = None
        self.last_state = None
        self.last_result = None
        self.last_risk = None
        self.last_plan = None
        self.last_directional = None
        self.last_directional_state = None
        self.last_jev_at_ms = None
        self.session_started_ms = None
        self.jev_latencies = LatencyTracker()
        self.fill_latencies = LatencyTracker()
        self.ws_disconnects = 0
        self.ws_recoveries = 0
        self.positive_ev_count = 0
        self.rejected_cost_count = 0
        self.jev_failures = 0
        self._warmup_ready = False
        self._last_blocked = None
        self._jev_was_error = False
        self.run_id = str(uuid.uuid4())
        self.repo.bind_run(
            run_id=self.run_id,
            experiment_id=self.settings.experiment_id,
            config_hash=self.config_hash,
        )
        self.calibration_view = {"horizons": {}, "min_n": 20}
        self.prediction_research_view = {"scorecard": [], "n_raw": 0}

    async def shutdown(self) -> None:
        """Process teardown. Stops strategy and closes DB / Jev clients."""
        await self.stop_strategy()
        try:
            await self.adapter.close()
        except Exception:
            logger.exception("Adapter close during shutdown failed")
        if self._owns_jev:
            await self.jev.aclose()
        await self.cross_market.aclose()
        if self._owns_db:
            await self.db.aclose()
        logger.info("Process shutdown complete")

    async def run(self) -> None:
        """Blocking strategy session used by tests/scripts. Dashboard uses start/stop."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                pass
        await self.start_strategy()
        task = self._strategy_task
        if task is not None:
            await task
        await self.shutdown()

    async def _strategy_session(self) -> None:
        try:
            await self._run_session()
        except Exception as exc:
            self.strategy_error = str(exc)
            logger.exception("Strategy session failed")
        finally:
            self.lifecycle = "STOPPED"
            self._strategy_task = None

    async def _run_session(self) -> None:
        logger.info(
            "Starting Phase 4 exchange=%s symbols=%s mode=%s strategy=%s db=%s jev=%s",
            self.settings.trading_exchange,
            ",".join(self.settings.symbols),
            self.settings.trading_mode,
            self.settings.trading_strategy,
            database_kind(self.settings.database_url),
            self.settings.jev_model,
        )
        if self.settings.trading_mode is TradingMode.PAPER:
            logger.info(
                "PAPER costs taker_fee_bps=%s slippage_bps=%s maker_fee_bps=%s "
                "initial_balance=%s %s notional=%s",
                self.settings.execution_taker_fee_bps,
                self.settings.slippage_bps,
                self.settings.execution_maker_fee_bps,
                self.settings.paper_initial_balance,
                self.settings.paper_currency,
                self.settings.execution_notional,
            )
        if not self.settings.openrouter_api_key:
            logger.error(
                "OPENROUTER_API_KEY missing — Jev calls will be skipped and no position may open"
            )
        if not self._bootstrapped:
            await self.db.create_all()
            self.repo.bind_run(
                run_id=self.run_id,
                experiment_id=self.settings.experiment_id,
                config_hash=self.config_hash,
            )
            started = wall_clock_ms()
            self.started_at_ms = started
            self.session_started_ms = started
            await self.repo.persist_experiment_run(
                run_id=self.run_id,
                experiment_id=self.settings.experiment_id,
                config_hash=self.config_hash,
                frozen=self.settings.experiment_frozen,
                started_at_ms=started,
                fingerprint=self.config_fingerprint,
            )
            logger.info(
                "EXPERIMENT id=%s run_id=%s config_hash=%s frozen=%s question=%s strategy=%s",
                self.settings.experiment_id,
                self.run_id,
                self.config_hash,
                self.settings.experiment_frozen,
                active_question_version(self.settings),
                active_strategy_version(self.settings),
            )
            if self.settings.experiment_frozen:
                logger.info(
                    "CONFIG FROZEN fingerprint=%s",
                    json.dumps(self.config_fingerprint, sort_keys=True, default=str),
                )
            if self.settings.trading_mode is TradingMode.PAPER:
                restored = await self.repo.load_paper_account()
                stored_experiment = await self.repo.paper_account_experiment_id()
                if (
                    restored is not None
                    and stored_experiment == self.settings.experiment_id
                ):
                    self.account = restored
                    open_sides = [
                        f"{pos.exchange}:{pos.symbol}:{pos.side.value}"
                        for pos in self.account.positions.values()
                    ]
                    logger.info(
                        "Restored paper account equity=%s positions=%s open=%s",
                        self.account.equity,
                        self.account.open_position_count,
                        open_sides or "FLAT",
                    )
                else:
                    if restored is not None:
                        logger.info(
                            "Fresh paper account for experiment %s "
                            "(previous payload experiment=%s). "
                            "Research rows are not deleted.",
                            self.settings.experiment_id,
                            stored_experiment,
                        )
                    self.account = await self.repo.reset_paper_account(
                        starting_balance=self.settings.paper_initial_balance,
                        currency=self.settings.paper_currency,
                    )
            self._bootstrapped = True
        await self.repo.record_system_event(
            event_type="ENGINE_START",
            payload={
                "mode": self.settings.trading_mode.value,
                "symbols": list(self.settings.symbols),
                "exchange": self.settings.trading_exchange,
                "paper_equity": self.account.equity,
                "strategy": self.settings.trading_strategy.value,
                "run_id": self.run_id,
                "experiment_id": self.settings.experiment_id,
                "config_hash": self.config_hash,
            },
        )
        monitor: asyncio.Task[None] | None = None
        outcomes: asyncio.Task[None] | None = None
        try:
            await self.adapter.connect()
            await self.adapter.wait_connected()
            await self.adapter.subscribe_market_data(self.settings.symbols)
            self.lifecycle = "RUNNING"
            monitor = asyncio.create_task(self._health_monitor(), name="health-monitor")
            outcomes = asyncio.create_task(self._outcome_loop(), name="outcome-enricher")
            await self._decision_loop()
        finally:
            for task in (monitor, outcomes):
                if task is None:
                    continue
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            try:
                if self.settings.trading_mode is TradingMode.PAPER:
                    await self.repo.save_paper_account(self.account)
                await self.repo.record_system_event(event_type="ENGINE_STOP")
            except Exception:
                logger.exception("Failed to persist ENGINE_STOP")
            await self.adapter.close()
            logger.info("Strategy session stopped")

    async def _decision_loop(self) -> None:
        interval = self.settings.decision_interval_seconds
        while not self._stop.is_set():
            started = wall_clock_ms()
            now = started
            for symbol in self.settings.symbols:
                await self._tick_symbol(symbol, now)
            elapsed = (wall_clock_ms() - started) / 1000.0
            remaining = max(0.0, interval - elapsed)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=remaining)
            except TimeoutError:
                continue

    async def _tick_symbol(self, symbol: str, now: int) -> None:
        runtime = self.adapter.runtime(symbol)
        bid = runtime.book.best_bid
        ask = runtime.book.best_ask
        if bid is not None and ask is not None:
            self.account.mark(
                symbol=symbol,
                exchange=self.settings.trading_exchange,
                bid=bid,
                ask=ask,
                now_ms=now,
            )
        state = self._snapshot(symbol, now)
        logger.info(format_state_line(state))
        logger.debug(json.dumps(state.canonical_dict(), separators=(",", ":")))
        barriers = None
        plan: ScalpPlan | None = None
        directional_plan: DirectionalPlan | None = None
        directional_state: dict[str, Any] | None = None
        if self.settings.trading_strategy is TradingStrategy.SCALP:
            barriers = self.barriers.plan(state)
        if self.settings.trading_strategy is TradingStrategy.DIRECTIONAL:
            stale = self.cross_market.last_ok_ms is None
            due = now - (self.cross_market.last_ok_ms or 0) > 15_000
            if stale or due:
                try:
                    await self.cross_market.refresh(now)
                except Exception:
                    logger.debug("Cross-market refresh failed", exc_info=True)
            runtime_depths = {
                "bid_1": sum(qty for _, qty in runtime.book.bids[:1]),
                "ask_1": sum(qty for _, qty in runtime.book.asks[:1]),
                "bid_5": sum(qty for _, qty in runtime.book.bids[:5]),
                "ask_5": sum(qty for _, qty in runtime.book.asks[:5]),
                "bid_10": sum(qty for _, qty in runtime.book.bids[:10]),
                "ask_10": sum(qty for _, qty in runtime.book.asks[:10]),
                "bid_20": sum(qty for _, qty in runtime.book.bids[:20]),
                "ask_20": sum(qty for _, qty in runtime.book.asks[:20]),
            }
            directional_state = build_directional_state(
                buffer=runtime.buffer,
                book=runtime.book,
                now_ms=now,
                cross=self.cross_market.payload,
                prev_depths=self._dir_prev_depths.get(symbol),
            )
            self._dir_prev_depths[symbol] = runtime_depths
            result = await self.jev.decide(
                state, forecast_state=serialize_directional_state(directional_state)
            )
        else:
            result = await self.jev.decide(state, barriers=barriers)
        self.jev_latencies.record(result.latency_ms)
        if result.status in {
            JevCallStatus.TIMEOUT,
            JevCallStatus.INVALID_RESPONSE,
            JevCallStatus.UNAVAILABLE,
        }:
            self.jev_failures += 1
        logger.info(format_jev_line(result))
        if self.settings.trading_strategy is TradingStrategy.SCALP:
            assert barriers is not None
            plan = self.scalp_policy.evaluate(state, result, barriers)
            candidate = plan.candidate
            self._note_scalp_counts(plan)
            logger.info(
                "SCALP best=%s:%s entry_net=%.2f hold=%.2f close_now=%.2f reverse=%.2f reason=%s",
                None if plan.best is None else plan.best.side,
                None if plan.best is None else plan.best.horizon_seconds,
                -999.0 if plan.best is None else plan.best.net_ev_bps,
                -999.0 if plan.hold_net_ev_bps is None else plan.hold_net_ev_bps,
                -999.0 if plan.close_now_value_bps is None else plan.close_now_value_bps,
                -999.0 if plan.reverse_net_ev_bps is None else plan.reverse_net_ev_bps,
                plan.policy_reason,
            )
        elif self.settings.trading_strategy is TradingStrategy.DIRECTIONAL:
            open_pos = self.account.get_position(state.symbol, state.exchange)
            directional_plan = self.directional_policy.evaluate(
                state,
                result,
                entry_thesis=None if open_pos is None else open_pos.directional_thesis,
                directional_state=directional_state,
            )
            candidate = directional_plan.candidate
            logger.info(
                "DIRECTIONAL candidate=%s p_1m=%s consensus=%s conflict=%s reason=%s",
                directional_plan.candidate.value,
                directional_plan.p_up.get("1m"),
                directional_plan.direction_consensus,
                directional_plan.direction_conflict,
                directional_plan.policy_reason,
            )
        else:
            candidate = self.policy.propose(state, result)
        risked = self.risk.evaluate(state, candidate, self.account, result, now_ms=now)
        self.last_state = state
        self.last_result = result
        self.last_risk = risked
        self.last_plan = plan
        self.last_directional = directional_plan
        self.last_directional_state = directional_state
        self.last_jev_at_ms = now
        await self._note_decision_events(state, result, risked)
        logger.info(
            "POLICY %s:%s candidate=%s final=%s blocked=%s source=%s",
            state.exchange,
            state.symbol,
            risked.candidate_action.value,
            risked.final_action.value,
            risked.blocked_reason,
            risked.source.value,
        )
        decided_at_ms = wall_clock_ms()
        policy_reason = None if plan is None else plan.policy_reason
        if directional_plan is not None:
            policy_reason = directional_plan.policy_reason
        latency_payload = {
            "market_state_created_at_ms": float(now),
            "jev_latency_ms": result.latency_ms,
            "decision_calculated_at_ms": float(decided_at_ms),
        }
        sample_valid, invalid_reason = research_sample_status(state, result)
        if directional_state is not None and sample_valid:
            required_gap = missing_required(directional_state)
            if required_gap:
                sample_valid, invalid_reason = False, INVALID_CONTEXT
        scalp_payload = None
        if plan is not None:
            scalp_payload = plan_payload(
                plan,
                latency=latency_payload,
                reference_price=state.price,
                timestamp=state.timestamp,
            )
        scalp_payload = annotate_research_sample(
            scalp_payload, valid=sample_valid, reason=invalid_reason
        )
        directional_payload = None
        if directional_plan is not None and directional_state is not None:
            forecast = dict(directional_plan.forecast)
            forecast["active_candles"] = (directional_state.get("price_action") or {}).get(
                "active_candles"
            )
            forecast["sigma_bps"] = directional_state.get("sigma_bps")
            forecast["reference_mid"] = directional_state.get("microstructure", {}).get("mid")
            forecast["reference_ts"] = now
            forecast["returns"] = (directional_state.get("price_action") or {}).get("returns")
            forecast["research_sample_valid"] = sample_valid
            forecast["invalid_reason"] = invalid_reason
            directional_payload = {
                "question_version": DIRECTIONAL_QUESTION_VERSION,
                "strategy_version": DIRECTIONAL_STRATEGY_VERSION,
                "feature_schema_version": DIRECTIONAL_FEATURE_SCHEMA,
                "feature_state": serialize_directional_state(directional_state),
                "inspector": compact_inspector(directional_state),
                "forecast": forecast,
                "research_sample_valid": sample_valid,
                "invalid_reason": invalid_reason,
                "policy": {
                    "candidate": directional_plan.candidate.value,
                    "reason": directional_plan.policy_reason,
                    "entry_reason": directional_plan.entry_reason,
                    "hold_reason": directional_plan.hold_reason,
                    "close_reason": directional_plan.close_reason,
                    "reverse_reason": directional_plan.reverse_reason,
                    "confirmations": directional_plan.confirmations,
                    "direction_consensus": directional_plan.direction_consensus,
                    "direction_conflict": directional_plan.direction_conflict,
                    "entry_thesis": directional_plan.entry_thesis,
                    "position_thesis": directional_plan.position_thesis,
                    "signals": {
                        "ENTRY": directional_plan.entry_signal,
                        "HOLD": directional_plan.hold_signal,
                        "CLOSE": directional_plan.close_signal,
                        "REVERSE": directional_plan.reverse_signal,
                    },
                    "recommended": directional_plan.recommended,
                },
                "rolling_outcomes": {},
                "candle_outcomes": {},
                "persistence_outcomes": {},
                "counterfactuals": {},
                "execution": {
                    "position_open": state.position.side is not Side.FLAT,
                    "risk_blocked": risked.blocked_reason is not None,
                    "risk_block_reason": risked.blocked_reason,
                },
            }
        try:
            decision_id = await self.repo.persist_tick(
                state,
                result,
                candidate_action=risked.candidate_action.value,
                final_action=risked.final_action.value,
                blocked_reason=risked.blocked_reason or policy_reason,
                scalp_payload=scalp_payload,
                directional_payload=directional_payload,
            )
        except Exception:
            logger.exception(
                "Persistence failed for %s:%s — continuing without this row",
                state.exchange,
                state.symbol,
            )
            return
        if self.paper is not None:
            await self._maybe_execute(
                state,
                risked,
                decision_id,
                now,
                bid,
                ask,
                plan,
                decided_at_ms,
                directional_plan,
            )
            await self.repo.save_paper_account(self.account)
            await self.repo.persist_equity_snapshot(self.account, now)

    async def _maybe_execute(
        self,
        state: MarketState,
        risked: RiskDecision,
        decision_id: str,
        now: int,
        bid: float | None,
        ask: float | None,
        plan: ScalpPlan | None,
        decided_at_ms: int,
        directional_plan: DirectionalPlan | None = None,
    ) -> None:
        assert self.paper is not None
        if risked.final_action not in EXECUTABLE_ACTIONS:
            return
        if bid is None or ask is None or not self.adapter.runtime(state.symbol).book.valid:
            logger.warning(
                "Skipping %s for %s:%s — book not executable",
                risked.final_action.value,
                state.exchange,
                state.symbol,
            )
            return
        already = await self.repo.has_orders_for_decision(decision_id)
        try:
            report = self.paper.execute(
                account=self.account,
                risk=risked,
                symbol=state.symbol,
                exchange=state.exchange,
                bid=bid,
                ask=ask,
                now_ms=now,
                decision_id=decision_id,
                already_processed=already,
            )
        except Exception:
            logger.exception("Paper execution failed for decision %s", decision_id)
            return
        fill_at_ms = wall_clock_ms()
        self.fill_latencies.record(float(fill_at_ms - decided_at_ms))
        if report is None:
            return
        if report.skipped_duplicate:
            logger.warning("Idempotent skip for decision %s", decision_id)
            return
        report = _annotate_scalp_execution(report, plan, risked)
        report = _annotate_directional_execution(report, directional_plan, risked)
        await self._emit("paper_order", symbol=state.symbol, exchange=state.exchange)
        await self._emit("paper_fill", symbol=state.symbol, exchange=state.exchange)
        await self.repo.persist_execution(
            report=report, now_ms=now, symbol=state.symbol, exchange=state.exchange
        )
        await self.repo.upsert_open_position(
            self.account.get_position(state.symbol, state.exchange),
            symbol=state.symbol,
            exchange=state.exchange,
            now_ms=now,
        )
        if report.closed is not None:
            logger.info(
                "PAPER TRADE %s %s qty=%s entry=%s exit=%s gross=%s fees=%s net=%s source=%s",
                report.closed.side.value,
                f"{state.exchange}:{state.symbol}",
                report.closed.quantity,
                report.closed.entry_price,
                report.closed.exit_price,
                report.closed.gross_pnl,
                report.closed.total_fees,
                report.closed.net_pnl,
                report.closed.exit_source,
            )
        if report.opened is not None:
            logger.info(
                "PAPER OPEN %s %s qty=%s entry=%s fee=%s source=%s",
                report.opened.side.value,
                f"{state.exchange}:{state.symbol}",
                report.opened.quantity,
                report.opened.entry_price,
                report.opened.entry_fee,
                report.opened.entry_source,
            )
        if report.remained_flat_after_reverse:
            logger.warning("Reverse close filled but new open blocked; remaining FLAT")

    async def _health_monitor(self) -> None:
        while not self._stop.is_set():
            now = wall_clock_ms()
            for symbol in self.settings.symbols:
                runtime = self.adapter.runtime(symbol)
                bid = runtime.book.best_bid
                ask = runtime.book.best_ask
                if bid is not None and ask is not None:
                    self.account.mark(
                        symbol=symbol,
                        exchange=self.settings.trading_exchange,
                        bid=bid,
                        ask=ask,
                        now_ms=now,
                    )
                state = self._snapshot(symbol, now)
                if state.health.stale:
                    if symbol not in self._stale_symbols:
                        logger.warning(
                            "Stale market data for %s:%s reasons=%s — trading blocked",
                            state.exchange,
                            state.symbol,
                            list(state.health.reasons),
                        )
                        self._stale_symbols.add(symbol)
                        await self._emit(
                            "stale_market_data",
                            level="WARNING",
                            symbol=state.symbol,
                            exchange=state.exchange,
                        )
                elif symbol in self._stale_symbols:
                    logger.info("Market data recovered for %s:%s", state.exchange, state.symbol)
                    self._stale_symbols.remove(symbol)
                    await self._emit(
                        "binance_recovered",
                        symbol=state.symbol,
                        exchange=state.exchange,
                    )
            await self._note_socket_events()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except TimeoutError:
                continue

    async def _outcome_loop(self) -> None:
        while not self._stop.is_set():
            try:
                updated = await enrich_decision_outcomes(self.repo, wall_clock_ms())
                if updated:
                    logger.info("Enriched %s historical Jev decisions with later prices", updated)
                dir_updated = await self.repo.enrich_directional_outcomes(wall_clock_ms())
                if dir_updated:
                    logger.info("Enriched %s directional forecasts", dir_updated)
                if self.settings.trading_strategy is TradingStrategy.DIRECTIONAL:
                    dir_rows = await self.repo.list_directional_forecasts(self.run_id)
                    self.calibration_view = summarize_directional_calibration(dir_rows)
                    self.prediction_research_view = dashboard_prediction_research(dir_rows)
                else:
                    self.calibration_view = await self.repo.calibration_summary(self.run_id)
            except Exception:
                logger.exception("Outcome enrichment failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=5.0)
            except TimeoutError:
                continue

    def snapshot_now(self, symbol: str, now_ms: int) -> MarketState:
        return self._snapshot(symbol, now_ms)

    async def operator_snapshot(self) -> dict[str, Any]:
        now_ms = wall_clock_ms()
        decisions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        run_closed: list[ClosedTrade] = []
        experiment_closed: list[ClosedTrade] = []
        db_ok = False
        try:
            async with asyncio.timeout(5):
                await self.repo.ping()
                db_ok = True
                run_filter = (
                    self.run_id
                    if self.settings.trading_strategy is TradingStrategy.DIRECTIONAL
                    else None
                )
                decisions = await self.repo.list_recent_decisions(20, run_id=run_filter)
                trades = await self.repo.list_recent_trades(20, run_id=run_filter)
                events = await self.repo.list_recent_events(30, run_id=run_filter)
                if self.settings.trading_strategy is TradingStrategy.DIRECTIONAL and self.run_id:
                    run_rows = await self.repo.list_trades_for_run(self.run_id)
                    run_closed = [_closed_from_row(row) for row in run_rows]
                    experiment_closed = [
                        _closed_from_row(row)
                        for row in await self.repo.list_trades_for_experiment(
                            self.settings.experiment_id
                        )
                    ]
        except Exception:
            logger.exception("Operator snapshot database read failed")
        return build_operator_snapshot(
            self,
            now_ms=now_ms,
            db_ok=db_ok,
            decisions=decisions,
            trades=trades,
            events=events,
            calibration=self.calibration_view,
            run_closed_trades=run_closed,
            experiment_closed_trades=experiment_closed,
        )

    async def _emit(
        self,
        event_type: str,
        *,
        level: str = "INFO",
        symbol: str | None = None,
        exchange: str | None = None,
    ) -> None:
        try:
            await self.repo.record_system_event(
                event_type=event_type,
                level=level,
                symbol=symbol,
                exchange=exchange,
            )
        except Exception:
            logger.exception("Failed to persist system event %s", event_type)

    async def _note_socket_events(self) -> None:
        public = self.adapter.ws_public_state()
        market = self.adapter.ws_market_state()
        if public != self._ws_public_state:
            await self._emit(
                _socket_event(public),
                level="WARNING" if public != "connected" else "INFO",
            )
            if public == "reconnecting":
                self.ws_disconnects += 1
            if public == "connected" and self._ws_public_state == "reconnecting":
                self.ws_recoveries += 1
            self._ws_public_state = public
        if market != self._ws_market_state:
            await self._emit(
                _socket_event(market, prefix="binance_market"),
                level="WARNING" if market != "connected" else "INFO",
            )
            self._ws_market_state = market

    async def _note_decision_events(
        self,
        state: MarketState,
        result: JevCallResult,
        risked: RiskDecision,
    ) -> None:
        if state.health.warmup_ready and not self._warmup_ready:
            self._warmup_ready = True
            await self._emit(
                "warmup_completed",
                symbol=state.symbol,
                exchange=state.exchange,
            )
        jev_error = result.status in {
            JevCallStatus.TIMEOUT,
            JevCallStatus.INVALID_RESPONSE,
            JevCallStatus.UNAVAILABLE,
        }
        if jev_error and not self._jev_was_error:
            event = "jev_timeout" if result.status is JevCallStatus.TIMEOUT else "jev_error"
            await self._emit(
                event,
                level="WARNING",
                symbol=state.symbol,
                exchange=state.exchange,
            )
        if not jev_error and self._jev_was_error:
            await self._emit(
                "jev_recovered",
                symbol=state.symbol,
                exchange=state.exchange,
            )
        self._jev_was_error = jev_error
        if risked.blocked_reason and risked.blocked_reason != self._last_blocked:
            await self._emit(
                "risk_block",
                level="WARNING",
                symbol=state.symbol,
                exchange=state.exchange,
            )
        self._last_blocked = risked.blocked_reason

    def _note_scalp_counts(self, plan: ScalpPlan) -> None:
        if plan.best is not None and plan.best.net_ev_bps >= self.settings.scalp_min_net_ev_bps:
            self.positive_ev_count += 1
        if plan.policy_reason == "net_ev_below_edge":
            self.rejected_cost_count += 1

    def _snapshot(self, symbol: str, now_ms: int) -> MarketState:
        runtime = self.adapter.runtime(symbol)
        price = runtime.book.mid if runtime.book.mid is not None else runtime.buffer.last_mid
        position = self.account.position_state(
            symbol,
            self.settings.trading_exchange,
            now_ms=now_ms,
            current_price=price,
        )
        return self.features.snapshot(
            buffer=runtime.buffer,
            book=runtime.book,
            now_ms=now_ms,
            stale_timeout_ms=self.settings.stale_timeout_ms,
            ws_public_connected=self.adapter.ws_public_connected,
            ws_market_connected=self.adapter.ws_market_connected,
            kill_switch=self.settings.kill_switch,
            position=position,
        )


def _annotate_scalp_execution(
    report: ExecutionReport,
    plan: ScalpPlan | None,
    risked: RiskDecision,
) -> ExecutionReport:
    opened = report.opened
    if opened is not None and plan is not None and plan.best is not None:
        opened.entry_horizon_seconds = plan.best.horizon_seconds
        opened.entry_net_ev_bps = plan.best.net_ev_bps
    closed = report.closed
    if closed is None:
        return report
    return replace(report, closed=replace(closed, exit_reason=_scalp_exit_reason(risked, plan)))


def _scalp_exit_reason(risked: RiskDecision, plan: ScalpPlan | None) -> str:
    if risked.source is ActionSource.RISK_ENGINE:
        return risked.blocked_reason or "RISK_ENGINE"
    if risked.final_action is Action.REVERSE:
        return "JEV_REVERSE"
    if risked.final_action is Action.CLOSE:
        return "JEV_CLOSE"
    if plan is not None and plan.policy_reason:
        return plan.policy_reason
    return risked.final_action.value


def _directional_exit_source(risked: RiskDecision) -> str:
    if risked.source is ActionSource.RISK_ENGINE:
        return "RISK_ENGINE"
    if risked.final_action in {Action.CLOSE, Action.REVERSE}:
        return "DIRECTIONAL_POLICY"
    return "SYSTEM"


def _annotate_directional_execution(
    report: ExecutionReport,
    plan: DirectionalPlan | None,
    risked: RiskDecision,
) -> ExecutionReport:
    if plan is None:
        return report
    opened = report.opened
    if opened is not None and plan.entry_thesis:
        opened.directional_thesis = dict(plan.entry_thesis)
        primary = plan.entry_thesis.get("entry_primary_horizon")
        seconds = HORIZON_SECONDS_BY_LABEL.get(str(primary))
        if seconds is not None:
            opened.entry_horizon_seconds = seconds
    closed = report.closed
    if closed is None:
        return report
    thesis = plan.position_thesis if isinstance(plan.position_thesis, dict) else {}
    entry = plan.entry_thesis if isinstance(plan.entry_thesis, dict) else {}
    seeded = closed.research if isinstance(closed.research, dict) else {}
    primary = thesis.get("primary_horizon") or entry.get("entry_primary_horizon")
    research = {
        **seeded,
        "entry_thesis": entry or seeded.get("entry_thesis"),
        "close_thesis": thesis,
        "thesis_score_entry": thesis.get("entry_thesis_score") or entry.get("entry_thesis_score"),
        "thesis_score_close": thesis.get("current_thesis_score"),
        "thesis_decay": thesis.get("thesis_decay"),
        "primary_horizon": primary,
        "primary_horizon_elapsed_multiple": thesis.get("primary_horizon_elapsed_multiple"),
        "mfe_bps": closed.mfe_bps,
        "mae_bps": closed.mae_bps,
        "net_pnl": closed.net_pnl,
        "gross_pnl": closed.gross_pnl,
        "capture_ratio": capture_ratio(closed.mfe_bps, closed.gross_pnl_bps),
        "profit_giveback_bps": thesis.get("profit_giveback_bps"),
        "position_age_seconds": closed.holding_seconds,
        "close_reason": plan.close_reason or thesis.get("reason"),
        "reverse_reason": plan.reverse_reason,
        "recommended": plan.recommended,
        "exit_source": _directional_exit_source(risked),
    }
    if risked.source is ActionSource.RISK_ENGINE:
        reason = risked.blocked_reason or "RISK_ENGINE"
        source = "RISK_ENGINE"
    else:
        reason = (
            plan.close_reason
            or plan.reverse_reason
            or closed.exit_reason
            or "DIRECTIONAL_POLICY"
        )
        source = _directional_exit_source(risked)
    research["exit_source"] = source
    research["close_reason"] = reason
    horizon_seconds = HORIZON_SECONDS_BY_LABEL.get(str(primary)) if primary else None
    return replace(
        report,
        closed=replace(
            closed,
            exit_reason=reason,
            exit_source=source,
            research=research,
            entry_horizon_seconds=(
                horizon_seconds if horizon_seconds is not None else closed.entry_horizon_seconds
            ),
        ),
    )
