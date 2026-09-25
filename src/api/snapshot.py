from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.analytics.session import scalp_session_metrics
from src.config.settings import TradingStrategy
from src.decision.action_values import ACTION_VALUE_REFERENCE
from src.decision.ev import opportunity_matrix
from src.directional.constants import STRATEGY_LABEL
from src.directional.coverage import feature_coverage
from src.directional.serialize import compact_inspector
from src.experiment.fingerprint import (
    active_feature_schema_version,
    active_question_version,
    active_strategy_version,
)
from src.jev.models import JevCallStatus
from src.market.models import PRIMARY_WARMUP_MS, MarketState, Side
from src.portfolio.account import ClosedTrade
from src.storage.database import database_kind

if TYPE_CHECKING:
    from src.trader.engine import TraderEngine

JEV_ERROR_STATUSES = {
    JevCallStatus.TIMEOUT,
    JevCallStatus.INVALID_RESPONSE,
    JevCallStatus.UNAVAILABLE,
    JevCallStatus.SKIPPED_NO_API_KEY,
}


def _iso(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).isoformat()


def _seconds_ago(ts_ms: int | None, now_ms: int) -> float | None:
    if ts_ms is None:
        return None
    return max(0.0, (now_ms - ts_ms) / 1000.0)


def _clock(ms: int) -> str:
    ms = max(0, ms)
    return f"{ms // 60_000:02d}:{(ms // 1000) % 60:02d}"


def _binance_ws_status(engine: TraderEngine, state: MarketState | None) -> str:
    public = engine.adapter.ws_public_state()
    market = engine.adapter.ws_market_state()
    if "reconnecting" in {public, market}:
        return "RECONNECTING"
    if state is not None and state.health.stale:
        return "STALE"
    if public == "connected" and market == "connected":
        return "CONNECTED"
    if public == "stopped" and market == "stopped":
        return "STOPPED"
    return "RECONNECTING"


def _jev_health(engine: TraderEngine) -> str:
    result = engine.last_result
    if result is None:
        return "WAITING"
    if result.status in JEV_ERROR_STATUSES:
        return "ERROR"
    return "HEALTHY"


def _warmup(engine: TraderEngine, now_ms: int, state: MarketState | None) -> tuple[str, str]:
    symbol = engine.settings.symbols[0]
    runtime = engine.adapter.runtime(symbol)
    live = runtime.buffer.live_mids()
    elapsed = 0 if not live else max(0, now_ms - live[0].ts_ms)
    elapsed = min(elapsed, PRIMARY_WARMUP_MS)
    ready = bool(state is not None and state.health.warmup_ready)
    label = "READY" if ready else "INCOMPLETE"
    return label, f"{_clock(elapsed)} / {_clock(PRIMARY_WARMUP_MS)}"


def _market_view(state: MarketState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    micro = state.microstructure
    vol = state.volatility
    returns = state.returns
    return {
        "price": state.price,
        "best_bid": state.best_bid,
        "best_ask": state.best_ask,
        "spread_bps": micro.spread_bps,
        "r_5s": returns.r_5s,
        "r_15s": returns.r_15s,
        "r_30s": returns.r_30s,
        "r_1m": returns.r_1m,
        "r_3m": returns.r_3m,
        "r_5m": returns.r_5m,
        "rv_30s": vol.rv_30s,
        "rv_1m": vol.rv_1m,
        "rv_5m": vol.rv_5m,
        "book_imbalance_1": micro.book_imbalance_1,
        "book_imbalance_5": micro.book_imbalance_5,
        "book_imbalance_10": micro.book_imbalance_10,
        "aggressive_flow_5s": micro.aggressive_flow_5s,
        "aggressive_flow_30s": micro.aggressive_flow_30s,
        "aggressive_flow_1m": micro.aggressive_flow_1m,
        "trade_velocity": micro.trade_velocity,
        "volume_velocity": micro.volume_velocity,
        "cvd_1m": micro.cvd_1m,
        "cvd_5m": micro.cvd_5m,
        "r_15m": returns.r_15m,
        "r_1h": returns.r_1h,
        "price_vs_vwap_1m_bps": (
            None if state.intelligence is None else state.intelligence.price_vs_vwap_1m_bps
        ),
        "price_vs_vwap_5m_bps": (
            None if state.intelligence is None else state.intelligence.price_vs_vwap_5m_bps
        ),
        "vol_ratio_short_long": (
            None if state.intelligence is None else state.intelligence.vol_ratio_short_long
        ),
        "flow_accel_5s_vs_30s": (
            None if state.intelligence is None else state.intelligence.flow_accel_5s_vs_30s
        ),
        "flow_accel_30s_vs_1m": (
            None if state.intelligence is None else state.intelligence.flow_accel_30s_vs_1m
        ),
        "imbalance_1_change": (
            None if state.intelligence is None else state.intelligence.imbalance_1_change
        ),
        "regime_direction": None
        if state.intelligence is None
        else state.intelligence.regime.direction,
        "regime_volatility": None
        if state.intelligence is None
        else state.intelligence.regime.volatility,
        "regime_liquidity": None
        if state.intelligence is None
        else state.intelligence.regime.liquidity,
        "range_position_1m": state.ranges.range_position_1m,
        "range_position_5m": state.ranges.range_position_5m,
        "dist_high_1m_bps": state.ranges.dist_from_high_1m_bps,
        "dist_low_1m_bps": state.ranges.dist_from_low_1m_bps,
        "dist_high_5m_bps": state.ranges.dist_from_high_5m_bps,
        "dist_low_5m_bps": state.ranges.dist_from_low_5m_bps,
        "dist_high_15m_bps": state.ranges.dist_from_high_15m_bps,
        "dist_low_15m_bps": state.ranges.dist_from_low_15m_bps,
        "expected_exit_spread_bps": None
        if state.intelligence is None
        else state.intelligence.expected_exit_spread_bps,
    }


def _jev_view(engine: TraderEngine, state: MarketState | None) -> dict[str, Any] | None:
    result = engine.last_result
    risk = engine.last_risk
    if result is None and risk is None:
        return None
    decision = None if result is None else result.decision
    blocked = None if risk is None else risk.blocked_reason
    if blocked is None and state is not None and state.health.reasons:
        blocked = state.health.reasons[0]
    trading_allowed = None if state is None else state.health.trading_allowed
    configured = engine.settings.jev_model
    resolved = None if decision is None else decision.model
    scalp = engine.settings.trading_strategy is TradingStrategy.SCALP
    directional = engine.settings.trading_strategy is TradingStrategy.DIRECTIONAL
    kind = "SCALP" if scalp else "DIRECTIONAL" if directional else "BASELINE"
    base: dict[str, Any] = {
        "kind": kind,
        "status": None if result is None else result.status.value,
        "candidate_action": None if risk is None else risk.candidate_action.value,
        "final_action": None if risk is None else risk.final_action.value,
        "trading_allowed": trading_allowed,
        "blocked_reason": blocked,
        "configured_model": configured,
        "resolved_response_model": resolved,
        "model": resolved or configured,
        "noise": None if decision is None else decision.noise_probability,
    }
    if directional:
        dir_plan = engine.last_directional
        forecast = {} if dir_plan is None else dict(dir_plan.forecast)
        base.update(
            {
                "term_structure": forecast.get("term_structure") or [],
                "p_up": forecast.get("p_up") or {},
                "p_down": forecast.get("p_down") or {},
                "candle_up": forecast.get("candle_up") or {},
                "persistence": forecast.get("persistence") or {},
                "magnitude": forecast.get("magnitude") or {},
                "diagnostics": forecast.get("diagnostics") or {},
                "direction_consensus": None if dir_plan is None else dir_plan.direction_consensus,
                "direction_conflict": None if dir_plan is None else dir_plan.direction_conflict,
                "probability_consistency_warnings": forecast.get(
                    "probability_consistency_warnings"
                )
                or [],
                "answer_audit": forecast.get("answer_audit") or [],
                "research_sample_valid": forecast.get("research_sample_valid"),
                "invalid_reason": forecast.get("invalid_reason"),
                "policy_reason": None if dir_plan is None else dir_plan.policy_reason,
                "entry_reason": None if dir_plan is None else dir_plan.entry_reason,
                "hold_reason": None if dir_plan is None else dir_plan.hold_reason,
                "close_reason": None if dir_plan is None else dir_plan.close_reason,
                "reverse_reason": None if dir_plan is None else dir_plan.reverse_reason,
                "recommended_action": None if dir_plan is None else dir_plan.candidate.value,
                "position_thesis": None if dir_plan is None else dir_plan.position_thesis,
                "entry_thesis": None if dir_plan is None else dir_plan.entry_thesis,
                "signals": None
                if dir_plan is None
                else {
                    "ENTRY": dir_plan.entry_signal,
                    "HOLD": dir_plan.hold_signal,
                    "CLOSE": dir_plan.close_signal,
                    "REVERSE": dir_plan.reverse_signal,
                },
            }
        )
        return base
    if not scalp:
        base.update(
            {
                "desired_exposure": None if decision is None else decision.exposure_choice,
                "long_probability": None if decision is None else decision.long_probability,
                "short_probability": None if decision is None else decision.short_probability,
                "evidence": None if decision is None else decision.evidence_probability,
                "continuation": None if decision is None else decision.continuation_probability,
                "reversal": None if decision is None else decision.reversal_probability,
                "exhaustion": None if decision is None else decision.exhaustion_probability,
            }
        )
        return base
    plan = engine.last_plan
    horizons: list[dict[str, Any]] = []
    if plan is not None:
        by_h: dict[int, dict[str, Any]] = {}
        for item in plan.opportunities:
            row = by_h.setdefault(
                item.horizon_seconds,
                {
                    "horizon_seconds": item.horizon_seconds,
                    "p_up_first": item.p_up,
                    "p_down_first": item.p_down,
                    "p_neither": item.p_neither,
                    "p_long_target": None,
                    "p_long_stop": None,
                    "p_long_neither": None,
                    "p_short_target": None,
                    "p_short_stop": None,
                    "p_short_neither": None,
                    "long_neither_return_bps": None,
                    "short_neither_return_bps": None,
                    "neither_return_bps": item.neither_return_bps,
                    "neither_payoff_method": item.neither_payoff_method,
                    "score_raw": item.score_raw,
                    "score_mapping_version": item.score_mapping_version,
                    "target_bps": item.target_bps,
                    "stop_bps": item.stop_bps,
                    "volatility_target_bps": item.volatility_target_bps,
                    "cost_floor_bps": item.cost_floor_bps,
                    "final_target_bps": item.target_bps,
                    "target_bound_by_cost_floor": item.target_bound_by_cost_floor,
                    "horizon_cost_constrained": item.horizon_cost_constrained,
                    "estimated_entry_cost_bps": item.cost_bps,
                    "long_entry_net_ev_bps": None,
                    "short_entry_net_ev_bps": None,
                    "long_gross_ev_bps": None,
                    "short_gross_ev_bps": None,
                },
            )
            if item.side == "LONG":
                row["long_entry_net_ev_bps"] = item.net_ev_bps
                row["long_gross_ev_bps"] = item.gross_ev_bps
                row["p_long_target"] = item.p_target
                row["p_long_stop"] = item.p_stop
                row["p_long_neither"] = item.p_neither
                row["long_neither_return_bps"] = item.neither_return_bps
            else:
                row["short_entry_net_ev_bps"] = item.net_ev_bps
                row["short_gross_ev_bps"] = item.gross_ev_bps
                row["p_short_target"] = item.p_target
                row["p_short_stop"] = item.p_stop
                row["p_short_neither"] = item.p_neither
                row["short_neither_return_bps"] = item.neither_return_bps
        horizons = [by_h[key] for key in sorted(by_h)]
    base.update(
        {
            "horizons": horizons,
            "hold_net_ev_bps": None if plan is None else plan.hold_net_ev_bps,
            "close_now_value_bps": None if plan is None else plan.close_now_value_bps,
            "reverse_net_ev_bps": None if plan is None else plan.reverse_net_ev_bps,
            "hold_value_bps": None if plan is None else plan.hold_net_ev_bps,
            "close_value_bps": None if plan is None else plan.close_now_value_bps,
            "reverse_value_bps": None if plan is None else plan.reverse_net_ev_bps,
            "action_margin_bps": None if plan is None else plan.action_margin_bps,
            "chosen_action": None if plan is None else plan.candidate.value,
            "action_value_reference": ACTION_VALUE_REFERENCE,
            "entry_net_ev_if_flat_bps": None if plan is None else plan.entry_net_ev_if_flat_bps,
            "hold_advantage_bps": None if plan is None else plan.hold_advantage_bps,
            "policy_reason": None if plan is None else plan.policy_reason,
            "recommended_action": None if plan is None else plan.candidate.value,
            "score_mapping_version": None
            if plan is None or not plan.forecasts
            else plan.forecasts[0].score_mapping_version,
        }
    )
    return base


def _position_view(engine: TraderEngine, now_ms: int, state: MarketState | None) -> dict[str, Any]:
    symbol = engine.settings.symbols[0]
    exchange = engine.settings.trading_exchange
    price = None if state is None else state.price
    position = engine.account.position_state(
        symbol, exchange, now_ms=now_ms, current_price=price
    )
    if position.side is Side.FLAT:
        return {"side": Side.FLAT.value, "open": False}
    return {
        "side": position.side.value,
        "open": True,
        "quantity": position.quantity,
        "entry_price": position.entry_price,
        "current_executable_mark": position.current_price,
        "age_seconds": position.age_seconds,
        "unrealized_pnl": position.unrealized_pnl,
        "unrealized_pnl_bps": position.unrealized_pnl_bps,
        "mfe_bps": position.max_favorable_excursion_bps,
        "mae_bps": position.max_adverse_excursion_bps,
        "profit_giveback_bps": position.profit_giveback_bps,
        "fees_paid": position.fees_paid,
        "notional": position.entry_notional,
        "hold_net_ev_bps": (
            None if engine.last_plan is None else engine.last_plan.hold_net_ev_bps
        ),
        "close_now_value_bps": (
            None if engine.last_plan is None else engine.last_plan.close_now_value_bps
        ),
        "reverse_net_ev_bps": (
            None if engine.last_plan is None else engine.last_plan.reverse_net_ev_bps
        ),
        "hold_value_bps": (
            None if engine.last_plan is None else engine.last_plan.hold_net_ev_bps
        ),
        "close_value_bps": (
            None if engine.last_plan is None else engine.last_plan.close_now_value_bps
        ),
        "reverse_value_bps": (
            None if engine.last_plan is None else engine.last_plan.reverse_net_ev_bps
        ),
        "action_margin_bps": (
            None if engine.last_plan is None else engine.last_plan.action_margin_bps
        ),
        "chosen_action": (
            None if engine.last_plan is None else engine.last_plan.candidate.value
        ),
        "action_value_reference": ACTION_VALUE_REFERENCE,
        "forward_ev_bps": (
            None if engine.last_plan is None else engine.last_plan.hold_net_ev_bps
        ),
        "opposite_ev_bps": (
            None if engine.last_plan is None else engine.last_plan.reverse_net_ev_bps
        ),
        "recommended_action": None
        if engine.last_risk is None
        else engine.last_risk.final_action.value,
    }


def _last_market_update_ms(engine: TraderEngine) -> int | None:
    stamps = [
        engine.adapter.last_public_message_at_ms(),
        engine.adapter.last_market_message_at_ms(),
    ]
    present = [stamp for stamp in stamps if stamp is not None]
    if not present:
        return None
    return max(present)


def _safety(engine: TraderEngine) -> dict[str, Any]:
    scalp = engine.settings.trading_strategy is TradingStrategy.SCALP
    directional = engine.settings.trading_strategy is TradingStrategy.DIRECTIONAL
    if directional:
        lines = ["PAPER", STRATEGY_LABEL, "LIVE BLOCKED"]
    elif scalp:
        lines = ["PAPER TRADING", "SCALP EXPERIMENT", "NO REAL ORDERS", "LIVE DISABLED"]
    else:
        lines = ["PAPER TRADING", "NO REAL ORDERS", "LIVE TRADING DISABLED"]
    return {
        "banner": " — ".join(lines),
        "lines": lines,
        "trading_mode": engine.settings.trading_mode.value,
        "strategy": engine.settings.trading_strategy.value,
        "live": "BLOCKED",
    }


def _cost_view(engine: TraderEngine, state: MarketState | None) -> dict[str, Any]:
    plan = engine.last_plan
    if plan is not None:
        cost = plan.cost
        return {
            "label": "ASSUMED PAPER COST MODEL" if cost.assumed else "PAPER COST MODEL",
            "assumed": cost.assumed,
            "source": cost.source,
            "maker_fee_bps": cost.maker_fee_bps,
            "taker_fee_bps": cost.taker_fee_bps,
            "slippage_bps": cost.slippage_bps,
            "spread_bps": cost.spread_bps,
            "entry_half_spread_bps": cost.entry_half_spread_bps,
            "expected_exit_half_spread_bps": cost.expected_exit_half_spread_bps,
            "entry_leg_cost_bps": cost.entry_leg_cost_bps,
            "exit_leg_cost_bps": cost.exit_leg_cost_bps,
            "round_trip_cost_bps": cost.round_trip_cost_bps,
            "funding_accounting": (
                "accrual_enabled"
                if engine.settings.paper_funding_enabled
                else "disabled_zero"
            ),
        }
    spread = None if state is None else state.microstructure.spread_bps
    taker = engine.settings.execution_taker_fee_bps
    maker = engine.settings.execution_maker_fee_bps
    slip = engine.settings.slippage_bps
    spread_v = 0.0 if spread is None else spread
    round_trip = 2.0 * taker + spread_v + 2.0 * slip
    assumed = engine.settings.trading_strategy in {
        TradingStrategy.SCALP,
        TradingStrategy.DIRECTIONAL,
    }
    return {
        "label": "ASSUMED PAPER COST MODEL" if assumed else "PAPER COST MODEL",
        "assumed": assumed and engine.settings.scalp_cost_model_assumed,
        "source": engine.settings.scalp_cost_model_source if assumed else "paper_env_fees",
        "maker_fee_bps": maker,
        "taker_fee_bps": taker,
        "slippage_bps": slip,
        "spread_bps": spread_v,
        "round_trip_cost_bps": round_trip,
    }


def _best_opportunity_view(engine: TraderEngine) -> dict[str, Any] | None:
    plan = engine.last_plan
    if plan is None:
        return None
    best = plan.best
    min_ev = engine.settings.scalp_min_net_ev_bps
    action = plan.candidate.value
    note = None
    if plan.policy_reason == "net_ev_below_edge":
        note = "NO TRADE — NET EV BELOW REQUIRED EDGE"
        action = "NO_ACTION"
    elif plan.candidate.value == "NO_ACTION" and plan.policy_reason:
        note = plan.policy_reason
    if best is None:
        return {
            "side": None,
            "horizon_seconds": None,
            "action": action,
            "note": note or "NO_ACTION / FLAT",
        }
    return {
        "side": best.side,
        "horizon_seconds": best.horizon_seconds,
        "target_bps": best.target_bps,
        "stop_bps": best.stop_bps,
            "p_up": best.p_up,
            "p_down": best.p_down,
            "p_neither": best.p_neither,
            "p_target": best.p_target,
            "p_stop": best.p_stop,
            "neither_return_bps": best.neither_return_bps,
            "neither_payoff_method": best.neither_payoff_method,
            "score_raw": best.score_raw,
            "expected_return_bps": best.expected_return_bps,
            "gross_ev_bps": best.gross_ev_bps,
            "cost_bps": best.cost_bps,
            "net_ev_bps": best.net_ev_bps,
            "cost_kind": best.cost_kind,
            "target_bound_by_cost_floor": best.target_bound_by_cost_floor,
            "horizon_cost_constrained": best.horizon_cost_constrained,
        "min_net_ev_bps": min_ev,
        "action": action,
        "note": note,
        "ranking": [
            {
                "side": item.side,
                "horizon_seconds": item.horizon_seconds,
                "net_ev_bps": item.net_ev_bps,
            }
            for item in plan.opportunities
        ],
    }


def _latency_view(engine: TraderEngine) -> dict[str, Any]:
    jev = engine.jev_latencies.stats()
    fill = engine.fill_latencies.stats()
    shortest = min(engine.settings.scalp_horizons) if engine.settings.scalp_horizons else 15
    warning = None
    if jev.p50_ms is not None and jev.p50_ms > shortest * 1000.0 * 0.3:
        warning = (
            f"Jev p50 {jev.p50_ms:.0f}ms is large versus the {shortest}s horizon. "
            "Horizons were not removed."
        )
    return {
        "jev_count": jev.count,
        "jev_p50_ms": jev.p50_ms,
        "jev_p95_ms": jev.p95_ms,
        "jev_p99_ms": jev.p99_ms,
        "jev_last_ms": jev.last_ms,
        "decision_to_fill_p50_ms": fill.p50_ms,
        "decision_to_fill_p95_ms": fill.p95_ms,
        "decision_to_fill_p99_ms": fill.p99_ms,
        "paper_decision_to_fill_p50_ms": fill.p50_ms,
        "paper_decision_to_fill_p95_ms": fill.p95_ms,
        "paper_decision_to_fill_p99_ms": fill.p99_ms,
        "decision_to_fill_is_paper_simulated": True,
        "warning": warning,
        "shortest_horizon_seconds": shortest,
    }


def _account_view(engine: TraderEngine, now_ms: int) -> dict[str, Any]:
    snap = engine.account.equity_snapshot(now_ms)
    notional = engine.settings.execution_notional
    start = engine.account.starting_balance
    snap["position_notional"] = notional
    snap["exposure_pct"] = None if start == 0 else notional / start
    snap["session_net_pnl_usdt"] = engine.account.net_pnl
    snap["session_return_pct_on_starting_equity"] = (
        None if start == 0 else engine.account.net_pnl / start
    )
    snap["session_return_bps_on_starting_equity"] = (
        None if start == 0 else engine.account.net_pnl / start * 10_000.0
    )
    snap["net_pnl_pct"] = snap["session_return_pct_on_starting_equity"]
    snap["net_pnl_bps"] = snap["session_return_bps_on_starting_equity"]
    snap["metrics_kind"] = "MARK_TO_MARKET_ACCOUNT_METRICS"
    snap["includes_open_position_mtm"] = True
    snap["session_pnl"] = engine.account.net_pnl
    return snap


def _session_metrics(
    engine: TraderEngine,
    now_ms: int,
    trades: list[ClosedTrade] | None = None,
) -> dict[str, Any]:
    started = engine.session_started_ms or engine.started_at_ms or now_ms
    runtime = max(0.0, (now_ms - started) / 1000.0)
    closed = engine.account.closed_trades if trades is None else trades
    metrics = scalp_session_metrics(
        closed,
        runtime_seconds=runtime,
        max_drawdown=engine.account.drawdown,
        starting_balance=engine.account.starting_balance,
        position_notional=engine.settings.execution_notional,
    )
    metrics["positive_ev_opportunities"] = engine.positive_ev_count
    metrics["rejected_cost_opportunities"] = engine.rejected_cost_count
    metrics["ws_disconnects"] = engine.ws_disconnects
    metrics["ws_recoveries"] = engine.ws_recoveries
    metrics["jev_failures"] = engine.jev_failures
    metrics["metrics_kind"] = "REALIZED_TRADE_METRICS"
    metrics["includes_open_position_mtm"] = False
    return metrics


def _session_view(
    engine: TraderEngine,
    now_ms: int,
    *,
    run_trades: list[ClosedTrade] | None = None,
    experiment_trades: list[ClosedTrade] | None = None,
) -> dict[str, Any]:
    directional = engine.settings.trading_strategy is TradingStrategy.DIRECTIONAL
    if directional:
        current_trades = (
            run_trades if run_trades is not None else engine.account.closed_trades
        )
        current = _session_metrics(engine, now_ms, current_trades)
        current["scope"] = "CURRENT RUN"
        current["run_id"] = engine.run_id
        experiment_source = (
            experiment_trades if experiment_trades is not None else current_trades
        )
        experiment = _session_metrics(engine, now_ms, experiment_source)
        experiment["scope"] = "EXPERIMENT TOTAL"
        experiment["experiment_id"] = engine.settings.experiment_id
        current["experiment_total"] = {
            "scope": "EXPERIMENT TOTAL",
            "trades": experiment["trades"],
            "wins": experiment["wins"],
            "losses": experiment["losses"],
            "gross_pnl": experiment["gross_pnl"],
            "fees": experiment["fees"],
            "slippage": experiment["slippage"],
            "net_pnl": experiment["net_pnl"],
            "max_drawdown": experiment["max_drawdown"],
        }
        return current
    return _session_metrics(engine, now_ms)


def _experiment_view(engine: TraderEngine) -> dict[str, Any]:
    directional = engine.settings.trading_strategy is TradingStrategy.DIRECTIONAL
    return {
        "experiment_id": engine.settings.experiment_id,
        "run_id": engine.run_id,
        "config_hash": engine.config_hash,
        "question_version": active_question_version(engine.settings),
        "strategy_version": active_strategy_version(engine.settings),
        "feature_schema_version": active_feature_schema_version(engine.settings),
        "strategy_label": STRATEGY_LABEL if directional else engine.settings.trading_strategy.value,
        "config_frozen": engine.settings.experiment_frozen,
        "action_value_reference": None if directional else ACTION_VALUE_REFERENCE,
    }


def _directional_view(engine: TraderEngine) -> dict[str, Any] | None:
    if engine.settings.trading_strategy is not TradingStrategy.DIRECTIONAL:
        return None
    state = engine.last_directional_state or {}
    plan = engine.last_directional
    forecast = {} if plan is None else dict(plan.forecast)
    active = (state.get("price_action") or {}).get("active_candles") or {}
    return {
        "term_structure": forecast.get("term_structure") or [],
        "active_candles": active,
        "candle_up": forecast.get("candle_up") or {},
        "persistence": forecast.get("persistence") or {},
        "magnitude": forecast.get("magnitude") or {},
        "diagnostics": forecast.get("diagnostics") or {},
        "regime": state.get("regime") or {},
        "inspector": None if not state else compact_inspector(state),
        "coverage": None if not state else feature_coverage(state),
        "answer_audit": forecast.get("answer_audit") or [],
        "research_sample_valid": forecast.get("research_sample_valid"),
        "invalid_reason": forecast.get("invalid_reason"),
        "raw_state": state,
        "policy": None
        if plan is None
        else {
            "candidate": plan.candidate.value,
            "entry_reason": plan.entry_reason,
            "hold_reason": plan.hold_reason,
            "close_reason": plan.close_reason,
            "reverse_reason": plan.reverse_reason,
            "consensus": plan.direction_consensus,
            "conflict": plan.direction_conflict,
            "signals": {
                "ENTRY": plan.entry_signal,
                "HOLD": plan.hold_signal,
                "CLOSE": plan.close_signal,
                "REVERSE": plan.reverse_signal,
            },
            "recommended": plan.recommended,
        },
        "entry_thesis": None if plan is None else plan.entry_thesis,
        "position_thesis": None if plan is None else plan.position_thesis,
        "forecast_quality": engine.calibration_view,
        "prediction_research": engine.prediction_research_view,
    }


def build_operator_snapshot(
    engine: TraderEngine,
    *,
    now_ms: int,
    db_ok: bool,
    decisions: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    events: list[dict[str, Any]],
    calibration: dict[str, Any] | None = None,
    run_closed_trades: list[ClosedTrade] | None = None,
    experiment_closed_trades: list[ClosedTrade] | None = None,
) -> dict[str, Any]:
    symbol = engine.settings.symbols[0]
    state = engine.last_state
    if state is None:
        try:
            state = engine.snapshot_now(symbol, now_ms)
        except Exception:
            state = None
    warmup, warmup_progress = _warmup(engine, now_ms, state)
    last_market_ms = _last_market_update_ms(engine)
    blocked = None if engine.last_risk is None else engine.last_risk.blocked_reason
    if blocked is None and state is not None and state.health.reasons:
        blocked = state.health.reasons[0]
    trading_allowed = False if state is None else state.health.trading_allowed
    matrix = [] if engine.last_plan is None else opportunity_matrix(engine.last_plan.opportunities)
    return {
        "safety": _safety(engine),
        "system": {
            "trading_mode": engine.settings.trading_mode.value,
            "strategy": engine.settings.trading_strategy.value,
            "live": "BLOCKED",
            "exchange": "Binance USD-M futures",
            "market_product": "usd_m_perpetual_futures",
            "symbol": symbol,
            "jev_model": engine.settings.jev_model,
            "configured_model": engine.settings.jev_model,
            "resolved_response_model": (
                None if engine.last_result is None or engine.last_result.decision is None
                else engine.last_result.decision.model
            ),
            "engine": engine.lifecycle,
            "strategy_running": engine.strategy_running,
            "can_start": engine.lifecycle == "STOPPED",
            "can_stop": engine.lifecycle in {"STARTING", "RUNNING"},
            "can_reset": engine.lifecycle != "STOPPING",
            "strategy_error": engine.strategy_error,
            "binance_websocket": _binance_ws_status(engine, state),
            "last_binance_update": {
                "timestamp": _iso(last_market_ms),
                "seconds_ago": _seconds_ago(last_market_ms, now_ms),
            },
            "jev": _jev_health(engine),
            "last_jev_decision": {
                "timestamp": _iso(engine.last_jev_at_ms),
                "seconds_ago": _seconds_ago(engine.last_jev_at_ms, now_ms),
            },
            "database": "CONNECTED" if db_ok else "ERROR",
            "database_type": database_kind(engine.settings.database_url),
            "paper_persistence": getattr(engine, "paper_persistence", "HEALTHY"),
            "warmup": warmup,
            "warmup_progress": warmup_progress,
            "trading_allowed": trading_allowed,
            "blocked_reason": blocked,
        },
        "market": _market_view(state),
        "jev_decision": _jev_view(engine, state),
        "position": _position_view(engine, now_ms, state),
        "account": _account_view(engine, now_ms),
        "cost_model": _cost_view(engine, state),
        "best_opportunity": _best_opportunity_view(engine),
        "opportunity_matrix": matrix,
        "latency": _latency_view(engine),
        "session": _session_view(
            engine,
            now_ms,
            run_trades=run_closed_trades,
            experiment_trades=experiment_closed_trades,
        ),
        "recent_decisions": decisions,
        "recent_trades": trades,
        "recent_events": events,
        "experiment": _experiment_view(engine),
        "calibration": calibration or {"horizons": {}, "min_n": 20},
        "directional": _directional_view(engine),
        "prediction_research": getattr(engine, "prediction_research_view", {"scorecard": []}),
    }
