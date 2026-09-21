from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from src.analytics.latency import LatencyTracker
from src.analytics.session import scalp_session_metrics
from src.config.settings import load_settings
from src.portfolio.account import ClosedTrade
from src.storage.database import Database
from src.storage.models import ExperimentRunRow, JevDecisionRow, SystemEventRow, TradeRow
from src.storage.repository import (
    Repository,
    _closed_from_row,
    summarize_run_calibration,
)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    tracker = LatencyTracker(maxlen=max(500, len(values)))
    for value in values:
        tracker.record(value)
    stats = tracker.stats()
    if pct <= 50:
        return stats.p50_ms
    if pct <= 95:
        return stats.p95_ms
    return stats.p99_ms


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "JEV TRADER SESSION REPORT",
        f"experiment_id={payload['experiment_id']}",
        f"run_id={payload['run_id']}",
        f"config_hash={payload['config_hash']}",
        f"frozen={payload['frozen']}",
        f"started_at={payload['started_at']}",
        f"runtime_seconds={_fmt(payload['runtime_seconds'])}",
        "",
        f"decisions={payload['decisions']}",
        f"valid_calibration_outcomes={payload['valid_calibration_outcomes']}",
        f"ambiguous_outcomes={payload['ambiguous_outcomes']}",
        f"jev_failures={payload['jev_failures']}",
        f"websocket_failures={payload['websocket_failures']}",
        "",
        f"trades={payload['trades']} long={payload['long_trades']} "
        f"short={payload['short_trades']} reversals={payload['reversals']}",
        f"gross_pnl={_fmt(payload['gross_pnl'])}",
        f"fees={_fmt(payload['fees'])}",
        f"spread_attribution={_fmt(payload['spread_attribution'])}",
        f"slippage_attribution={_fmt(payload['slippage_attribution'])}",
        f"net_pnl={_fmt(payload['net_pnl'])}",
        f"account_return_pct={_fmt(payload['account_return_pct'])}",
        f"account_return_bps={_fmt(payload['account_return_bps'])}",
        f"avg_trade_return_on_notional_bps={_fmt(payload['avg_trade_return_on_notional_bps'])}",
        f"profit_factor={_fmt(payload['profit_factor'])}",
        f"win_rate={_fmt(payload['win_rate'])}",
        f"expectancy={_fmt(payload['expectancy'])}",
        f"max_drawdown={_fmt(payload['max_drawdown'])}",
        f"avg_hold_s={_fmt(payload['avg_hold_s'])}",
        f"median_hold_s={_fmt(payload['median_hold_s'])}",
        f"jev_latency_p50={_fmt(payload['jev_latency_p50'])}",
        f"jev_latency_p95={_fmt(payload['jev_latency_p95'])}",
        f"jev_latency_p99={_fmt(payload['jev_latency_p99'])}",
        f"positive_ev_opportunities={payload['positive_ev_opportunities']}",
        f"rejected_opportunities={payload['rejected_opportunities']}",
        "",
        "CALIBRATION (raw overlapping samples; n_nonoverlapping is horizon-spaced)",
    ]
    for horizon, summary in (payload.get("calibration") or {}).items():
        lines.append(
            f"  {horizon}: N={summary.get('n')} spaced={summary.get('n_nonoverlapping')} "
            f"ambiguous={summary.get('n_ambiguous')} insufficient={summary.get('insufficient')} "
            f"brier={_fmt(summary.get('brier'))} log_loss={_fmt(summary.get('log_loss'))}"
        )
        for cls in ("down", "up", "neither"):
            block = summary.get(cls) or {}
            lines.append(
                f"    {cls}: pred_avg={_fmt(block.get('predicted_avg'))} "
                f"obs={_fmt(block.get('observed_frequency'))}"
            )
        for bin_row in summary.get("ev_bins") or []:
            lines.append(
                f"    EV {bin_row.get('bin')}: N={bin_row.get('n')} "
                f"pred={_fmt(bin_row.get('avg_predicted_ev'))} "
                f"real={_fmt(bin_row.get('avg_realized_pnl'))} "
                f"median={_fmt(bin_row.get('median_realized_pnl'))} "
                f"win={_fmt(bin_row.get('win_frequency'))}"
            )
    return "\n".join(lines) + "\n"


def build_report_payload(
    *,
    run: ExperimentRunRow,
    decisions: list[JevDecisionRow],
    trades: list[TradeRow],
    events: list[SystemEventRow],
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    started = run.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    runtime = max(0.0, (now - started).total_seconds())
    closed = [_closed_from_row(row) for row in trades]
    equity = float((run.fingerprint or {}).get("paper_equity") or 10_000)
    notional = float((run.fingerprint or {}).get("paper_position_notional") or 1_000)
    metrics = scalp_session_metrics(
        closed,
        runtime_seconds=runtime,
        max_drawdown=_max_drawdown(closed, equity),
        starting_balance=equity,
        position_notional=notional,
    )
    latencies = [
        float(row.jev_latency_ms) for row in decisions if row.jev_latency_ms is not None
    ]
    calibration = summarize_run_calibration(decisions)
    valid = 0
    ambiguous = 0
    positive_ev = 0
    rejected = 0
    for row in decisions:
        payload = row.scalp_payload if isinstance(row.scalp_payload, dict) else {}
        outcomes = payload.get("barrier_outcomes")
        if isinstance(outcomes, dict):
            sample_valid = payload.get("research_sample_valid") is not False
            for item in outcomes.values():
                if not isinstance(item, dict):
                    continue
                for race in ("long", "short"):
                    race_row = item.get(race)
                    if not isinstance(race_row, dict):
                        continue
                    outcome = race_row.get("outcome")
                    if outcome in {"LONG_AMBIGUOUS", "SHORT_AMBIGUOUS", "AMBIGUOUS"}:
                        ambiguous += 1
                    elif outcome in {
                        "LONG_TARGET_FIRST",
                        "LONG_STOP_FIRST",
                        "LONG_NEITHER",
                        "SHORT_TARGET_FIRST",
                        "SHORT_STOP_FIRST",
                        "SHORT_NEITHER",
                    } and sample_valid:
                        valid += 1
        best = payload.get("best")
        if (
            isinstance(best, dict)
            and isinstance(best.get("net_ev_bps"), int | float)
            and float(best["net_ev_bps"]) > 0
        ):
            positive_ev += 1
        reason = payload.get("policy_reason") or row.blocked_reason
        if reason in {"net_ev_below_edge", "probability_edge_too_small", "high_noise"}:
            rejected += 1
    jev_failures = sum(1 for row in decisions if row.status not in {"OK", "SKIPPED_STALE"})
    ws_fail = sum(
        1
        for event in events
        if "disconnect" in event.event_type.lower() or "reconnect" in event.event_type.lower()
    )
    spread = sum(
        float(trade.payload.get("total_spread_cost") or 0)
        for trade in trades
        if isinstance(trade.payload, dict)
    )
    slip = sum(
        float(trade.payload.get("total_slippage_cost") or trade.payload.get("slippage_cost") or 0)
        for trade in trades
        if isinstance(trade.payload, dict)
    )
    holds = [float(trade.holding_seconds or 0.0) for trade in trades]
    return {
        "experiment_id": run.experiment_id,
        "run_id": run.run_id,
        "config_hash": run.config_hash,
        "frozen": run.frozen,
        "started_at": started.isoformat(),
        "runtime_seconds": runtime,
        "decisions": len(decisions),
        "valid_calibration_outcomes": valid,
        "ambiguous_outcomes": ambiguous,
        "jev_failures": jev_failures,
        "websocket_failures": ws_fail,
        "trades": metrics["trades"],
        "long_trades": metrics["long_trades"],
        "short_trades": metrics["short_trades"],
        "reversals": metrics["reversals"],
        "gross_pnl": metrics["gross_pnl"],
        "fees": metrics["fees"],
        "spread_attribution": spread,
        "slippage_attribution": slip,
        "net_pnl": metrics["net_pnl"],
        "account_return_pct": metrics["session_return_pct_on_starting_equity"],
        "account_return_bps": metrics["session_return_bps_on_starting_equity"],
        "avg_trade_return_on_notional_bps": metrics["average_trade_return_bps_on_notional"],
        "profit_factor": metrics["profit_factor"],
        "win_rate": metrics["win_rate"],
        "expectancy": metrics["expectancy_per_trade"],
        "max_drawdown": metrics["max_drawdown"],
        "avg_hold_s": metrics["average_holding_seconds"],
        "median_hold_s": None if not holds else float(median(holds)),
        "jev_latency_p50": _percentile(latencies, 50),
        "jev_latency_p95": _percentile(latencies, 95),
        "jev_latency_p99": _percentile(latencies, 99),
        "positive_ev_opportunities": positive_ev,
        "rejected_opportunities": rejected,
        "calibration": calibration.get("horizons") or {},
    }


def _max_drawdown(trades: list[ClosedTrade], start: float) -> float:
    equity = start
    peak = start
    max_dd = 0.0
    for trade in trades:
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


async def collect_report(
    repo: Repository, *, run_id: str | None, experiment_id: str | None
) -> dict[str, Any]:
    run: ExperimentRunRow | None
    if run_id:
        run = await repo.get_experiment_run(run_id)
    elif experiment_id:
        runs = await repo.list_experiment_runs(experiment_id)
        run = runs[0] if runs else None
    else:
        run = await repo.latest_experiment_run()
    if run is None:
        raise SystemExit("No experiment run found")
    decisions = await repo.list_decisions_for_run(run.run_id)
    trades = await repo.list_trades_for_run(run.run_id)
    events = await repo.list_events_for_run(run.run_id)
    return build_report_payload(run=run, decisions=decisions, trades=trades, events=events)


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report a PAPER/SCALP experiment run. Does not trade."
    )
    parser.add_argument("--current", action="store_true", help="Use the latest run_id")
    parser.add_argument("--run-id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    settings = load_settings()
    db = Database(settings.database_url)
    repo = Repository(db)
    try:
        await db.create_all()
        payload = await collect_report(
            repo,
            run_id=args.run_id,
            experiment_id=None if args.current else args.experiment_id,
        )
    finally:
        await db.aclose()
    text = json.dumps(payload, indent=2, default=str) if args.json else render_report(payload)
    print(text)
    out = args.out
    if out is None:
        reports = Path("reports")
        reports.mkdir(exist_ok=True)
        out = str(reports / f"run-{payload['run_id']}.txt")
    Path(out).write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))
