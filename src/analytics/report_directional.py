from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.analytics.report import _fmt
from src.analytics.session import scalp_session_metrics
from src.config.settings import load_settings
from src.storage.database import Database
from src.storage.models import DirectionalForecastRow, ExperimentRunRow, SystemEventRow, TradeRow
from src.storage.repository import (
    Repository,
    _closed_from_row,
    summarize_directional_calibration,
)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "JEV TRADER DIRECTIONAL REPORT",
        f"experiment_id={payload['experiment_id']}",
        f"run_id={payload['run_id']}",
        f"config_hash={payload['config_hash']}",
        f"frozen={payload['frozen']}",
        f"question_version={payload['question_version']}",
        f"strategy_version={payload['strategy_version']}",
        f"feature_schema_version={payload['feature_schema_version']}",
        f"jev_configured={payload['jev_configured']}",
        f"jev_resolved={payload['jev_resolved']}",
        "",
        f"forecasts={payload['forecasts']}",
        f"mature_rolling={payload['mature_rolling']}",
        f"paper_trades={payload['trades']}",
        f"gross_pnl={_fmt(payload['gross_pnl'])}",
        f"fees={_fmt(payload['fees'])}",
        f"slippage={_fmt(payload['slippage'])}",
        f"net_pnl={_fmt(payload['net_pnl'])}",
        "",
        "DIRECTION CALIBRATION",
    ]
    for horizon, summary in (payload.get("horizons") or {}).items():
        lines.append(
            f"  {horizon}: N={summary.get('n')} spaced={summary.get('n_nonoverlapping')} "
            f"brier={_fmt(summary.get('brier'))} log_loss={_fmt(summary.get('log_loss'))} "
            f"acc={_fmt(summary.get('accuracy'))} "
            f"baseline_recent={_fmt(summary.get('baseline_recent'))} "
            f"baseline_momentum={_fmt(summary.get('baseline_momentum'))}"
        )
        for bin_row in summary.get("calibration_bins") or []:
            lines.append(
                f"    {bin_row.get('bin')}: N={bin_row.get('n')} "
                f"pred={_fmt(bin_row.get('predicted_avg'))} "
                f"obs_up={_fmt(bin_row.get('observed_up_frequency'))}"
            )
    lines.append("ALIGNED CANDLES")
    for tf, summary in (payload.get("candles") or {}).items():
        lines.append(
            f"  {tf}: N={summary.get('n')} acc={_fmt(summary.get('accuracy'))} "
            f"brier={_fmt(summary.get('brier'))}"
        )
        for seg, block in (summary.get("by_remaining") or {}).items():
            lines.append(
                f"    remaining {seg}: N={block.get('n')} acc={_fmt(block.get('accuracy'))}"
            )
    lines.append("PERSISTENCE")
    for key, summary in (payload.get("persistence") or {}).items():
        lines.append(
            f"  {key}: N={summary.get('n')} pred_mean vs obs acc={_fmt(summary.get('accuracy'))} "
            f"brier={_fmt(summary.get('brier'))}"
        )
    lines.append("MAGNITUDE")
    for tf, summary in (payload.get("magnitude") or {}).items():
        lines.append(f"  {tf}: N={summary.get('n')} acc={_fmt(summary.get('accuracy'))}")
    lines.append("COUNTERFACTUAL")
    for tf, summary in (payload.get("counterfactual") or {}).items():
        lines.append(
            f"  {tf}: N={summary.get('n')} raw={_fmt(summary.get('avg_raw_bps'))} "
            f"net={_fmt(summary.get('avg_net_bps'))} "
            f"correct_unprofitable={summary.get('correct_unprofitable')}"
        )
    lines.append("HIGH-CONFIDENCE")
    for row in payload.get("high_confidence_correct") or []:
        lines.append(f"  correct {row}")
    for row in payload.get("high_confidence_wrong") or []:
        lines.append(f"  wrong {row}")
    for row in payload.get("jev_vs_momentum") or []:
        lines.append(f"  disagree {row}")
    return "\n".join(lines) + "\n"


def _cf_summary(rows: list[DirectionalForecastRow]) -> dict[str, Any]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        cf = row.counterfactuals if isinstance(row.counterfactuals, dict) else {}
        for tf, item in cf.items():
            if not isinstance(item, dict):
                continue
            bucket = out.setdefault(tf, {"n": 0, "raw": 0.0, "net": 0.0, "correct_unprofitable": 0})
            bucket["n"] += 1
            bucket["raw"] += float(item.get("raw_market_return_bps") or 0.0)
            if isinstance(item.get("net_executable_return_bps"), int | float):
                bucket["net"] += float(item["net_executable_return_bps"])
            if item.get("correct_but_unprofitable"):
                bucket["correct_unprofitable"] += 1
    return {
        tf: {
            "n": item["n"],
            "avg_raw_bps": None if item["n"] == 0 else item["raw"] / item["n"],
            "avg_net_bps": None if item["n"] == 0 else item["net"] / item["n"],
            "correct_unprofitable": item["correct_unprofitable"],
        }
        for tf, item in out.items()
    }


def _high_confidence(rows: list[DirectionalForecastRow]) -> tuple[list[str], list[str], list[str]]:
    correct: list[str] = []
    wrong: list[str] = []
    disagree: list[str] = []
    for row in rows:
        forecast = row.forecast if isinstance(row.forecast, dict) else {}
        raw_p = forecast.get("p_up")
        p_up: dict[str, Any] = raw_p if isinstance(raw_p, dict) else {}
        rolling = row.rolling_outcomes if isinstance(row.rolling_outcomes, dict) else {}
        for tf, outcome in rolling.items():
            if not isinstance(outcome, dict):
                continue
            p = float(p_up.get(tf) or 0.5)
            if p < 0.80 and p > 0.20:
                continue
            predicted = "UP" if p >= 0.5 else "DOWN"
            actual = outcome.get("direction")
            ret = outcome.get("return_bps")
            text = f"{tf} p={p:.3f} pred={predicted} actual={actual} ret={ret}"
            if predicted == actual:
                correct.append(text)
            else:
                wrong.append(text)
            raw_base = outcome.get("baselines")
            baselines = raw_base if isinstance(raw_base, dict) else {}
            mom = baselines.get("momentum")
            if mom and mom != predicted:
                disagree.append(f"{tf} jev={predicted} momentum={mom} actual={actual}")
    return correct[:8], wrong[:8], disagree[:8]


def build_report_payload(
    *,
    run: ExperimentRunRow,
    forecasts: list[DirectionalForecastRow],
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
        max_drawdown=0.0,
        starting_balance=equity,
        position_notional=notional,
    )
    calibration = summarize_directional_calibration(forecasts)
    fp = run.fingerprint if isinstance(run.fingerprint, dict) else {}
    correct, wrong, disagree = _high_confidence(forecasts)
    mature = sum(
        1
        for row in forecasts
        if isinstance(row.rolling_outcomes, dict) and row.rolling_outcomes
    )
    return {
        "experiment_id": run.experiment_id,
        "run_id": run.run_id,
        "config_hash": run.config_hash,
        "frozen": run.frozen,
        "started_at": started.isoformat(),
        "runtime_seconds": runtime,
        "question_version": fp.get("question_version"),
        "strategy_version": fp.get("strategy_version"),
        "feature_schema_version": fp.get("feature_state_schema_version"),
        "jev_configured": fp.get("jev_configured_model"),
        "jev_resolved": fp.get("jev_resolved_model"),
        "forecasts": len(forecasts),
        "mature_rolling": mature,
        "trades": metrics["trades"],
        "gross_pnl": metrics["gross_pnl"],
        "fees": metrics["fees"],
        "slippage": metrics["slippage"],
        "net_pnl": metrics["net_pnl"],
        "horizons": calibration.get("horizons") or {},
        "candles": calibration.get("candles") or {},
        "persistence": calibration.get("persistence") or {},
        "magnitude": calibration.get("magnitude") or {},
        "counterfactual": _cf_summary(forecasts),
        "high_confidence_correct": correct,
        "high_confidence_wrong": wrong,
        "jev_vs_momentum": disagree,
        "events": len(events),
    }


async def collect_report(
    repo: Repository, *, run_id: str | None, experiment_id: str | None
) -> dict[str, Any]:
    if run_id:
        run = await repo.get_experiment_run(run_id)
    elif experiment_id:
        runs = await repo.list_experiment_runs(experiment_id)
        run = runs[0] if runs else None
    else:
        run = await repo.latest_experiment_run()
    if run is None:
        raise SystemExit("No experiment run found")
    forecasts = await repo.list_directional_forecasts(run.run_id)
    trades = await repo.list_trades_for_run(run.run_id)
    events = await repo.list_events_for_run(run.run_id)
    return build_report_payload(run=run, forecasts=forecasts, trades=trades, events=events)


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report a DIRECTIONAL PAPER experiment. Does not trade."
    )
    parser.add_argument("--current", action="store_true")
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
        out = str(reports / f"directional-{payload['run_id']}.txt")
    Path(out).write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))
