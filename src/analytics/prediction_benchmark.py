# ruff: noqa: E501
"""DIRECTIONAL_PREDICTION_BENCHMARK_V1. Independent of PAPER execution."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.analytics.audit_math import as_float, predicted_side_and_confidence, signed_future_return
from src.analytics.prediction_metrics import (
    BOOTSTRAP_N_DASH,
    BOOTSTRAP_N_FULL,
    CLIMATOLOGY_DEFINITION,
    EPISODE_THRESHOLDS,
    HORIZON_GAP_MS,
    MAX_ASSOCIATION_LATENCY_MS,
    MIN_INFERENTIAL_N,
    ROLLING_WINDOWS_MS,
    calibration_bins,
    confidence_buckets,
    evaluate_stream,
    scorecard_row,
    segment_episodes,
)
from src.config.settings import Settings, load_settings
from src.directional.constants import HORIZON_LABELS
from src.storage.database import Database
from src.storage.models import DirectionalForecastRow, ExperimentRunRow, JevDecisionRow, TradeRow
from src.storage.repository import Repository, _directional_row_valid

RISK_REASONS = {
    "warmup_incomplete",
    "stale_market_data",
    "max_consecutive_losses",
    "max_daily_loss",
    "max_daily_drawdown",
    "max_simultaneous_positions",
    "min_seconds_between_orders",
    "flat_cooldown",
    "kill_switch",
    "trading_not_allowed",
}
POSITION_CANDIDATES = {"HOLD", "CLOSE", "REVERSE"}
BENCHMARK_VERSION = "DIRECTIONAL_PREDICTION_BENCHMARK_V1"


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _ci(block: dict[str, Any] | None) -> str:
    if not block:
        return "n/a"
    return f"[{_num(block.get('lo'), 3)}, {_num(block.get('hi'), 3)}]"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def feature_mid(row: DirectionalForecastRow) -> float | None:
    forecast = _dict(row.forecast)
    mid = as_float(forecast.get("reference_mid"))
    if mid is not None:
        return mid
    state = _dict(row.feature_state)
    return as_float(_dict(state.get("microstructure")).get("mid"))


def feature_regime(row: DirectionalForecastRow) -> dict[str, Any]:
    state = _dict(row.feature_state)
    regime = _dict(state.get("regime")) or _dict(state.get("REGIME"))
    diagnostics = _dict(_dict(row.forecast).get("diagnostics"))
    hour = regime.get("utc_hour")
    if hour is None:
        hour = datetime.fromtimestamp(row.timestamp_ms / 1000, tz=UTC).hour
    return {
        "direction": str(regime.get("direction") or regime.get("trend") or "UNKNOWN"),
        "volatility": str(regime.get("volatility") or "UNKNOWN"),
        "liquidity": str(regime.get("liquidity") or "UNKNOWN"),
        "market_character": str(diagnostics.get("market_character") or "UNKNOWN"),
        "utc_hour": int(hour) if isinstance(hour, int | float) else hour,
    }


def execution_meta(
    row: DirectionalForecastRow,
    *,
    decision: JevDecisionRow | None,
    traded_ids: set[str],
) -> dict[str, Any]:
    stored = _dict(_dict(row.forecast).get("execution")) or _dict(_dict(row.policy).get("execution"))
    policy = _dict(row.policy)
    candidate = str(stored.get("candidate") or policy.get("candidate") or "")
    position_open = stored.get("position_open")
    if position_open is None:
        position_open = candidate in POSITION_CANDIDATES or bool(policy.get("hold_signal") or policy.get("close_signal"))
    reason = stored.get("risk_block_reason")
    if reason is None and decision is not None:
        reason = decision.blocked_reason
    risk_blocked = stored.get("risk_blocked")
    if risk_blocked is None:
        risk_blocked = bool(reason) and str(reason) in RISK_REASONS
    decision_id = row.decision_id
    return {
        "position_open": bool(position_open),
        "risk_blocked": bool(risk_blocked),
        "risk_block_reason": reason,
        "trade_opened": bool(decision_id and decision_id in traded_ids),
        "policy_candidate": candidate,
    }


def association_ok(latency_ms: float | None, status: str | None) -> bool:
    if status is not None and status != "OK":
        return False
    if latency_ms is None:
        return True
    return float(latency_ms) < MAX_ASSOCIATION_LATENCY_MS


def extract_samples(
    forecasts: list[DirectionalForecastRow],
    *,
    decisions: list[JevDecisionRow] | None = None,
    trades: list[TradeRow] | None = None,
) -> dict[str, Any]:
    by_decision = {row.id: row for row in (decisions or [])}
    traded_ids = {
        str(row.entry_jev_decision_id)
        for row in (trades or [])
        if row.entry_jev_decision_id
    }
    raw_n = len(forecasts)
    valid_n = 0
    scored: dict[str, list[dict[str, Any]]] = {label: [] for label in HORIZON_LABELS}
    exec_counts: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    for row in forecasts:
        forecast = _dict(row.forecast)
        decision = by_decision.get(str(row.decision_id or ""))
        latency = None if decision is None else decision.jev_latency_ms
        status = None if decision is None else decision.status
        meta = execution_meta(row, decision=decision, traded_ids=traded_ids)
        if meta["position_open"]:
            exec_counts["position_open"] += 1
        if meta["risk_blocked"]:
            exec_counts["risk_blocked"] += 1
        if meta["trade_opened"]:
            exec_counts["trade_opened"] += 1
        valid = _directional_row_valid(row)
        if valid:
            valid_n += 1
        else:
            invalid[str(row.invalid_reason or forecast.get("invalid_reason") or "invalid")] += 1
        associated = association_ok(latency, status)
        health = "HEALTHY" if valid and associated else "DEGRADED"
        ref = feature_mid(row)
        regime = feature_regime(row)
        p_up = _dict(forecast.get("p_up"))
        rolling = _dict(row.rolling_outcomes)
        horizon_outcomes: dict[str, Any] = {}
        for label in HORIZON_LABELS:
            outcome = _dict(rolling.get(label))
            if "direction" not in outcome:
                continue
            future = as_float(outcome.get("future_mid"))
            p = as_float(p_up.get(label))
            if p is None:
                continue
            side, conf = predicted_side_and_confidence(p)
            signed = None
            if ref and future:
                signed = signed_future_return(p_up=p, reference_mid=ref, future_mid=future) * 10_000.0
            bases = _dict(outcome.get("baselines"))
            rec = {
                "timestamp_ms": row.timestamp_ms,
                "decision_id": row.decision_id,
                "p_up": p,
                "predicted_side": side,
                "confidence": conf,
                "actual_up": outcome.get("direction") == "UP",
                "future_mid": future,
                "reference_mid": ref,
                "return_bps": as_float(outcome.get("return_bps")),
                "signed_move_bps": signed,
                "momentum": bases.get("momentum") or "DOWN",
                "recent_direction": bases.get("recent_direction") or "DOWN",
                "research_sample_valid": valid,
                "health": health,
                "associated": associated,
                **regime,
                **meta,
            }
            horizon_outcomes[label] = {
                "direction": outcome.get("direction"),
                "future_mid": future,
                "reference_mid": ref,
                "return_bps": rec["return_bps"],
            }
            if valid and associated:
                scored[label].append(rec)
        if scored["1m"] and scored["1m"][-1]["timestamp_ms"] == row.timestamp_ms:
            scored["1m"][-1]["horizon_outcomes"] = horizon_outcomes
    return {
        "n_raw": raw_n,
        "n_valid_flag": valid_n,
        "invalid_reasons": dict(invalid),
        "execution_counts": dict(exec_counts),
        "by_horizon": scored,
        "collection_note": (
            "Every persisted forecast is a research candidate. "
            "research_sample_valid / association filters scoring, not collection. "
            "position_open / risk_blocked / trade_opened are metadata only."
        ),
    }


def _regime_tables(samples: list[dict[str, Any]], *, min_gap_ms: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in samples:
        groups[("direction", str(item.get("direction")))].append(item)
        groups[("volatility", str(item.get("volatility")))].append(item)
        groups[("character", str(item.get("market_character")))].append(item)
        groups[("utc_hour", str(item.get("utc_hour")))].append(item)
    rows: list[dict[str, Any]] = []
    for (kind, segment), group in sorted(groups.items()):
        ev = evaluate_stream(group, min_gap_ms=min_gap_ms, n_boot=BOOTSTRAP_N_DASH)
        research = ev["nonoverlapping_research"]
        n_star = int(research.get("n_nonoverlapping") or 0)
        rows.append(
            {
                "segment_kind": kind,
                "segment": segment,
                "n": research.get("n"),
                "n_nonoverlapping": n_star,
                "low_sample": n_star < MIN_INFERENTIAL_N,
                "accuracy": research.get("accuracy"),
                "balanced_accuracy": research.get("balanced_accuracy"),
                "roc_auc": research.get("roc_auc"),
                "brier": research.get("brier"),
                "brier_skill": research.get("brier_skill_score"),
            }
        )
    return rows


def _rolling_tables(samples: list[dict[str, Any]], *, min_gap_ms: int) -> list[dict[str, Any]]:
    if not samples:
        return []
    end = max(int(item["timestamp_ms"]) for item in samples)
    rows: list[dict[str, Any]] = []
    for name, width in ROLLING_WINDOWS_MS.items():
        group = samples if width is None else [item for item in samples if int(item["timestamp_ms"]) >= end - width]
        ev = evaluate_stream(group, min_gap_ms=min_gap_ms, n_boot=BOOTSTRAP_N_DASH)
        research = ev["nonoverlapping_research"]
        rows.append(
            {
                "window": name,
                "n": research.get("n"),
                "n_nonoverlapping": research.get("n_nonoverlapping"),
                "accuracy": research.get("accuracy"),
                "balanced_accuracy": research.get("balanced_accuracy"),
                "roc_auc": research.get("roc_auc"),
                "brier": research.get("brier"),
                "brier_skill": research.get("brier_skill_score"),
                "ece": research.get("ece"),
            }
        )
    return rows


def build_benchmark(
    *,
    forecasts: list[DirectionalForecastRow],
    decisions: list[JevDecisionRow] | None = None,
    trades: list[TradeRow] | None = None,
    experiment_id: str | None,
    run_id: str | None,
    runs: list[ExperimentRunRow] | None = None,
    n_boot: int = BOOTSTRAP_N_FULL,
    lite: bool = False,
) -> dict[str, Any]:
    extracted = extract_samples(forecasts, decisions=decisions, trades=trades)
    horizons: dict[str, Any] = {}
    scorecard: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    conf_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for label in HORIZON_LABELS:
        samples = extracted["by_horizon"][label]
        ev = evaluate_stream(samples, min_gap_ms=HORIZON_GAP_MS[label], n_boot=n_boot)
        research = ev["nonoverlapping_research"]
        card = scorecard_row(label, research)
        horizons[label] = {**ev, "n_scored": len(samples)}
        scorecard.append(card)
        for row in calibration_bins(samples, min_gap_ms=HORIZON_GAP_MS[label]):
            cal_rows.append({"horizon": label, **row})
        for row in confidence_buckets(samples, min_gap_ms=HORIZON_GAP_MS[label]):
            conf_rows.append({"horizon": label, **row})
        if lite:
            continue
        for row in _regime_tables(samples, min_gap_ms=HORIZON_GAP_MS[label]):
            regime_rows.append({"horizon": label, **row})
        for row in _rolling_tables(samples, min_gap_ms=HORIZON_GAP_MS[label]):
            rolling_rows.append({"horizon": label, **row})
    if not lite:
        one_m = extracted["by_horizon"]["1m"]
        for threshold in EPISODE_THRESHOLDS:
            episode_rows.extend(segment_episodes(one_m, threshold=threshold))
    fingerprints = [run.fingerprint for run in (runs or []) if run.fingerprint]
    return {
        "meta": {
            "benchmark_version": BENCHMARK_VERSION,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "run_ids": [run.run_id for run in (runs or [])],
            "n_forecasts": extracted["n_raw"],
            "n_valid_flag": extracted["n_valid_flag"],
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "read_only": True,
            "climatology_definition": CLIMATOLOGY_DEFINITION,
            "association_rule": (
                f"Exclude from scored research if Jev status != OK or latency >= {MAX_ASSOCIATION_LATENCY_MS:.0f} ms. "
                "Failed/stale samples stay persisted but are not scored."
            ),
            "independence": extracted["collection_note"],
            "fingerprint": fingerprints[0] if fingerprints else None,
        },
        "execution_metadata": extracted["execution_counts"],
        "invalid_reasons": extracted["invalid_reasons"],
        "horizons": horizons,
        "scorecard": scorecard,
        "calibration": cal_rows,
        "confidence": conf_rows,
        "episodes": episode_rows,
        "regimes": regime_rows,
        "rolling": rolling_rows,
        "success_criteria_progress": {
            "note": "Display only. Not strategy rules.",
            "wanted": [
                "substantial non-overlapping N",
                "ROC-AUC consistently > 0.5",
                "positive Brier Skill Score vs point-in-time climatology",
                "improvement over momentum",
                "confidence associated with realized outcome probability",
                "similar effect across windows / regimes",
                "effect survives future out-of-sample data",
            ],
        },
    }


def dashboard_prediction_research(rows: list[DirectionalForecastRow]) -> dict[str, Any]:
    bench = build_benchmark(
        forecasts=rows,
        experiment_id=None,
        run_id=None,
        n_boot=0,
        lite=True,
    )
    compact_cal = [row for row in bench["calibration"] if row["horizon"] == "1m"]
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "n_raw": bench["meta"]["n_forecasts"],
        "n_valid_flag": bench["meta"]["n_valid_flag"],
        "scorecard": bench["scorecard"],
        "calibration_1m": compact_cal,
        "independence": "Prediction research is independent of PAPER PnL and risk blocks.",
    }


def render_markdown(bench: dict[str, Any]) -> str:
    meta = bench["meta"]
    lines = [
        "# DIRECTIONAL PREDICTION BENCHMARK V1",
        "",
        "This report measures **probabilistic directional information only**.",
        "PAPER PnL, fees, slippage, position management and risk blocks do not decide the conclusion.",
        "",
        f"- experiment `{meta.get('experiment_id')}` run `{meta.get('run_id')}`",
        f"- forecasts {meta.get('n_forecasts')} valid_flag {meta.get('n_valid_flag')}",
        f"- generated {meta.get('generated_at')} · read-only · {BENCHMARK_VERSION}",
        f"- climatology: {meta.get('climatology_definition')}",
        f"- association: {meta.get('association_rule')}",
        "",
        "## Scorecard (NON-OVERLAPPING / RESEARCH)",
        "",
    ]
    headers = ["H", "N*", "UP base", "Acc", "Bal Acc", "ROC-AUC", "Brier", "BSS", "LogLoss", "ECE", "Mom Acc", "Δ mom", "STATUS"]
    table = []
    for row in bench["scorecard"]:
        table.append(
            [
                str(row["horizon"]),
                str(row["n_star"]),
                _pct(row["up_base_rate"]),
                _pct(row["accuracy"]),
                _pct(row["balanced_accuracy"]),
                _num(row["roc_auc"], 3),
                _num(row["brier"], 3),
                _num(row["brier_skill"], 3),
                _num(row["log_loss"], 3),
                _num(row["ece"], 3),
                _pct(row["momentum_accuracy"]),
                _num(row["delta_vs_momentum"], 3),
                str(row["status"]),
            ]
        )
    lines += [_md_table(headers, table), ""]
    lines += ["STATUS is mechanical: INSUFFICIENT_SAMPLE | NO_MEASURED_LIFT | WEAK_MEASURED_LIFT | MEASURED_LIFT_PENDING_REPLICATION. Nothing is called edge-proven.", ""]
    for row in bench["scorecard"]:
        lines.append(
            f"- {row['horizon']} 95% block-bootstrap CI acc={_ci(row.get('accuracy_ci95'))} "
            f"bal={_ci(row.get('balanced_accuracy_ci95'))} auc={_ci(row.get('roc_auc_ci95'))} "
            f"bss={_ci(row.get('brier_skill_ci95'))}"
        )
    lines += ["", "## Execution metadata (does not filter scoring)", ""]
    lines.append(json.dumps(bench.get("execution_metadata") or {}, indent=2))
    lines += ["", "## Horizons", ""]
    for label in HORIZON_LABELS:
        block = bench["horizons"][label]
        ov = block["overlapping_descriptive"]
        rs = block["nonoverlapping_research"]
        lines += [
            f"### {label}",
            "",
            f"OVERLAPPING / DESCRIPTIVE N={ov['n']} acc={_pct(ov['accuracy'])} brier={_num(ov['brier'], 3)} auc={_num(ov['roc_auc'], 3)}",
            f"NON-OVERLAPPING / RESEARCH N*={rs['n_nonoverlapping']} acc={_pct(rs['accuracy'])} bal={_pct(rs['balanced_accuracy'])} "
            f"auc={_num(rs['roc_auc'], 3)} bss={_num(rs['brier_skill_score'], 3)}",
            f"Jev vs 50/50 Δ={_num(rs.get('delta_accuracy_vs_50_50'), 3)} · vs climatology Δ={_num(rs.get('delta_accuracy_vs_climatology'), 3)} · vs momentum Δ={_num(rs.get('delta_accuracy_vs_momentum'), 3)}",
            f"Pearson(p_up-0.5, return_bps)={_num(rs.get('pearson_edge_vs_return'), 3)} Spearman={_num(rs.get('spearman_edge_vs_return'), 3)}",
            f"UP prec={_pct(rs.get('up_precision'))} rec={_pct(rs.get('up_recall'))} · DOWN prec={_pct(rs.get('down_precision'))} rec={_pct(rs.get('down_recall'))}",
            "",
        ]
    lines += ["## Calibration (p_up bins)", ""]
    lines.append(
        _md_table(
            ["H", "bin", "N", "N*", "mean p", "realized UP", "|err|", "avg ret bps"],
            [
                [
                    row["horizon"],
                    row["bin"],
                    str(row["n"]),
                    str(row["n_nonoverlapping"]),
                    _num(row["mean_predicted_p_up"], 3),
                    _pct(row["realized_up_frequency"]),
                    _num(row["absolute_calibration_error"], 3),
                    _num(row["avg_future_return_bps"], 2),
                ]
                for row in bench["calibration"]
            ],
        )
    )
    lines += ["", "## Predicted-side confidence", ""]
    lines.append(
        _md_table(
            ["H", "bucket", "N", "N*", "acc", "bal", "avg signed bps", "med signed"],
            [
                [
                    row["horizon"],
                    row["bucket"] + (" LOW SAMPLE" if row["low_sample"] else ""),
                    str(row["n"]),
                    str(row["n_nonoverlapping"]),
                    _pct(row["accuracy"]),
                    _pct(row["balanced_accuracy"]),
                    _num(row["avg_signed_move_bps"], 2),
                    _num(row["median_signed_move_bps"], 2),
                ]
                for row in bench["confidence"]
            ],
        )
    )
    lines += ["", "## Rolling windows", ""]
    lines.append(
        _md_table(
            ["H", "window", "N", "N*", "acc", "bal", "auc", "brier", "BSS", "ECE"],
            [
                [
                    row["horizon"],
                    row["window"],
                    str(row["n"]),
                    str(row["n_nonoverlapping"]),
                    _pct(row["accuracy"]),
                    _pct(row["balanced_accuracy"]),
                    _num(row["roc_auc"], 3),
                    _num(row["brier"], 3),
                    _num(row["brier_skill"], 3),
                    _num(row["ece"], 3),
                ]
                for row in bench["rolling"]
            ],
        )
    )
    lines += ["", "## Regimes (LOW SAMPLE if N* < 30)", ""]
    lines.append(
        _md_table(
            ["H", "kind", "segment", "N", "N*", "acc", "auc", "BSS"],
            [
                [
                    row["horizon"],
                    row["segment_kind"],
                    str(row["segment"]) + (" LOW SAMPLE" if row["low_sample"] else ""),
                    str(row["n"]),
                    str(row["n_nonoverlapping"]),
                    _pct(row["accuracy"]),
                    _num(row["roc_auc"], 3),
                    _num(row["brier_skill"], 3),
                ]
                for row in bench["regimes"]
            ],
        )
    )
    lines += [
        "",
        f"## Episodes (1m confidence crossings; N={len(bench['episodes'])})",
        "",
        "Research only. Not used for trading.",
        "",
        "## Language",
        "",
        "Observed fact / statistical evidence / hypothesis stay separate. "
        "PAPER execution is a downstream monetization problem.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_artifacts(bench: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(bench, indent=2, default=str), encoding="utf-8")
    (directory / "REPORT.md").write_text(render_markdown(bench), encoding="utf-8")
    _write_csv(directory / "horizons.csv", bench["scorecard"])
    _write_csv(directory / "calibration.csv", bench["calibration"])
    _write_csv(directory / "confidence.csv", bench["confidence"])
    _write_csv(directory / "episodes.csv", bench["episodes"])
    _write_csv(directory / "regimes.csv", bench["regimes"])
    _write_csv(directory / "rolling.csv", bench["rolling"])


async def collect_benchmark(
    repo: Repository, *, experiment_id: str | None, run_id: str | None
) -> dict[str, Any]:
    runs: list[ExperimentRunRow] = []
    resolved_run: str | None
    if run_id:
        run = await repo.get_experiment_run(run_id)
        if run is None:
            raise SystemExit(f"run_id not found: {run_id}")
        if experiment_id and run.experiment_id != experiment_id:
            raise SystemExit(f"run {run_id} belongs to {run.experiment_id}, not {experiment_id}")
        runs = [run]
        forecasts = await repo.list_directional_forecasts(run_id)
        decisions = await repo.list_decisions_for_run(run_id, limit=100_000)
        trades = await repo.list_trades_for_run(run_id)
        resolved_experiment = run.experiment_id
        resolved_run = run.run_id
    else:
        if not experiment_id:
            raise SystemExit("Provide --experiment and/or --run-id")
        runs = await repo.list_experiment_runs(experiment_id)
        forecasts = await repo.list_directional_forecasts(experiment_id=experiment_id)
        decisions = []
        trades = []
        for run in runs:
            decisions.extend(await repo.list_decisions_for_run(run.run_id, limit=100_000))
            trades.extend(await repo.list_trades_for_run(run.run_id))
        resolved_experiment = experiment_id
        resolved_run = None if len(runs) != 1 else runs[0].run_id
    return build_benchmark(
        forecasts=forecasts,
        decisions=decisions,
        trades=trades,
        experiment_id=resolved_experiment,
        run_id=resolved_run,
        runs=runs,
    )


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only directional prediction benchmark. Independent of PAPER execution."
    )
    parser.add_argument("--experiment", dest="experiment_id")
    parser.add_argument("--run-id", dest="run_id")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if not args.experiment_id and not args.run_id:
        raise SystemExit("Provide --experiment and/or --run-id")
    settings: Settings = load_settings()
    db = Database(settings.database_url)
    repo = Repository(db)
    try:
        bench = await collect_benchmark(
            repo, experiment_id=args.experiment_id, run_id=args.run_id
        )
    finally:
        await db.aclose()
    slug = args.run_id or args.experiment_id or "unknown"
    out = Path(args.out) if args.out else Path("artifacts/prediction_benchmark") / slug
    write_artifacts(bench, out)
    print(f"wrote {out / 'REPORT.md'}")
    print(f"wrote {out / 'summary.json'}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))
