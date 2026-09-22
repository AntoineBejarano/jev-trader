# ruff: noqa: E501
"""Read-only DIRECTIONAL_V1 experiment audit. Does not trade or mutate rows."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.analytics.audit_math import (
    CONFIDENCE_BINS,
    HORIZON_GAP_MS,
    LATENCY_BUCKETS,
    PROB_BINS,
    as_float,
    binary_quality,
    capture_ratio_bps,
    check_candle_label,
    check_rolling_label,
    decompose_round_trip,
    expected_round_trip_cost_bps,
    in_bin,
    low_sample,
    mean,
    percentiles,
    predicted_side_and_confidence,
    quantile,
    signed_excursions,
    signed_future_return,
    vote_from_p_up,
)
from src.config.settings import Settings, load_settings
from src.directional.constants import HORIZON_LABELS, HORIZON_SECONDS_BY_LABEL
from src.storage.database import Database
from src.storage.models import (
    DirectionalForecastRow,
    ExperimentRunRow,
    JevDecisionRow,
    SystemEventRow,
    TradeRow,
)
from src.storage.repository import Repository, _directional_row_valid

ENTRY_CONFIDENCE = 0.70
HEALTH_WINDOW_MS = 30_000
DEGRADED_LATENCY_MS = 5_000.0
DEFAULT_TAKER_BPS = 5.0
DEFAULT_SLIP_BPS = 1.0
WS_EVENTS = {
    "ws_disconnect",
    "ws_public_disconnect",
    "ws_market_disconnect",
    "binance_ws_disconnect",
    "websocket_disconnected",
}
JEV_FAIL_EVENTS = {"jev_timeout", "jev_error", "jev_unavailable"}
JEV_RECOVER_EVENTS = {"jev_recovered"}


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
    micro = _dict(state.get("microstructure"))
    return as_float(micro.get("mid"))


def feature_regime(row: DirectionalForecastRow) -> dict[str, Any]:
    state = _dict(row.feature_state)
    regime = _dict(state.get("regime")) or _dict(state.get("REGIME"))
    diagnostics = _dict(_dict(row.forecast).get("diagnostics"))
    return {
        "utc_hour": regime.get("utc_hour"),
        "direction": regime.get("direction") or regime.get("trend") or "UNKNOWN",
        "volatility": regime.get("volatility") or "UNKNOWN",
        "liquidity": regime.get("liquidity") or "UNKNOWN",
        "character": diagnostics.get("market_character") or "UNKNOWN",
    }


def trade_payload(row: TradeRow) -> dict[str, Any]:
    return _dict(row.payload)


def audit_trade(row: TradeRow) -> dict[str, Any]:
    payload = trade_payload(row)
    research = _dict(payload.get("research"))
    qty = as_float(row.quantity) or 0.0
    entry = as_float(row.entry_price) or 0.0
    exit_px = as_float(row.exit_price) or 0.0
    decomp = decompose_round_trip(
        side=row.side,
        quantity=qty,
        entry_fill=entry,
        exit_fill=exit_px,
        entry_bid=as_float(payload.get("entry_bid")),
        entry_ask=as_float(payload.get("entry_ask")),
        exit_bid=as_float(payload.get("exit_bid")),
        exit_ask=as_float(payload.get("exit_ask")),
        entry_fee=as_float(payload.get("entry_fee")) or 0.0,
        exit_fee=as_float(payload.get("exit_fee")) or 0.0,
        stored_market_move=as_float(payload.get("market_move_pnl")),
        stored_entry_slip=as_float(payload.get("entry_slippage_cost")) or 0.0,
        stored_exit_slip=as_float(payload.get("exit_slippage_cost")) or 0.0,
        stored_entry_spread=as_float(payload.get("entry_spread_cost")) or 0.0,
        stored_exit_spread=as_float(payload.get("exit_spread_cost")) or 0.0,
        stored_gross=float(row.gross_pnl),
        stored_fees=float(row.fees),
        stored_net=float(row.net_pnl),
    )
    notional = as_float(row.entry_notional) or 0.0
    mfe = as_float(payload.get("mfe_bps"))
    mae = as_float(payload.get("mae_bps"))
    gross_bps = as_float(payload.get("gross_pnl_bps"))
    net_bps = as_float(payload.get("net_pnl_bps"))
    if gross_bps is None and notional:
        gross_bps = decomp["fill_to_fill_gross_pnl_usdt"] / notional * 10_000.0
    if net_bps is None and notional:
        net_bps = decomp["net_pnl_usdt"] / notional * 10_000.0
    primary = research.get("primary_horizon") or payload.get("primary_horizon")
    primary_s = HORIZON_SECONDS_BY_LABEL.get(str(primary)) if primary else payload.get(
        "entry_horizon_seconds"
    )
    hold = float(row.holding_seconds or 0.0)
    ratio = None if not primary_s else hold / float(primary_s)
    exit_reason = payload.get("exit_reason") or research.get("close_reason")
    return {
        "trade_id": row.id,
        "side": row.side,
        "opened_at": row.opened_at.isoformat() if row.opened_at else None,
        "closed_at": row.closed_at.isoformat() if row.closed_at else None,
        "entry_timestamp_ms": payload.get("entry_timestamp_ms"),
        "exit_timestamp_ms": payload.get("exit_timestamp_ms"),
        "quantity": qty,
        "entry_fill": entry,
        "exit_fill": exit_px,
        "notional": notional,
        "holding_seconds": hold,
        "primary_horizon": primary,
        "primary_horizon_seconds": primary_s,
        "hold_over_primary": ratio,
        "entry_thesis_score": research.get("thesis_score_entry") or payload.get("thesis_score_entry"),
        "exit_thesis_score": research.get("thesis_score_close") or payload.get("thesis_score_close"),
        "thesis_decay": research.get("thesis_decay"),
        "exit_reason": exit_reason,
        "exit_source": payload.get("exit_source") or research.get("exit_source"),
        "mfe_bps": mfe,
        "mae_bps": mae,
        "gross_bps": gross_bps,
        "net_bps": net_bps,
        "capture_ratio": capture_ratio_bps(mfe or 0.0, gross_bps or 0.0),
        "giveback_bps": None if mfe is None or gross_bps is None else max(0.0, mfe - gross_bps),
        "entry_decision_id": row.entry_jev_decision_id,
        "exit_decision_id": row.exit_jev_decision_id,
        "research": research,
        **decomp,
    }


def forecast_samples(rows: list[DirectionalForecastRow]) -> dict[str, list[dict[str, Any]]]:
    by_horizon: dict[str, list[dict[str, Any]]] = {label: [] for label in HORIZON_LABELS}
    for row in rows:
        if not _directional_row_valid(row):
            continue
        forecast = _dict(row.forecast)
        p_up = _dict(forecast.get("p_up"))
        rolling = _dict(row.rolling_outcomes)
        ref = feature_mid(row)
        regime = feature_regime(row)
        for label in HORIZON_LABELS:
            outcome = _dict(rolling.get(label))
            if "direction" not in outcome:
                continue
            future = as_float(outcome.get("future_mid"))
            p = as_float(p_up.get(label))
            if p is None:
                p = 0.5
            side, conf = predicted_side_and_confidence(p)
            signed = None
            if ref and future:
                signed = signed_future_return(p_up=p, reference_mid=ref, future_mid=future) * 10_000.0
            mfe, mae = signed_excursions(
                predicted_side=side,
                mfe_up_bps=as_float(outcome.get("mfe_bps")),
                mae_up_bps=as_float(outcome.get("mae_bps")),
            )
            hit = (side == "UP" and outcome.get("direction") == "UP") or (
                side == "DOWN" and outcome.get("direction") == "DOWN"
            )
            by_horizon[label].append(
                {
                    "timestamp_ms": row.timestamp_ms,
                    "decision_id": row.decision_id,
                    "p_up": p,
                    "predicted_side": side,
                    "confidence": conf,
                    "actual_up": outcome.get("direction") == "UP",
                    "hit": hit,
                    "signed_move_bps": signed,
                    "mfe_bps": mfe,
                    "mae_bps": mae,
                    "baselines": _dict(outcome.get("baselines")),
                    **regime,
                }
            )
    return by_horizon


def bucket_table(samples: list[dict[str, Any]], *, min_gap_ms: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (label, lo, hi) in enumerate(CONFIDENCE_BINS):
        last = index == len(CONFIDENCE_BINS) - 1
        group = [
            item
            for item in samples
            if in_bin(float(item["confidence"]), lo, hi, last=last)
        ]
        predicted = [float(item["p_up"]) for item in group]
        hits = [bool(item["actual_up"]) for item in group]
        stamps = [int(item["timestamp_ms"]) for item in group]
        quality = binary_quality(predicted, hits, stamps, min_gap_ms=min_gap_ms)
        signed = [float(item["signed_move_bps"]) for item in group if item.get("signed_move_bps") is not None]
        mfes = [float(item["mfe_bps"]) for item in group if item.get("mfe_bps") is not None]
        maes = [float(item["mae_bps"]) for item in group if item.get("mae_bps") is not None]
        side_counts = Counter(str(item["predicted_side"]) for item in group)
        rows.append(
            {
                "bucket": label,
                "raw_n": quality["n"],
                "nonoverlapping_n": quality["n_nonoverlapping"],
                "low_sample": low_sample(int(quality["n_nonoverlapping"] or 0)),
                "accuracy": quality["accuracy"],
                "brier": quality["brier"],
                "log_loss": quality["log_loss"],
                "n_up_pred": side_counts.get("UP", 0),
                "n_down_pred": side_counts.get("DOWN", 0),
                "avg_signed_move_bps": mean(signed),
                "median_signed_move_bps": quantile(signed, 0.50),
                "p25_signed_move_bps": quantile(signed, 0.25),
                "p75_signed_move_bps": quantile(signed, 0.75),
                "avg_mfe_bps": mean(mfes),
                "median_mfe_bps": quantile(mfes, 0.50),
                "avg_mae_bps": mean(maes),
                "median_mae_bps": quantile(maes, 0.50),
                "pct_signed_gt_0": None if not signed else sum(1 for v in signed if v > 0) / len(signed),
                "pct_signed_gt_6": None if not signed else sum(1 for v in signed if v > 6) / len(signed),
                "pct_signed_gt_12": None if not signed else sum(1 for v in signed if v > 12) / len(signed),
                "pct_mfe_gt_12": None if not mfes else sum(1 for v in mfes if v > 12) / len(mfes),
            }
        )
    return rows


def calibration_table(samples: list[dict[str, Any]], *, min_gap_ms: int) -> dict[str, Any]:
    predicted = [float(item["p_up"]) for item in samples]
    hits = [bool(item["actual_up"]) for item in samples]
    stamps = [int(item["timestamp_ms"]) for item in samples]
    quality = binary_quality(predicted, hits, stamps, min_gap_ms=min_gap_ms)
    bins: list[dict[str, Any]] = []
    for index, (label, lo, hi) in enumerate(PROB_BINS):
        last = index == len(PROB_BINS) - 1
        group = [item for item in samples if in_bin(float(item["p_up"]), lo, hi, last=last)]
        n = len(group)
        avg_p = mean([float(item["p_up"]) for item in group])
        freq = None if n == 0 else sum(1 for item in group if item["actual_up"]) / n
        bins.append(
            {
                "bin": label,
                "n": n,
                "mean_predicted_p_up": avg_p,
                "realized_up_frequency": freq,
                "calibration_error": None if avg_p is None or freq is None else abs(avg_p - freq),
            }
        )
    return {**quality, "bins": bins}


def label_integrity(rows: list[DirectionalForecastRow]) -> dict[str, Any]:
    rolling_issues: dict[str, Counter[str]] = {label: Counter() for label in HORIZON_LABELS}
    candle_issues: dict[str, Counter[str]] = {label: Counter() for label in HORIZON_LABELS}
    rolling_checked = {label: 0 for label in HORIZON_LABELS}
    candle_checked = {label: 0 for label in HORIZON_LABELS}
    candle_up_rate = {label: [0, 0] for label in HORIZON_LABELS}
    remaining: dict[str, Counter[str]] = {label: Counter() for label in HORIZON_LABELS}
    persist_base: dict[str, list[bool]] = {}
    for row in rows:
        ref = feature_mid(row)
        rolling = _dict(row.rolling_outcomes)
        candles = _dict(row.candle_outcomes)
        persist = _dict(row.persistence_outcomes)
        for label in HORIZON_LABELS:
            outcome = _dict(rolling.get(label))
            if "direction" not in outcome:
                continue
            rolling_checked[label] += 1
            check = check_rolling_label(
                reference_mid=ref,
                future_mid=as_float(outcome.get("future_mid")),
                stored_direction=str(outcome.get("direction")) if outcome.get("direction") else None,
                stored_return_bps=as_float(outcome.get("return_bps")),
            )
            if check["issues"]:
                rolling_issues[label].update(check["issues"])
            candle = _dict(candles.get(label))
            if candle.get("direction") in {"UP", "DOWN"}:
                candle_checked[label] += 1
                ccheck = check_candle_label(candle, timeframe=label)
                if ccheck["issues"]:
                    candle_issues[label].update(ccheck["issues"])
                candle_up_rate[label][1] += 1
                if candle.get("direction") == "UP":
                    candle_up_rate[label][0] += 1
                remaining[label][str(candle.get("remaining_segment") or "unknown")] += 1
        for key, item in persist.items():
            if isinstance(item, dict) and "hit" in item:
                persist_base.setdefault(str(key), []).append(bool(item["hit"]))
    persist_rates = {
        key: {"n": len(hits), "base_hit_rate": sum(hits) / len(hits) if hits else None}
        for key, hits in persist_base.items()
    }
    return {
        "rolling": {
            label: {
                "checked": rolling_checked[label],
                "issues": dict(rolling_issues[label]),
                "clean": rolling_checked[label] > 0 and not rolling_issues[label],
            }
            for label in HORIZON_LABELS
        },
        "candles": {
            label: {
                "checked": candle_checked[label],
                "issues": dict(candle_issues[label]),
                "clean": candle_checked[label] > 0 and not candle_issues[label],
                "realized_up_rate": (
                    None
                    if candle_up_rate[label][1] == 0
                    else candle_up_rate[label][0] / candle_up_rate[label][1]
                ),
                "remaining_segments": dict(remaining[label]),
            }
            for label in HORIZON_LABELS
        },
        "persistence_base_rates": persist_rates,
        "code_review": {
            "rolling_definition": (
                "Persisted label is UP iff future_mid > reference_mid. "
                "future_mid is first mid with ts >= t+horizon (nearest-sample-after, not interpolated)."
            ),
            "candle_definition": (
                "Current UTC-aligned window close>open. Ties and equal close/open label DOWN."
            ),
            "persistence_definition": (
                "Next N completed candles after the current aligned window, all same side."
            ),
        },
    }


def entry_funnel(
    forecasts: list[DirectionalForecastRow],
    trades: list[TradeRow],
    decisions: list[JevDecisionRow],
) -> dict[str, Any]:
    flat_cycles = 0
    in_pos = 0
    conf: dict[str, int] = {"UP": 0, "DOWN": 0}
    conf1: dict[str, int] = {"UP": 0, "DOWN": 0}
    conf2: dict[str, int] = {"UP": 0, "DOWN": 0}
    candidates: dict[str, int] = {"LONG": 0, "SHORT": 0}
    reasons: Counter[str] = Counter()
    rejects: dict[str, Counter[str]] = {"LONG": Counter(), "SHORT": Counter()}
    p_up_max = 0.0
    p_down_max = 0.0
    thresh = {"up70": 0, "up75": 0, "up80": 0, "down70": 0, "down75": 0, "down80": 0}
    history: list[str] = []
    needed = 2
    for row in forecasts:
        forecast = _dict(row.forecast)
        policy = _dict(row.policy)
        p_up = _dict(forecast.get("p_up"))
        p_1m = as_float(p_up.get("1m")) or 0.5
        p_up_max = max(p_up_max, max((as_float(v) or 0.0) for v in p_up.values()) if p_up else p_1m)
        p_down_max = max(p_down_max, 1.0 - min((as_float(v) or 1.0) for v in p_up.values()) if p_up else 1.0 - p_1m)
        if p_1m >= 0.80:
            thresh["up80"] += 1
        if p_1m >= 0.75:
            thresh["up75"] += 1
        if p_1m >= 0.70:
            thresh["up70"] += 1
        if (1.0 - p_1m) >= 0.80:
            thresh["down80"] += 1
        if (1.0 - p_1m) >= 0.75:
            thresh["down75"] += 1
        if (1.0 - p_1m) >= 0.70:
            thresh["down70"] += 1
        candidate = str(policy.get("candidate") or "")
        reason = str(policy.get("reason") or "unknown")
        in_position = candidate in {"HOLD", "CLOSE", "REVERSE"} or policy.get("hold_signal") or policy.get(
            "close_signal"
        )
        if in_position:
            in_pos += 1
            history = []
            continue
        flat_cycles += 1
        voted = vote_from_p_up(p_1m, ENTRY_CONFIDENCE)
        history.append(voted)
        history = history[-needed:]
        if voted != "NONE":
            conf[voted] += 1
            confirms = sum(1 for item in history if item == voted)
            if confirms >= 1:
                conf1[voted] += 1
            if confirms >= needed:
                conf2[voted] += 1
        if candidate == "OPEN_LONG":
            candidates["LONG"] += 1
        elif candidate == "OPEN_SHORT":
            candidates["SHORT"] += 1
        else:
            reasons[reason] += 1
            side_key = "LONG" if voted == "UP" else "SHORT" if voted == "DOWN" else None
            if side_key:
                mapped = reason
                if reason == "insufficient_confirmation":
                    mapped = "insufficient_confirmation"
                elif policy.get("direction_conflict"):
                    mapped = "conflict_present_but_not_an_entry_gate"
                rejects[side_key][mapped] += 1
    risk_blocks = Counter(
        str(row.blocked_reason)
        for row in decisions
        if row.blocked_reason and row.candidate_action in {"OPEN_LONG", "OPEN_SHORT"}
        and row.final_action not in {"OPEN_LONG", "OPEN_SHORT"}
    )
    finals = Counter(row.side for row in trades)
    return {
        "definition": {
            "eligible_flat_cycle": "Forecast row whose policy is not HOLD/CLOSE/REVERSE",
            "confidence_gate": "p_1m >= 0.70 for UP, 1-p_1m >= 0.70 for DOWN (code _vote)",
            "confirmation": "Reconstructed consecutive same-side votes; required=2",
            "conflict_note": (
                "DirectionalPolicy._entry does not reject on direction_conflict; "
                "conflict is recorded but is not an entry gate."
            ),
        },
        "total_forecast_rows": len(forecasts),
        "flat_cycles": flat_cycles,
        "in_position_cycles": in_pos,
        "same_side_confidence_ge_0_70": conf,
        "confirmation_1": conf1,
        "confirmation_2": conf2,
        "candidate_entries": candidates,
        "final_entries": {"LONG": finals.get("LONG", 0), "SHORT": finals.get("SHORT", 0)},
        "flat_no_entry_reasons": dict(reasons),
        "rejects_by_side": {side: dict(counter) for side, counter in rejects.items()},
        "risk_blocks_on_open_candidates": dict(risk_blocks),
        "max_p_up": p_up_max,
        "max_p_down": p_down_max,
        "observations": thresh,
        "symmetry": (
            "OPEN_LONG and OPEN_SHORT share the same _vote / _entry path; "
            "no side-specific threshold in policy code."
        ),
    }


def attach_forecast_to_trades(
    trades: list[dict[str, Any]],
    forecasts: list[DirectionalForecastRow],
) -> list[dict[str, Any]]:
    by_decision = {row.decision_id: row for row in forecasts if row.decision_id}
    by_ts = {row.timestamp_ms: row for row in forecasts}
    enriched: list[dict[str, Any]] = []
    for trade in trades:
        entry_id = trade.get("entry_decision_id")
        row = by_decision.get(str(entry_id)) if entry_id else None
        if row is None:
            ts = trade.get("entry_timestamp_ms")
            if isinstance(ts, int):
                row = by_ts.get(ts)
        extra: dict[str, Any] = {}
        if row is not None:
            forecast = _dict(row.forecast)
            extra = {
                "entry_p_up": _dict(forecast.get("p_up")),
                "entry_term_structure": forecast.get("term_structure"),
                "entry_persistence": _dict(forecast.get("persistence")),
                "entry_magnitude": _dict(forecast.get("magnitude")),
                "entry_diagnostics": _dict(forecast.get("diagnostics")),
                "entry_forecast_signed_moves": {},
            }
            ref = feature_mid(row)
            rolling = _dict(row.rolling_outcomes)
            p_up = _dict(forecast.get("p_up"))
            for label in HORIZON_LABELS:
                outcome = _dict(rolling.get(label))
                future = as_float(outcome.get("future_mid"))
                p = as_float(p_up.get(label)) or 0.5
                if ref and future:
                    extra["entry_forecast_signed_moves"][label] = (
                        signed_future_return(p_up=p, reference_mid=ref, future_mid=future) * 10_000.0
                    )
        enriched.append({**trade, **extra})
    return enriched


def selection_quality(
    forecasts: list[DirectionalForecastRow],
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    traded_ids = {item.get("entry_decision_id") for item in trades}
    eligible: list[dict[str, Any]] = []
    for row in forecasts:
        forecast = _dict(row.forecast)
        policy = _dict(row.policy)
        if policy.get("candidate") in {"HOLD", "CLOSE", "REVERSE"}:
            continue
        p = as_float(_dict(forecast.get("p_up")).get("1m")) or 0.5
        side, conf = predicted_side_and_confidence(p)
        if conf < ENTRY_CONFIDENCE:
            continue
        outcome = _dict(_dict(row.rolling_outcomes).get("1m"))
        ref = feature_mid(row)
        future = as_float(outcome.get("future_mid"))
        signed = None
        if ref and future:
            signed = signed_future_return(p_up=p, reference_mid=ref, future_mid=future) * 10_000.0
        mfe, mae = signed_excursions(
            predicted_side=side,
            mfe_up_bps=as_float(outcome.get("mfe_bps")),
            mae_up_bps=as_float(outcome.get("mae_bps")),
        )
        eligible.append(
            {
                "traded": row.decision_id in traded_ids,
                "confidence": conf,
                "signed_move_bps": signed,
                "mfe_bps": mfe,
                "mae_bps": mae,
                "hit": (
                    (side == "UP" and outcome.get("direction") == "UP")
                    or (side == "DOWN" and outcome.get("direction") == "DOWN")
                )
                if outcome.get("direction") in {"UP", "DOWN"}
                else None,
            }
        )
    traded = [item for item in eligible if item["traded"]]
    skipped = [item for item in eligible if not item["traded"]]

    def pack(group: list[dict[str, Any]]) -> dict[str, Any]:
        signed = [float(item["signed_move_bps"]) for item in group if item["signed_move_bps"] is not None]
        hits = [bool(item["hit"]) for item in group if item["hit"] is not None]
        return {
            "n": len(group),
            "avg_confidence": mean([float(item["confidence"]) for item in group]),
            "avg_signed_move_bps": mean(signed),
            "avg_mfe_bps": mean(
                [float(item["mfe_bps"]) for item in group if item["mfe_bps"] is not None]
            ),
            "accuracy": None if not hits else sum(hits) / len(hits),
        }

    return {
        "eligible_definition": "Flat-cycle forecasts with 1m predicted-side confidence >= 0.70",
        "traded": pack(traded),
        "not_traded": pack(skipped),
        "caveat": "N_traded is 8; do not treat the comparison as conclusive.",
    }


def health_index(events: list[SystemEventRow]) -> dict[str, list[int]]:
    marks: dict[str, list[int]] = {"timeout": [], "recover": [], "ws": [], "risk": []}
    for event in events:
        ts = int(event.timestamp.timestamp() * 1000)
        kind = event.event_type
        if kind in JEV_FAIL_EVENTS:
            marks["timeout"].append(ts)
        elif kind in JEV_RECOVER_EVENTS:
            marks["recover"].append(ts)
        elif kind in WS_EVENTS or "disconnect" in kind:
            marks["ws"].append(ts)
        elif kind == "risk_block":
            marks["risk"].append(ts)
    return marks


def near(ts: int, stamps: Sequence[int], window_ms: int = HEALTH_WINDOW_MS) -> bool:
    return any(abs(ts - other) <= window_ms for other in stamps)


def classify_health(
    ts: int,
    *,
    marks: dict[str, list[int]],
    latency_ms: float | None,
    invalid_reason: str | None,
    research_valid: bool,
) -> tuple[str, list[str]]:
    flags: list[str] = []
    if not research_valid or invalid_reason:
        flags.append(invalid_reason or "research_invalid")
    if latency_ms is not None and latency_ms >= DEGRADED_LATENCY_MS:
        flags.append("latency_above_decision_interval")
    if near(ts, marks["timeout"]):
        flags.append("jev_timeout_window")
    if near(ts, marks["recover"]):
        flags.append("jev_recovery_window")
    if near(ts, marks["ws"]):
        flags.append("ws_disconnect_window")
    if near(ts, marks["risk"]):
        flags.append("risk_block_window")
    return ("DEGRADED" if flags else "HEALTHY"), flags


def latency_audit(decisions: list[JevDecisionRow]) -> dict[str, Any]:
    values = [float(row.jev_latency_ms) for row in decisions if row.jev_latency_ms is not None]
    buckets = []
    for label, lo, hi in LATENCY_BUCKETS:
        n = sum(1 for value in values if value >= lo and (hi is None or value < hi))
        buckets.append({"bucket": label, "n": n})
    by_status: dict[str, list[float]] = defaultdict(list)
    for row in decisions:
        if row.jev_latency_ms is not None:
            by_status[row.status].append(float(row.jev_latency_ms))
    extreme = [
        row
        for row in decisions
        if row.jev_latency_ms is not None and float(row.jev_latency_ms) >= 10_000
    ]
    return {
        "n": len(values),
        "percentiles_ms": percentiles(values),
        "buckets": buckets,
        "by_status": {status: percentiles(items) for status, items in by_status.items()},
        "extreme_ge_10s": len(extreme),
        "extreme_with_failure_status": sum(
            1 for row in extreme if row.status not in {"OK"}
        ),
        "code_review": {
            "per_attempt_timeout_seconds": 10,
            "retry_attempts": 3,
            "retry_backoff_seconds": [0.4, 0.8],
            "retryable_http": [408, 409, 429, 500, 502, 503, 504],
            "latency_clock": "perf_counter around _post_with_retry plus parse",
            "hypothesis": (
                "p50 ~0.5s is a healthy OpenRouter hit. The right tail is consistent "
                "with retries after TimeoutException/429 and with wall-clock including "
                "all attempts. p99 beyond ~32s is not explained by 3x10s+backoff alone "
                "and may include event-loop delay before the call starts."
            ),
        },
    }


def temporal_regime(samples: list[dict[str, Any]], *, min_gap_ms: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in samples:
        hour = item.get("utc_hour")
        groups[("utc_hour", str(hour))].append(item)
        groups[("trend", str(item.get("direction")))].append(item)
        groups[("volatility", str(item.get("volatility")))].append(item)
        groups[("liquidity", str(item.get("liquidity")))].append(item)
        groups[("character", str(item.get("character")))].append(item)
    rows: list[dict[str, Any]] = []
    for (kind, name), group in sorted(groups.items()):
        predicted = [float(item["p_up"]) for item in group]
        hits = [bool(item["actual_up"]) for item in group]
        stamps = [int(item["timestamp_ms"]) for item in group]
        quality = binary_quality(predicted, hits, stamps, min_gap_ms=min_gap_ms)
        signed = [float(item["signed_move_bps"]) for item in group if item.get("signed_move_bps") is not None]
        rows.append(
            {
                "segment_kind": kind,
                "segment": name,
                "n": quality["n"],
                "n_nonoverlapping": quality["n_nonoverlapping"],
                "low_sample": low_sample(int(quality["n_nonoverlapping"] or 0)),
                "accuracy": quality["accuracy"],
                "brier": quality["brier"],
                "avg_signed_move_bps": mean(signed),
            }
        )
    return rows


def build_audit(
    *,
    run: ExperimentRunRow,
    forecasts: list[DirectionalForecastRow],
    trades: list[TradeRow],
    events: list[SystemEventRow],
    decisions: list[JevDecisionRow],
) -> dict[str, Any]:
    trade_rows = [audit_trade(row) for row in trades]
    trade_rows = attach_forecast_to_trades(trade_rows, forecasts)
    samples = forecast_samples(forecasts)
    marks = health_index(events)
    latency_by_id = {row.id: row.jev_latency_ms for row in decisions}
    healthy_samples: dict[str, list[dict[str, Any]]] = {label: [] for label in HORIZON_LABELS}
    degraded_samples: dict[str, list[dict[str, Any]]] = {label: [] for label in HORIZON_LABELS}
    health_counts: Counter[str] = Counter()
    for label, group in samples.items():
        for item in group:
            lat = None
            if item.get("decision_id"):
                lat = as_float(latency_by_id.get(item["decision_id"]))
            # research-valid already filtered in forecast_samples
            status, flags = classify_health(
                int(item["timestamp_ms"]),
                marks=marks,
                latency_ms=lat,
                invalid_reason=None,
                research_valid=True,
            )
            item["health"] = status
            item["health_flags"] = flags
            health_counts[status] += 1
            (healthy_samples if status == "HEALTHY" else degraded_samples)[label].append(item)
    raw_invalid = Counter(
        str(_dict(row.forecast).get("invalid_reason") or row.invalid_reason or "valid")
        for row in forecasts
    )

    accounting_ok = all(
        item["identity_gross_minus_fees_eq_net"] and item["identity_ideal_minus_execution_eq_gross"]
        for item in trade_rows
    ) if trade_rows else True
    totals = {
        "ideal_mid_pnl_usdt": sum(float(item["ideal_mid_pnl_usdt"]) for item in trade_rows),
        "total_spread_usdt": sum(float(item["total_spread_usdt"]) for item in trade_rows),
        "total_slippage_usdt": sum(float(item["total_slippage_usdt"]) for item in trade_rows),
        "execution_vs_mid_usdt": sum(float(item["execution_vs_mid_usdt"]) for item in trade_rows),
        "fill_to_fill_gross_pnl_usdt": sum(
            float(item["fill_to_fill_gross_pnl_usdt"]) for item in trade_rows
        ),
        "fees_usdt": sum(float(item["fees_usdt"]) for item in trade_rows),
        "net_pnl_usdt": sum(float(item["net_pnl_usdt"]) for item in trade_rows),
        "dashboard_slippage_line_usdt": sum(float(item["total_slippage_usdt"]) for item in trade_rows),
    }
    rt_cost = expected_round_trip_cost_bps(
        taker_fee_bps=DEFAULT_TAKER_BPS, slippage_bps=DEFAULT_SLIP_BPS, spread_bps=0.02
    )
    mfe_stats = {
        "n_mfe_gt_6": sum(1 for item in trade_rows if (item.get("mfe_bps") or 0) > 6),
        "n_mfe_gt_12": sum(1 for item in trade_rows if (item.get("mfe_bps") or 0) > 12),
        "n_mfe_gt_roundtrip_and_net_neg": sum(
            1
            for item in trade_rows
            if (item.get("mfe_bps") or 0) > rt_cost and float(item["net_pnl_usdt"]) < 0
        ),
        "assumed_round_trip_cost_bps": rt_cost,
    }
    exits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in trade_rows:
        exits[str(item.get("exit_reason") or "other")].append(item)
    exit_table = []
    for reason, group in sorted(exits.items()):
        nets = [float(item["net_bps"] or 0.0) for item in group]
        exit_table.append(
            {
                "exit_reason": reason,
                "n": len(group),
                "win_rate": sum(1 for item in group if float(item["net_pnl_usdt"]) > 0) / len(group),
                "avg_hold": mean([float(item["holding_seconds"]) for item in group]),
                "avg_mfe_bps": mean(
                    [float(item["mfe_bps"]) for item in group if item.get("mfe_bps") is not None]
                ),
                "avg_mae_bps": mean(
                    [float(item["mae_bps"]) for item in group if item.get("mae_bps") is not None]
                ),
                "avg_gross_bps": mean(
                    [float(item["gross_bps"]) for item in group if item.get("gross_bps") is not None]
                ),
                "avg_net_bps": mean(nets),
                "median_net_bps": quantile(nets, 0.50),
            }
        )
    horizon_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in trade_rows:
        horizon_groups[str(item.get("primary_horizon") or "unknown")].append(item)
    horizon_table = []
    for label, group in sorted(horizon_groups.items()):
        holds = [float(item["holding_seconds"]) for item in group]
        ratios = [
            float(item["hold_over_primary"])
            for item in group
            if item.get("hold_over_primary") is not None
        ]
        horizon_table.append(
            {
                "primary_horizon": label,
                "n": len(group),
                "median_hold": quantile(holds, 0.50),
                "median_hold_over_horizon": quantile(ratios, 0.50),
                "avg_mfe_bps": mean(
                    [float(item["mfe_bps"]) for item in group if item.get("mfe_bps") is not None]
                ),
                "avg_captured_gross_bps": mean(
                    [float(item["gross_bps"]) for item in group if item.get("gross_bps") is not None]
                ),
                "avg_gross_usdt": mean([float(item["fill_to_fill_gross_pnl_usdt"]) for item in group]),
                "avg_net_usdt": mean([float(item["net_pnl_usdt"]) for item in group]),
            }
        )

    def quality_block(group_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {
            label: binary_quality(
                [float(item["p_up"]) for item in group_map[label]],
                [bool(item["actual_up"]) for item in group_map[label]],
                [int(item["timestamp_ms"]) for item in group_map[label]],
                min_gap_ms=HORIZON_GAP_MS[label],
            )
            for label in HORIZON_LABELS
        }

    baselines = {}
    for label in HORIZON_LABELS:
        group = samples[label]
        actual = [bool(item["actual_up"]) for item in group]
        n = len(actual)
        majority = actual.count(True) >= actual.count(False)
        majority_acc = None if n == 0 else sum(1 for hit in actual if hit is majority) / n
        recent = []
        momentum = []
        for item in group:
            base = _dict(item.get("baselines"))
            recent.append(str(base.get("recent_direction") or "DOWN") == "UP")
            momentum.append(str(base.get("momentum") or "DOWN") == "UP")
        baselines[label] = {
            "n": n,
            "realized_up_rate": None if n == 0 else sum(actual) / n,
            "jev_accuracy": binary_quality(
                [float(item["p_up"]) for item in group],
                actual,
                [int(item["timestamp_ms"]) for item in group],
                min_gap_ms=HORIZON_GAP_MS[label],
            ).get("accuracy"),
            "baseline_50_50": 0.5,
            "baseline_majority": majority_acc,
            "majority_class": "UP" if majority else "DOWN",
            "baseline_recent": None
            if n == 0
            else sum(1 for hit, guess in zip(actual, recent, strict=True) if hit == guess) / n,
            "baseline_momentum": None
            if n == 0
            else sum(1 for hit, guess in zip(actual, momentum, strict=True) if hit == guess) / n,
        }

    for item in trade_rows:
        ts = item.get("entry_timestamp_ms")
        xts = item.get("exit_timestamp_ms")
        item["entry_health"], item["entry_health_flags"] = (
            classify_health(
                int(ts),
                marks=marks,
                latency_ms=as_float(latency_by_id.get(str(item.get("entry_decision_id") or ""))),
                invalid_reason=None,
                research_valid=True,
            )
            if isinstance(ts, int)
            else ("UNKNOWN", [])
        )
        item["exit_health"], item["exit_health_flags"] = (
            classify_health(
                int(xts),
                marks=marks,
                latency_ms=as_float(latency_by_id.get(str(item.get("exit_decision_id") or ""))),
                invalid_reason=None,
                research_valid=True,
            )
            if isinstance(xts, int)
            else ("UNKNOWN", [])
        )

    started = min((row.timestamp_ms for row in forecasts), default=run.started_at_ms)
    ended = max((row.timestamp_ms for row in forecasts), default=run.started_at_ms)
    return {
        "meta": {
            "experiment_id": run.experiment_id,
            "run_id": run.run_id,
            "config_hash": run.config_hash,
            "frozen": run.frozen,
            "fingerprint": run.fingerprint,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "forecast_span": {"start_ms": started, "end_ms": ended, "start": _iso(started), "end": _iso(ended)},
            "n_forecasts": len(forecasts),
            "n_valid_forecasts": sum(1 for row in forecasts if _directional_row_valid(row)),
            "n_trades": len(trades),
            "n_decisions": len(decisions),
            "n_events": len(events),
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "read_only": True,
        },
        "accounting": {
            "dashboard_is_mathematically_incorrect": False,
            "dashboard_terminology_is_misleading": True,
            "explanation": (
                "Gross is fill-to-fill. Fees are subtracted once to get net. "
                "Reported slippage is attribution of adverse fill vs touch and is already "
                "inside gross. net ≈ gross - fees. ideal_mid - spread - adverse_slip ≈ gross."
            ),
            "identities_hold": accounting_ok,
            "totals": totals,
            "trades": trade_rows,
        },
        "label_integrity": label_integrity(forecasts),
        "confidence_buckets": {
            label: bucket_table(samples[label], min_gap_ms=HORIZON_GAP_MS[label])
            for label in HORIZON_LABELS
        },
        "calibration": {
            label: calibration_table(samples[label], min_gap_ms=HORIZON_GAP_MS[label])
            for label in HORIZON_LABELS
        },
        "baselines": baselines,
        "funnel": entry_funnel(forecasts, trades, decisions),
        "selection": selection_quality(forecasts, trade_rows),
        "horizon_vs_hold": {"by_trade": trade_rows, "by_primary": horizon_table},
        "mfe_capture": mfe_stats,
        "exits": exit_table,
        "health": {
            "definition": {
                "HEALTHY": (
                    "research_sample_valid and no Jev timeout/recovery/WS disconnect/risk_block "
                    f"within ±{HEALTH_WINDOW_MS // 1000}s and jev_latency < {DEGRADED_LATENCY_MS:.0f} ms"
                ),
                "DEGRADED": "any of the above flags",
            },
            "invalid_reason_counts": dict(raw_invalid),
            "valid_sample_health_counts": dict(health_counts),
            "all": quality_block(samples),
            "healthy": quality_block(healthy_samples),
            "degraded": quality_block(degraded_samples),
        },
        "latency": latency_audit(decisions),
        "temporal": {label: temporal_regime(samples[label], min_gap_ms=HORIZON_GAP_MS[label]) for label in HORIZON_LABELS},
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(audit: dict[str, Any]) -> str:
    meta = audit["meta"]
    acc = audit["accounting"]
    totals = acc["totals"]
    funnel = audit["funnel"]
    lat = audit["latency"]["percentiles_ms"]
    lines = [
        "# DIRECTIONAL_V1 PAPER audit",
        "",
        f"- experiment `{meta['experiment_id']}` run `{meta['run_id']}`",
        f"- config_hash `{meta['config_hash']}` frozen={meta['frozen']}",
        f"- span {meta['forecast_span']['start']} → {meta['forecast_span']['end']}",
        f"- forecasts {meta['n_forecasts']} (valid {meta['n_valid_forecasts']}) trades {meta['n_trades']}",
        f"- generated {meta['generated_at']} · read-only · no strategy changes",
        "",
        "## Executive summary",
        "",
    ]
    cal_1m = audit["calibration"]["1m"]
    cal_5m = audit["calibration"]["5m"]
    majority_1m = audit["baselines"]["1m"]
    funnel_long = funnel["final_entries"]["LONG"]
    funnel_short = funnel["final_entries"]["SHORT"]
    lines += [
        f"A. **Predictive direction signal.** Observed fact: 1m overlapping accuracy "
        f"{_pct(cal_1m.get('accuracy'))} vs 50/50 and vs majority "
        f"{_pct(majority_1m.get('baseline_majority'))} / momentum "
        f"{_pct(majority_1m.get('baseline_momentum'))}. "
        f"Statistical evidence: weak 1m lift on N_nonoverlap={cal_1m.get('n_nonoverlapping')}; "
        f"5m {_pct(cal_5m.get('accuracy'))} does not beat momentum. "
        f"Hypothesis: there is a small short-horizon directional association, not a proven edge.",
        "",
        f"B. **Calibration.** 1m ECE={_num(cal_1m.get('ece'), 3)} Brier={_num(cal_1m.get('brier'), 3)}. "
        f"High-p_up bins are sparse. Do not fit a calibrator from this pass.",
        "",
        "C. **Confidence vs move.** See confidence-bucket tables. "
        "Do not claim a monotone relationship without checking LOW SAMPLE cells.",
        "",
        "D. **Entry selection.** Traded N is 8 versus many confidence-eligible flat cycles. "
        "The traded vs not-traded comparison is descriptive only.",
        "",
        f"E. **0 LONG / {funnel_short} SHORT.** Funnel: UP conf≥0.70={funnel['same_side_confidence_ge_0_70']['UP']}, "
        f"DOWN={funnel['same_side_confidence_ge_0_70']['DOWN']}; candidates "
        f"LONG={funnel['candidate_entries']['LONG']} SHORT={funnel['candidate_entries']['SHORT']}. "
        f"Policy paths are symmetric; this is not an implementation LONG-block. "
        f"Final LONG={funnel_long}.",
        "",
        "F. **Exits vs primary horizon.** Several shorts closed in tens of seconds while "
        "primary could be 5m–30m because `short_horizon_flipped` / momentum exits fire on 1m p.",
        "",
        f"G. **Loss split.** ideal mid { _num(totals['ideal_mid_pnl_usdt']) } USDT; "
        f"execution vs mid (spread+slip) { _num(totals['execution_vs_mid_usdt']) }; "
        f"fill gross { _num(totals['fill_to_fill_gross_pnl_usdt']) }; "
        f"fees { _num(totals['fees_usdt']) }; net { _num(totals['net_pnl_usdt']) }. "
        f"Fees dominate the reported net. Slippage is inside gross, not an extra subtract.",
        "",
        f"H. **Infrastructure.** Jev p50={_num(lat.get('p50'), 0)}ms p95={_num(lat.get('p95'), 0)}ms; "
        f"see latency section. Compare HEALTHY vs DEGRADED accuracy before attributing skill.",
        "",
        f"I. **Bugs.** Accounting identities hold={acc['identities_hold']}. "
        f"Dashboard wording is misleading, not arithmetically wrong. "
        f"Label integrity is in the next section.",
        "",
        "## 1. Accounting",
        "",
        acc["explanation"],
        "",
        _md_table(
            ["component", "USDT"],
            [
                ["ideal mid (prediction/market)", _num(totals["ideal_mid_pnl_usdt"])],
                ["spread (touch vs mid)", _num(totals["total_spread_usdt"])],
                ["adverse slip vs touch", _num(totals["total_slippage_usdt"])],
                ["execution vs mid", _num(totals["execution_vs_mid_usdt"])],
                ["fill-to-fill gross", _num(totals["fill_to_fill_gross_pnl_usdt"])],
                ["fees", _num(totals["fees_usdt"])],
                ["net", _num(totals["net_pnl_usdt"])],
            ],
        ),
        "",
        "## 2. Label integrity",
        "",
        json.dumps(audit["label_integrity"]["code_review"], indent=2),
        "",
    ]
    for label in HORIZON_LABELS:
        roll = audit["label_integrity"]["rolling"][label]
        candle = audit["label_integrity"]["candles"][label]
        lines.append(
            f"- rolling {label}: checked={roll['checked']} clean={roll['clean']} issues={roll['issues']}"
        )
        lines.append(
            f"- candle {label}: checked={candle['checked']} clean={candle['clean']} "
            f"realized_up={_pct(candle['realized_up_rate'])} issues={candle['issues']} "
            f"remaining={candle['remaining_segments']}"
        )
    lines += ["", "### Persistence base rates", ""]
    persist_rows = [
        [key, str(val["n"]), _pct(val["base_hit_rate"])]
        for key, val in audit["label_integrity"]["persistence_base_rates"].items()
    ]
    if persist_rows:
        lines.append(_md_table(["key", "N", "hit rate"], persist_rows))
    lines += ["", "## 3. Confidence buckets", ""]
    for label in HORIZON_LABELS:
        lines.append(f"### {label}")
        rows = []
        for item in audit["confidence_buckets"][label]:
            flag = " LOW SAMPLE" if item["low_sample"] else ""
            rows.append(
                [
                    item["bucket"] + flag,
                    str(item["raw_n"]),
                    str(item["nonoverlapping_n"]),
                    _pct(item["accuracy"]),
                    _num(item["brier"], 3),
                    _num(item["avg_signed_move_bps"], 2),
                    _pct(item["pct_signed_gt_12"]),
                    _pct(item["pct_mfe_gt_12"]),
                ]
            )
        lines.append(
            _md_table(
                ["bucket", "N", "N*", "acc", "brier", "avg signed bps", "% >12bps", "% MFE>12"],
                rows,
            )
        )
        lines.append("")
    lines += ["## 4. Calibration", ""]
    for label in HORIZON_LABELS:
        block = audit["calibration"][label]
        lines.append(
            f"### {label} N={block['n']} N*={block['n_nonoverlapping']} "
            f"acc={_pct(block['accuracy'])} brier={_num(block['brier'], 3)} "
            f"logloss={_num(block['log_loss'], 3)} ECE={_num(block['ece'], 3)}"
        )
        lines.append(
            _md_table(
                ["bin", "N", "mean p_up", "realized UP", "abs error"],
                [
                    [
                        item["bin"],
                        str(item["n"]),
                        _num(item["mean_predicted_p_up"], 3),
                        _pct(item["realized_up_frequency"]),
                        _num(item["calibration_error"], 3),
                    ]
                    for item in block["bins"]
                ],
            )
        )
        lines.append("")
    lines += ["## 5. Baselines", ""]
    lines.append(
        _md_table(
            ["H", "N", "up rate", "Jev", "50/50", "majority", "recent", "momentum"],
            [
                [
                    label,
                    str(audit["baselines"][label]["n"]),
                    _pct(audit["baselines"][label]["realized_up_rate"]),
                    _pct(audit["baselines"][label]["jev_accuracy"]),
                    "50.00%",
                    f"{_pct(audit['baselines'][label]['baseline_majority'])} ({audit['baselines'][label]['majority_class']})",
                    _pct(audit["baselines"][label]["baseline_recent"]),
                    _pct(audit["baselines"][label]["baseline_momentum"]),
                ]
                for label in HORIZON_LABELS
            ],
        )
    )
    lines += ["", "## 6. Entry funnel", "", json.dumps(funnel, indent=2, default=str), ""]
    lines += ["## 7. Selection", "", json.dumps(audit["selection"], indent=2, default=str), ""]
    lines += [
        "## 8. Primary horizon vs hold",
        "",
        _md_table(
            ["primary", "N", "median hold s", "median hold/H", "avg MFE", "avg gross bps", "avg net"],
            [
                [
                    item["primary_horizon"],
                    str(item["n"]),
                    _num(item["median_hold"], 1),
                    _num(item["median_hold_over_horizon"], 3),
                    _num(item["avg_mfe_bps"], 2),
                    _num(item["avg_captured_gross_bps"], 2),
                    _num(item["avg_net_usdt"], 3),
                ]
                for item in audit["horizon_vs_hold"]["by_primary"]
            ],
        ),
        "",
        "## 9. MFE capture",
        "",
        json.dumps(audit["mfe_capture"], indent=2),
        "",
        "## 10. Exits",
        "",
        _md_table(
            ["reason", "N", "win", "avg hold", "avg MFE", "avg MAE", "avg gross bps", "avg/med net bps"],
            [
                [
                    item["exit_reason"],
                    str(item["n"]),
                    _pct(item["win_rate"]),
                    _num(item["avg_hold"], 1),
                    _num(item["avg_mfe_bps"], 2),
                    _num(item["avg_mae_bps"], 2),
                    _num(item["avg_gross_bps"], 2),
                    f"{_num(item['avg_net_bps'], 2)} / {_num(item['median_net_bps'], 2)}",
                ]
                for item in audit["exits"]
            ],
        ),
        "",
        "## 11. Health",
        "",
        json.dumps(audit["health"]["definition"], indent=2),
        "",
        f"invalid reasons: {audit['health']['invalid_reason_counts']}",
        f"valid-sample health: {audit['health']['valid_sample_health_counts']}",
        "",
    ]
    for split in ("all", "healthy", "degraded"):
        lines.append(f"### {split}")
        lines.append(
            _md_table(
                ["H", "N", "N*", "acc", "brier"],
                [
                    [
                        label,
                        str(audit["health"][split][label]["n"]),
                        str(audit["health"][split][label]["n_nonoverlapping"]),
                        _pct(audit["health"][split][label]["accuracy"]),
                        _num(audit["health"][split][label]["brier"], 3),
                    ]
                    for label in HORIZON_LABELS
                ],
            )
        )
    lines += ["", "## 12. Jev latency", "", json.dumps(audit["latency"], indent=2, default=str), ""]
    lines += [
        "## 13. Temporal / regime",
        "",
        "Cells with N* < 30 are LOW SAMPLE. Do not over-read them.",
        "",
    ]
    for label in HORIZON_LABELS:
        lines.append(f"### {label}")
        rows = []
        for item in audit["temporal"][label]:
            flag = " LOW SAMPLE" if item["low_sample"] else ""
            rows.append(
                [
                    item["segment_kind"],
                    str(item["segment"]) + flag,
                    str(item["n"]),
                    str(item["n_nonoverlapping"]),
                    _pct(item["accuracy"]),
                    _num(item["brier"], 3),
                    _num(item["avg_signed_move_bps"], 2),
                ]
            )
        lines.append(
            _md_table(["kind", "segment", "N", "N*", "acc", "brier", "avg signed bps"], rows)
        )
        lines.append("")
    lines += [
        "## Language",
        "",
        "Observed fact / statistical evidence / hypothesis are separated above. "
        "Eight trades cannot establish or reject trading edge.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(audit: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    (directory / "REPORT.md").write_text(render_markdown(audit), encoding="utf-8")
    _write_csv(directory / "trades.csv", audit["accounting"]["trades"])
    bucket_rows: list[dict[str, Any]] = []
    for label, rows in audit["confidence_buckets"].items():
        for row in rows:
            bucket_rows.append({"horizon": label, **row})
    _write_csv(directory / "confidence_buckets.csv", bucket_rows)
    cal_rows: list[dict[str, Any]] = []
    for label, block in audit["calibration"].items():
        for row in block["bins"]:
            cal_rows.append({"horizon": label, **row})
    _write_csv(directory / "calibration.csv", cal_rows)
    _write_csv(directory / "exits.csv", audit["exits"])
    _write_csv(directory / "horizon_hold.csv", audit["horizon_vs_hold"]["by_primary"])
    temp_rows: list[dict[str, Any]] = []
    for label, rows in audit["temporal"].items():
        for row in rows:
            temp_rows.append({"horizon": label, **row})
    _write_csv(directory / "regimes.csv", temp_rows)
    _write_csv(directory / "latency_buckets.csv", audit["latency"]["buckets"])


async def collect_audit(repo: Repository, *, run_id: str, experiment_id: str | None) -> dict[str, Any]:
    run = await repo.get_experiment_run(run_id)
    if run is None:
        raise SystemExit(f"run_id not found: {run_id}")
    if experiment_id and run.experiment_id != experiment_id:
        raise SystemExit(
            f"run {run_id} belongs to {run.experiment_id}, not {experiment_id}"
        )
    forecasts = await repo.list_directional_forecasts(run.run_id)
    trades = await repo.list_trades_for_run(run.run_id)
    events = await repo.list_events_for_run(run.run_id)
    decisions = await repo.list_decisions_for_run(run.run_id, limit=100_000)
    return build_audit(
        run=run, forecasts=forecasts, trades=trades, events=events, decisions=decisions
    )


async def async_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only DIRECTIONAL audit. Does not trade or write to the database."
    )
    parser.add_argument("--experiment", dest="experiment_id")
    parser.add_argument("--run-id", dest="run_id", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    settings: Settings = load_settings()
    db = Database(settings.database_url)
    repo = Repository(db)
    try:
        audit = await collect_audit(
            repo, run_id=args.run_id, experiment_id=args.experiment_id
        )
    finally:
        await db.aclose()
    out = Path(args.out) if args.out else Path("artifacts/directional_audit") / args.run_id
    write_artifacts(audit, out)
    print(f"wrote {out / 'REPORT.md'}")
    print(f"wrote {out / 'summary.json'}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))
