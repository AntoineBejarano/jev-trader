from __future__ import annotations

"""USD-M funding attribution hook.

Funding is not the primary concern for second/minute scalps. Accrual stays
disabled (zero) unless PAPER_FUNDING_ENABLED is set. The architecture still
accepts a funding_cost on closed trades so a future long PAPER run that
crosses a funding timestamp can attribute it without rewriting PnL.
"""


def funding_cost_usdt(
    *,
    enabled: bool,
    side: str,
    notional: float,
    funding_rate: float | None,
    entry_timestamp_ms: int,
    exit_timestamp_ms: int,
    next_funding_time_ms: int | None,
) -> float:
    if not enabled or funding_rate is None or next_funding_time_ms is None:
        return 0.0
    if not (entry_timestamp_ms < next_funding_time_ms <= exit_timestamp_ms):
        return 0.0
    signed = 1.0 if side == "LONG" else -1.0
    # Longs pay a positive funding rate; shorts receive it.
    return signed * notional * funding_rate
