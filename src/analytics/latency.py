from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True, slots=True)
class LatencyStats:
    count: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    last_ms: float | None


class LatencyTracker:
    def __init__(self, maxlen: int = 500) -> None:
        self._samples: deque[float] = deque(maxlen=maxlen)

    def record(self, value_ms: float | None) -> None:
        if value_ms is None:
            return
        self._samples.append(float(value_ms))

    def stats(self) -> LatencyStats:
        if not self._samples:
            return LatencyStats(0, None, None, None, None)
        ordered = sorted(self._samples)
        return LatencyStats(
            count=len(ordered),
            p50_ms=_percentile(ordered, 50),
            p95_ms=_percentile(ordered, 95),
            p99_ms=_percentile(ordered, 99),
            last_ms=self._samples[-1],
        )


def _percentile(ordered: list[float], pct: float) -> float:
    if not ordered:
        return 0.0
    rank = max(1, min(len(ordered), ceil(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]
