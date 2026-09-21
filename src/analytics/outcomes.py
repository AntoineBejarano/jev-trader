from __future__ import annotations

from src.storage.repository import Repository


async def enrich_decision_outcomes(repo: Repository, now_ms: int) -> int:
    """Label historical Jev decisions with later prices. Never mutates live trading."""
    return await repo.enrich_outcomes(now_ms)
