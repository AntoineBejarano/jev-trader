from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.market.models import MarketHealth


@runtime_checkable
class MarketDataAdapter(Protocol):
    """Public market-data contract. Execution stays on a separate adapter."""

    exchange: str

    async def connect(self) -> None: ...

    async def subscribe_market_data(self, symbols: tuple[str, ...]) -> None: ...

    async def close(self) -> None: ...

    def health(self) -> MarketHealth: ...
