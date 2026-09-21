from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config.settings import Settings
from src.features.returns import simple_return

logger = logging.getLogger(__name__)

SPOT_REST = "https://api.binance.com"
RETURN_WINDOWS = {"r_1m": 1, "r_5m": 5, "r_15m": 15, "r_30m": 30}


class CrossMarketCache:
    """Public BTC spot vs perp and ETHUSDT context. Optional; never blocks."""

    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._http = http
        self._owns = http is None
        self.payload: dict[str, Any] = {"missing": ["spot_basis", "eth"]}
        self.last_ok_ms: int | None = None

    async def refresh(self, now_ms: int) -> dict[str, Any]:
        client = self._http
        owns = False
        if client is None:
            client = httpx.AsyncClient(timeout=5.0)
            owns = True
        missing: list[str] = []
        payload: dict[str, Any] = {}
        try:
            try:
                spot = await client.get(
                    f"{SPOT_REST}/api/v3/ticker/price", params={"symbol": "BTCUSDT"}
                )
                fut = await client.get(
                    f"{self.settings.binance_futures_rest_base}/fapi/v1/ticker/price",
                    params={"symbol": "BTCUSDT"},
                )
                spot.raise_for_status()
                fut.raise_for_status()
                spot_px = float(spot.json()["price"])
                fut_px = float(fut.json()["price"])
                payload["btc_spot"] = spot_px
                payload["btc_perp"] = fut_px
                payload["btc_spot_perp_basis_bps"] = (
                    None if spot_px <= 0 else (fut_px / spot_px - 1.0) * 10_000.0
                )
            except Exception:
                missing.append("spot_basis")
                logger.debug("BTC spot/perp basis unavailable", exc_info=True)
            try:
                klines = await client.get(
                    f"{self.settings.binance_futures_rest_base}/fapi/v1/klines",
                    params={"symbol": "ETHUSDT", "interval": "1m", "limit": 40},
                )
                klines.raise_for_status()
                rows = klines.json()
                closes = [float(row[4]) for row in rows if isinstance(row, list) and len(row) > 4]
                if not closes:
                    raise ValueError("no eth closes")
                last = closes[-1]
                eth_returns: dict[str, float | None] = {}
                for name, minutes in RETURN_WINDOWS.items():
                    if len(closes) <= minutes:
                        eth_returns[name] = None
                    else:
                        eth_returns[name] = simple_return(last, closes[-1 - minutes])
                payload["ethusdt"] = {
                    "last": last,
                    "returns": eth_returns,
                    "momentum": "UP"
                    if (eth_returns.get("r_5m") or 0.0) > 0
                    else "DOWN",
                }
            except Exception:
                missing.append("eth")
                logger.debug("ETHUSDT context unavailable", exc_info=True)
            payload["missing"] = missing
            self.payload = payload
            if "spot_basis" not in missing or "eth" not in missing:
                self.last_ok_ms = now_ms
        finally:
            if owns:
                await client.aclose()
        return self.payload

    async def aclose(self) -> None:
        if self._owns and self._http is not None:
            await self._http.aclose()
