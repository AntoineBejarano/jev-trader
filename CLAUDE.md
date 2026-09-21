# Jev Trader

Autonomous multi-asset intraday trader. Jev decides; Python measures and enforces safety.

## Current phase

Phase 4 — paper execution + position management + PnL. LIVE remains blocked. Next is a frozen BTC PAPER prospective run, then Phase 5 only if hysteresis/risk needs a dedicated pass.

Do not implement a traditional indicator strategy in Python. Do not let Jev override hard risk limits in later phases.

## Architectural decisions

- Name the system Jev Trader, not BTC Trader. BTCUSDT is the first configured symbol.
- Use Binance USD-M perpetual futures (`binance_usdm`) rather than spot so native SHORT, funding, mark, open interest and liquidations are available.
- After 2026-04-23, USD-M WebSockets are split:
  - `wss://fstream.binance.com/public/stream` for `depth` and `bookTicker`
  - `wss://fstream.binance.com/market/stream` for `aggTrade`, `markPrice`, `forceOrder`
  - Open interest has no public WebSocket; poll `GET /fapi/v1/openInterest`
- Maintain a compact top-20 book via `@depth20@100ms` plus `@bookTicker` for best bid/ask. This is enough for imbalance 1/5/10 without a full L2 diff machine in V1. If the public socket drops, clear the book immediately so a reconnect cannot trade on a stale ladder.
- `MarketState` is instrument-tagged (`symbol`, `exchange`) and has a reserved `context` field for later cross-asset features. Do not infer the asset from global process state.
- Pin Jev to `typesafe/jev-1.13`. Never `jev-latest`. Official Decisions API: `POST https://openrouter.ai/api/alpha/decisions` with `{model, state, questions}`. Question types are `noul`, `choice`, `score`.
- Do not use a generic `EXCHANGE` env var; it collides with unrelated environments. Use `TRADING_EXCHANGE=binance_usdm`. The alias `binance` is accepted.
- LIVE requires `TRADING_MODE=LIVE` and `LIVE_TRADING_CONFIRMATION=I_UNDERSTAND_REAL_MONEY_IS_AT_RISK`. Phase 4 still refuses LIVE because authenticated execution does not exist.
- Persistent asyncio worker. Not serverless. Dashboard may later be separate; the engine must keep outbound WebSockets.

## Implementation order

1. Market data + MarketState (done)
2. Jev Decisions API, no trading
3. Persistence of snapshots and decisions
4. Paper execution + PnL (done)
5. DecisionPolicy, hysteresis, RiskEngine (core controls exist in Phase 4; dedicated pass later if needed)
6. Dashboard
7. Replay
8. Authenticated live execution only after explicit human approval

## Commands

```bash
uv sync
uv run python -m src.main
uv run pytest
uv run ruff check src tests
uv run mypy src
```
