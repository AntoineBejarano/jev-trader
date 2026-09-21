# Jev Trader

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/uv-Astral-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" alt="uv" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Binance-F0B90B?style=for-the-badge&logo=binance&logoColor=black" alt="Binance USD-M" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/mode-PAPER-3dd68c?style=flat-square" alt="PAPER" />
  <img src="https://img.shields.io/badge/live-BLOCKED-ff5d6c?style=flat-square" alt="LIVE blocked" />
  <img src="https://img.shields.io/badge/Jev-typesafe%2Fjev--1.13-7c5cfc?style=flat-square" alt="Jev model" />
  <img src="https://img.shields.io/badge/venue-Binance%20USD--M-0b1020?style=flat-square" alt="Binance USD-M" />
</p>

**Jev decides. Python measures the market, prices the trade, and enforces safety.**

Jev Trader is an autonomous, instrument-agnostic intraday research system. TypeSafe **Jev** is the probabilistic decision engine. A persistent **Python asyncio** worker owns market data, feature construction, risk, paper execution, and the operator dashboard.

BTCUSDT on Binance USD-M perpetual futures is the first configured symbol — not the product boundary. Additional symbols are a configuration change, not a rewrite.

> [!IMPORTANT]
> This repository is **Phase 4**: live market data + paper execution + PnL. **LIVE trading is hard-blocked.** There is no authenticated Binance order path. Nothing in this README is financial advice.

---

## Why this exists

Most “AI traders” either:

1. dump a chart into a chatbot and hope, or
2. encode RSI / EMA crossovers as if they were a strategy.

Jev Trader does neither.

Jev answers **narrow, versioned, probabilistic questions** (`noul`, `choice`, `score`). Python computes compact microstructure and price-action features, converts those answers into candidate actions, then **refuses** anything that violates hard risk limits. Transaction costs live in execution and analytics — not inside Jev’s market belief.

The research question is not “did the bot print money today?”. It is:

- Does Jev know **direction**?
- Over which **horizons**?
- Can that information survive **spread, fees, and slippage**?

---

## Architecture

```mermaid
flowchart LR
  subgraph Venue["Binance USD-M"]
    PUB["public stream<br/>depth · bookTicker"]
    MKT["market stream<br/>aggTrade · markPrice · forceOrder"]
    REST["REST<br/>open interest"]
  end

  subgraph Engine["Persistent asyncio worker"]
    BUF["Market buffers<br/>top-20 book · mids · trades"]
    FEAT["Feature engine<br/>ms.v2 · intel.v1 · regime.v1"]
    JEV["Jev Decisions API<br/>typesafe/jev-1.13"]
    POL["Strategy policy<br/>SCALP or DIRECTIONAL"]
    RISK["RiskEngine<br/>hard limits"]
    PAPER["Paper execution<br/>bid/ask · fees · slip"]
  end

  subgraph Store["PostgreSQL"]
    DB[("snapshots · decisions<br/>forecasts · trades · PnL")]
  end

  DASH["Operator dashboard<br/>localhost:8000"]

  PUB --> BUF
  MKT --> BUF
  REST --> BUF
  BUF --> FEAT
  FEAT --> JEV
  JEV --> POL
  POL --> RISK
  RISK --> PAPER
  PAPER --> DB
  FEAT --> DB
  JEV --> DB
  DB --> DASH
  PAPER --> DASH
```

The process is a **long-lived worker**, not a serverless function. Outbound WebSockets must stay open. The dashboard is a read-only FastAPI view of the same process.

---

## Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| Runtime | **Python 3.12** + **asyncio** | One process owns sockets, timers, and paper fills. |
| Packaging | **[uv](https://docs.astral.sh/uv/)** | Fast, locked, reproducible installs. |
| Settings | **Pydantic Settings** | Typed `.env` contract, no mystery globals. |
| Market data | **websockets** + **httpx** | Split Binance USD-M public / market streams. |
| Decisions | **TypeSafe Jev** via OpenRouter | `POST /api/alpha/decisions` — not a chat completion. |
| HTTP / UI | **FastAPI** + **Uvicorn** | Operator snapshot at `/dashboard` and `/api/status`. |
| Persistence | **SQLAlchemy 2 async** + **asyncpg** | PostgreSQL in research runs; SQLite allowed in tests. |
| Quality | **pytest**, **ruff**, **mypy --strict** | Fast tests only. No overnight soaks in CI. |

Pinned Jev model: `typesafe/jev-1.13`. Never `jev-latest`.

---

## Market data

Binance USD-M is used **instead of spot** so native SHORT, funding, mark price, open interest, and liquidations exist.

After 2026-04-23 the USD-M sockets are split:

| Socket | Streams |
| --- | --- |
| `wss://fstream.binance.com/public/stream` | `depth`, `bookTicker` |
| `wss://fstream.binance.com/market/stream` | `aggTrade`, `markPrice`, `forceOrder` |

Open interest has no public WebSocket. It is polled with `GET /fapi/v1/openInterest`.

The book is a compact **top-20** ladder (`@depth20@100ms`) plus `@bookTicker` for best bid/ask. That is enough for imbalance 1 / 5 / 10 / 20 without a full L2 diff machine. If the public socket drops, the book is **cleared immediately** so a reconnect cannot trade on a stale ladder.

`MarketState` is tagged with `symbol` and `exchange`. The asset is never inferred from process-global state.

---

## Strategies

Strategies are **named and versioned**. They coexist in the codebase. One process runs one `TRADING_STRATEGY`.

| Strategy | Label | Question set | Strategy version | Feature schema |
| --- | --- | --- | --- | --- |
| `BASELINE` | BASELINE | early measurement questions | — | `ms.v2` |
| `SCALP` | SCALP | `q.scalp.v3` | `scalp.action_values.v2` | `ms.v2` |
| `DIRECTIONAL` | **DIRECTIONAL_V1** | `q.directional.v1` | `directional.v1` | `directional.state.v1` |

### SCALP

Jev estimates short-horizon action values around barriers. Python converts those scores into expected value **after** assumed VIP0 fees and slippage, then applies hysteresis (enter / hold / close / reverse). Hard risk still wins.

### DIRECTIONAL_V1

A research split: **market forecasting ≠ execution economics**.

Jev never sees fees, slippage, paper PnL, or required EV. It only answers:

- rolling mid direction at 60s / 300s / 600s / 1800s
- UTC-aligned candle close (1m / 5m / synthetic 10m / 30m)
- consecutive-candle persistence (2 / 3 / 5)
- move magnitude buckets vs Python-computed sigma
- compact regime / reversal diagnostics

Python then scores calibration (Brier, log loss, accuracy), compares simple baselines, and computes **raw market return vs net executable PAPER return**. A secondary paper policy may open LONG/SHORT from strong, confirmed directional belief. Forecasts are stored even when the policy stays FLAT.

---

## Safety model

```text
Jev candidate  →  strategy policy  →  RiskEngine  →  paper fill
                         ↑
              never overrides hard limits
```

LIVE requires **both**:

```bash
TRADING_MODE=LIVE
LIVE_TRADING_CONFIRMATION=I_UNDERSTAND_REAL_MONEY_IS_AT_RISK
```

Phase 4 still refuses LIVE: authenticated execution does not exist. Stale books skip Jev and cannot open. Kill switch, daily loss, drawdown, consecutive-loss, and order-rate caps are Python-owned.

Default paper book for research:

| | |
| --- | --- |
| Equity | 10,000 USDT |
| Notional | 1,000 USDT |
| Leverage | 1× |
| Assumed taker | 5 bps (VIP0 Regular) |
| Assumed slippage | 1 bp |
| Fills | never at mid — always bid/ask |

---

## Operator dashboard

```
http://127.0.0.1:8000/dashboard
```

The banner always states mode, strategy label, and LIVE status. DIRECTIONAL_V1 adds:

- direction **term structure** (P(UP) / P(DOWN) by horizon)
- current aligned candles
- persistence probabilities
- feature inspector (microstructure, price action, technical, derivatives, cross-market, regime)
- forecast quality vs trading quality

---

## Quick start

**Requirements:** Python 3.12, [uv](https://docs.astral.sh/uv/), PostgreSQL (local), an [OpenRouter](https://openrouter.ai) key with Jev access.

```bash
git clone https://github.com/AntoineBejarano/jev-trader.git
cd jev-trader
uv sync
cp .env.example .env
```

Edit `.env`:

```bash
TRADING_MODE=PAPER
TRADING_STRATEGY=DIRECTIONAL          # or SCALP
TRADING_EXCHANGE=binance_usdm
TRADING_SYMBOLS=BTCUSDT
OPENROUTER_API_KEY=...
DATABASE_URL=postgresql://jev:jev@127.0.0.1:5433/jev_trader_paper
EXPERIMENT_ID=directional-v1-manual-001
EXPERIMENT_FROZEN=true
```

Run the worker + dashboard:

```bash
uv run python -m src.main
# equivalent: make dev
```

Reports (do not start trading):

```bash
uv run python -m scripts.report_directional --current
uv run python -m scripts.report_run --current
```

### Quality gates

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src
```

Tests never place orders. They do not call OpenRouter or authenticated Binance APIs.

---

## Deploy on Railway

The process is a **long-lived worker** (WebSockets + FastAPI). Use a Railway **service**, not a serverless function. Keep **one replica**.

1. Create a Railway project and add a **PostgreSQL** plugin.
2. Deploy this repo with the included `Dockerfile` / `railway.toml`.
3. Set service variables (do not commit `.env`):

```bash
TRADING_MODE=PAPER
TRADING_STRATEGY=DIRECTIONAL
TRADING_EXCHANGE=binance_usdm
TRADING_SYMBOLS=BTCUSDT
OPENROUTER_API_KEY=...
DATABASE_URL=${{Postgres.DATABASE_PRIVATE_URL}}
HOST=0.0.0.0
ENVIRONMENT=production
EXPERIMENT_ID=directional-v1-railway-001
EXPERIMENT_FROZEN=true
```

Railway injects `PORT`. The image binds `0.0.0.0` and the health check is `GET /health`. Live trading stays blocked.

After deploy, open the public URL at `/dashboard`.

---

## Repository map

```text
src/
  market/         Binance USD-M adapters, top-20 book, mid buffer
  features/       returns, volatility, flow, intelligence, regime
  jev/            Decisions API client, question sets, parsers
  directional/    DIRECTIONAL_V1 features, policy, calibration
  decision/       SCALP EV, barriers, action values, hysteresis
  risk/           hard limits Jev cannot override
  execution/      paper fills, costs, attribution
  portfolio/      equity, positions, realized / unrealized PnL
  storage/        SQLAlchemy models + research repositories
  api/            FastAPI dashboard and operator snapshot
  trader/         asyncio engine that ties the loop together
scripts/          frozen-experiment reports
tests/            fast unit and persistence tests
```

---

## Roadmap

| Phase | Status |
| --- | --- |
| 1. Market data + `MarketState` | Done |
| 2. Jev Decisions API, no trading | Done |
| 3. Persistence of snapshots and decisions | Done |
| 4. Paper execution + PnL | **Current** |
| 5. Dedicated hysteresis / risk pass | Only if the frozen PAPER run needs it |
| 6. Dashboard | Operator view exists; iterate from research |
| 7. Replay | Not started |
| 8. Authenticated live execution | Blocked until explicit human approval |

---

## Disclaimer

Jev Trader is a **research instrument**. Paper PnL is not a promise of live PnL. Directional accuracy can be real and still be untradeable after costs. Do not run this against real funds. Do not treat Jev answers as orders.

---

<p align="center">
  <sub>Jev Trader · PAPER · LIVE BLOCKED</sub>
</p>
