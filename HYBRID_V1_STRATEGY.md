# HYBRID_V1 — Strategy Thesis

## Purpose

HYBRID_V1 is the next research strategy for Jev Trader. Its objective is not to prove that an AI model can trade, nor to maximize the number of trades. Its objective is to test whether several independent sources of information can be combined into a repeatable **net positive expectancy** after realistic trading costs.

The strategy combines four layers with deliberately different jobs:

- **Kronos** estimates the likely future path and distribution of price.
- **VWAP** describes intraday location and directional context.
- **EMA9** describes short-term momentum and helps identify timing.
- **Jev** acts as the probabilistic decision layer that evaluates the combined market state and chooses LONG, SHORT, or NO_TRADE.
- **Python risk/execution** remains deterministic and has final authority over whether a trade is allowed and how much risk is taken.

The central hypothesis is:

> A trade should only be taken when predictive direction, market regime, short-term momentum, and probabilistic decision confidence agree strongly enough that the expected move can overcome fees, spread, slippage, and uncertainty.

This is a research hypothesis, not an assumption that the strategy is profitable.

---

## 1. The edge we are trying to capture

The strategy is based on **confluence**, not on a single indicator.

VWAP + EMA9 by themselves are not assumed to contain sufficient alpha. A crossover is not automatically a trade.

Kronos by itself is not assumed to be a trading strategy. A forecast can be directionally correct while still being too small, uncertain, or slow to monetize after costs.

Jev by itself is not assumed to know the future. It is used to judge a structured state and decide whether the evidence is sufficiently coherent to justify taking risk.

The desired setup is therefore:

**Forecast edge + directional context + momentum confirmation + high decision confidence + positive economics = candidate trade.**

If those conditions are not present, the desired action is **NO_TRADE**.

Abstention is a feature of the strategy. The system should prefer missing a mediocre opportunity over manufacturing a trade from weak evidence.

---

## 2. Roles of each component

### Kronos — predictive layer

Kronos answers questions about what may happen next.

We want information such as:

- expected direction over several horizons;
- expected return / magnitude of move;
- probability that the future return is positive or negative;
- forecast dispersion / uncertainty;
- expected volatility;
- whether different forecast horizons agree.

Kronos should not directly place trades.

A useful forecast is not simply "UP". It should tell us whether the predicted move is large and reliable enough to potentially matter economically.

### VWAP — market context

VWAP answers where price is trading relative to the intraday volume-weighted reference price.

The intended interpretation is broadly:

- price sustainably above VWAP supports a bullish context;
- price sustainably below VWAP supports a bearish context;
- repeated oscillation around VWAP suggests uncertainty/chop;
- a reclaim/rejection of VWAP may be useful confirmation.

VWAP is context, not an automatic BUY/SELL rule.

### EMA9 — short-term momentum and timing

EMA9 is used to understand whether immediate momentum supports the broader thesis.

Examples:

- bullish Kronos forecast + price above VWAP + rising EMA9 strengthens a LONG thesis;
- bearish Kronos forecast + price below VWAP + falling EMA9 strengthens a SHORT thesis;
- Kronos bullish while EMA9 momentum is strongly bearish is a reason to wait rather than force an entry.

EMA9/VWAP crossovers may be informative events, but **cross = trade is explicitly not the strategy**.

### Jev — decision gate

Jev receives the structured evidence and makes a narrow probabilistic decision.

The conceptual output is:

- LONG
- SHORT
- NO_TRADE

with confidence/probability information where possible.

Jev's job is to answer:

> Given the forecast, technical context, microstructure, volatility and current market state, is there enough coherent evidence to take directional risk now?

Jev should not be encouraged to trade frequently. NO_TRADE should be normal.

### Deterministic risk and execution

Jev never gets authority to override risk limits.

Python remains responsible for:

- position sizing;
- maximum risk per trade;
- daily loss and drawdown limits;
- spread/fee/slippage checks;
- stale-data protection;
- order-rate limits;
- stops and exit constraints;
- kill switch;
- paper fills and execution accounting.

A 99% confident model decision is still rejected if the economics or risk rules do not permit the trade.

---

## 3. Multi-horizon agreement

The strategy should not think about the market at only one horizon.

The research should compare short and medium intraday horizons, broadly aligned with the horizons already used by DIRECTIONAL_V1.

The strongest setup is expected to look conceptually like:

| Signal | Short | Medium | Longer |
| --- | --- | --- | --- |
| Kronos | bullish | bullish | bullish |
| Jev | bullish | bullish | bullish/neutral |
| VWAP context | bullish | — | — |
| EMA9 momentum | bullish | — | — |

This is a strong LONG candidate.

By contrast:

| Signal | Short | Medium | Longer |
| --- | --- | --- | --- |
| Kronos | bullish | bearish | bearish |
| Jev | bullish | neutral | bearish |
| VWAP context | bullish | — | — |
| EMA9 momentum | bullish | — | — |

This should normally produce NO_TRADE.

Agreement matters more than generating activity.

---

## 4. Conceptual LONG setup

A high-quality LONG candidate should generally have most of the following characteristics:

1. Kronos predicts positive returns over the relevant horizon.
2. The expected move is meaningfully larger than estimated round-trip trading costs.
3. Kronos uncertainty is acceptable and preferably several horizons agree.
4. Price is above VWAP, has reclaimed VWAP, or otherwise shows constructive VWAP behavior.
5. EMA9 momentum is positive and not fighting the forecast.
6. Existing microstructure information does not strongly contradict the trade.
7. Jev assigns sufficiently strong probability/confidence to LONG relative to SHORT and NO_TRADE.
8. Risk limits permit a position.
9. The trade still has positive expected economics after fees, spread and slippage.

If important evidence conflicts, wait.

---

## 5. Conceptual SHORT setup

The SHORT thesis is symmetric:

1. Kronos predicts negative returns over the relevant horizon.
2. The expected move is sufficiently large relative to costs.
3. Forecast uncertainty is acceptable and horizons preferably agree.
4. Price is below VWAP, has rejected VWAP, or shows bearish VWAP behavior.
5. EMA9 momentum is negative.
6. Microstructure does not strongly contradict the trade.
7. Jev strongly prefers SHORT.
8. Risk limits permit the position.
9. Expected economics remain positive after realistic costs.

Again, conflicting evidence should generally lead to NO_TRADE.

---

## 6. NO_TRADE is the default state

The system should not try to be permanently LONG or SHORT.

NO_TRADE is appropriate when:

- Kronos forecasts disagree across horizons;
- forecast magnitude is too small to overcome costs;
- Kronos uncertainty is high;
- price is chopping around VWAP;
- EMA9 and the broader thesis disagree materially;
- Jev confidence is weak;
- spread or volatility makes execution unattractive;
- market data is stale or incomplete;
- risk limits have been reached;
- the setup is ambiguous.

A major research objective is to determine whether **selectivity** improves expectancy.

The strategy may ultimately trade only a small fraction of evaluated opportunities. That is acceptable.

---

## 7. Entry philosophy

Do not optimize for entering at the exact turning point.

The strategy should prefer confirmation over prediction alone.

A forecast becoming bullish is not necessarily the entry. The entry should occur when the broader evidence becomes coherent enough that the probability-weighted reward appears attractive.

VWAP/EMA9 can therefore be used as timing/confirmation around a Kronos directional thesis rather than as standalone signals.

This is intentionally different from:

> EMA9 crosses VWAP → BUY.

The crossover may help confirm the setup, but it does not create the setup by itself.

---

## 8. Exit philosophy

The exit should be evaluated independently from the entry.

Potential reasons to reduce or close a position include:

- the original forecast edge disappears;
- Kronos reverses materially;
- Jev's directional belief collapses or reverses;
- price loses the VWAP/EMA structure supporting the trade;
- expected remaining upside/downside no longer compensates for risk;
- stop/risk limits are reached;
- the intended forecast horizon has expired;
- profit protection rules trigger.

The system should avoid repeatedly reversing LONG → SHORT → LONG due to noisy short-term changes. Hysteresis and confirmation should protect against overtrading.

---

## 9. Economics before PnL

A prediction is only useful if it is monetizable.

For every candidate trade, the research should distinguish:

**raw forecasting quality** from **net executable profitability**.

Conceptually:

Expected edge must exceed:

- fees;
- bid/ask spread;
- slippage;
- uncertainty buffer.

A model predicting +0.10% with +0.12% realistic round-trip friction has no useful trading edge even if the direction is correct.

The goal is therefore not maximum directional accuracy.

The goal is **positive expected value after costs**.

---

## 10. Experimental comparison

HYBRID_V1 must not be judged in isolation.

The same market periods should be used to compare at least:

1. Current Jev DIRECTIONAL_V1.
2. VWAP + EMA9 baseline.
3. Kronos-only forecasting/trading baseline.
4. Kronos + VWAP/EMA9.
5. Jev + Kronos.
6. **Jev + Kronos + VWAP/EMA9 (HYBRID_V1).**
7. A passive/reference benchmark where useful.

This matters because a profitable HYBRID_V1 does not prove every component adds value.

If Kronos + VWAP/EMA9 outperforms the full system, Jev may be unnecessary.

If Jev + Kronos outperforms the technical hybrid, VWAP/EMA9 may be unnecessary.

Every component must earn its complexity.

---

## 11. How we judge whether it works

Do not judge the strategy by one profitable day, a screenshot, win rate alone, or gross PnL.

At minimum evaluate:

- net PnL after all modeled costs;
- expectancy per trade;
- profit factor;
- Sharpe / risk-adjusted return;
- maximum drawdown;
- win rate together with average win/loss;
- number of trades;
- turnover and cost drag;
- performance by volatility/regime;
- performance by LONG vs SHORT;
- performance by confidence bucket;
- forecast calibration;
- stability across different time periods;
- out-of-sample performance.

The central question is:

> Does the edge survive unseen data, realistic costs, and different market regimes?

---

## 12. Research process

The intended progression is:

### Stage A — establish baselines

Measure the current DIRECTIONAL_V1 and simple alternatives.

### Stage B — add technical context

Add VWAP/EMA9 as measurable context and test whether they improve selection/timing.

### Stage C — add Kronos

Measure Kronos forecasting ability independently before allowing its forecasts to influence trades.

### Stage D — build HYBRID_V1

Expose the relevant Kronos + technical + existing market state to Jev and implement the selective LONG/SHORT/NO_TRADE policy.

### Stage E — replay/backtest

Run all strategies over identical historical periods without look-ahead.

### Stage F — walk-forward / out-of-sample

Tune only on historical/training windows. Evaluate on untouched future windows.

### Stage G — prospective PAPER

Freeze the strategy and let it run prospectively without changing rules because of short-term results.

Only after these stages should live trading even be considered.

---

## 13. Anti-overfitting rules

We should assume overfitting is a major threat.

Therefore:

- do not tune thresholds on the test period;
- do not repeatedly change the strategy because a few trades lost;
- keep a frozen out-of-sample period;
- prefer simple rules that survive multiple regimes;
- compare against simple baselines;
- include all costs;
- preserve losing trades in the dataset;
- do not cherry-pick dates;
- do not evaluate only bull markets;
- do not promote a strategy because of one spectacular run.

A strategy that needs constant parameter changes to remain profitable is not considered validated.

---

## 14. When we reject the hypothesis

The experiment should be considered a failure, or at least insufficient to continue toward live capital, if after a meaningful sample and out-of-sample testing one or more of the following are true:

### No net edge

Net expectancy is approximately zero or negative after realistic fees, spread and slippage.

### The hybrid adds no value

HYBRID_V1 does not materially outperform simpler alternatives on a risk-adjusted basis.

If a simpler system performs as well or better, use the simpler system.

### Edge disappears out of sample

Results look strong in development/backtest periods but collapse on untouched data or prospective PAPER.

This is a classic overfitting signal.

### Profit comes from a tiny number of lucky trades

If overall profitability depends on one or two extreme outcomes rather than a repeatable distribution of trades, the strategy is not validated.

### Excessive drawdown

Returns may be positive but economically unattractive once drawdown and tail risk are considered.

### Costs consume the edge

Gross predictions may be good while turnover, fees and slippage make net expectancy poor.

That means the forecast may contain information but the chosen execution horizon is not monetizable.

### Unstable across regimes

If the strategy works only in one narrow market regime and we cannot reliably identify that regime in advance, it should not be trusted.

### Jev/Kronos do not add measurable incremental value

Each expensive/complex model must improve either forecasting, selectivity, drawdown, or net expectancy. Otherwise remove it.

---

## 15. Suggested validation thresholds

These are research gates, not guarantees of future profitability.

Before considering the strategy promising, we would like to see something approximately in this territory across a meaningful sample:

- positive net expectancy after conservative costs;
- profit factor preferably above ~1.3, with >1.5 being much more interesting;
- Sharpe preferably above ~1.0 out of sample, with >1.5 being compelling if robust;
- drawdown that is acceptable relative to return;
- enough trades that results are not dominated by a tiny sample;
- profitability across more than one market regime;
- no catastrophic deterioration in prospective PAPER versus backtest;
- meaningful improvement versus simpler baselines.

These are not pass/fail laws. Statistical confidence, sample size, turnover and tail behavior matter.

---

## 16. Promotion to real capital

HYBRID_V1 should **not** be promoted directly from a good backtest to LIVE.

The intended gate is:

**backtest → untouched out-of-sample → frozen prospective PAPER → review → only then consider minimal live exposure.**

Any future live phase should begin with capital small enough that losing the experimental allocation is financially irrelevant.

Scaling should depend on observed live behavior, not confidence in the model.

---

## 17. What success looks like

The ideal result is not necessarily the highest-return strategy.

Success is a strategy where we can say:

1. Kronos contributes measurable predictive information.
2. VWAP/EMA9 improve timing or remove low-quality trades.
3. Jev improves decision quality/selectivity beyond simpler deterministic combinations.
4. Net expectancy remains positive after conservative execution costs.
5. The effect persists on unseen data.
6. Drawdown and turnover are acceptable.
7. The strategy remains understandable enough to diagnose when performance changes.

If only some components add value, HYBRID_V1 should be simplified accordingly.

---

## Final principle

**Prediction is not the product. Profitable selectivity is the product.**

Kronos predicts. VWAP provides context. EMA9 provides momentum/timing. Jev judges the combined evidence. Python controls economics and risk.

The system makes money only if the combined process identifies a subset of market situations where:

> **expected reward × probability is consistently greater than expected loss + trading friction + uncertainty.**

Until the data demonstrates that, the strategy remains a research hypothesis and PAPER stays the correct mode.
