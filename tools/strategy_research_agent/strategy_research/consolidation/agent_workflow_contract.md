# Strategy Agent Workflow Contract

This is the durable interpretation of the current strategy research Agent.

## Current Architecture

The Agent is not being built from zero. It already has:

- A materials layer with local transcripts, web snapshots, and user-provided local research documents.
- A knowledge layer with cleaned claims, knowledge cards, and a price-action knowledge graph.
- A self-iteration loop that can generate hypotheses, run Freqtrade research backtests, diagnose failures, plan improvements, and maintain candidate/watchlist/rejected queues.

The current improvement is to integrate those existing parts into one stronger research workflow:

1. Read the knowledge graph to source professional trading ideas.
2. Read research memory to understand repeated failures, avoid rules, and next blockers.
3. Read the consolidation layer to enforce hard research boundaries and required validation gates.
4. Convert knowledge-guided and memory-guided hypotheses into measurable event definitions.
5. Run or read an event study before generating concrete strategy classes.
6. Only events with forward-distribution evidence may pass into the existing self-iteration loop: isolated strategy generation, backtesting, post-run attribution, improvement planning, and promotion-gate review.

## Event Study Gate

Concrete strategy generation is not the first research step. The Agent must first prove that an entry event has statistical edge using event-study evidence:

- event sample count;
- forward returns at short horizons;
- win rate;
- MFE/MAE distribution;
- pair, side, timeframe, and regime notes;
- fee/slippage sensitivity before any candidate promotion.

If an event does not clear the edge gate, the Agent may only use it as a counterexample, redesign input, or negative-control experiment. It must not turn that event into another strategy class just because the knowledge card sounds plausible.

## Timeframe Contract

For 50x Binance USDT-M futures research, concrete strategy entry must use short-cycle K-line granularity:

- Allowed primary entry timeframes: `3m`, `5m`, `15m`.
- `1h` may only be used as a background, regime, or confirmation timeframe.
- `1h`, `4h`, and `1d` must not be used as the primary entry timeframe for new 50x strategy classes.

If an event study is based on `1h` candles, the Agent may only use it as regime context or as a negative/control study until the entry trigger is translated to `3m`, `5m`, or `15m`.

## Post-Run Attribution Gate

Every strategy research round that runs backtests must end with post-run attribution before it updates research memory, mature researcher queues, or the next experiment plan.

The attribution gate is part of the same Agent, not a separate Agent. It must reuse the same knowledge graph, research memory, event-study evidence, backtest outputs, exported trades, and promotion blockers. Splitting attribution into a separate Agent is not allowed unless the workflow still treats the result as the same mandatory gate.

The gate must classify the result into explicit failure or edge buckets:

- signal edge: whether the entry event had forward-distribution expectancy before leverage;
- entry timing: whether MAE/MFE shows entries were late, early, or structurally adverse;
- exit quality: whether ROI, stoploss, time exits, or invalidation rules cut winners or held losers;
- cost and funding drag: whether fee, slippage, spread, or funding stress erased gross edge;
- fixed 50x risk amplification: whether the configured futures risk口径 magnified a weak signal rather than revealing edge;
- regime dependency: whether results depend on BTC lead, volatility, trend/range, funding, session, or pair-specific structure;
- sample validity: whether trade count, window coverage, and robustness are enough to justify another experiment.

No next experiment queue item may be created from a backtest round unless the attribution gate states what failed, what survived, and which single mechanism the next run is testing.

## Durable Rule

Do not describe the Agent as missing materials, knowledge, or self-iteration. Those already exist. The accurate framing is:

> The Agent already has materials, knowledge graph, and self-iteration. The next work is deeper integration: using the knowledge graph and durable memory to guide the existing automatic research loop.

The active research rule is:

> Knowledge proposes events. Event studies test edge. Only edge candidates become strategies. Backtests must then be attributed before memory or next experiments change.

## Factor Research Gate

Factor research is a required front-door stage inside the same Agent, not a
separate Agent. Its job is to stop the workflow from turning knowledge cards or
research memory directly into strategy classes.

The fixed sequence is:

1. Knowledge graph and research memory propose research directions.
2. Factor research scores `3m`, `5m`, and `15m` Binance USDT-M futures OHLCV
   factors against forward return, MFE, MAE, sample count, side, and timeframe.
3. Factor-to-strategy planning converts only factor rows with sufficient sample,
   after-fee expectancy, win rate, and MFE/MAE evidence into event-study
   hypotheses.
4. Event study tests those hypotheses as measurable entry events.
5. Only event edge candidates may become concrete Freqtrade strategy classes,
   unless the run is explicitly labeled as a negative-control or redesign study.

The Agent must not say "external knowledge generated this strategy" unless the
factor/event evidence chain exists. External knowledge can inspire what to test;
the factor layer decides whether there is enough local market evidence to turn
it into a strategy hypothesis.

## Safety Boundary

This workflow is research-only. It must not start live trading, read exchange API keys, modify dry-run/live config, or promote a theory-derived strategy without evidence gates.

## Futures Runtime Safety Gate

For Binance USDT-M futures, a process heartbeat, UI `pong`, or Telegram startup
message is not enough to call dry-run or live review healthy. The Agent must
separate strategy evidence from runtime safety.

Every futures dry-run or live-review candidate must satisfy:

- The runtime config must allow ccxt to use the active VPN/proxy environment:
  `ccxt_config.requests_trust_env=true` and
  `ccxt_async_config.aiohttp_trust_env=true`.
- Startup must run a ccxt Binance futures preflight using the same Python
  environment as Freqtrade. It must fetch exchange time and at least two
  `BTC/USDT:USDT` 15m candles before the bot starts.
- Preflight failure blocks startup or promotion. A bot that is `RUNNING` but
  cannot fetch futures OHLCV is considered unsafe, not healthy.
- Startup must run the dry-run strategy risk preflight before the network
  preflight. It must verify that strategy hooks are callable through Freqtrade,
  config overrides are explicit, and the final effective values still satisfy
  the fixed futures risk contract.
- Exchange-side stoploss must be configured before live review:
  `order_types.stoploss=market`, `order_types.stoploss_on_exchange=true`, and
  futures `stoploss_price_type=mark`.
- The strategy risk preflight must block dry-run review if ROI, stoploss,
  50x leverage, 8h losing-trade custom exit, exchange-side stoploss, or the
  three-stoploss StoplossGuard cannot be loaded by Freqtrade.
- Live review requires a tiny-size operational test proving that a filled
  position receives an exchange-side stop order. Dry-run can validate config
  parsing, but it cannot prove the real exchange order exists.

This gate applies to all strategy families, not only A1. It exists because 50x
futures runtime failure can turn a network problem into an unmanaged position.
