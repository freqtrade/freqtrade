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

## Safety Boundary

This workflow is research-only. It must not start live trading, read exchange API keys, modify dry-run/live config, or promote a theory-derived strategy without evidence gates.
