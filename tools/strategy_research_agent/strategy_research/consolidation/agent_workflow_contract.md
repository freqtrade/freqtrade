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
6. Only events with forward-distribution evidence may pass into the existing self-iteration loop: isolated strategy generation, backtesting, diagnosis, improvement planning, and promotion-gate review.

## Event Study Gate

Concrete strategy generation is not the first research step. The Agent must first prove that an entry event has statistical edge using event-study evidence:

- event sample count;
- forward returns at short horizons;
- win rate;
- MFE/MAE distribution;
- pair, side, timeframe, and regime notes;
- fee/slippage sensitivity before any candidate promotion.

If an event does not clear the edge gate, the Agent may only use it as a counterexample, redesign input, or negative-control experiment. It must not turn that event into another strategy class just because the knowledge card sounds plausible.

## Durable Rule

Do not describe the Agent as missing materials, knowledge, or self-iteration. Those already exist. The accurate framing is:

> The Agent already has materials, knowledge graph, and self-iteration. The next work is deeper integration: using the knowledge graph and durable memory to guide the existing automatic research loop.

The active research rule is:

> Knowledge proposes events. Event studies test edge. Only edge candidates become strategies.

## Safety Boundary

This workflow is research-only. It must not start live trading, read exchange API keys, modify dry-run/live config, or promote a theory-derived strategy without evidence gates.
