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
4. Generate knowledge-guided and memory-guided hypotheses.
5. Pass any concrete strategy idea back into the existing self-iteration loop: isolated strategy generation, backtesting, diagnosis, improvement planning, and promotion-gate review.

## Durable Rule

Do not describe the Agent as missing materials, knowledge, or self-iteration. Those already exist. The accurate framing is:

> The Agent already has materials, knowledge graph, and self-iteration. The next work is deeper integration: using the knowledge graph and durable memory to guide the existing automatic research loop.

## Safety Boundary

This workflow is research-only. It must not start live trading, read exchange API keys, modify dry-run/live config, or promote a theory-derived strategy without evidence gates.
