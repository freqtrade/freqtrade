---
name: strategy-promotion-gate
description: Use when deciding whether a crypto/Freqtrade strategy should remain rejected, move to watchlist, become a research candidate, enter dry-run review, or be blocked from live trading. Covers evidence gates, bias checks, regime/walk-forward/cost validation, dashboard candidate pools, and manual approval boundaries.
---

# Strategy Promotion Gate

## Purpose

Prevent unverified strategies from becoming dry-run or live strategies.

Use when the user asks:
- “这个策略能保留吗”
- “能进 dashboard 候选池吗”
- “能 dry-run/live 吗”
- “现在 Agent 能自动做策略了吗”
- “回测很好是不是可以上”

## States

| State | Meaning |
|---|---|
| rejected | Failed current hypothesis; keep as learning evidence |
| watchlist | Interesting but missing sample, cost, or robustness proof |
| research_candidate | Worth deeper validation; not dry-run permission |
| dryrun_candidate | Passed gates and ready for manual dry-run review |
| live_candidate | Requires explicit human approval after dry-run evidence |

## Required Gates

Before dry-run review:

- positive return after realistic cost assumptions
- PF above threshold appropriate to strategy frequency
- drawdown within account-risk limits
- enough trades for the strategy class
- no recursive-analysis bias
- no lookahead-analysis bias
- regime matrix not fragile
- walk-forward not dependent on one lucky window
- stress fee/slippage/funding evidence acceptable
- strategy source is local or safely translated, not directly executed external code

For scalping, add:
- stress-fee survival
- maker/taker assumption clarity
- MFE/MAE and exit-reason review

For futures, add:
- leverage justification
- margin/account loss math
- liquidation distance awareness
- circuit breakers

## Hard Blocks

Never promote when:

- missing lookahead/recursive checks
- only one favorable window works
- stress cost turns strategy negative
- trade count is too low
- live API keys are needed for research
- external code has not been isolated
- high leverage is the only reason returns look attractive

## Local Commands

Use when available:

```bash
user_data/strategy_research/start_manual_research.sh --promotion-gate
user_data/strategy_research/start_manual_research.sh --walk-forward
user_data/strategy_research/start_manual_research.sh --mature-researcher
user_data/strategy_research/start_manual_research.sh --preflight-only
```

## Output Contract

Say:

1. current state
2. missing gates
3. evidence that supports or blocks promotion
4. exact next command or experiment
5. whether manual approval is required

Do not blur “research candidate” into “can trade live”.
