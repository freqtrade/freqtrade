# Strategy Agent Knowledge Cards

This directory contains versioned, short-form knowledge cards for the local
strategy research Agent.

## What Is Versioned

- `knowledge_cards/`: active cards that can guide research hypotheses.
- `knowledge_cards_quarantined/`: cards retained as reviewed reference material but excluded from active hypothesis generation.

Cards are intentionally short. They contain concepts, source references, a
testable strategy hypothesis, Freqtrade translation hints, risk notes, and avoid
rules.

## What Is Not Versioned

Do not commit:

- raw Bilibili subtitles
- downloaded videos
- PDFs or book files
- full web snapshots
- browser cookies
- generated graph/report/dashboard/backtest artifacts

Those belong in local `user_data/strategy_research/knowledge/raw_sources/` and
other runtime directories.

## Runtime Build

After installing the agent runtime, rebuild the usable knowledge layer:

```bash
user_data/strategy_research/start_manual_research.sh --agent-brain
```

For the weekly external update loop:

```bash
user_data/strategy_research/start_manual_research.sh --weekly-knowledge-update
```
