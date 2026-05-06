# Bot Factory /goal Command Runbook

Use this runbook when starting a Codex CLI/TUI `/goal` for Bot Factory work.
The goal text must be specific enough for the agent to run to completion
without drifting into unsafe trading behavior or stopping after analysis only.

Do not start a one-letter or ambiguous goal such as `/goal r`. If a goal is
created with the wrong objective, stop it and restart with a full objective.

## Required `/goal` Shape

Use a complete objective sentence:

```text
/goal Complete the next Bot Factory Strategy Generation / Candidate Factory increment end-to-end: read the required docs, preserve existing changes, implement the smallest incomplete TODO-backed increment, verify it, update docs with exact commands and results, clean unrelated artifacts from the intended diff, and prepare a PR-ready summary without starting trade, paper, dry-run, canary, live, exchange-facing, leverage, shorting, or secret-using behavior.
```

If the desired output includes branch, commit, push, and PR creation, say that
explicitly:

```text
/goal Complete the next Bot Factory Strategy Generation / Candidate Factory increment end-to-end and open a PR against develop when verification passes. Start from git status, keep the diff scoped, run required Bot Factory checks, update docs/BOT_FACTORY_MVP_TODO.md with exact commands and results, commit only the intended files, push a codex/ branch, and create the PR. Do not start trade, paper, dry-run, canary, live, exchange-facing, leverage, shorting, or secret-using behavior.
```

For cleanup after an interrupted or mistaken goal:

```text
/goal Review the current Bot Factory working tree, identify which changes are necessary for the TODO-backed implementation and which are session artifacts, remove unrelated or incomplete generated artifacts from the intended diff, verify the remaining changes, update docs/BOT_FACTORY_MVP_TODO.md with exact commands and limitations, and prepare the branch or PR as requested. Preserve user changes and do not start any trade, paper, dry-run, canary, live, exchange-facing, leverage, shorting, or secret-using behavior.
```

## Required First Steps

The agent must begin with:

```powershell
git status --short --untracked-files=all
```

Then read, at minimum:

- `AGENTS.md`
- `docs/BOT_FACTORY_MVP_TODO.md`
- the phase or handoff document relevant to the selected TODO item
- the implementation files already related to the selected TODO item
- focused tests under `tests/`

`docs/BOT_FACTORY_MVP_TODO.md` remains the source of truth. The goal must pick
the smallest incomplete Bot Factory increment supported by that file unless the
user names a more specific target.

Do not rely only on the latest follow-up notes at the end of
`docs/BOT_FACTORY_MVP_TODO.md`. The agent must also scan the main MVP checklist
and the detailed Strategy Generation / Candidate Factory TODO sections:

```powershell
rg -n "^- \\[ \\]|^###|^## Strategy Generation|Candidate Evaluation|Candidate Ranking|Iteration|Paper trading deployment" docs\BOT_FACTORY_MVP_TODO.md
```

Use that scan to reconcile what is still incomplete before choosing work.
Recent follow-up notes may describe a partial implementation while the main MVP
checkbox remains intentionally open.

## MVP TODO Selection Rules

When choosing the next `/goal` increment:

- prefer the earliest unchecked MVP TODO that can be safely completed within
  Bot Factory scope
- confirm whether recent follow-up notes already implemented part of that item
- complete only the smallest remaining behavior that has code, tests,
  artifacts or schema updates, verification, and TODO evidence
- leave a checkbox open if the implementation is only a foundation, metadata
  contract, command preview, or artifact validator
- mark a checkbox complete only when the full behavior described by that
  checkbox is implemented and verified

For Strategy Generation / Candidate Factory work, do not mark these capabilities
complete unless their full MVP behavior is present:

- proposal-driven variation across candidate logic, features, thresholds, and
  labels
- candidate evaluation that can run or validate the required historical,
  walk-forward, and training artifact chain according to the active TODO
- ranking across multiple candidates with explicit normalized metrics and
  reasons
- iteration with lineage, reviewer findings, overfitting controls, retry
  limits, and safety guards
- Phase 3 paper connection only after passing local historical, walk-forward,
  training, report, and readiness artifacts

## Scope Rules

Keep changes scoped to:

- `freqtrade_ext/bot_factory/`
- `scripts/`
- `docs/`
- focused tests under `tests/`
- local JSON/CSV/Markdown artifacts only when they are complete evidence for
  the implemented path

Do not include these in the intended PR diff unless the user asked for them:

- slash-command session notes
- partial smoke-generated proposal or strategy artifacts
- local temp files, logs, caches, or permission-test directories
- unrelated generated files from interrupted runs

## Hard Safety Boundaries

The goal must not:

- start `freqtrade trade`
- start paper trading, dry-run trading, canary live, or live trading
- place or simulate exchange-facing orders
- call exchange order endpoints
- use API keys, secrets, private environment values, or credential-like config
  values
- add shorting or leverage above `1.0`
- start, stop, poll, terminate, clean up, promote, or manage any paper/live
  process unless the active phase documentation explicitly permits it and the
  user explicitly requests it

Historical `freqtrade backtesting` is allowed only through existing safe Bot
Factory wrappers and only after static and data-quality checks are satisfied.

## Execution Contract

The agent should keep working until one of these is true:

- the selected increment is implemented, verified, documented, and ready for
  review
- a required destructive action needs user approval
- the next required step would violate the safety boundaries
- the environment blocks verification or dependency installation, and all
  runnable checks plus the exact blocker have been recorded
- unrelated user changes make the requested increment impossible to complete
  safely without clarification

The agent should not stop after a plan if implementation is feasible. It should
make the change, run the narrowest relevant checks first, broaden as needed,
update documentation, and clean the intended diff.

## Verification Order

Before FreqAI-specific work:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_check_freqai_env.py
```

For Python code changes:

```powershell
.\.venv\Scripts\python.exe -m py_compile <changed python files>
```

For focused Bot Factory behavior:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py -q
```

Before any historical backtest or generated strategy evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

When known OHLCV parquet inputs are used, run the relevant OHLCV quality check
before historical evaluation.

If a command is blocked, the agent must record:

- exact command
- exit code when available
- root cause
- which checks still passed
- remaining limitation

Blocked verification is not completion unless all non-blocked implementation,
tests, artifact schema updates, and documentation updates are finished.

## Documentation Contract

After every completed increment, update `docs/BOT_FACTORY_MVP_TODO.md` with:

- exact date and timezone
- files or modules changed
- exact commands run
- command results
- artifact paths produced or intentionally excluded
- remaining limitations

Do not mark a capability complete until implementation, tests, verification,
and artifact/documentation evidence are present.

If the goal changes any behavior that corresponds to an unchecked MVP TODO,
update the relevant checklist item only when the whole item is complete. If the
work is partial, add a follow-up record with the remaining limitation and leave
the main checkbox unchecked.

## PR-Ready Completion Criteria

Before commit or PR creation, the agent must confirm:

- `git status --short --untracked-files=all` shows only intended files
- `git diff --check` passes
- generated artifacts in the diff are complete evidence, not interrupted smoke
  output
- no secrets, API keys, or private environment values are present
- TODO evidence records the checks actually run
- tests/checks are reported honestly, including warnings and environment
  blockers

When the user asks for PR creation, use a `codex/` branch unless they request a
different branch name, commit only the intended files, push to `origin`, and
open the PR against `develop` unless the repository context clearly says
otherwise.
