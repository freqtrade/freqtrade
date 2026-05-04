# Bot Factory Strategy Generation Next Agent Prompt

Use the following prompt for the next coding agent.

````markdown
Continue Bot Factory from the current Strategy Generation / Candidate Factory
state. The project goal is an AI-assisted candidate factory: generate multiple
rule-based, FreqAI, and hybrid ML+rule strategy candidates; evaluate them with
local historical artifacts; rank/select/reject them; and feed failures plus
reviewer findings back into later candidate iterations.

Do not drift into only generating one fixed hand-written indicator template.
`Strategy Code Generator v1` exists only as a deterministic long-only
rule-based baseline and safety-path proof.

First command, required:

```powershell
git status --short --untracked-files=all
```

Read these files before making changes:

- `AGENTS.md`
- `docs/BOT_FACTORY_MVP_TODO.md`
- `docs/BOT_FACTORY_STRATEGY_GENERATION_NEXT_AGENT_PROMPT.md`
- `docs/BOT_FACTORY_PHASE3_NEXT_AGENT_PROMPT.md`
- `docs/BOT_FACTORY_PHASE2_RUNBOOK.md`
- `docs/BOT_FACTORY_PHASE2_AGENT_INSTRUCTIONS.md`
- `registry/strategies/proposals/TEMPLATE.md`
- latest accepted proposal metadata under `registry/strategies/proposals/`
- latest generated strategy metadata under `registry/strategies/generated/`

Current known state:

- Strategy Proposal Generator is implemented:
  - `freqtrade_ext/bot_factory/strategy_proposals.py`
  - `scripts/bot_factory_generate_strategy_proposal.py`
- Strategy Code Generator v1 baseline is implemented:
  - `freqtrade_ext/bot_factory/strategy_code.py`
  - `scripts/bot_factory_generate_strategy_code.py`
- v1 generated a safe long-only RSI pullback baseline from the accepted
  `LongOnlyRsiPullbackCandidate` proposal. This is not the full AI/ML strategy
  generation layer.
- Candidate Evaluation Pipeline, Candidate Ranking / Registry, Iteration /
  Improvement Loop, FreqAI/hybrid candidate generation, and paper deployment
  remain incomplete.

Current worktree context:

- The handoff may include uncommitted Bot Factory changes:
  - `docs/BOT_FACTORY_MVP_TODO.md`
  - `docs/BOT_FACTORY_PHASE3_NEXT_AGENT_PROMPT.md`
  - `docs/BOT_FACTORY_STRATEGY_GENERATION_NEXT_AGENT_PROMPT.md`
  - `freqtrade_ext/bot_factory/strategy_code.py`
  - `scripts/bot_factory_generate_strategy_code.py`
  - `tests/test_bot_factory.py`
  - generated smoke artifacts under `registry/strategies/generated/`
- Preserve existing user and prior-agent changes. Do not revert unrelated
  docs, generated artifacts, or test changes.
- Known Windows ACL warnings may appear in `git status`:
  - `.codex_tmp/pytest-of-yoro4/`
  - `bot_factory_pytest_tmp/`
  - `codex_tmp/pytest/`

Hard safety boundaries:

- Do not start `freqtrade trade`.
- Do not start paper trading, dry-run trading, canary live, live trading, or
  any bot startup process.
- Do not stop, poll, terminate, clean up, promote, or manage any paper process.
- Do not use API keys, secrets, private environment values, exchange order
  endpoints, real order placement, leverage above `1.0`, or shorting.
- Do not promote a generated candidate to paper from one proposal, one
  generated strategy, or one backtest.
- Keep local JSON, CSV, Markdown, and logs as the source of truth. MLflow is
  optional and must not replace local artifacts.

Handoff priority:

1. Keep `Strategy Code Generator v1` classified as a baseline only.
2. The next useful strategy-generation increment should either:
   - extend the code generator with explicit `rule_based`, `freqai`, and
     `hybrid_ml` generator modes, or
   - implement the Candidate Evaluation Pipeline so the current baseline can be
     evaluated and compared against future ML/hybrid candidates.
3. If extending the generator, do not just add another fixed rule template.
   Read proposal metadata and generate candidate-specific feature sets, target
   definitions, label horizons, entry/exit policy, rule filters, and risk
   policy where applicable.
4. If adding FreqAI/hybrid generation, generated code must include the expected
   FreqAI methods only when the proposal requests ML:
   `feature_engineering_expand_all`, `feature_engineering_expand_basic`,
   `feature_engineering_standard`, `set_freqai_targets`,
   `populate_indicators`, long-only `populate_entry_trend`, and long-only
   `populate_exit_trend`.
5. Future labels and negative shifts may only appear in `set_freqai_targets`.
   Negative shifts remain forbidden in indicator, entry, and exit generation.
6. Continue to run static strategy scanning before any generated candidate can
   enter evaluation.

Recommended next implementation path:

- Add `generator_mode` or equivalent generated metadata support:
  `rule_based`, `freqai`, `hybrid_ml`.
- Teach proposals/metadata to carry ML-relevant generation intent when needed:
  feature families, target definition, label horizon, prediction threshold,
  rule filters, and expected failure modes.
- Extend `freqtrade_ext/bot_factory/strategy_code.py` without relaxing safety:
  long-only, leverage `1.0`, no short signals, no order endpoints, no secrets,
  no process control.
- Add focused tests for:
  - v1 baseline remains safe and static-check clean
  - blocked proposals cannot generate code
  - FreqAI/hybrid mode emits required FreqAI methods and metadata
  - negative shifts are only allowed in `set_freqai_targets`
  - generated ML/hybrid metadata records features, label horizon, target, and
    safety scope

Alternative next implementation path:

- Implement `Candidate Evaluation Pipeline` before expanding generator modes.
  This should consume proposal metadata, generated strategy metadata, config,
  data paths, and candidate ID; write `candidate_manifest.json`; run static
  checks first; then run only historical evaluation commands that are already
  safe and documented.
- Do not implement ranking, iteration, or paper deployment until the evaluation
  pipeline exists and is verified.

Suggested verification after code changes:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  freqtrade_ext\bot_factory\strategy_code.py `
  scripts\bot_factory_generate_strategy_code.py `
  tests\test_bot_factory.py

.\.venv\Scripts\python.exe -m pytest tests\test_bot_factory.py

.\.venv\Scripts\python.exe scripts\bot_factory_static_check.py user_data\strategies
```

Documentation requirement:

- Update `docs/BOT_FACTORY_MVP_TODO.md` after every completed increment with
  exact commands, results, artifacts, and remaining limitations.
- Do not mark Candidate Evaluation Pipeline, Candidate Ranking / Registry,
  Iteration / Improvement Loop, or Paper trading deployment complete until
  implementation, tests, verification commands, and artifacts are recorded.
- Do not describe Strategy Code Generator v1 as the full AI/ML strategy
  generation solution. It is only the baseline generator.
````
