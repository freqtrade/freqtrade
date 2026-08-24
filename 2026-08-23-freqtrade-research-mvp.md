# Freqtrade Research-Gate MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 MVP from `FREQTRADE_RESEARCH_ARCHITECTURE.md` §18 — a
`research/` package that runs a walk-forward promotion gate (Deflated Sharpe + BH-FDR +
PBO) over an existing freqtrade `IStrategy`, without modifying freqtrade core.

**Architecture:** `research/` is a client of freqtrade, not a fork of it. It imports
`freqtrade.optimize.backtesting.Backtesting` and `freqtrade.data.history` directly
(in-process, same venv) to run per-window backtests, keeps its own SQLite ledger of every
candidate tried (survivor and loser), and exposes one CLI command
(`python -m research.cli gate ...`) that reports PASS/FAIL with the statistical evidence.

**Tech Stack:** Python (repo floor: >=3.11, `pyproject.toml:16`), SQLAlchemy 2.0 (already a
dependency, `pyproject.toml:34` — used directly, no SQLModel), NumPy/SciPy/pandas (already
dependencies, `pyproject.toml:43-45`), pytest/pytest-mock (already dev dependencies).
**No new dependencies are added.**

**Spec:** `FREQTRADE_RESEARCH_ARCHITECTURE.md` (repo root), §§9-11, §15, §18.

## Global Constraints

- Python >=3.11 (`pyproject.toml:16`) — use modern type hints (`dict`, `list`, `X | None`).
- No new third-party dependencies — SQLAlchemy, NumPy, SciPy, pandas already ship with
  freqtrade (`pyproject.toml:34,43-45`); use them directly.
- `freqtrade/` package is never modified. All new code lives under `research/`.
- Research-layer tests live under `research/tests/`, run via `pytest research/tests -v`
  from the repo root (not mixed into freqtrade's own `tests/` tree).
- Every task follows TDD: write the failing test, watch it fail, write the minimal
  implementation, watch it pass.
- Every task gets a second opinion from an online LLM via the `lmchatbot` skill
  (`C:\dev\lmchatbot`, HTTP API at `localhost:3000`) on its core design decision **before**
  the final commit — per this project's standing `CLAUDE.md` rule to pair on non-trivial
  code decisions. If `curl -s localhost:3000/` doesn't respond, start it first:
  `node C:/dev/lmchatbot/server.js &` (background), then retry.
- Lint before every commit: `ruff check research/ --fix && ruff format research/`
  (repo-wide `[tool.ruff]` config in `pyproject.toml:237` covers `research/` by default —
  only `.env`, `.venv`, `*.md` are excluded).
- Every git commit message ends with the standard Freqtrade Co-Authored-By/session
  trailer this environment's tooling appends automatically — don't add your own.

---

## File Structure

```
research/
├── __init__.py
├── db.py            Task 1 — SQLAlchemy engine/session factory
├── models.py         Task 1 — CandidateResult ORM table
├── ledger.py          Task 1 — family_of / log_candidate_result / family_trial_count
├── statistics.py       Task 2 — deflated_sharpe_ratio, benjamini_hochberg, permutation_test
├── pbo.py                Task 3 — probability_of_backtest_overfitting (CSCV), choose_n_splits
├── walkforward.py          Task 4 — Window, WindowResult, WalkForwardRunner, generate_windows
├── gate.py                   Task 5 — GateResult, run_promotion_gate
├── cli.py                      Task 5 — `python -m research.cli gate ...`
└── tests/
    ├── __init__.py
    ├── test_ledger.py           Task 1
    ├── test_statistics.py         Task 2
    ├── test_pbo.py                   Task 3
    ├── test_walkforward.py             Task 4
    ├── test_gate.py                      Task 5
    └── test_cli.py                        Task 5
```

---

### Task 1: Candidate ledger (db.py, models.py, ledger.py)

**Files:**
- Create: `research/__init__.py` (empty)
- Create: `research/db.py`
- Create: `research/models.py`
- Create: `research/ledger.py`
- Test: `research/tests/__init__.py` (empty)
- Test: `research/tests/test_ledger.py`

**Interfaces:**
- Produces: `research.db.get_engine(db_path: str = "user_data/research.sqlite") -> Engine`
- Produces: `research.db.get_session(engine: Engine) -> Session`
- Produces: `research.models.Base` (DeclarativeBase), `research.models.CandidateResult` (ORM class, columns per architecture doc §10)
- Produces: `research.ledger.family_of(strategy_id: str) -> str`
- Produces: `research.ledger.log_candidate_result(session, *, strategy_id, params, universe, timeframe, discovery_start, discovery_end, n_trials_this_run, is_sharpe, oos_sharpe, deflated_sharpe, permutation_p, pbo, survived, validation_start=None, validation_end=None, oos_start=None, oos_end=None, evidence=None, run_stamp=None) -> CandidateResult`
- Produces: `research.ledger.family_trial_count(session, family: str, declared: int = 0) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/test_ledger.py
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from research.ledger import family_of, family_trial_count, log_candidate_result
from research.models import Base, CandidateResult


def _memory_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_log_candidate_result_round_trips():
    session = _memory_session()
    row = log_candidate_result(
        session,
        strategy_id="ema_cross_v3",
        params={"buy_rsi": 30},
        universe="BTC/USDT",
        timeframe="1h",
        discovery_start="2020-01-01",
        discovery_end="2024-12-31",
        n_trials_this_run=48,
        is_sharpe=1.2,
        oos_sharpe=0.8,
        deflated_sharpe=0.6,
        permutation_p=0.03,
        pbo=0.2,
        survived=True,
    )
    session.commit()

    fetched = session.query(CandidateResult).filter_by(id=row.id).one()
    assert fetched.strategy_id == "ema_cross_v3"
    assert fetched.strategy_family == "trend_following"
    assert fetched.params_json == '{"buy_rsi": 30}'
    assert fetched.survived is True


def test_family_of_maps_known_alias_and_falls_back_to_strategy_id():
    assert family_of("ema_cross_v3") == "trend_following"
    assert family_of("some_unmapped_strategy") == "some_unmapped_strategy"


def test_family_trial_count_prefers_ledger_count_when_higher_than_declared():
    session = _memory_session()
    for i in range(5):
        log_candidate_result(
            session,
            strategy_id="ema_cross_v3",
            params={"buy_rsi": i},
            universe="BTC/USDT",
            timeframe="1h",
            discovery_start="2020-01-01",
            discovery_end="2024-12-31",
            n_trials_this_run=1,
            is_sharpe=0.1,
            oos_sharpe=0.0,
            deflated_sharpe=0.0,
            permutation_p=1.0,
            pbo=1.0,
            survived=False,
        )
    session.commit()

    assert family_trial_count(session, "trend_following") == 5
    assert family_trial_count(session, "trend_following", declared=2) == 5
    assert family_trial_count(session, "trend_following", declared=10) == 10


def test_get_engine_creates_sqlite_file_and_tables(tmp_path):
    from research.db import get_engine, get_session

    db_path = tmp_path / "research.sqlite"
    engine = get_engine(str(db_path))
    session = get_session(engine)

    assert db_path.exists()
    assert session.query(CandidateResult).count() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest research/tests/test_ledger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.ledger'` (or `research.models`/`research.db`)

- [ ] **Step 3: Write minimal implementation**

```python
# research/models.py
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CandidateResult(Base):
    """One row per (research run, strategy, parameter-set) — survivors AND losers
    both get logged, so `family_trial_count` reflects the real search history."""

    __tablename__ = "candidate_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_stamp: Mapped[datetime] = mapped_column(DateTime)
    strategy_id: Mapped[str] = mapped_column(String(120))
    strategy_family: Mapped[str] = mapped_column(String(120), index=True)
    params_json: Mapped[str] = mapped_column(String)
    universe: Mapped[str] = mapped_column(String(200))
    timeframe: Mapped[str] = mapped_column(String(10))
    discovery_start: Mapped[str] = mapped_column(String(20))
    discovery_end: Mapped[str] = mapped_column(String(20))
    validation_start: Mapped[str | None] = mapped_column(String(20), nullable=True)
    validation_end: Mapped[str | None] = mapped_column(String(20), nullable=True)
    oos_start: Mapped[str | None] = mapped_column(String(20), nullable=True)
    oos_end: Mapped[str | None] = mapped_column(String(20), nullable=True)
    n_trials_this_run: Mapped[int] = mapped_column(Integer)
    is_sharpe: Mapped[float] = mapped_column(Float)
    oos_sharpe: Mapped[float] = mapped_column(Float)
    deflated_sharpe: Mapped[float] = mapped_column(Float)
    permutation_p: Mapped[float] = mapped_column(Float)
    pbo: Mapped[float] = mapped_column(Float)
    survived: Mapped[bool] = mapped_column(Boolean)
    evidence_json: Mapped[str] = mapped_column(String, default="{}")
```

```python
# research/db.py
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from research.models import Base

DEFAULT_DB_PATH = "user_data/research.sqlite"


def get_engine(db_path: str = DEFAULT_DB_PATH) -> Engine:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


def get_session(engine: Engine) -> Session:
    return Session(engine)
```

```python
# research/ledger.py
import json
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from research.models import CandidateResult

# Strategy-name -> family aliasing. Grow this table as strategy variants are added —
# related parameter sweeps must share a family so trial counts compound correctly.
_FAMILY_ALIASES: dict[str, str] = {
    "ema_cross_v3": "trend_following",
}


def family_of(strategy_id: str) -> str:
    return _FAMILY_ALIASES.get(strategy_id, strategy_id)


def log_candidate_result(
    session: Session,
    *,
    strategy_id: str,
    params: dict,
    universe: str,
    timeframe: str,
    discovery_start: str,
    discovery_end: str,
    n_trials_this_run: int,
    is_sharpe: float,
    oos_sharpe: float,
    deflated_sharpe: float,
    permutation_p: float,
    pbo: float,
    survived: bool,
    validation_start: str | None = None,
    validation_end: str | None = None,
    oos_start: str | None = None,
    oos_end: str | None = None,
    evidence: dict | None = None,
    run_stamp: datetime | None = None,
) -> CandidateResult:
    row = CandidateResult(
        run_stamp=run_stamp or datetime.now(UTC),
        strategy_id=strategy_id,
        strategy_family=family_of(strategy_id),
        params_json=json.dumps(params, sort_keys=True),
        universe=universe,
        timeframe=timeframe,
        discovery_start=discovery_start,
        discovery_end=discovery_end,
        validation_start=validation_start,
        validation_end=validation_end,
        oos_start=oos_start,
        oos_end=oos_end,
        n_trials_this_run=n_trials_this_run,
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        deflated_sharpe=deflated_sharpe,
        permutation_p=permutation_p,
        pbo=pbo,
        survived=survived,
        evidence_json=json.dumps(evidence or {}),
    )
    session.add(row)
    session.flush()
    return row


def family_trial_count(session: Session, family: str, declared: int = 0) -> int:
    """The number of trials to deflate against: whichever is larger between the
    ledger's accumulated row count for this family and a caller-declared count.
    Call this BEFORE writing the current run's own row (count-then-write) so a run
    never deflates against trials it hasn't finished yet."""
    ledger_count = (
        session.query(CandidateResult)
        .filter(CandidateResult.strategy_family == family)
        .count()
    )
    return max(ledger_count, declared)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest research/tests/test_ledger.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Get a second opinion via lmchatbot**

```bash
curl -s localhost:3000/ 2>/dev/null || (node C:/dev/lmchatbot/server.js &)
cat > /tmp/lmchatbot_task1.json <<'EOF'
{"provider":"gemini","prompt":"Reviewing a SQLAlchemy 2.0 declarative schema for a research trial ledger meant to prevent a multiple-testing/selection-bias problem when repeatedly backtesting trading strategies. Table CandidateResult logs one row per (run, strategy, param-set), survivors AND losers. family_trial_count(session, family, declared=0) returns max(ledger row count for that family, declared) and MUST be called before the current run writes its own row, so a run never deflates its own significance test against trials it hasn't finished yet (count-then-write). Does this design have an obvious flaw for its stated purpose? Schema:\n\nclass CandidateResult(Base):\n    __tablename__ = \"candidate_results\"\n    id, run_stamp, strategy_id, strategy_family, params_json, universe, timeframe,\n    discovery_start, discovery_end, validation_start, validation_end, oos_start, oos_end,\n    n_trials_this_run, is_sharpe, oos_sharpe, deflated_sharpe, permutation_p, pbo, survived, evidence_json\n\ndef family_trial_count(session, family, declared=0):\n    ledger_count = session.query(CandidateResult).filter(CandidateResult.strategy_family == family).count()\n    return max(ledger_count, declared)"}
EOF
curl -s localhost:3000/ask -d @/tmp/lmchatbot_task1.json
rm /tmp/lmchatbot_task1.json
```

Read the reply. If it surfaces a real gap (e.g. a refused/degenerate run still silently
inflating trial count, or a race between two concurrent processes reading-then-writing),
fix it in `ledger.py` before continuing; otherwise proceed.

- [ ] **Step 6: Lint and commit**

```bash
ruff check research/ --fix && ruff format research/
git add research/__init__.py research/db.py research/models.py research/ledger.py \
        research/tests/__init__.py research/tests/test_ledger.py
git commit -m "feat(research): add candidate ledger with family trial counting"
```

---

### Task 2: Statistics core (DSR, BH-FDR, permutation test)

**Files:**
- Create: `research/statistics.py`
- Test: `research/tests/test_statistics.py`

**Interfaces:**
- Produces: `research.statistics.deflated_sharpe_ratio(sharpe_ratio: float, n_obs: int, n_trials: int = 1, skewness: float = 0.0, kurtosis: float = 3.0, periods_per_year: int = 365) -> float`
- Produces: `research.statistics.benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]`
- Produces: `research.statistics.permutation_test(observed_stat: float, returns: np.ndarray, n_permutations: int = 1000, seed: int | None = None) -> float`

**Reference implementation to port from** (per `FREQTRADE_RESEARCH_ARCHITECTURE.md` §15 —
these are near-verbatim-portable, asset-class-agnostic pure functions):
`C:\dev\MarketMind\backend\backtest\enhanced\statistics.py`:
- `deflated_sharpe_ratio` — lines 168-228 (Bailey & López de Prado 2014)
- `benjamini_hochberg` — lines 14-38 (standard BH step-up procedure)
- `permutation_test` (sign-flip variant) — lines 41-104

Adapt, don't copy verbatim:
- Replace the source file's hand-rolled `_norm_cdf`/`_norm_ppf` (its lines ~110-140) with
  `scipy.stats.norm.cdf` / `scipy.stats.norm.ppf` — SciPy is already a freqtrade dependency
  (`pyproject.toml:43`), so there's no reason to hand-roll a normal-distribution
  approximation the way MarketMind did to avoid a new dependency it didn't have.
- Change the default `periods_per_year` from the source's 252 (equities trading days) to
  **365** (crypto trades every calendar day).
- This project's own hard-won lesson (architecture doc §16, lesson 1): a Deflated Sharpe
  with no real sample-size discrimination is just a sign test wearing a rigor costume — the
  tests below are written to catch exactly that regression, independent of the exact
  formula ported.

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/test_statistics.py
import numpy as np
import pytest

from research.statistics import benjamini_hochberg, deflated_sharpe_ratio, permutation_test


class TestDeflatedSharpeRatio:
    def test_returns_zero_on_degenerate_n_obs(self):
        assert deflated_sharpe_ratio(sharpe_ratio=2.0, n_obs=1, n_trials=10) == 0.0

    def test_bounded_in_unit_interval(self):
        for sharpe in (-3.0, -0.5, 0.0, 0.5, 1.5, 3.0):
            for n_obs in (10, 100, 1000):
                for n_trials in (1, 10, 1000):
                    result = deflated_sharpe_ratio(sharpe, n_obs, n_trials)
                    assert 0.0 <= result <= 1.0, (sharpe, n_obs, n_trials, result)

    def test_more_trials_never_increases_the_score(self):
        """Regression test for the exact bug this project's spec (§16, lesson 1) flags:
        a DSR with no real trial-count term would score n_trials=1 and n_trials=1000
        identically. Holding sharpe and n_obs fixed, searching harder must not look
        MORE convincing."""
        few_trials = deflated_sharpe_ratio(sharpe_ratio=2.0, n_obs=500, n_trials=1)
        many_trials = deflated_sharpe_ratio(sharpe_ratio=2.0, n_obs=500, n_trials=1000)
        assert many_trials <= few_trials

    def test_more_observations_never_decreases_the_score(self):
        """Regression test for the same lesson from the other axis: a 6-trade and a
        10,000-trade backtest must not deflate identically."""
        few_obs = deflated_sharpe_ratio(sharpe_ratio=1.0, n_obs=20, n_trials=50)
        many_obs = deflated_sharpe_ratio(sharpe_ratio=1.0, n_obs=5000, n_trials=50)
        assert many_obs >= few_obs


class TestBenjaminiHochberg:
    def test_matches_hand_worked_example(self):
        # sorted ascending already; thresholds are (rank/m)*q = 0.01,0.02,0.03,0.04,0.05
        p_values = [0.01, 0.02, 0.03, 0.04, 0.5]
        result = benjamini_hochberg(p_values, q=0.05)
        assert result == [True, True, True, True, False]

    def test_empty_input_returns_empty_output(self):
        assert benjamini_hochberg([], q=0.05) == []

    def test_all_p_values_above_q_rejects_nothing(self):
        assert benjamini_hochberg([0.9, 0.8, 0.99], q=0.05) == [False, False, False]


class TestPermutationTest:
    def test_p_value_in_unit_interval(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, 40)
        observed = float(returns.mean() / returns.std())
        p = permutation_test(observed, returns, n_permutations=200, seed=1)
        assert 0.0 <= p <= 1.0

    def test_low_p_value_for_a_strong_consistent_positive_edge(self):
        """All-positive, low-noise returns: virtually every sign-flip permutation
        produces a worse Sharpe than the unflipped (real) series, since flipping any
        positive return can only hurt the mean. p must be small."""
        rng = np.random.default_rng(42)
        returns = 0.02 + rng.normal(0, 0.001, 30)
        observed = float(returns.mean() / returns.std())
        p = permutation_test(observed, returns, n_permutations=2000, seed=42)
        assert p < 0.05

    def test_seeded_calls_are_reproducible(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.02, 50)
        observed = float(returns.mean() / returns.std())
        p1 = permutation_test(observed, returns, n_permutations=500, seed=123)
        p2 = permutation_test(observed, returns, n_permutations=500, seed=123)
        assert p1 == p2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest research/tests/test_statistics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.statistics'`

- [ ] **Step 3: Port the implementation**

Open `C:\dev\MarketMind\backend\backtest\enhanced\statistics.py` and port
`deflated_sharpe_ratio` (lines 168-228), `benjamini_hochberg` (lines 14-38), and the
sign-flip `permutation_test` (lines 41-104) into `research/statistics.py`, applying the two
adaptations called out above (scipy for the normal CDF/PPF, `periods_per_year=365`). Keep
the function names and parameter names exactly as declared in the Interfaces block so
`research/gate.py` (Task 5) can call them unchanged.

```python
# research/statistics.py
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# --- ported and adapted from C:\dev\MarketMind\backend\backtest\enhanced\statistics.py ---
# (paste + adapt deflated_sharpe_ratio, benjamini_hochberg, permutation_test here)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest research/tests/test_statistics.py -v`
Expected: PASS (9 tests). If `test_more_trials_never_increases_the_score` or
`test_more_observations_never_decreases_the_score` fails, the port has the same historical
bug MarketMind shipped for months (architecture doc §16, lesson 1) — fix the formula, don't
loosen the test.

- [ ] **Step 5: Get a second opinion via lmchatbot (cross-provider verify)**

```bash
curl -s localhost:3000/ 2>/dev/null || (node C:/dev/lmchatbot/server.js &)
cat > /tmp/lmchatbot_task2.json <<'EOF'
{"provider":"chatgpt","verify":"gemini","prompt":"I ported a Deflated Sharpe Ratio implementation (Bailey & Lopez de Prado 2014) from an internal reference into a fresh module, replacing its hand-rolled normal CDF/PPF with scipy.stats.norm, and changing periods_per_year from 252 to 365 for a 24/7 crypto market. Paste of research/statistics.py below. Two known historical bugs to check for specifically: (1) does it correctly discriminate a strategy tried across 1 trial vs 1000 trials at equal Sharpe/n_obs (the deflated score must NOT increase with more trials searched)? (2) does it correctly discriminate n_obs=20 vs n_obs=5000 at equal Sharpe (the deflated score must NOT decrease with more observations)? Also sanity check the Benjamini-Hochberg step-up implementation is the standard procedure. PASTE research/statistics.py CONTENTS HERE"}
EOF
curl -s localhost:3000/ask -d @/tmp/lmchatbot_task2.json
rm /tmp/lmchatbot_task2.json
```

Replace `PASTE research/statistics.py CONTENTS HERE` with the actual file contents before
sending. Read `draft`, `verifier`, and the `FINAL:`-prefixed corrected answer in the reply.
If either provider flags a real formula error, fix `statistics.py` and rerun Step 4 before
continuing.

- [ ] **Step 6: Lint and commit**

```bash
ruff check research/ --fix && ruff format research/
git add research/statistics.py research/tests/test_statistics.py
git commit -m "feat(research): port deflated Sharpe, BH-FDR, and permutation test from MarketMind"
```

---

### Task 3: Probability of Backtest Overfitting (CSCV)

**Files:**
- Create: `research/pbo.py`
- Test: `research/tests/test_pbo.py`

**Interfaces:**
- Consumes: none (pure numeric function, no dependency on Task 1/2 modules)
- Produces: `research.pbo.choose_n_splits(n_periods: int, max_splits: int = 16) -> int`
- Produces: `research.pbo.probability_of_backtest_overfitting(returns_matrix: np.ndarray, n_splits: int | None = None) -> dict` — returns `{"pbo": float, "n_splits": int, "n_combinations": int, "logits": list[float]}`. `returns_matrix` shape is `(n_variants, n_periods)`: one row per candidate parameter set, one column per time block (a walk-forward window, in Task 4's usage). Cell = that variant's return/Sharpe proxy in that block.

**Reference implementation to port from** (per `FREQTRADE_RESEARCH_ARCHITECTURE.md` §15):
`C:\dev\MarketMind\backend\backtest\enhanced\reality_check.py:271-345` —
`probability_of_backtest_overfitting` (line 271) and `choose_n_splits` (line 345). This is
the single highest-value MarketMind file to port per the architecture doc — it operates
purely on a returns matrix, no asset-class assumptions. **MVP scope note (YAGNI):** port
only these two functions. `reality_check.py` also contains White's Reality Check, Hansen's
SPA test, and a stationary block bootstrap (architecture doc §15 item 4) — those are real
and valuable but explicitly out of scope for this MVP; add them in a later phase if PBO
alone proves insufficient.

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/test_pbo.py
import numpy as np

from research.pbo import choose_n_splits, probability_of_backtest_overfitting


class TestChooseNSplits:
    def test_returns_an_even_divisor_of_n_periods(self):
        n_periods = 16
        s = choose_n_splits(n_periods)
        assert s % 2 == 0
        assert n_periods % s == 0

    def test_never_exceeds_max_splits(self):
        assert choose_n_splits(100, max_splits=8) <= 8


class TestProbabilityOfBacktestOverfitting:
    def test_bounded_in_unit_interval(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(0, 0.01, size=(6, 16))
        result = probability_of_backtest_overfitting(matrix)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_low_when_one_variant_dominates_every_period(self):
        """Variant 0 has a real, consistent edge (positive mean, low noise) in every
        period; the others are pure noise. Picking the best-in-sample variant should
        reliably also do well out-of-sample -> low overfitting probability."""
        rng = np.random.default_rng(11)
        n_variants, n_periods = 5, 16
        matrix = rng.normal(0.0, 0.01, size=(n_variants, n_periods))
        matrix[0] = rng.normal(0.05, 0.005, size=n_periods)
        result = probability_of_backtest_overfitting(matrix)
        assert result["pbo"] < 0.3

    def test_near_coin_flip_when_all_variants_are_pure_noise(self):
        """No variant has a real edge -> which one looks best in-sample is arbitrary
        and should NOT reliably predict out-of-sample rank. PBO should be high."""
        rng = np.random.default_rng(5)
        matrix = rng.normal(0.0, 0.01, size=(6, 16))
        result = probability_of_backtest_overfitting(matrix)
        assert result["pbo"] > 0.35

    def test_degenerate_input_fails_closed(self):
        result = probability_of_backtest_overfitting(np.zeros((1, 2)))
        assert result["pbo"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest research/tests/test_pbo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.pbo'`

- [ ] **Step 3: Write the implementation**

Port from `C:\dev\MarketMind\backend\backtest\enhanced\reality_check.py:271-345` (see Task
header), adapted to operate directly on a `(n_variants, n_periods)` NumPy array rather than
MarketMind's DB-backed candidate objects:

```python
# research/pbo.py
from __future__ import annotations

import itertools

import numpy as np


def choose_n_splits(n_periods: int, max_splits: int = 16) -> int:
    """Largest even number <= max_splits that evenly divides n_periods, so CSCV's
    IS/OOS blocks are equal-sized. Falls back to 2 if nothing else divides evenly."""
    for s in range(min(max_splits, n_periods), 1, -1):
        if s % 2 == 0 and n_periods % s == 0:
            return s
    return 2


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray, n_splits: int | None = None
) -> dict:
    """Combinatorially Symmetric Cross-Validation (Bailey, Borwein, Lopez de Prado,
    Zhu 2014). Splits the n_periods columns into n_splits contiguous blocks, and for
    every way of picking half the blocks as in-sample: finds the variant with the best
    in-sample Sharpe, then checks how that same variant ranks out-of-sample. PBO is the
    fraction of splits where the in-sample winner ranked in the OOS-worse half — logit
    of the OOS relative rank <= 0.
    """
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    if returns_matrix.ndim != 2:
        return {"pbo": 1.0, "n_splits": 0, "n_combinations": 0, "logits": []}

    n_variants, n_periods = returns_matrix.shape
    if n_variants < 2 or n_periods < 4:
        return {"pbo": 1.0, "n_splits": 0, "n_combinations": 0, "logits": []}

    s = n_splits or choose_n_splits(n_periods)
    if s < 2 or n_periods % s != 0:
        return {"pbo": 1.0, "n_splits": 0, "n_combinations": 0, "logits": []}

    blocks = np.array_split(returns_matrix, s, axis=1)
    logits: list[float] = []

    for is_block_idx in itertools.combinations(range(s), s // 2):
        oos_block_idx = [i for i in range(s) if i not in is_block_idx]
        is_returns = np.concatenate([blocks[i] for i in is_block_idx], axis=1)
        oos_returns = np.concatenate([blocks[i] for i in oos_block_idx], axis=1)

        is_sharpe = is_returns.mean(axis=1) / (is_returns.std(axis=1) + 1e-12)
        oos_sharpe = oos_returns.mean(axis=1) / (oos_returns.std(axis=1) + 1e-12)

        best_variant = int(np.argmax(is_sharpe))
        # relative rank of the IS-best variant's OOS performance, in (0, 1)
        rank = int((oos_sharpe < oos_sharpe[best_variant]).sum()) + 1
        omega = rank / (n_variants + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))

    pbo = float(np.mean([1.0 if lg <= 0 else 0.0 for lg in logits]))
    return {"pbo": pbo, "n_splits": s, "n_combinations": len(logits), "logits": logits}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest research/tests/test_pbo.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Get a second opinion via lmchatbot (cross-provider verify)**

```bash
curl -s localhost:3000/ 2>/dev/null || (node C:/dev/lmchatbot/server.js &)
cat > /tmp/lmchatbot_task3.json <<'EOF'
{"provider":"gemini","verify":"chatgpt","prompt":"I implemented Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation (Bailey/Borwein/Lopez de Prado/Zhu 2014) from scratch based on the paper's description, operating on a (n_variants, n_periods) returns matrix. For every way of splitting n_periods contiguous blocks into an in-sample half and out-of-sample half, I pick the variant with best in-sample Sharpe, compute its relative rank among all variants' out-of-sample Sharpes as omega=rank/(n_variants+1), take logit(omega), and count PBO as the fraction of splits where that logit <= 0. Does this match the paper's actual CSCV procedure, and is there a subtle bug in the rank/logit computation? PASTE research/pbo.py CONTENTS HERE"}
EOF
curl -s localhost:3000/ask -d @/tmp/lmchatbot_task3.json
rm /tmp/lmchatbot_task3.json
```

Replace the placeholder with the real file contents before sending. Fix and rerun Step 4 if
either provider identifies a real deviation from the CSCV procedure.

- [ ] **Step 6: Lint and commit**

```bash
ruff check research/ --fix && ruff format research/
git add research/pbo.py research/tests/test_pbo.py
git commit -m "feat(research): add PBO via CSCV"
```

---

### Task 4: Walk-forward runner wrapping freqtrade's Backtesting

**Files:**
- Create: `research/walkforward.py`
- Test: `research/tests/test_walkforward.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-3 (pure freqtrade integration)
- Produces: `research.walkforward.Window` (dataclass: `train_start, train_end, test_start, test_end: datetime`)
- Produces: `research.walkforward.WindowResult` (dataclass: `window: Window, variant_returns: dict[str, float], best_params: dict, train_sharpe: float, test_sharpe: float, test_n_trades: int, test_returns: list[float]`)
- Produces: `research.walkforward.variant_key(params: dict) -> str`
- Produces: `research.walkforward.generate_windows(start: datetime, end: datetime, train_days: int, test_days: int) -> list[Window]`
- Produces: `research.walkforward.WalkForwardRunner(config: dict, pairs: list[str], timeframe: str, datadir: Path)` with methods `.run_window(window: Window, param_grid: list[dict]) -> WindowResult` and `.run(windows: list[Window], param_grid: list[dict]) -> list[WindowResult]`

**Real freqtrade APIs this task calls** (verified against `develop` @ `10db8654c`):
- `freqtrade.optimize.backtesting.Backtesting(config)` then `.strategylist[0]` +
  `._set_strategy(strategy)` — instantiation pattern used throughout
  `tests/optimize/test_backtesting.py` (e.g. line 827).
- `freqtrade.data.history.load_data(datadir, timeframe, pairs, timerange=..., startup_candles=...)` — `freqtrade/data/history/history_utils.py:87`.
- `freqtrade.data.converter.trim_dataframes(preprocessed, timerange, startup_candles)` — `freqtrade/data/converter/converter.py:199`, used the same way internally at `freqtrade/optimize/backtesting.py:1840`.
- `freqtrade.data.metrics.calculate_sharpe(trades, min_date, max_date, starting_balance)` — `freqtrade/data/metrics.py:455`.
- `freqtrade.configuration.TimeRange("date", "date", start_epoch, end_epoch)` — `freqtrade/configuration/timerange.py:36`.
- `backtesting.backtest(processed=..., start_date=..., end_date=...) -> {"results": DataFrame, ...}` — restricts *simulation* to `[start_date, end_date]`; `processed` may (and here does) span further, matching freqtrade's own convention (§3 of the architecture doc). Indicators in the test strategy used here (`ta.RSI`) are backward-looking, so this is safe; a `ponytail:` comment documents the caveat for non-causal indicators.
- Strategy parameters are set via `getattr(strategy, name).value = value` (the `IntParameter`/`DecimalParameter` descriptor pattern every freqtrade strategy uses, e.g. `tests/strategy/strats/strategy_test_v3.py:77` — `buy_rsi = IntParameter([0, 50], default=30, space="buy")`, read as `self.buy_rsi.value`).

- [ ] **Step 1: Write the failing test**

```python
# research/tests/test_walkforward.py
from datetime import timedelta
from pathlib import Path

from freqtrade.data import history
from freqtrade.data.history import get_timerange
from freqtrade.enums import RunMode

from research.walkforward import Window, WalkForwardRunner, variant_key
from tests.conftest import get_default_conf, patch_exchange

TESTDATADIR = Path(__file__).resolve().parents[2] / "tests" / "testdata"
EXMS = "freqtrade.exchange.exchange.Exchange"


def _conf():
    conf = get_default_conf(TESTDATADIR)
    conf["runmode"] = RunMode.BACKTEST
    conf["max_open_trades"] = 10
    conf["use_exit_signal"] = False
    return conf


def test_run_window_selects_best_train_params_and_reports_oos_result(mocker):
    conf = _conf()
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])

    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    train_days = max(1, int((max_date - min_date).days * 0.7))
    train_end = min_date + timedelta(days=train_days)
    window = Window(
        train_start=min_date, train_end=train_end, test_start=train_end, test_end=max_date
    )

    runner = WalkForwardRunner(conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR)
    param_grid = [{"buy_rsi": 25}, {"buy_rsi": 35}]
    result = runner.run_window(window, param_grid)

    assert result.best_params in param_grid
    assert set(result.variant_returns) == {variant_key(p) for p in param_grid}
    assert isinstance(result.train_sharpe, float)
    assert isinstance(result.test_sharpe, float)
    assert isinstance(result.test_returns, list)
    assert result.test_n_trades == len(result.test_returns)


def test_generate_windows_are_contiguous_and_non_overlapping():
    from datetime import UTC, datetime

    from research.walkforward import generate_windows

    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2020, 3, 1, tzinfo=UTC)
    windows = generate_windows(start, end, train_days=20, test_days=10)

    assert len(windows) > 0
    for w in windows:
        assert w.train_end == w.test_start
        assert w.test_end <= end
    for a, b in zip(windows, windows[1:]):
        assert b.train_start == a.test_start
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest research/tests/test_walkforward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.walkforward'`

- [ ] **Step 3: Write the implementation**

```python
# research/walkforward.py
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.metrics import calculate_sharpe
from freqtrade.optimize.backtesting import Backtesting


@dataclass
class Window:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class WindowResult:
    window: Window
    variant_returns: dict[str, float]
    best_params: dict
    train_sharpe: float
    test_sharpe: float
    test_n_trades: int
    test_returns: list[float]


def variant_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def generate_windows(
    start: datetime, end: datetime, train_days: int, test_days: int
) -> list[Window]:
    """Rolling, non-overlapping windows: each window's test period starts exactly
    where its train period ends; the next window starts test_days later."""
    windows: list[Window] = []
    cursor = start
    while True:
        train_end = cursor + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        if test_end > end:
            break
        windows.append(Window(cursor, train_end, train_end, test_end))
        cursor = cursor + timedelta(days=test_days)
    return windows


class WalkForwardRunner:
    def __init__(self, config: dict, pairs: list[str], timeframe: str, datadir: Path):
        self.config = config
        self.pairs = pairs
        self.timeframe = timeframe
        self.datadir = datadir

    def run_window(self, window: Window, param_grid: list[dict]) -> WindowResult:
        backtesting = Backtesting(self.config)
        backtesting._set_strategy(backtesting.strategylist[0])

        timerange = TimeRange(
            "date", "date", int(window.train_start.timestamp()), int(window.test_end.timestamp())
        )
        data = history.load_data(
            datadir=self.datadir,
            timeframe=self.timeframe,
            pairs=self.pairs,
            timerange=timerange,
            startup_candles=backtesting.required_startup,
        )

        variant_returns: dict[str, float] = {}
        best_key, best_sharpe, best_params, best_processed = None, -np.inf, None, None

        for params in param_grid:
            for name, value in params.items():
                getattr(backtesting.strategy, name).value = value

            # ponytail: indicators are computed once over [train_start, test_end],
            # per freqtrade's own convention (§3 of the architecture doc). Safe for
            # backward-looking indicators (this strategy's RSI); a strategy with a
            # non-causal indicator (centered rolling, .shift(-n)) would leak across
            # the train/test boundary here — run freqtrade's lookahead-analysis on
            # any strategy before trusting this runner's results for it.
            processed = backtesting.strategy.advise_all_indicators(data)
            processed = trim_dataframes(processed, timerange, backtesting.required_startup)

            train_result = backtesting.backtest(
                processed=deepcopy(processed),
                start_date=window.train_start,
                end_date=window.train_end,
            )
            train_trades = train_result["results"]
            sharpe = calculate_sharpe(
                train_trades, window.train_start, window.train_end, self.config["dry_run_wallet"]
            )
            key = variant_key(params)
            variant_returns[key] = (
                float((train_trades["profit_abs"] / self.config["dry_run_wallet"]).mean())
                if len(train_trades)
                else 0.0
            )

            if sharpe > best_sharpe:
                best_sharpe, best_key, best_params, best_processed = sharpe, key, params, processed

        for name, value in best_params.items():
            getattr(backtesting.strategy, name).value = value
        test_result = backtesting.backtest(
            processed=deepcopy(best_processed),
            start_date=window.test_start,
            end_date=window.test_end,
        )
        test_trades = test_result["results"]
        test_returns = (test_trades["profit_abs"] / self.config["dry_run_wallet"]).tolist()
        test_sharpe = calculate_sharpe(
            test_trades, window.test_start, window.test_end, self.config["dry_run_wallet"]
        )

        return WindowResult(
            window=window,
            variant_returns=variant_returns,
            best_params=best_params,
            train_sharpe=best_sharpe,
            test_sharpe=test_sharpe,
            test_n_trades=len(test_trades),
            test_returns=test_returns,
        )

    def run(self, windows: list[Window], param_grid: list[dict]) -> list[WindowResult]:
        return [self.run_window(w, param_grid) for w in windows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest research/tests/test_walkforward.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Get a second opinion via lmchatbot**

```bash
curl -s localhost:3000/ 2>/dev/null || (node C:/dev/lmchatbot/server.js &)
cat > /tmp/lmchatbot_task4.json <<'EOF'
{"provider":"gemini","prompt":"I built a walk-forward runner on top of freqtrade's Backtesting class. Per window: I load OHLCV data spanning [train_start, test_end] in one call (with startup_candles for indicator warmup), compute indicators ONCE via strategy.advise_all_indicators() over that whole span, then call backtesting.backtest(processed=..., start_date=X, end_date=Y) twice against the SAME processed dataframe -- once restricted to [train_start, train_end] to pick the best parameter set by train Sharpe, then again restricted to [test_start, test_end] using the winning params, for the out-of-sample result. My reasoning that this doesn't leak test-period data into the train-period parameter selection: freqtrade always computes indicators over the full loaded range and only the start_date/end_date args restrict which candles get SIMULATED (trades placed), and for backward-looking indicators (e.g. TA-Lib RSI) more trailing rows after a given row don't change that row's own indicator value. Is this reasoning actually sound, and is there a walk-forward-integrity gap I'm missing even for backward-looking-only indicators?"}
EOF
curl -s localhost:3000/ask -d @/tmp/lmchatbot_task4.json
rm /tmp/lmchatbot_task4.json
```

If the reply surfaces a real gap (e.g., an interaction with `startup_candle_count` sizing,
or stoploss/ROI evaluation using look-back state that spans the train/test boundary), note
it as a `ponytail:` comment in `walkforward.py` at minimum, and fix it if it's cheap.

- [ ] **Step 6: Lint and commit**

```bash
ruff check research/ --fix && ruff format research/
git add research/walkforward.py research/tests/test_walkforward.py
git commit -m "feat(research): add walk-forward runner wrapping freqtrade Backtesting"
```

---

### Task 5: Promotion gate + CLI

**Files:**
- Create: `research/gate.py`
- Create: `research/cli.py`
- Test: `research/tests/test_gate.py`
- Test: `research/tests/test_cli.py`

**Interfaces:**
- Consumes: `research.ledger.family_of`, `.family_trial_count`, `.log_candidate_result` (Task 1); `research.statistics.deflated_sharpe_ratio`, `.benjamini_hochberg`, `.permutation_test` (Task 2); `research.pbo.probability_of_backtest_overfitting` (Task 3); `research.walkforward.WalkForwardRunner`, `.generate_windows`, `.Window`, `.WindowResult` (Task 4)
- Produces: `research.gate.GateResult` (dataclass: `strategy_id: str, passed: bool, deflated_sharpe: float, permutation_p: float, pbo: float, mean_test_sharpe: float, n_trials: int, reasons: list[str]`)
- Produces: `research.gate.run_promotion_gate(config, strategy_id, pairs, timeframe, datadir, start, end, train_days, test_days, param_grid, db_path="user_data/research.sqlite", dsr_threshold=0.95, fdr_q=0.05, pbo_threshold=0.5, periods_per_year=365) -> GateResult`
- Produces: `research.cli.main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/test_gate.py
from datetime import timedelta
from pathlib import Path

import pytest
from freqtrade.data import history
from freqtrade.data.history import get_timerange
from freqtrade.enums import RunMode

from research.db import get_session, get_engine
from research.gate import run_promotion_gate
from research.models import CandidateResult
from tests.conftest import get_default_conf, patch_exchange

TESTDATADIR = Path(__file__).resolve().parents[2] / "tests" / "testdata"
EXMS = "freqtrade.exchange.exchange.Exchange"


def _conf():
    conf = get_default_conf(TESTDATADIR)
    conf["runmode"] = RunMode.BACKTEST
    conf["max_open_trades"] = 10
    conf["use_exit_signal"] = False
    return conf


def _patch(mocker):
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])


def test_run_promotion_gate_raises_with_too_few_windows(mocker, tmp_path):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)

    with pytest.raises(ValueError, match="walk-forward windows"):
        run_promotion_gate(
            config=conf,
            strategy_id="StrategyTestV3",
            pairs=["UNITTEST/BTC"],
            timeframe="5m",
            datadir=TESTDATADIR,
            start=min_date,
            end=max_date,
            train_days=3650,  # deliberately far larger than the available data span
            test_days=3650,
            param_grid=[{"buy_rsi": 30}],
            db_path=str(tmp_path / "research.sqlite"),
        )


def test_run_promotion_gate_returns_result_and_writes_ledger_row(mocker, tmp_path):
    conf = _conf()
    _patch(mocker)
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)
    db_path = str(tmp_path / "research.sqlite")

    result = run_promotion_gate(
        config=conf,
        strategy_id="StrategyTestV3",
        pairs=["UNITTEST/BTC"],
        timeframe="5m",
        datadir=TESTDATADIR,
        start=min_date,
        end=max_date,
        train_days=train_days,
        test_days=test_days,
        param_grid=[{"buy_rsi": 25}, {"buy_rsi": 35}],
        db_path=db_path,
    )

    assert result.strategy_id == "StrategyTestV3"
    assert isinstance(result.passed, bool)
    assert 0.0 <= result.deflated_sharpe <= 1.0
    assert 0.0 <= result.permutation_p <= 1.0
    assert 0.0 <= result.pbo <= 1.0
    assert result.n_trials >= 1

    session = get_session(get_engine(db_path))
    assert session.query(CandidateResult).filter_by(strategy_id="StrategyTestV3").count() == 1
```

```python
# research/tests/test_cli.py
from research.cli import main
from research.gate import GateResult


def test_gate_command_prints_verdict_and_returns_pass_exit_code(mocker, capsys):
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=True,
        deflated_sharpe=0.97,
        permutation_p=0.01,
        pbo=0.1,
        mean_test_sharpe=1.2,
        n_trials=12,
        reasons=[],
    )
    mocker.patch("research.cli.run_promotion_gate", return_value=canned)
    mocker.patch("research.cli.Configuration.from_files", return_value={"datadir": "user_data/data"})

    exit_code = main(
        [
            "gate",
            "--strategy", "StrategyTestV3",
            "--config", "config.json",
            "--pairs", "BTC/USDT,ETH/USDT",
            "--timeframe", "1h",
            "--start", "2024-01-01",
            "--end", "2024-06-01",
            "--train-days", "60",
            "--test-days", "20",
            "--param-grid", "[{\"buy_rsi\": 30}]",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out


def test_gate_command_returns_nonzero_exit_code_on_fail(mocker, capsys):
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=False,
        deflated_sharpe=0.4,
        permutation_p=0.3,
        pbo=0.7,
        mean_test_sharpe=0.1,
        n_trials=12,
        reasons=["deflated_sharpe 0.400 below threshold 0.95"],
    )
    mocker.patch("research.cli.run_promotion_gate", return_value=canned)
    mocker.patch("research.cli.Configuration.from_files", return_value={"datadir": "user_data/data"})

    exit_code = main(
        [
            "gate",
            "--strategy", "StrategyTestV3",
            "--config", "config.json",
            "--pairs", "BTC/USDT",
            "--timeframe", "1h",
            "--start", "2024-01-01",
            "--end", "2024-06-01",
            "--train-days", "60",
            "--test-days", "20",
            "--param-grid", "[{\"buy_rsi\": 30}]",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest research/tests/test_gate.py research/tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.gate'` (and `research.cli`)

- [ ] **Step 3: Write the implementation**

```python
# research/gate.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from research.db import get_engine, get_session
from research.ledger import family_of, family_trial_count, log_candidate_result
from research.pbo import probability_of_backtest_overfitting
from research.statistics import benjamini_hochberg, deflated_sharpe_ratio, permutation_test
from research.walkforward import WalkForwardRunner, generate_windows


@dataclass
class GateResult:
    strategy_id: str
    passed: bool
    deflated_sharpe: float
    permutation_p: float
    pbo: float
    mean_test_sharpe: float
    n_trials: int
    reasons: list[str]


def run_promotion_gate(
    config: dict,
    strategy_id: str,
    pairs: list[str],
    timeframe: str,
    datadir: Path,
    start: datetime,
    end: datetime,
    train_days: int,
    test_days: int,
    param_grid: list[dict],
    db_path: str = "user_data/research.sqlite",
    dsr_threshold: float = 0.95,
    fdr_q: float = 0.05,
    pbo_threshold: float = 0.5,
    periods_per_year: int = 365,
) -> GateResult:
    windows = generate_windows(start, end, train_days, test_days)
    if len(windows) < 4:
        raise ValueError(
            f"Need at least 4 walk-forward windows for a meaningful gate, got "
            f"{len(windows)}. Widen start/end or shrink train_days/test_days."
        )

    runner = WalkForwardRunner(config, pairs, timeframe, datadir)
    results = runner.run(windows, param_grid)

    all_test_returns = [r for wr in results for r in wr.test_returns]
    n_obs = len(all_test_returns)
    mean_test_sharpe = float(np.mean([wr.test_sharpe for wr in results]))

    variant_keys = sorted({k for wr in results for k in wr.variant_returns})
    variant_matrix = np.array(
        [[wr.variant_returns[key] for wr in results] for key in variant_keys]
    )
    pbo_result = probability_of_backtest_overfitting(variant_matrix)

    engine = get_engine(db_path)
    session = get_session(engine)
    family = family_of(strategy_id)
    this_run_trials = len(param_grid) * len(windows)
    # count-then-write: read ledger history BEFORE writing this run's own row, so a
    # run never deflates its own significance test against trials it hasn't finished.
    n_trials = family_trial_count(session, family, declared=this_run_trials)

    deflated = deflated_sharpe_ratio(
        mean_test_sharpe, n_obs=n_obs, n_trials=n_trials, periods_per_year=periods_per_year
    )
    perm_p = permutation_test(mean_test_sharpe, np.array(all_test_returns))
    survived_bh = benjamini_hochberg([perm_p], q=fdr_q)[0]

    reasons: list[str] = []
    if deflated < dsr_threshold:
        reasons.append(f"deflated_sharpe {deflated:.3f} below threshold {dsr_threshold}")
    if not survived_bh:
        reasons.append(f"permutation p-value {perm_p:.3f} fails BH-FDR at q={fdr_q}")
    if pbo_result["pbo"] > pbo_threshold:
        reasons.append(f"PBO {pbo_result['pbo']:.3f} above threshold {pbo_threshold}")
    passed = not reasons

    log_candidate_result(
        session,
        strategy_id=strategy_id,
        params={"grid": param_grid},
        universe=",".join(pairs),
        timeframe=timeframe,
        discovery_start=start.isoformat(),
        discovery_end=end.isoformat(),
        n_trials_this_run=this_run_trials,
        is_sharpe=float(np.mean([wr.train_sharpe for wr in results])),
        oos_sharpe=mean_test_sharpe,
        deflated_sharpe=deflated,
        permutation_p=perm_p,
        pbo=pbo_result["pbo"],
        survived=passed,
    )
    session.commit()

    return GateResult(
        strategy_id=strategy_id,
        passed=passed,
        deflated_sharpe=deflated,
        permutation_p=perm_p,
        pbo=pbo_result["pbo"],
        mean_test_sharpe=mean_test_sharpe,
        n_trials=n_trials,
        reasons=reasons,
    )
```

```python
# research/cli.py
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from freqtrade.configuration import Configuration

from research.gate import run_promotion_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research", description="Freqtrade research gate")
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser("gate", help="Run the promotion gate for a strategy")
    gate.add_argument("--strategy", required=True)
    gate.add_argument("--config", required=True, help="Path to a freqtrade config.json")
    gate.add_argument("--pairs", required=True, help="Comma-separated pairs, e.g. BTC/USDT,ETH/USDT")
    gate.add_argument("--timeframe", required=True)
    gate.add_argument("--start", required=True, help="YYYY-MM-DD")
    gate.add_argument("--end", required=True, help="YYYY-MM-DD")
    gate.add_argument("--train-days", type=int, required=True)
    gate.add_argument("--test-days", type=int, required=True)
    gate.add_argument("--param-grid", required=True, help="JSON list of param dicts")
    gate.add_argument("--db-path", default="user_data/research.sqlite")

    args = parser.parse_args(argv)

    if args.command == "gate":
        ft_config = Configuration.from_files([args.config])
        ft_config["strategy"] = args.strategy
        result = run_promotion_gate(
            config=ft_config,
            strategy_id=args.strategy,
            pairs=args.pairs.split(","),
            timeframe=args.timeframe,
            datadir=Path(ft_config["datadir"]),
            start=datetime.fromisoformat(args.start),
            end=datetime.fromisoformat(args.end),
            train_days=args.train_days,
            test_days=args.test_days,
            param_grid=json.loads(args.param_grid),
            db_path=args.db_path,
        )
        verdict = "PASS" if result.passed else "FAIL"
        print(f"{result.strategy_id}: {verdict}")
        print(f"  deflated_sharpe   {result.deflated_sharpe:.3f}")
        print(f"  permutation p     {result.permutation_p:.3f}")
        print(f"  PBO               {result.pbo:.3f}")
        print(f"  mean OOS sharpe   {result.mean_test_sharpe:.3f}")
        print(f"  trials (ledger)   {result.n_trials}")
        for reason in result.reasons:
            print(f"  - {reason}")
        return 0 if result.passed else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest research/tests/test_gate.py research/tests/test_cli.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Get a second opinion via lmchatbot**

```bash
curl -s localhost:3000/ 2>/dev/null || (node C:/dev/lmchatbot/server.js &)
cat > /tmp/lmchatbot_task5.json <<'EOF'
{"provider":"gemini","prompt":"Reviewing the orchestration of a crypto trading-strategy promotion gate. It runs a walk-forward backtest across windows, then: reads a SQLite ledger's trial history for this strategy's family BEFORE writing this run's own row (count-then-write, to avoid a run deflating its own significance test against trials it hasn't finished), computes Deflated Sharpe Ratio using that trial count, a sign-flip permutation test p-value on the concatenated out-of-sample per-trade returns, and Probability of Backtest Overfitting across the parameter grid's variants. It fails the strategy if deflated_sharpe < 0.95 OR the permutation p-value fails Benjamini-Hochberg at q=0.05 OR PBO > 0.5, and always logs the attempt (pass or fail) to the ledger. Do these three default thresholds (0.95 DSR, 0.05 FDR-q, 0.5 PBO) seem reasonable as a first-pass gate for a crypto strategy with limited historical data, or overly strict/lenient? Any obvious ordering bug in when the ledger read happens vs when the row gets written?"}
EOF
curl -s localhost:3000/ask -d @/tmp/lmchatbot_task5.json
rm /tmp/lmchatbot_task5.json
```

If the reply makes a strong case for different default thresholds, adjust the keyword
defaults in `run_promotion_gate`'s signature and note the reasoning in a comment; otherwise
keep the current defaults.

- [ ] **Step 6: Lint, run the full research suite, and commit**

```bash
ruff check research/ --fix && ruff format research/
pytest research/tests -v
git add research/gate.py research/cli.py research/tests/test_gate.py research/tests/test_cli.py
git commit -m "feat(research): add promotion gate orchestration and CLI"
```

---

## Self-Review Notes (from writing this plan)

- **Spec coverage:** all of `FREQTRADE_RESEARCH_ARCHITECTURE.md` §18's six MVP bullets map
  to a task: package scaffold+ledger → Task 1; statistics core → Task 2; PBO/CSCV → Task 3;
  walk-forward runner → Task 4; CLI + PASS/FAIL report → Task 5 (gate.py + cli.py). Deferred
  items (regime analysis, cost-sensitivity sweep, live/paper edge monitoring, portfolio
  construction, AI hypothesis generation) are intentionally out of scope per §18's own
  ordering — not gaps in this plan.
- **No placeholders:** every step has real, complete code; no `TODO`/`TBD`/"similar to
  above". The one intentionally-incomplete block (Task 2, Step 3) is scoped to porting an
  external file whose exact source I verified line-ranges for but didn't reproduce
  wholesale here — the task explicitly names the file, lines, and the two required
  adaptations, and Step 1's tests fully pin the required behavior regardless of the exact
  formula ported.
- **Type/signature consistency:** `deflated_sharpe_ratio`, `benjamini_hochberg`,
  `permutation_test`, `probability_of_backtest_overfitting`, `WalkForwardRunner`,
  `generate_windows`, `family_of`, `family_trial_count`, and `log_candidate_result` are
  each defined once (Tasks 1-4) and called with matching names/kwargs in Task 5's
  `gate.py` — verified by re-reading Task 5 against each producing task's Interfaces block.
