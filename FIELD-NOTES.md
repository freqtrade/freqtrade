# FIELD-NOTES

Per-project traps for this repo, each with a `file:line` an agent actually looked at.
See `C:\dev\agent-knowledge\README.md` for the format this file follows and how the
`field-notes` pre-commit hook (`C:/dev/agent-knowledge/validate.py --hook`) keeps its
citations honest — it re-reads every cited line on commit and fails if the code moved.

---

## freqtrade's `Trade.session` is GLOBAL class-level state, not scoped per call

`freqtrade.persistence.init_db(db_url)` sets `Trade.session` directly on the `Trade`
class as a module-level side effect, not a factory returning an independent session
object. The assignment itself lives at `freqtrade/persistence/models.py:86`
(`Trade.session = scoped_session(`), inside `init_db`. Calling `init_db` anywhere
redirects `Trade.session` for the entire process, not just the caller.

This has bitten this repo twice already. PR #5 fixed a leak of a *different* pair of
class-level globals on the same classes (`Trade.use_db` and `LocalTrade.bt_trades`)
across `pytest --random-order -n auto` runs. Then `research/promotion.py`'s
`evaluate_paper_trading_health` became the first code in this package to call `init_db`
itself, deliberately, against a real dry-run database — the exact same footgun in a new
place. The call site is `research/promotion.py:196`
(`init_db(f"sqlite:///{db_path}")`). The very next statement, the query itself, is
`research/promotion.py:199` (`Trade.session.query`).

The trap: two calls to a function like this in the same process — sequential or
interleaved — silently redirect each other's `Trade.session`. A test file that exercises
such a function must reset `Trade.session` after every test in the file, not just the
new ones, or a throwaway dry-run sqlite file from one test leaks into whichever test
runs next in the same pytest-xdist worker, the exact same failure shape as PR #5's bug.

The guard that worked here: a file-scoped autouse pytest fixture at
`research/tests/test_promotion.py:173`
(`_reset_trade_session_after_evaluation_tests`), verified by two independent code
reviews (a task reviewer and an lmchatbot cross-check) to actually be function-scoped
and apply to the whole file, not nested in a class or conditionally applied. Its
teardown calls `init_db` with an in-memory URL at
`research/tests/test_promotion.py:180` (`init_db("sqlite://")`). The test data helper
that inserts throwaway trades releases its own redirect the same way immediately after
use, at `research/tests/test_promotion.py:214` (`init_db("sqlite://")`).

What doesn't fix it: wrapping the production call in a try/finally that "restores"
`Trade.session` afterward sounds right but isn't — the function has no way to know what
`Trade.session` pointed at before it ran, so resetting to a fresh in-memory database can
itself clobber a caller's real state instead of restoring it. The actual fix in
`research/promotion.py` is narrower and honest about the limit: fully materialize the
query with `.all()` before returning, and document in the function's own docstring that
it must never be called concurrently with, or interleaved with, other in-process code
relying on `Trade.session` pointing elsewhere — a `WalkForwardRunner`/`Backtesting` run
in the same process being the concrete example. No code fix removes this constraint;
it's freqtrade's own architecture, not a bug in this package.

Verified 2026-08-24 during the paper-trading-promotion sub-project: confirmed via two
independent code reviews (one traced the claim against `freqtrade/persistence/models.py`'s
actual source, one against a live 78/78-passing full test-suite run) and the same
failure pattern this repo already fixed once for a different pair of `Trade`-class
globals in PR #5.

---

## `research/pbo.py`'s PBO silently fails closed to 1.0 on a prime window count

CSCV (the Probability of Backtest Overfitting method) only ever needs an EVEN number of
blocks -- so it can split them into two equal-sized HALVES, in-sample and
out-of-sample -- not blocks that evenly divide the underlying period count. The
original `choose_n_splits` required both: `research/pbo.py:16`
(`s = min(max_splits, n_periods)`) is what it looks like after the fix; before the fix
it searched for an even number that also evenly divided `n_periods`, falling back to a
hardcoded 2 when none existed. For a PRIME `n_periods` (no even divisor exists at
all -- and critically, this includes the common case of 17 walk-forward windows, which
`train_days=90, test_days=30` over roughly a 20-month discovery window produces), that
fallback of 2 doesn't divide 17 evenly either, so the caller's own guard clause used to
silently return the fail-closed `research/pbo.py:47`
(`return {"pbo": 1.0, "n_splits": 0, "n_combinations": 0, "logits": []}`) — a maximally
pessimistic PBO regardless of what the real data showed.

The trap: this isn't a rare edge case. Any `train_days`/`test_days` choice whose walk-
forward window count comes out prime (or has no even divisor) silently corrupts PBO for
that entire gate run, and the caller has no way to tell the difference between "this
really is fully overfit" and "the window count broke CSCV's block math" -- both report
the identical `pbo: 1.0`. Three real gate runs in one session (`EmaTrendFollow`,
`BandtasticMeanReversion`, `MacdMomentum`, all using a 17-window setup) all reported the
fake 1.0. `choose_n_splits(17)` now returns 16 rather than falling back -- confirmed at
`research/tests/test_pbo.py:26` (`assert s % 2 == 0`). Re-running the same
`EmaTrendFollow` config after the fix gave a real PBO of 0.299 (passes the 0.5
threshold); deflated Sharpe and the permutation test, not PBO, were the actual reasons
none of the three strategies passed.

The fix: `np.array_split` (used a few lines later to build the actual CSCV blocks)
already handles a remainder by giving the first few blocks one extra column --
`research/pbo.py:49` (`blocks = np.array_split(returns_matrix, s, axis=1)`) never
needed an exact divisor in the first place. Only evenness and `s >= 2` matter; the
caller's guard is now `research/pbo.py:46` (`if s < 2 or s % 2 != 0:`), no longer
`n_periods % s != 0`.

Verified 2026-08-24 during momentum-strategy gate testing: confirmed by directly
reproducing the bug (`research/tests/test_pbo.py:60`
(`test_low_when_one_variant_dominates_every_period_and_n_periods_is_prime`), a dominant-
variant fixture at `n_periods=17` that reported `pbo=1.0` before the fix and `pbo<0.3`
after) and by re-running a real gate command with the exact `train_days=90,
test_days=30` config that originally triggered it.
