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
