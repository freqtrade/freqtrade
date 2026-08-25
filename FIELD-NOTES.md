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

---

## `LoggingMixin.show_output` is GLOBAL class-level state too -- leaks past `research/`'s own tests

Same class of bug as the `Trade.session`/`Trade.use_db` entries above, a third instance
of it. `freqtrade/optimize/backtesting.py:144` (`LoggingMixin.show_output = False`, in
`Backtesting.__init__`) and `:495` (same assignment, in `reset_backtest()`) unconditionally
disable a class attribute -- not per-instance state -- shared by every `LoggingMixin`
subclass and instance in the process. The only thing that ever restores it is the separate `cleanup()` staticmethod at
`freqtrade/optimize/backtesting.py:284` (`def cleanup():`), which in *production* code is
called from exactly one place: `freqtrade/rpc/api_server/api_backtest.py:275`
(`ApiBG.bt["bt"].cleanup()`, the webserver's background-backtest endpoint) -- test-only
call sites exist too (`tests/optimize/conftest.py`'s own directory-scoped
`backtesting_cleanup` fixture among them), but they don't help anything outside their own
directory, same gap this fixture closes. Nothing that constructs a `Backtesting` instance
directly in production code -- including `research/walkforward.py:113` and `:181` (both
`Backtesting(...)`) and `research/cost_stress.py:61` (`Backtesting`) -- ever calls it.
**That's still true after this fix** -- the fixtures added here only patch pytest's own
test isolation; a long-lived non-test process (a script, a notebook) that calls
`research/walkforward.py`/`cost_stress.py` more than once still leaks `show_output=False`
for the rest of its life. Tracked as a follow-up, not fixed here.

The trap: any test in the same pytest-xdist worker process that runs *after* one that
touches `Backtesting` silently has every `LoggingMixin.log_once()` call become a no-op for
the rest of that worker's life, in any file, unrelated to whatever the leaking test was
about. This surfaced as two CI failures that looked completely unconnected on an unrelated
PR: `tests/exchange/test_exchange.py::test__async_kucoin_get_candle_history` and
`tests/plugins/test_pairlist.py::test_log_cached` /
`test_remove_logs_for_pairs_already_in_blacklist` -- both assert a `log_once`-driven
message was captured/called and got `0` instead. Reproduces deterministically with plain
sequential ordering, no `--random-order`/`-n auto` needed at all: run any `research/` test
that touches `Backtesting` immediately before either pairlist test.

The guard, mirroring `reset_use_db_flags` above (same pattern, different global): an
autouse fixture at `tests/conftest.py`'s `reset_logging_mixin_show_output`, restoring
`LoggingMixin.show_output = True` after every test. **This alone was not sufficient** --
`tests/conftest.py`'s autouse fixtures only apply within their own directory subtree, and
`research/tests/` is a *sibling* of `tests/`, not nested under it (the exact gap
`research/tests/test_promotion.py` already had to work around once for the `Trade.session`
entry above, with its own local fixture rather than relying on `tests/conftest.py`). The
real fix needed the identical fixture duplicated into a new `research/tests/conftest.py`
too, so `research/`'s own tests clean up after themselves regardless of which directory the
next test in the worker happens to live in.

Verified 2026-08-25 while debugging PR #14's unrelated CI failures: reproduced the leak
deterministically with real sequential test ordering (no fixture, confirmed fail), then
confirmed the exact same sequence passes with both fixtures in place. A combined
`pytest research/ tests/plugins/test_pairlist.py tests/exchange/test_exchange.py
--random-order` run found zero further collisions (1848 passed). The regression test
(`tests/test_logging_mixin_show_output_leak.py`, same `pytester` sub-run technique as
`tests/persistence/test_use_db_flag_leak.py`) could not be run locally in this environment
-- a pre-existing, unrelated `pytest-retry` version-drift issue breaks *any* `pytester`
sub-run here, confirmed identically on the already-merged `test_use_db_flag_leak.py`; CI is
the first place that test runs end-to-end.

---

## Hyperliquid's `tid` is not always a real per-fill identifier -- "Spot Dust Conversion" fills use `tid=0` as a sentinel

`research/trader_mining/ingestion.py:39-44`'s own comment already flagged this exact risk
before it was confirmed real: `RawFill.tid`/`NormalizedFill.tid` are globally
`unique=True` (`research/models.py:101`, `:117`), and `ingest_hyperliquid_fills`'s
dedup check (`research/trader_mining/ingestion.py:45-46`,
`session.query(RawFill.tid).filter(RawFill.trader == trader)`) is scoped per-trader, not
global -- it relies on Hyperliquid's `tid` being genuinely unique across every wallet.

That guarantee doesn't hold for `"direction": "Spot Dust Conversion"` fills: Hyperliquid
periodically auto-converts small residual token balances into another asset, and these
system-generated fills carry `tid=0` instead of a real per-fill sequence number --
confirmed by reproducing the exact predicted failure live: importing a second real wallet
(`0x493db0ed7514c975e9abcc110bd40c473b6763e3`) into a database that already held a
`tid=0` dust-conversion fill from a different wallet
(`0xfae95f601f3a25ace60d19dbb929f2a5c57e3571`) raised
`sqlalchemy.exc.IntegrityError: UNIQUE constraint failed: normalized_fills.tid` on
`trader-import`, uncaught, aborting the whole run for the second wallet. The exact same
degenerate-identifier pattern this session already found and fixed once, one layer away,
for `RawLedgerEvent.event_id`/Hyperliquid's `cStakingTransfer` events sharing one
zero-hash `id` (see `research/trader_mining/ingestion.py`'s `_dedup_key` and
`docs/superpowers/specs/2026-08-25-trader-mining-ledger-reconciliation-design.md`'s
"Not incremental" section) -- Hyperliquid using a sentinel/placeholder identifier for
system-generated events rather than a genuine unique one, not confined to one table.

Not yet fixed here -- found live-validating trader-mining Release 3 (performance metrics)
against real wallets, out of scope for that release (this is a Release 1/2 ingestion bug,
not a metrics bug). Two concrete consequences follow directly from the ledger-table fix's
own precedent: (1) a SECOND dust-conversion fill for the SAME wallet would be silently
dropped by the existing per-trader dedup (once `tid=0` is in `existing_tids`, every
subsequent same-wallet `tid=0` fill is skipped forever) -- plausibly part of what's been
causing "unexplained" position gaps in real wallet validation, not yet confirmed as the
specific cause of any one documented gap; (2) a second WALLET's dust-conversion fill
crashes the whole `trader-import` run, as reproduced above. Upgrade path if/when this is
prioritized: the same fix already applied to `RawLedgerEvent` -- a derived dedup key
(trader, tid, timestamp, symbol, a hash of the raw payload) instead of trusting `tid`
alone, since `tid=0` genuinely recurs across both wallets and time for this fill type.

Verified 2026-08-25 while real-data-validating trader-mining Release 3: reproduced the
cross-wallet `IntegrityError` live, confirmed `direction='Spot Dust Conversion'` on the
colliding row, and confirmed the same session already fixed the identical pattern for
`RawLedgerEvent` (`cStakingTransfer`'s zero-hash `id`) a few hours earlier in the same
line of work.
