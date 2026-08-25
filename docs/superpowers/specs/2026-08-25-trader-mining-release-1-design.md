# Trader/Wallet Mining — Release 1 (Hyperliquid Ingestion) — Design

## Context

First sub-project of `TRADER_WALLET_MINING_PROPOSAL.md`'s phased release plan: read-only
ingestion of one public Hyperliquid wallet's fill history into local storage. Explicitly
**not** trade reconstruction (Release 2), performance analysis (Release 3), or anything
multi-wallet (Release 5+) -- this release answers only "can we reliably get the data,"
matching the proposal's own success criterion for its first milestone.

Grounded in a live spike against the real Hyperliquid API (not just documentation), done
before writing this spec:

- `ccxt.async_support.hyperliquid().fetch_my_trades(params={"user": <address>})` works fully
  **unauthenticated** -- no `apiKey`/`secret` needed at all, confirmed by instantiating
  `ccxt_async.hyperliquid()` with zero credentials and successfully calling it.
- ccxt already normalizes the response into its unified trade structure (`id`, `order`,
  `timestamp`, `symbol`, `side`, `price`, `amount`, `cost`, `fee`) **and** preserves
  Hyperliquid's raw payload verbatim under `trade["info"]`. Confirmed: ccxt's `id` field is
  Hyperliquid's `tid`, and `order` is `oid` -- exactly the fill-identity convention the
  proposal's review notes already called for.
- The underlying raw HTTP call ccxt itself makes is `exchange.publicPostInfo({"type":
  "userFills"|"userFillsByTime", "user": ..., ...})`, read directly from
  `ccxt/async_support/hyperliquid.py`'s `fetch_my_trades` source -- this is the exact
  boundary Component 2's frozen-fixture tests patch, not `fetch_my_trades` itself.
- Captured a real 8-fill response from this call (wallet: Hyperliquid's zero address,
  `0x000...000`) as `research/tests/fixtures/hyperliquid_user_fills_raw.json` -- real field
  names verified: `coin`, `px`, `sz`, `side`, `time`, `startPosition`, `dir`, `closedPnl`,
  `hash`, `oid`, `crossed`, `fee`, `tid`, `feeToken`, `twapId`. Matches every field the
  proposal's review notes predicted.

**Known limitation of the captured fixture, stated honestly rather than glossed over:** the
zero address turned out to be some settlement/system account (every captured fill has
`dir: "Settlement"`, on an exotic `hyna:XMR` pair), not a normal discretionary trader. This
is fine for Release 1 -- it proves the schema and ingestion mechanics -- but it means no
`dir` value other than `"Settlement"` has been observed yet. Liquidation/ADL `dir` encodings
remain **unverified**, exactly as the review notes already flagged; this spec does not
attempt to classify them. `direction` is stored as the raw string, unfiltered, for whatever
value Hyperliquid actually sends -- future releases classify it once real trader data exists
to classify from.

## What this is not

**Not `engine.py`, not reconstruction, not analysis.** The proposal's review notes originally
scoped Release 1 to four files including `engine.py`; that was wrong -- `engine.py`'s whole
job (Release 2) is grouping fills into logical trades, which has nothing to operate on until
raw fills exist. This release ships exactly three: `provider.py`, and new tables in the
existing `research/models.py` (no separate `storage.py` module -- see Components), plus a
`research/cli.py` subcommand.

**Not a new database or a new `research/db.py`.** The proposal explicitly says to reuse this
project's existing storage conventions "rather than introducing a second database technology
without a concrete reason." `research/db.py`'s `get_engine`/`get_session` already work for
any table registered on the shared `Base` (`research/models.py`) -- new fill tables register
on that same `Base`, so `research/db.py` needs zero changes, and by default they land in the
same `user_data/research.sqlite` the `gate` subcommand already uses (overridable via
`--db-path`, matching `gate`'s existing flag).

**Not a `TraderDataProvider` Protocol yet.** Per the review notes: build `provider.py` as a
concrete Hyperliquid-only module; extract an abstract interface when a second provider
(Bitquery, Dune, ...) actually exists to design it against. A Protocol with one implementer
is speculative abstraction.

**Not automatic pagination beyond one wallet's full available history.** Fetches everything
available up to Hyperliquid's 10,000-fill ceiling per call sequence, surfacing
`history_completeness` explicitly. Does not attempt to work around that ceiling (e.g. via
some other data source) -- out of scope, and the review notes already flag this as a hard
provider limit, not an engineering gap to close here.

**Not a live/scheduled CI job.** The live-canary test (Testing, layer 3) is written and
included, gated behind an env var so it never runs by default -- setting up an actual
scheduled GitHub Actions workflow to run it is explicitly deferred; nothing in this repo's
CI config changes as part of this release.

## Architecture

```
research/
  models.py          [extended] -- RawFill, NormalizedFill (new), same shared Base
  trader_mining/
    __init__.py       [new]
    provider.py        [new] -- fetch_hyperliquid_fills(trader, since=None) -> FetchFillsResult
    ingestion.py        [new] -- ingest_hyperliquid_fills(session, trader, since=None) -> IngestResult
                                  (persists provider.py's output idempotently)
  cli.py             [extended] -- new `trader-import` subcommand

research/tests/
  fixtures/hyperliquid_user_fills_raw.json   [new] -- real captured raw response, 8 fills
  trader_mining/
    test_provider.py    [new]
    test_ingestion.py    [new]
  test_cli.py         [extended]
```

## Components

**`research/models.py`** (extended) -- two new tables on the existing shared `Base`:

`RawFill`: `id` (PK), `source` (`"hyperliquid"`, string -- a deliberately-provider-scoped
column even though only one provider exists yet, since it costs nothing and the proposal's
own storage section calls for a `trader_sources`-style distinction), `trader` (wallet
address, indexed), `tid` (Hyperliquid's fill id, **unique** -- a fill's `tid` is globally
unique per Hyperliquid's own documentation, not just per-wallet), `payload_json` (the full
`trade["info"]` dict, `json.dumps`'d verbatim -- exactly what the proposal asks for: "preserve
the original provider payload as well, so that normalization bugs can be investigated
later"), `retrieved_at` (datetime, when *this process* fetched it -- the proposal's "timestamped
with the source retrieval time" requirement).

`NormalizedFill`: `id` (PK), `trader` (indexed), `tid` (indexed, matches a `RawFill.tid` --
no formal `ForeignKey` constraint, matching this file's existing lightweight
`PromotionRecord.candidate_result_id`/`HealthCheck.promotion_record_id` convention of an
indexed plain column, not a hard FK), `timestamp` (datetime, from ccxt's `trade["timestamp"]`),
`symbol` (ccxt's unified symbol string), `side` (`"buy"`/`"sell"`, from ccxt), `price` (float),
`quantity` (float, ccxt's `amount`), `notional` (float, ccxt's `cost`), `position` (float,
Hyperliquid's `startPosition` -- the position size *before* this fill, per Hyperliquid's own
field name and semantics), `closed_pnl` (float, Hyperliquid's `closedPnl`), `direction`
(string, Hyperliquid's raw `dir` value, unfiltered/unclassified per "What this is not"
above), `crossed` (bool, Hyperliquid's own taker/maker-equivalent flag), `fee` (float),
`fee_currency` (string, ccxt's `fee.currency`), `order_id` (string, ccxt's `order` = `oid`).

**`research/trader_mining/provider.py`** (new) --

```python
@dataclass
class FetchFillsResult:
    trades: list[dict]  # ccxt unified trade dicts; trade["info"] is the raw Hyperliquid payload
    history_completeness: str  # "complete" | "truncated_by_provider_limit"

async def fetch_hyperliquid_fills(
    trader: str, since: datetime | None = None
) -> FetchFillsResult: ...
```

Wraps `ccxt.async_support.hyperliquid()` (constructed fresh per call, no credentials --
matches `research/cost_stress.py`'s existing pattern of a throwaway per-call exchange
instance rather than a shared long-lived one). Calls `fetch_my_trades(since=..., params=
{"user": trader})`, paginating by repeatedly advancing `since` to the **last returned fill's
own timestamp** (not `+1ms`) and re-fetching from there -- deliberately overlapping by one
timestamp instant rather than risking a skipped fill, per this session's own earlier lesson
(an lmchatbot cross-check on this exact `+1ms` pattern flagged it as unsafe when multiple
fills can share a timestamp at a page boundary). The resulting duplicate fill(s) at each page
boundary are harmless -- `ingestion.py`'s `tid`-based dedup discards them before insert, so
correctness comes from deduplication, not from timestamp arithmetic being exact. Paginates
until either (a) a page returns fewer than the requested page size (end of available
history -- `"complete"`) or (b) the running total reaches Hyperliquid's documented 10,000-fill
ceiling (`"truncated_by_provider_limit"`). No client-side request-count rate limiter beyond
what ccxt's own built-in `enableRateLimit` already provides for a single wallet's sequential
pagination -- multi-wallet concurrent rate budgeting is explicitly deferred to whichever
release actually fetches more than one wallet at a time (Release 5+).

**`research/trader_mining/ingestion.py`** (new) --

```python
@dataclass
class IngestResult:
    n_fetched: int
    n_new: int
    history_completeness: str

def ingest_hyperliquid_fills(
    session: Session, trader: str, since: datetime | None = None
) -> IngestResult: ...
```

Calls `provider.fetch_hyperliquid_fills` (via `asyncio.run`, since `research/cli.py`'s
`main()` is itself synchronous -- matches how the rest of `research/` stays sync-only, no
async creeping further into the package than this one boundary), then persists idempotently:
queries existing `RawFill.tid` values for this trader first, inserts only genuinely new
fills into both `RawFill` and `NormalizedFill`, so re-running the same import is a safe no-op
for already-seen fills (the proposal's explicit "idempotent, resumable, tolerant of
duplicate fills" requirement).

**`research/cli.py`** (extended) -- new `trader-import` subcommand: `--trader` (wallet
address, required), `--since` (`YYYY-MM-DD`, optional), `--db-path` (default
`user_data/research.sqlite`, matching `gate`'s own flag). Prints `n_fetched`, `n_new`, and
`history_completeness`; exits `0` always (ingestion has no pass/fail verdict the way `gate`
does -- `history_completeness == "truncated_by_provider_limit"` prints a warning line, not a
nonzero exit).

## Data flow

1. `python -m research.cli trader-import --trader 0x... --db-path ...` -> `cli.main()`
2. `ingestion.ingest_hyperliquid_fills(session, trader, since)`
3. `asyncio.run(provider.fetch_hyperliquid_fills(trader, since))` -- paginates internally,
   returns `FetchFillsResult`
4. Existing `RawFill.tid` values for `trader` queried once; new fills only inserted into
   `RawFill` (raw `info` payload) and `NormalizedFill` (mapped fields) in one transaction
5. `IngestResult` returned to the CLI, printed

## Error handling

- `fetch_hyperliquid_fills` lets ccxt's own exceptions (network errors, `ExchangeError` on a
  malformed address, etc.) propagate -- no swallowing, matching every other `research/`
  module's convention of failing loudly rather than degrading silently.
- `ingest_hyperliquid_fills` wraps the fetch-then-persist sequence in one DB transaction --
  a failure partway through (e.g. a malformed fill missing an expected field) rolls back
  rather than leaving `RawFill` and `NormalizedFill` out of sync for the fills already
  processed in that batch.
- `history_completeness == "truncated_by_provider_limit"` is not an error -- returned and
  printed as an explicit, honest result, per the proposal's "do not silently treat missing
  historical data as zero activity."

## Testing

Three layers, resolved via `superpowers:brainstorming` + an lmchatbot second opinion
(Gemini draft, ChatGPT-corrected) before locking in:

1. **Provider unit tests** (`test_provider.py`) -- mock `fetch_my_trades` itself (via
   `mocker.patch.object` on a `ccxt_async.hyperliquid` instance) to test `provider.py`'s own
   control flow: wallet-address forwarding into `params={"user": ...}`, pagination-loop
   termination on both the short-page and 10k-ceiling conditions, `history_completeness`
   correctness. Fast, no network, tests intent not the ccxt/Hyperliquid contract.

2. **Frozen real-response contract test** (`test_provider.py`) -- patches
   `HyperliquidProvider`'s underlying exchange's `publicPostInfo` (the raw HTTP boundary,
   confirmed above by reading ccxt's own source -- **not** `fetch_my_trades`, which would
   skip ccxt's real parser and collapse this into a second copy of layer 1) to return
   `research/tests/fixtures/hyperliquid_user_fills_raw.json`'s real captured content, then
   asserts exact values on the real ccxt-parsed output: `id == "802647614388392"`
   (`tid`), `info["dir"] == "Settlement"`, `info` round-trips the raw payload byte-for-byte
   reparseable. This is the layer that actually catches ccxt/Hyperliquid schema drift.

3. **Live canary** (`test_provider.py`, one test) -- calls the real Hyperliquid API against
   the zero address, gated behind `if not os.environ.get("HYPERLIQUID_LIVE_TEST"):
   pytest.skip(...)` at the top of the test body -- **not** a `pytest.mark` requiring changes
   to this repo's shared `pyproject.toml`/CI workflow config, so it never runs in normal CI
   and never makes an unrelated PR flaky. Asserts structural/semantic invariants only
   (`isinstance` checks, `"info" in trade`, timestamps parse, price/amount are numeric) --
   never exact values, since real data changes over time.

4. **`ingestion.py` tests** (`test_ingestion.py`) -- real SQLite (`:memory:` via
   `research.db.get_engine(":memory:")`, matching `research/tests/test_*.py`'s existing
   convention of real DB objects, no DB mocking), `provider.fetch_hyperliquid_fills` mocked
   to return canned `FetchFillsResult` objects (deterministic control over `n_fetched`/`n_new`
   without hitting the network at this layer -- the network/parsing correctness is layer 2's
   job, not this one's). Covers: first import populates both tables; re-running the same
   import with the same fills is a no-op (`n_new == 0`); a new fill on top of existing ones
   only inserts the new one; `history_completeness` passes through unchanged.

5. **`cli.py` test** (`test_cli.py`, extended) -- mirrors the existing `gate` subcommand test
   pattern: `mocker.patch("research.cli.ingest_hyperliquid_fills", ...)`, asserts the CLI
   parses `--trader`/`--since`/`--db-path` correctly and prints the result.

## Open items resolved during brainstorming

- Scope: **three files, not the review notes' four** -- `engine.py` moved out to Release 2,
  caught while writing this spec (the proposal doc's own "Phased release plan" section was
  corrected to match, before this spec was written).
- Storage: **new tables in the existing `research/models.py`/`research/db.py`**, no new
  database, no new `storage.py` module -- simpler than the review notes' original 4-file
  layout once it was clear there's no new engine/config to isolate into its own file.
- Provider abstraction: **deferred**, per review notes -- concrete Hyperliquid module only.
- Testing strategy (the design fork this brainstorming session was specifically asked to
  resolve): **three layers** (provider-unit / frozen-real-response-through-the-real-parser /
  env-gated live canary), corrected from Gemini's first draft by ChatGPT's cross-check
  (the draft's "frozen fixture" mocked `fetch_my_trades` directly, which would have made it
  functionally identical to the unit-test layer and lost all drift-detection value).
- Fixture wallet: **the zero address's real captured response**, despite it being an
  atypical settlement-type account -- sufficient for schema/mechanics fidelity, which is all
  Release 1 needs; explicitly documented as a known gap (no non-`"Settlement"` `dir` value
  observed yet) rather than silently assumed complete.
