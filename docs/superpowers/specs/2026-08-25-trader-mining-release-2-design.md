# Trader/Wallet Mining — Release 2 (Trade Reconstruction) — Design

## Context

Second release of `TRADER_WALLET_MINING_PROPOSAL.md`'s phased plan, building directly on
Release 1's `research/trader_mining/provider.py`/`ingestion.py` and the `NormalizedFill`
table. Groups a wallet's already-ingested fills into logical `ReconstructedTrade` records --
proposal §3's explicit requirement, and the thing `engine.py` was held back from Release 1
specifically to do once real fill data existed to operate on.

The core algorithm (position-crossing trade boundaries, reversal handling, entry/exit price
weighting) was cross-checked with an external model (Gemini draft, ChatGPT-corrected) before
being written up here -- one real correction to the original proposal draft survived that
review and shapes this spec directly (see "Reversal `closed_pnl` handling" below).

## What this is not

**Not a fill-to-trade join table.** `ReconstructedTrade` aggregates directly into its own
fields (weighted entry/exit price, summed P&L/fees, `n_fills` count) rather than persisting
an exhaustive per-fill membership table. Proposal §3 asks for trade-level records with
aggregate fields, not fill-level traceability beyond `RawFill.payload_json` (Release 1) and
`n_fills` (informational only, not a foreign-key-backed list).

**Not float arithmetic for position tracking.** `NormalizedFill`'s columns stay `Float`
(Release 1's already-shipped schema, unchanged) -- but `engine.py` converts to `Decimal` at
the point of use for every value that participates in the running-position/zero-boundary
calculation. Zero is a semantic boundary here (trade close/open detection), and float
summation over a long fill history can leave a residual like `1e-11` that never hits exact
zero -- confirmed as a real risk, not theoretical, by the external review.

**Not funding-aware.** Funding payments alter account balance, not base position quantity
(confirmed against Hyperliquid's own semantics in the same review) -- out of scope here,
consistent with Release 1's "funding is a separate endpoint" principle. A trade's `net_pnl`
in this release reflects execution P&L only, not funding accrued while the position was
open; documented as a known gap, not silently assumed complete.

**Not a special liquidation P&L model.** Liquidation/ADL fills are processed through the
*same* position state machine as any other fill (confirmed: they carry the same
`startPosition`/`closedPnl`/`fee` shape, not a separate non-fill mechanism) -- the only
special handling is a `was_liquidated` flag set when the fill's `direction` field indicates
one, for later filtering/analysis. No liquidation-specific pricing or fee adjustment.

**Not incremental/patchable reconstruction.** `reconstruct_and_persist_trades` recomputes a
(trader, symbol)'s trades from scratch every time it runs -- deletes existing
`ReconstructedTrade` rows for that scope, re-derives from every currently-stored
`NormalizedFill` row, re-inserts. Simpler and more correct than incremental patching: a
later `trader-import` run backfilling older history could retroactively change earlier
trade boundaries, which incremental reconciliation can't handle cleanly, but a clean
recompute always can.

## Architecture

```
research/
  models.py                    [extended] -- ReconstructedTrade (new), same shared Base
  trader_mining/
    engine.py                   [new]
      reconstruct_trades(trader, symbol, fills: list[NormalizedFill]) -> list[ReconstructedTrade]
        -- pure function, no DB access, the core testable algorithm
      reconstruct_and_persist_trades(session, trader, symbol=None) -> ReconstructResult
        -- orchestration: queries NormalizedFill grouped by symbol, calls
           reconstruct_trades per symbol, deletes + re-inserts ReconstructedTrade rows
  cli.py                       [extended] -- new `trader-analyze` subcommand
```

## Components

**`research/models.py`** (extended) -- one new table:

`ReconstructedTrade`: `id` (PK), `trader` (indexed), `symbol` (indexed alongside trader),
`direction` (`"long"`/`"short"`, from the sign of the position while the trade is open --
positive = long, negative = short; unambiguous even for a truncated-start trade with no
observed entry-like fill), `entry_timestamp`,
`entry_price` (quantity-weighted average across every position-increasing fill *actually
observed* in the trade), `exit_timestamp`, `exit_price` (quantity-weighted average across
every position-decreasing fill, including a reversal's synthetic closing leg), `quantity`
(total entered == total exited, by construction of the zero-to-zero boundary -- for a
truncated-start trade, the portion actually observed), `gross_pnl` (sum of `closed_pnl`
across every fill/leg in the trade), `fees` (summed, proportionally split for a reversal's
two legs), `net_pnl` (`gross_pnl - fees`), `holding_time_seconds` (float,
`exit_timestamp - entry_timestamp`), `n_fills` (int, informational), `is_truncated_start`
(bool -- see below), `was_liquidated` (bool). No nullable fields: `entry_price`/
`entry_timestamp` are always populated, even for a truncated-start trade with zero
*observed* entry-like fills (the bootstrapped position never scales in further before it
starts exiting) -- in that specific case they fall back to the first observed fill's own
price/timestamp, an explicit approximation of "state as first observed," not a claim about
the true (unobserved) economic entry. `is_truncated_start=True` is the caller's signal to
treat `entry_price`/`entry_timestamp` with that skepticism; downstream code must check it
rather than trusting these fields blindly on every trade.

**`research/trader_mining/engine.py`** (new) --

```python
def reconstruct_trades(
    trader: str, symbol: str, fills: list[NormalizedFill]
) -> list[ReconstructedTrade]: ...

@dataclass
class ReconstructResult:
    n_trades: int
    symbols: list[str]

def reconstruct_and_persist_trades(
    session: Session, trader: str, symbol: str | None = None
) -> ReconstructResult: ...
```

`reconstruct_trades` -- the core algorithm, operating on one (trader, symbol)'s fills,
**already sorted by `(timestamp, abs(position), tid)`** (see "Post-implementation
correction" below -- `tid` alone is not a monotonic sequence number, found validating
against real data; `abs(position)` is the real tiebreaker, `tid` demoted to a
third-level, only-for-full-determinism one) -- the caller's responsibility, not
re-sorted internally, since `reconstruct_and_persist_trades` already queries in that order).

All position/price/quantity/pnl/fee arithmetic uses `Decimal(str(value))` conversions from
the fill's `Float` columns, never raw float addition, specifically because the zero-crossing
check (`running_position == Decimal("0")`) is exact-equality-sensitive.

1. **Bootstrap**: `running_position = Decimal(str(fills[0].position))` (the first fill's own
   `startPosition`) -- not assumed zero. If non-zero, the position predates observed
   history; the trade this first fill contributes to gets `is_truncated_start=True`, and
   the first fill itself is classified as entry-like or exit-like exactly as any other fill
   (step 3) rather than treated as a synthetic "trade start" event -- there was no
   zero-to-nonzero transition to observe. `entry_price`/`entry_timestamp` accumulate only
   from *actually observed* entry-like fills within this trade, same mechanism as any other
   trade; if none occur before the trade closes (the bootstrapped position immediately
   starts exiting), they fall back to the first fill's own price/timestamp (see the
   `ReconstructedTrade` field list above for why that's an approximation, not a real entry).
   `net_pnl`/`fees` on a truncated-start trade remain fully valid regardless -- Hyperliquid's
   own `closed_pnl` on closing fills doesn't depend on when the position was first opened.
2. For each fill: `signed_qty = Decimal(str(fill.quantity)) * (1 if fill.side == "buy" else
   -1)`, `end_position = running_position + signed_qty`.
3. **Same-sign continuation** (`running_position` and `end_position` share a sign, or either
   is zero without the other crossing through the opposite sign): classify the fill as
   entry-like if `abs(end_position) > abs(running_position)` (adding to the position) or
   exit-like if `abs(end_position) < abs(running_position)` (reducing it) -- this is what
   makes scaling in/out and partial entries/exits work: a trade can have many entry-like and
   many exit-like fills before it finally closes, not just one of each.
4. **Close** (`end_position == 0`): finalize the open trade -- `exit_timestamp`/`exit_price`
   from this fill (or this fill folded into the running exit-weighted average if other
   exit-like fills preceded it), append to results, clear trade-in-progress state.
5. **Reversal** (`end_position`'s sign is the opposite of `running_position`'s, and
   `running_position != 0`): split this one fill into two synthetic legs at the exact
   quantity that would have zeroed the position --
   - **Closing leg**: `quantity = abs(running_position)`, at this fill's price and
     timestamp. Finalizes the currently-open trade. Gets **100% of this fill's
     `closed_pnl`** -- per the external review's correction to the original proposal draft
     (which had proposed a proportional split): Hyperliquid attributes realized P&L only to
     the closing portion of a reversal; the newly-opened portion has zero realized P&L by
     definition. Fee: `fill.fee * (abs(running_position) / fill.quantity)` (fees, unlike
     P&L, *do* split proportionally by quantity across both legs).
   - **Opening leg**: `quantity = abs(signed_qty) - abs(running_position)`, same price and
     timestamp, `closed_pnl = 0` (see above), remaining fee
     (`fill.fee - closing_leg_fee`). Starts a new trade-in-progress with this as its entry
     leg, `running_position` reset to `end_position`.
   - Both legs count this one real fill toward `n_fills` on their respective trades (one
     real execution genuinely touches both).
6. **Malformed input**: a fill with `quantity <= 0` raises `ValueError` -- a data error
   worth failing loudly on (per the proposal's "do not assume all fills can be perfectly
   reconstructed" -- fail loudly on the ones that can't, don't silently mishandle them),
   not something this algorithm should paper over.
7. **Liquidation flagging**: any fill in a trade whose raw `direction` string contains
   `"Liquidat"` (case-sensitive substring match against the known Hyperliquid values
   observed/documented so far -- see Release 1's spec for the same "don't assume unverified
   field values" caution) sets that trade's `was_liquidated = True`.

`reconstruct_and_persist_trades` -- queries distinct symbols for `trader` (or just the one
given `symbol`) from `NormalizedFill`, for each symbol queries all its fills ordered by
`(timestamp, abs(position), tid)`, calls `reconstruct_trades`, deletes existing `ReconstructedTrade` rows
for `(trader, symbol)`, inserts the freshly computed ones, commits once at the end (one
transaction covering the full delete+reinsert, so a partial failure doesn't leave stale and
fresh trades mixed).

**`research/cli.py`** (extended) -- new `trader-analyze` subcommand: `--trader` (required),
`--symbol` (optional, default all symbols), `--db-path` (default matching existing
subcommands). Prints `n_trades` and the list of symbols processed.

## Data flow

1. `python -m research.cli trader-analyze --trader 0x...` -> `cli.main()`
2. `engine.reconstruct_and_persist_trades(session, trader, symbol=None)`
3. Query distinct symbols for `trader` from `NormalizedFill`
4. Per symbol: query fills ordered `(timestamp, abs(position), tid)` -> `engine.reconstruct_trades(...)`
5. Delete existing `ReconstructedTrade` rows for `(trader, symbol)`, insert new ones
6. `ReconstructResult` returned to the CLI, printed

## Error handling

- `reconstruct_trades` raises `ValueError` on a `quantity <= 0` fill -- caller-visible data
  error, fail loudly.
- `reconstruct_and_persist_trades` wraps the full delete+reinsert in one transaction --
  a failure partway through rolls back rather than leaving a symbol's trades half-replaced.
- An unclosed trade at the end of a fill sequence (the wallet's current open position) is
  **not** emitted as a `ReconstructedTrade` -- only fully closed (zero-to-zero) spans are.
  Documented, not silent: `ReconstructResult` doesn't currently surface "N fills belong to
  a still-open position," a real gap for Release 3 (performance metrics) to account for
  when it needs open-position awareness; not pretended to be handled here.

## Testing

Per the proposal's own explicit instruction: heavily unit tested with **hand-built fill
sequences**, not just replayed real captured data (Release 1's real fixture is all
same-direction `"Settlement"` fills on one exotic pair -- it exercises none of the
position-transition logic this release is actually about).

`research/tests/trader_mining/test_engine.py` -- pure-function tests against
`reconstruct_trades`, constructing `NormalizedFill` ORM instances directly (no DB, matching
`research/tests/test_ingestion.py`'s established convention for fill-shaped test data):

1. One entry fill, one exit fill, `quantity` fully closes -> one `ReconstructedTrade`,
   correct `entry_price`/`exit_price`/`gross_pnl`/`fees`/`net_pnl`/`holding_time_seconds`.
2. Multiple entry fills (scale-in) before one exit -> `entry_price` is the correct
   quantity-weighted average across the entry fills.
3. Multiple exit fills (scale-out) after one entry -> `exit_price` is the correct
   quantity-weighted average.
4. A reversal fill (`+5` position, one fill of size `8` on the opposite side, landing at
   `-3`), followed by a further fill that fully closes the resulting `-3` short -> exactly
   two `ReconstructedTrade`s: the closing leg from the reversal (`quantity=5`, `closed_pnl`
   = the reversal fill's full `closed_pnl`) and the opening leg (`quantity=3`,
   `closed_pnl=0` from the reversal fill, plus whatever `closed_pnl` the follow-up closing
   fill reports), fee on the reversal fill split proportionally `5/8` and `3/8`, both
   trades' `n_fills` reflecting every fill that actually contributed to them (the reversal
   fill counts toward both). A single reversal fill with nothing closing the new leg
   afterward correctly produces only ONE `ReconstructedTrade` (the closing leg) -- the
   newly-opened leg stays an in-progress, unclosed position and is not emitted, per "Error
   handling" above; covered as its own explicit test case, not left implicit.
5. A non-zero `position` on the very first fill, immediately followed by an exit-like fill
   that closes it with no further entry-like fills observed -> `is_truncated_start=True`,
   and `entry_price`/`entry_timestamp` fall back to that first fill's own values exactly
   (the no-real-entry-observed case). A second variant of this test: the same truncated
   start, but with one more entry-like (scale-in) fill observed before it closes ->
   `is_truncated_start=True`, but `entry_price` is the weighted average of only the
   *observed* entry-like fill(s), not including the unobserved starting position's implied
   cost basis.
6. A fill whose `direction` contains `"Liquidat"` -> the containing trade has
   `was_liquidated=True`.
7. `quantity <= 0` on any fill -> `ValueError`.
8. A long fill sequence (dozens of fills, scaling in and out repeatedly without ever
   reaching exactly zero in float terms if summed naively) -> position tracking stays exact
   via `Decimal`, closes correctly when it should.
9. An unclosed trailing position (fills end with a non-zero running position) -> not
   emitted as a `ReconstructedTrade` (per "Error handling" above).

`research/tests/trader_mining/test_engine.py` (continued) -- `reconstruct_and_persist_trades`
against a real in-memory SQLite session (`research.db.get_engine(":memory:")`, matching
Release 1's own DB-test convention): first run persists trades; a second run after adding
more fills for the same trader/symbol correctly replaces (not duplicates) the prior result;
scoping to one `symbol` leaves other symbols' trades untouched.

`research/tests/test_cli.py` (extended) -- `trader-analyze` subcommand test mirroring the
existing `trader-import` test's shape (mocked `reconstruct_and_persist_trades`, asserts
argument forwarding and printed output).

## Open items resolved during brainstorming

- Trade-boundary definition (the design fork this brainstorming session specifically set
  out to resolve): **zero-crossing** (flat-to-flat), confirmed as standard practice via
  external review, not the proposal's originally-vaguer language.
- Reversal `closed_pnl` handling: **100% to the closing leg, 0 to the opening leg** --
  corrected from an initial draft proposal (proportional split) that the external review
  caught as inconsistent with Hyperliquid's own realized-P&L semantics.
- Fee handling on a reversal: **proportional split by quantity** across both legs (unlike
  `closed_pnl` -- fees apply to the whole execution, P&L doesn't).
- Position bootstrap: **from the first fill's own `startPosition`**, not assumed zero,
  tagged `is_truncated_start` rather than silently treated as a clean start.
- Arithmetic: **`Decimal`, not float**, for the position/boundary tracking specifically --
  confirmed as a real risk (not theoretical) for a long fill history.
- Liquidations: **processed as regular fills** through the same state machine (confirmed
  they carry the same fill shape), just flagged via `was_liquidated`, not modeled specially.
- Real active-wallet test fixture: initially deferred -- no reliable real address was found
  within reasonable effort at design time (Hyperliquid exposes no public
  leaderboard/discovery endpoint; a Nansen API exists that could serve this purpose, but
  requires a paid API key not currently available -- noted for `TRADER_WALLET_MINING_PROPOSAL.md`'s
  own "Future Providers" section, which already anticipated Nansen/Arkham as
  discovery/labeling sources). Resolved during implementation once the user supplied a real
  address from Hyperliquid's own leaderboard web UI (full address, not the UI's truncated
  display form) -- see "Post-implementation correction" below for what that real data caught.

## Post-implementation correction (found validating against real data)

`research/tests/fixtures/hyperliquid_active_wallet_spot_fills_raw.json` -- 9 real fills from
a real active Hyperliquid wallet (HYPE/USDC spot accumulation, all same-direction, never
closes to zero -- doesn't exercise the trade-boundary logic directly, but its raw shape
caught something the hand-built fixtures couldn't: **`tid` is not a monotonic sequence
number.** Three fills sharing one exact millisecond timestamp, when sorted by `(timestamp,
tid)` as this spec originally specified, came out in **exactly reversed** chronological
order -- confirmed by checking that each fill's own `startPosition` plus its signed quantity
matches the *next* fill's `startPosition` only when ordered by `abs(position)` ascending, not
by `tid` ascending.

**Corrected constraint, superseding the original "(timestamp, tid)" ordering stated
earlier in this document:** `reconstruct_and_persist_trades` orders fills by `(timestamp,
abs(position), tid)` -- `abs(position)` as the real tiebreaker (recovers true execution
order for same-direction accumulation, the only case observed in real data and the general
case for monotonic position-building in either direction), `tid` demoted to a
third-level, only-for-full-determinism tiebreaker. A residual risk remains, documented
in code as a `ponytail:` comment rather than silently assumed away: a reversal fill landing
inside the same tied-timestamp group as other fills isn't necessarily ordered correctly by
`abs(position)` either, since it isn't monotonic across a sign change. Narrower and rarer
than the bug this replaces; not observed in the real wallet data validated so far.

This is exactly the kind of finding the proposal's own emphasis on real-data validation (and
this session's explicit push to source one) was for -- the hand-built fixtures, however
thorough, encoded the same incorrect assumption the implementer made, so they could not have
caught it themselves.
