# Trader/Wallet Mining — Ledger Reconciliation (Release 2 fast-follow) — Design

## Context

Fast-follow to Release 2 (trade reconstruction, PR #17), triggered by a live finding: running
`trader-analyze` against real leaderboard wallets, the position-continuity guard (added for
Release 2's own code review) hard-failed repeatedly on wallets whose spot balance changed
outside of any trade fill. Root-caused and confirmed against real data before this spec was
written, not guessed: Hyperliquid spot balances move via non-trade ledger events (deposits,
withdrawals, wallet-to-wallet transfers, genesis/airdrop allocations, staking transfers) that
`fetch_my_trades` never sees. A live spike against the exact wallet/gap that surfaced this
(`0x9794bbbc222b6b93c1417d01aa1ff06d42e5333b`, a 62264.0 HYPE position discontinuity) found the
explaining event via ccxt's `fetch_ledger`:

```
type=spotTransfer token=HYPE amount=62264.0 user=<this wallet> destination=<other wallet>
timestamp=2024-11-29T10:02:32Z  (inside the gap window 09:53:54-10:07:13)
```

Design cross-checked with lmchatbot (gemini, verified by chatgpt) before writing this up --
confirmed the theory, confirmed `userNonFundingLedgerUpdates`/ccxt's `fetchLedger()` is the
right API, and recommended the scope below (reconcile against the ledger only when the guard
fires; never silently patch an unexplained gap).

## What this is

When `reconstruct_trades`' position-continuity guard would raise, first check whether the
ingested ledger explains the gap: sum the affected asset's signed ledger deltas within the
gap's time window. If they net out to the discrepancy (same epsilon as the existing check),
accept the adjusted position and continue, recording that a reconciliation happened. If they
don't -- or the ledger simply has nothing in that window -- still hard-fail. The guard's
existing "never guess" behavior is preserved; this only widens what counts as "explained."

## What this is not

**Not event-type-specific reconciliation logic in `engine.py`.** `reconstruct_trades` gets one
new input (a lookup of already-summed signed deltas per asset/time-window) and one new
decision (does the sum explain the gap). All the messy per-event-type parsing (which field
holds the amount, which fields determine direction) lives in a new `ledger.py`, mirroring how
`provider.py`/`ingestion.py` already keep raw-shape parsing out of `engine.py`.

**Not a live API call during reconstruction.** Matching Release 1/2's existing separation:
`trader-import` (extended) ingests the ledger up front and stores it; `reconstruct_trades`
stays a pure function reading only already-fetched data, no I/O. Confirmed via
`AskUserQuestion` with the user before writing this spec -- the alternative (fetch live, only
on a gap) was explicitly rejected because it breaks that property.

**Not full ledger semantics.** Six event types are normalized (deposit, withdraw,
accountClassTransfer, spotTransfer, spotGenesis, cStakingTransfer, send) -- the ones observed
in real ingested data (see "Event type survey" below). `deposit`/`withdraw`/
`accountClassTransfer` are USDC-only (they move collateral between the exchange and the
outside world, or between a wallet's own spot/perp sub-accounts) and never explain a
spot-token position gap -- normalized for completeness and future USDC-ledger use, but the
reconciliation check filters to the gap's own base asset, so they're inert for today's use
case. An unrecognized event type normalizes to "no token delta" (not an error) -- if it later
turns out to explain a gap, that gap still hard-fails, loud and diagnosable, rather than
silently passing.

**Not a new field on `ReconstructedTrade`.** A reconciled gap is an ingestion/reconstruction-
time fact, not a fact about the trade itself. Recorded via a log line
(`reconstruct_and_persist_trades`'s return value gains a `reconciled_gaps: list[str]` of
human-readable descriptions) -- confirmed sufficient for a research tool; no new table, no new
column, per YAGNI. `research/cli.py`'s `trader-analyze` prints them if present.

**Not incremental.** `RawLedgerEvent` ingestion is idempotent-by-`id` (the ledger entry's own
hash), exactly matching `RawFill.tid`'s existing dedup pattern -- re-running `trader-import` is
always safe.

## Event type survey (real data, this wallet)

Each event's raw shape lives in `info["delta"]`; ccxt's own unified top-level `amount`/
`currency` fields are inconsistently populated (`None` for `spotTransfer`/`spotGenesis`/
`cStakingTransfer`/`send` in real captured data) and are NOT used -- `info["delta"]` is the
only reliable source, matching `RawFill.payload_json`'s existing "trust the raw payload, not
the unified wrapper" precedent from Release 1.

| type | token/asset | signed delta (from trader's perspective) |
|---|---|---|
| `deposit` | USDC (implicit) | `+delta["usdc"]` |
| `withdraw` | USDC (implicit) | `-delta["usdc"]` |
| `accountClassTransfer` | USDC (implicit) | `+delta["usdc"]` if `toPerp is False` (moving into spot) else `-delta["usdc"]` |
| `spotTransfer` | `delta["token"]` | `-delta["amount"]` if `delta["user"] == trader` (this wallet is the sender) else `+delta["amount"]` (this wallet is `destination`) |
| `spotGenesis` | `delta["token"]` | `+delta["amount"]` (always additive -- airdrop/genesis allocation, no sender) |
| `cStakingTransfer` | `delta["token"]` | `-delta["amount"]` if `delta["isDeposit"]` (into staking, leaves spot) else `+delta["amount"]` (out of staking, returns to spot) |
| `send` | `delta["token"]` | same direction rule as `spotTransfer` (`delta["user"]`/`delta["destination"]`) |

`spotTransfer`/`send` direction check uses `delta["user"] == trader`, not the ledger entry's
own top-level `account`/`referenceAccount` (both `None` in every real captured example) --
confirmed by inspecting real payloads, not assumed.

## Architecture

```
research/
  models.py                    [extended] -- RawLedgerEvent (new)
  trader_mining/
    provider.py                 [extended]
      fetch_hyperliquid_ledger(trader: str) -> list[dict]
        -- thin wrapper around ccxt's fetch_ledger, mirrors fetch_hyperliquid_fills'
           shape/error-handling conventions. No pagination needed: ccxt's Hyperliquid
           fetch_ledger has no documented page-size ceiling analogous to fetch_my_trades'
           2000/10000 (unconfirmed at scale -- flagged as a residual risk below, not
           blocking, since no real wallet ingested so far has hit one).
    ingestion.py                 [extended]
      ingest_hyperliquid_ledger(session, trader) -> LedgerIngestResult
        -- idempotent by RawLedgerEvent.event_id, same pattern as ingest_hyperliquid_fills
    ledger.py                    [new]
      signed_token_delta(entry: RawLedgerEvent, trader: str) -> tuple[str, Decimal] | None
        -- the event-type table above, pure function
      reconciliation_deltas(session, trader, symbol, window_start, window_end) -> Decimal
        -- sums signed_token_delta results for symbol's base asset within the window
    engine.py                    [extended]
      reconstruct_trades gains an optional `reconcile: Callable[[str, datetime, datetime],
      Decimal] | None` parameter (defaults to None -- existing hard-fail-only behavior
      unchanged when not supplied). reconstruct_and_persist_trades passes
      ledger.reconciliation_deltas bound to the session/trader.
  cli.py                        [extended]
    trader-import also calls ingest_hyperliquid_ledger and prints its counts
    trader-analyze prints reconciled_gaps if any occurred
```

### Why `reconcile` is an injected callable, not a hard dependency

`reconstruct_trades` stays a pure function (no `Session`, no imports of `ledger.py`) --
consistent with its existing docstring/design ("no DB access, the core testable algorithm").
Tests exercise the reconciliation branch by passing a canned callable, not a real database.
`reconstruct_and_persist_trades` is the only caller that wires a real one.

## `RawLedgerEvent` schema

```python
class RawLedgerEvent(Base):
    """One row per ccxt fetch_ledger entry, near-verbatim -- info_json preserves the
    raw payload, matching RawFill's existing precedent, since real captured data shows
    ccxt's own unified top-level fields (amount/currency) are unreliable across event
    types (see the design doc's event type survey) and info["delta"] is the field
    actually used at reconciliation time."""

    __tablename__ = "raw_ledger_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trader: Mapped[str] = mapped_column(String(120), index=True)
    event_id: Mapped[str] = mapped_column(String(80), unique=True)  # ledger entry's own "id"
    event_type: Mapped[str] = mapped_column(String(40))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    info_json: Mapped[str] = mapped_column(String)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime)
```

`event_id` (the ledger entry's own hash, e.g. `"0x3412e8f4..."`) is the dedup key, mirroring
`RawFill.tid` -- confirmed unique per real captured data (it's a transaction hash).

## Reconciliation algorithm

```python
def reconciliation_deltas(
    session: Session, trader: str, symbol: str, window_start: datetime, window_end: datetime
) -> Decimal:
    """Sum signed_token_delta for symbol's base asset, ledger events strictly between
    window_start and window_end (the two fills whose position mismatch triggered this
    check). Base asset parsed the same way as engine._is_base_asset_fee -- shared
    helper, extracted to avoid the two modules drifting on what "base asset" means."""
```

`reconstruct_trades`' existing gap check becomes: on mismatch, if a `reconcile` callable was
supplied, call it for `(base_asset_symbol, fill.timestamp, next_fill.timestamp)`; if the
returned sum brings the discrepancy within epsilon, accept `next_position` as the new
`running_position`/`end_position` baseline, append a human-readable note to a `reconciled`
list threaded through to the caller, and continue the loop. Otherwise (no `reconcile`
supplied, or the ledger doesn't explain it), raise exactly as today.

## Shared helper extraction

`_base_asset_of(symbol) -> str | None` (parses the part before `"/"`) is pulled out of
`engine._is_base_asset_fee` into a small shared module (`research/trader_mining/symbols.py`)
so `ledger.py` doesn't duplicate or drift from that parsing logic. `_is_base_asset_fee` and
the new reconciliation code both call it.

## Testing plan

- `test_ledger.py`: one test per event type in the survey table (all 6, using the real
  captured payload shapes above as fixtures -- not invented shapes), covering both directions
  where direction is ambiguous (`spotTransfer`/`send`/`cStakingTransfer`/
  `accountClassTransfer`); an unrecognized type returns `None`, not an error.
- `test_engine.py`: extend with a `reconcile` callable in the two already-existing gap-guard
  tests' style -- one case where the injected sum exactly explains the gap (guard doesn't
  fire, reconstruction proceeds), one where it's short of explaining it (guard still fires).
- `test_ingestion.py`: idempotent re-ingest by `event_id`, matching the existing
  `ingest_hyperliquid_fills` test pattern.
- **Real-data validation**: re-run `trader-import` + `trader-analyze` against
  `0x9794bbbc222b6b93c1417d01aa1ff06d42e5333b` (HYPE/USDC) after implementation -- this is the
  exact wallet/gap that motivated this spec, so it's the acceptance test, not just a nice-to-
  have. Expect the guard to no longer fire there, with a `reconciled_gaps` entry describing
  the 62264.0 HYPE `spotTransfer`.

## Residual risks (documented, not blocking)

- **`fetch_ledger` pagination at scale unconfirmed.** All real wallets ingested so far have
  under 100 ledger entries; whether ccxt's Hyperliquid `fetchLedger()` has an undocumented
  page/count ceiling analogous to `fetch_my_trades`' 2000/10000 is unverified. If it does,
  `ingest_hyperliquid_ledger` could silently under-fetch for a very active wallet. Upgrade
  path if this bites: add the same "did we get a full page" completeness check
  `fetch_hyperliquid_fills` already has.
- **Reconciliation only covers the gap's specific base asset.** A gap caused by an
  `accountClassTransfer`/`deposit`/`withdraw` (USDC-only events) reconciling a *token*
  position gap is structurally impossible under this design (by definition, those events
  don't move token balances) -- correctly excluded, not a bug, but worth stating explicitly
  since the table above normalizes them anyway.
