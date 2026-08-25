# Trader/Wallet Mining — Ledger Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `reconstruct_trades`' position-continuity guard would fire on a spot-market
gap, check the ingested Hyperliquid ledger (deposits, withdrawals, transfers, airdrops,
staking) for events that explain it; accept the reconciled baseline if they do, still
hard-fail if they don't.

**Architecture:** Ingest-up-front, not fetch-on-gap (confirmed with the user). A new
`RawLedgerEvent` table, populated by `trader-import` alongside fills. A new `ledger.py`
normalizes six observed event types into a `(token, signed_delta)` shape and sums them
over a time window. `reconstruct_trades` gains an optional `reconcile` callable
(injected, so it stays a pure, DB-free function) that engine.py's caller wires to the
ledger.

**Tech Stack:** Python, SQLAlchemy 2.0 declarative ORM, ccxt (`fetch_ledger`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-trader-mining-ledger-reconciliation-design.md`

## Global Constraints

- `RawLedgerEvent.info_json` stores the raw ledger entry's `info` dict verbatim (matches
  `RawFill.payload_json`'s existing precedent) -- confirmed necessary because ccxt's own
  unified top-level `amount`/`currency` fields are `None` for most event types in real
  captured data (see spec's "Event type survey" table). Never parse the unified fields for
  the delta itself; always parse `info["delta"]`.
- `RawLedgerEvent.event_id` (the ledger entry's own `id`, a transaction hash string) is the
  dedup key -- `unique=True`, mirroring `RawFill.tid`.
- `reconstruct_trades` stays DB-free -- no `Session` import, no import of `ledger.py`. The
  `reconcile` parameter is `Callable[[str, datetime, datetime], Decimal] | None` (asset
  ticker, window_start, window_end) -> summed signed delta. `reconstruct_and_persist_trades`
  is the only caller that wires a real one, via `functools.partial`. Both `reconcile` and
  `reconciled_gaps` default to `None` -- every existing caller and test that doesn't pass
  them keeps today's hard-fail-only behavior unchanged.
- All new Decimal-sensitive money/quantity code follows the existing file's convention:
  `Decimal(str(x))` at the point of use, never raw float arithmetic.
- Every new/changed function keeps the existing files' comment density and "confirmed
  against real data" citation style -- this plan's code blocks already carry the real
  values found during design; keep them in the implementation, don't generalize them away.

---

### Task 1: `RawLedgerEvent` model + shared `symbols.py` helper

**Files:**
- Modify: `research/models.py`
- Create: `research/trader_mining/symbols.py`
- Modify: `research/trader_mining/engine.py` (refactor `_is_base_asset_fee` to use the new
  shared helper -- no behavior change)
- Test: `research/tests/test_models.py`
- Test: `research/tests/trader_mining/test_symbols.py`

**Interfaces:**
- Produces: `research.trader_mining.symbols.base_asset_of(symbol: str) -> str | None`
- Produces: `research.models.RawLedgerEvent` (columns: `id`, `trader`, `event_id`,
  `event_type`, `timestamp`, `info_json`, `retrieved_at`)

- [ ] **Step 1: Write the failing test for `base_asset_of`**

```python
# research/tests/trader_mining/test_symbols.py
from research.trader_mining.symbols import base_asset_of


def test_parses_base_from_slash_separated_symbol():
    assert base_asset_of("HYPE/USDC") == "HYPE"


def test_parses_base_from_perp_symbol_with_settle_suffix():
    assert base_asset_of("BTC/USDC:USDC") == "BTC"


def test_returns_none_for_unparsable_raw_index_symbol():
    assert base_asset_of("@705") is None
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest research/tests/trader_mining/test_symbols.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.trader_mining.symbols'`

- [ ] **Step 3: Create `symbols.py`**

```python
# research/trader_mining/symbols.py
"""Shared symbol parsing used by both fee-currency classification (engine.py) and
ledger-event reconciliation (ledger.py) -- kept in one place so the two don't drift on
what "base asset" means for a given trading symbol."""

from __future__ import annotations


def base_asset_of(symbol: str) -> str | None:
    """The base asset ticker parsed from the part of `symbol` before "/" (e.g. "HYPE"
    from "HYPE/USDC" or "BTC" from "BTC/USDC:USDC"). None for the handful of unparsable
    raw internal Hyperliquid index symbols observed in real data (e.g. "@705", no "/")."""
    return symbol.split("/")[0] if "/" in symbol else None
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest research/tests/trader_mining/test_symbols.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Refactor `engine.py` to use the shared helper (no behavior change)**

In `research/trader_mining/engine.py`, add the import and replace the inline parse:

```python
from research.trader_mining.symbols import base_asset_of
```

```python
def _is_base_asset_fee(fill: NormalizedFill) -> bool:
    """... (docstring unchanged) ..."""
    base = base_asset_of(fill.symbol)
    if base is not None:
        return fill.fee_currency == base
    return fill.fee_currency not in _KNOWN_QUOTE_CURRENCIES
```

- [ ] **Step 6: Run the full existing engine test suite, verify no regression**

Run: `pytest research/tests/trader_mining/test_engine.py -q`
Expected: PASS, same count as before this task (32 passed, 1 skipped)

- [ ] **Step 7: Write the failing test for `RawLedgerEvent`**

```python
# research/tests/test_models.py -- add to the existing file
from research.models import RawLedgerEvent


def test_raw_ledger_event_round_trips(tmp_path):
    from datetime import UTC, datetime
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from research.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)

    session.add(
        RawLedgerEvent(
            trader="0xAAA",
            event_id="0xdeadbeef",
            event_type="spotTransfer",
            timestamp=datetime(2024, 11, 29, 10, 2, 32, tzinfo=UTC),
            info_json='{"type": "spotTransfer", "token": "HYPE", "amount": "62264.0"}',
            retrieved_at=datetime.now(UTC),
        )
    )
    session.commit()

    row = session.query(RawLedgerEvent).one()
    assert row.event_id == "0xdeadbeef"
    assert row.event_type == "spotTransfer"
    assert "62264.0" in row.info_json
```

- [ ] **Step 8: Run test, verify it fails**

Run: `pytest research/tests/test_models.py::test_raw_ledger_event_round_trips -v`
Expected: FAIL with `ImportError: cannot import name 'RawLedgerEvent'`

- [ ] **Step 9: Add `RawLedgerEvent` to `research/models.py`**

```python
class RawLedgerEvent(Base):
    """One row per ccxt fetch_ledger entry, near-verbatim -- info_json preserves the
    raw payload (matches RawFill's existing precedent), since real captured data shows
    ccxt's own unified top-level fields (amount/currency) are unreliable across event
    types (deposit/spotTransfer/spotGenesis/cStakingTransfer/send all shape their real
    delta differently -- see docs/superpowers/specs/2026-08-25-trader-mining-ledger-
    reconciliation-design.md's event type survey) and info["delta"] is the only field
    actually used at reconciliation time."""

    __tablename__ = "raw_ledger_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trader: Mapped[str] = mapped_column(String(120), index=True)
    event_id: Mapped[str] = mapped_column(String(80), unique=True)
    event_type: Mapped[str] = mapped_column(String(40))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    info_json: Mapped[str] = mapped_column(String)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime)
```

- [ ] **Step 10: Run test, verify it passes**

Run: `pytest research/tests/test_models.py -v`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add research/models.py research/trader_mining/symbols.py research/trader_mining/engine.py research/tests/test_models.py research/tests/trader_mining/test_symbols.py
git commit -m "feat(research): add RawLedgerEvent table and shared base_asset_of helper"
```

---

### Task 2: `provider.py` -- fetch the ledger

**Files:**
- Modify: `research/trader_mining/provider.py`
- Test: `research/tests/trader_mining/test_provider.py`

**Interfaces:**
- Consumes: `ccxt.async_support.hyperliquid.fetch_ledger(code, since, limit, params)`
  (confirmed present: `ex.has["fetchLedger"] is True`, signature verified live during
  design)
- Produces: `research.trader_mining.provider.fetch_hyperliquid_ledger(trader: str) ->
  list[dict]` (ccxt unified ledger-entry dicts; each entry's `info` key holds the raw
  Hyperliquid payload)

- [ ] **Step 1: Write the failing test**

```python
# research/tests/trader_mining/test_provider.py -- add to the existing file
from research.trader_mining.provider import fetch_hyperliquid_ledger


async def test_fetch_ledger_forwards_trader_as_user_param(mocker):
    mock_fetch = mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.fetch_ledger",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch("research.trader_mining.provider.ccxt_async.hyperliquid.close", new=AsyncMock())

    await fetch_hyperliquid_ledger(TRADER)

    _, kwargs = mock_fetch.call_args
    assert kwargs["params"]["user"] == TRADER


async def test_fetch_ledger_returns_entries_unchanged(mocker):
    entries = [{"id": "0xabc", "type": "deposit", "timestamp": 1_700_000_000_000, "info": {}}]
    mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.fetch_ledger",
        new=AsyncMock(return_value=entries),
    )
    mocker.patch("research.trader_mining.provider.ccxt_async.hyperliquid.close", new=AsyncMock())

    result = await fetch_hyperliquid_ledger(TRADER)

    assert result == entries
```

(add `from unittest.mock import AsyncMock` if not already imported in the test file --
it already is, per the existing `test_forwards_trader_as_user_param` above it)

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest research/tests/trader_mining/test_provider.py -k fetch_ledger -v`
Expected: FAIL with `ImportError: cannot import name 'fetch_hyperliquid_ledger'`

- [ ] **Step 3: Implement `fetch_hyperliquid_ledger`**

```python
# research/trader_mining/provider.py -- add below fetch_hyperliquid_fills

async def fetch_hyperliquid_ledger(trader: str) -> list[dict]:
    """Fetch trader's full non-funding ledger (deposits, withdrawals, transfers,
    airdrops, staking, etc.) via ccxt's unified fetch_ledger. No pagination loop here,
    unlike fetch_hyperliquid_fills -- ccxt's Hyperliquid fetch_ledger has no documented
    page-size ceiling analogous to fetch_my_trades' 2000/10000, and no real wallet
    ingested during this feature's design exceeded 100 ledger entries. ponytail:
    unconfirmed at scale -- if a very active wallet's ledger silently gets truncated by
    ccxt/the API, this would under-fetch with no warning (unlike
    history_completeness). Upgrade path if this bites: add the same
    "did we get a full page" completeness check fetch_hyperliquid_fills already has.
    """
    exchange = ccxt_async.hyperliquid()
    try:
        return await exchange.fetch_ledger(code=None, params={"user": trader})
    finally:
        await exchange.close()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest research/tests/trader_mining/test_provider.py -k fetch_ledger -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add research/trader_mining/provider.py research/tests/trader_mining/test_provider.py
git commit -m "feat(research): add fetch_hyperliquid_ledger"
```

---

### Task 3: `ledger.py` -- normalize event types into signed token deltas

**Files:**
- Create: `research/trader_mining/ledger.py`
- Create: `research/tests/trader_mining/test_ledger.py`

**Interfaces:**
- Consumes: `research.models.RawLedgerEvent`
- Produces: `research.trader_mining.ledger.signed_token_delta(entry: RawLedgerEvent,
  trader: str) -> tuple[str, Decimal] | None`

All fixture payloads below are the REAL shapes captured live during design (see the
spec's "Event type survey" table) -- not invented.

- [ ] **Step 1: Write the failing tests (one per event type + one unrecognized-type case)**

```python
# research/tests/trader_mining/test_ledger.py
import json
from datetime import UTC, datetime
from decimal import Decimal

from research.models import RawLedgerEvent
from research.trader_mining.ledger import signed_token_delta


TRADER = "0x9794bbbc222b6b93c1417d01aa1ff06d42e5333b"
OTHER = "0xead210997055781f27eeab816cc548673bf6e500"


def _event(event_type: str, info: dict) -> RawLedgerEvent:
    return RawLedgerEvent(
        trader=TRADER,
        event_id="0xtest",
        event_type=event_type,
        timestamp=datetime(2024, 11, 29, tzinfo=UTC),
        info_json=json.dumps(info),
        retrieved_at=datetime.now(UTC),
    )


def test_deposit_is_positive_usdc():
    entry = _event("deposit", {"type": "deposit", "usdc": "288888.0"})
    assert signed_token_delta(entry, TRADER) == ("USDC", Decimal("288888.0"))


def test_withdraw_is_negative_usdc():
    entry = _event("withdraw", {"type": "withdraw", "usdc": "880278.6", "fee": "1.0"})
    assert signed_token_delta(entry, TRADER) == ("USDC", Decimal("-880278.6"))


def test_account_class_transfer_into_spot_is_positive():
    entry = _event(
        "transfer", {"type": "accountClassTransfer", "usdc": "288888.0", "toPerp": False}
    )
    assert signed_token_delta(entry, TRADER) == ("USDC", Decimal("288888.0"))


def test_account_class_transfer_into_perp_is_negative_for_spot():
    entry = _event(
        "transfer", {"type": "accountClassTransfer", "usdc": "288888.0", "toPerp": True}
    )
    assert signed_token_delta(entry, TRADER) == ("USDC", Decimal("-288888.0"))


def test_spot_transfer_sent_by_this_trader_is_negative():
    # the exact real event that explains the 62264.0 HYPE gap found live-testing
    entry = _event(
        "spotTransfer",
        {
            "type": "spotTransfer",
            "token": "HYPE",
            "amount": "62264.0",
            "user": TRADER,
            "destination": OTHER,
        },
    )
    assert signed_token_delta(entry, TRADER) == ("HYPE", Decimal("-62264.0"))


def test_spot_transfer_received_by_this_trader_is_positive():
    entry = _event(
        "spotTransfer",
        {
            "type": "spotTransfer",
            "token": "HYPE",
            "amount": "62264.0",
            "user": OTHER,
            "destination": TRADER,
        },
    )
    assert signed_token_delta(entry, TRADER) == ("HYPE", Decimal("62264.0"))


def test_spot_genesis_is_always_positive():
    entry = _event("spotGenesis", {"type": "spotGenesis", "token": "UP", "amount": "10282.7199104"})
    assert signed_token_delta(entry, TRADER) == ("UP", Decimal("10282.7199104"))


def test_c_staking_transfer_deposit_is_negative():
    entry = _event(
        "cStakingTransfer",
        {"type": "cStakingTransfer", "token": "HYPE", "amount": "500000.0", "isDeposit": True},
    )
    assert signed_token_delta(entry, TRADER) == ("HYPE", Decimal("-500000.0"))


def test_c_staking_transfer_withdrawal_is_positive():
    entry = _event(
        "cStakingTransfer",
        {"type": "cStakingTransfer", "token": "HYPE", "amount": "500000.0", "isDeposit": False},
    )
    assert signed_token_delta(entry, TRADER) == ("HYPE", Decimal("500000.0"))


def test_send_uses_same_direction_rule_as_spot_transfer():
    entry = _event(
        "send",
        {
            "type": "send",
            "user": TRADER,
            "destination": OTHER,
            "token": "HYPE",
            "amount": "200000.0",
        },
    )
    assert signed_token_delta(entry, TRADER) == ("HYPE", Decimal("-200000.0"))


def test_unrecognized_event_type_returns_none():
    entry = _event("somethingNew", {"type": "somethingNew"})
    assert signed_token_delta(entry, TRADER) is None
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_ledger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.trader_mining.ledger'`

- [ ] **Step 3: Implement `signed_token_delta`**

```python
# research/trader_mining/ledger.py
"""Normalizes ccxt fetch_ledger entries (research.trader_mining.provider's
fetch_hyperliquid_ledger, persisted as RawLedgerEvent) into a signed per-token position
delta, for reconciling gaps research.trader_mining.engine's position-continuity guard
would otherwise hard-fail on. See docs/superpowers/specs/2026-08-25-trader-mining-
ledger-reconciliation-design.md's "Event type survey" for the real captured payload
shapes this table is built from -- info["delta"] is the only reliable field; ccxt's own
unified top-level amount/currency fields are None for most of these event types in real
data.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from research.models import RawLedgerEvent


def signed_token_delta(entry: RawLedgerEvent, trader: str) -> tuple[str, Decimal] | None:
    """(token, signed_delta) this ledger entry applied to `trader`'s balance, from
    `trader`'s own perspective. None for an unrecognized event_type -- NOT an error;
    an unmodeled event type just can't help explain a gap (the gap still hard-fails if
    nothing else explains it either)."""
    delta = json.loads(entry.info_json).get("delta", {})
    event_type = entry.event_type

    if event_type == "deposit":
        return ("USDC", Decimal(delta["usdc"]))
    if event_type == "withdraw":
        return ("USDC", -Decimal(delta["usdc"]))
    if event_type == "transfer" and delta.get("type") == "accountClassTransfer":
        sign = 1 if delta.get("toPerp") is False else -1
        return ("USDC", sign * Decimal(delta["usdc"]))
    if event_type in ("spotTransfer", "send"):
        sign = -1 if delta.get("user") == trader else 1
        return (delta["token"], sign * Decimal(delta["amount"]))
    if event_type == "spotGenesis":
        return (delta["token"], Decimal(delta["amount"]))
    if event_type == "cStakingTransfer":
        sign = -1 if delta.get("isDeposit") else 1
        return (delta["token"], sign * Decimal(delta["amount"]))
    return None
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_ledger.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add research/trader_mining/ledger.py research/tests/trader_mining/test_ledger.py
git commit -m "feat(research): normalize ledger event types into signed token deltas"
```

---

### Task 4: `ingestion.py` -- persist the ledger

**Files:**
- Modify: `research/trader_mining/ingestion.py`
- Test: `research/tests/trader_mining/test_ingestion.py`

**Interfaces:**
- Consumes: `research.trader_mining.provider.fetch_hyperliquid_ledger`
- Produces: `research.trader_mining.ingestion.ingest_hyperliquid_ledger(session: Session,
  trader: str) -> LedgerIngestResult` (fields: `n_fetched: int`, `n_new: int`)

- [ ] **Step 1: Write the failing tests**

```python
# research/tests/trader_mining/test_ingestion.py -- add to the existing file
from research.models import RawLedgerEvent
from research.trader_mining.ingestion import ingest_hyperliquid_ledger


def _ledger_entry(event_id: str, ts_ms: int = 1_700_000_000_000) -> dict:
    return {
        "id": event_id,
        "timestamp": ts_ms,
        "type": "deposit",
        "info": {"time": ts_ms, "hash": event_id, "delta": {"type": "deposit", "usdc": "100.0"}},
    }


def test_ledger_first_import_populates_table(mocker):
    session = _memory_session()
    mocker.patch(
        "research.trader_mining.ingestion.fetch_hyperliquid_ledger",
        new=AsyncMock(return_value=[_ledger_entry("0xa"), _ledger_entry("0xb")]),
    )

    result = ingest_hyperliquid_ledger(session, TRADER)

    assert result.n_fetched == 2
    assert result.n_new == 2
    assert session.query(RawLedgerEvent).count() == 2


def test_ledger_rerun_is_idempotent_by_event_id(mocker):
    session = _memory_session()
    mocker.patch(
        "research.trader_mining.ingestion.fetch_hyperliquid_ledger",
        new=AsyncMock(return_value=[_ledger_entry("0xa")]),
    )
    ingest_hyperliquid_ledger(session, TRADER)

    result = ingest_hyperliquid_ledger(session, TRADER)

    assert result.n_new == 0
    assert session.query(RawLedgerEvent).count() == 1


def test_ledger_normalized_fields_mapped_correctly(mocker):
    session = _memory_session()
    mocker.patch(
        "research.trader_mining.ingestion.fetch_hyperliquid_ledger",
        new=AsyncMock(return_value=[_ledger_entry("0xa")]),
    )

    ingest_hyperliquid_ledger(session, TRADER)

    row = session.query(RawLedgerEvent).one()
    assert row.trader == TRADER
    assert row.event_id == "0xa"
    assert row.event_type == "deposit"
    assert "100.0" in row.info_json
```

(`_memory_session`, `TRADER`, and `from unittest.mock import AsyncMock` already exist at
the top of this file, reused from the fills tests above.)

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_ingestion.py -k ledger -v`
Expected: FAIL with `ImportError: cannot import name 'ingest_hyperliquid_ledger'`

- [ ] **Step 3: Implement `ingest_hyperliquid_ledger`**

```python
# research/trader_mining/ingestion.py -- add imports and this function
from research.models import NormalizedFill, RawFill, RawLedgerEvent
from research.trader_mining.provider import fetch_hyperliquid_fills, fetch_hyperliquid_ledger


@dataclass
class LedgerIngestResult:
    n_fetched: int
    n_new: int


def ingest_hyperliquid_ledger(session: Session, trader: str) -> LedgerIngestResult:
    """Fetch trader's non-funding ledger and persist any not already stored (by
    event_id), matching ingest_hyperliquid_fills' idempotent-by-tid pattern."""
    entries = asyncio.run(fetch_hyperliquid_ledger(trader))
    retrieved_at = datetime.now(UTC)

    existing_ids = {
        eid
        for (eid,) in session.query(RawLedgerEvent.event_id)
        .filter(RawLedgerEvent.trader == trader)
        .all()
    }

    n_new = 0
    for entry in entries:
        event_id = entry["id"]
        if event_id in existing_ids:
            continue
        n_new += 1
        existing_ids.add(event_id)
        session.add(
            RawLedgerEvent(
                trader=trader,
                event_id=event_id,
                event_type=entry["type"],
                timestamp=datetime.fromtimestamp(entry["timestamp"] / 1000, tz=UTC),
                info_json=json.dumps(entry["info"]),
                retrieved_at=retrieved_at,
            )
        )

    session.commit()
    return LedgerIngestResult(n_fetched=len(entries), n_new=n_new)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_ingestion.py -v`
Expected: PASS, all tests (existing fill-ingestion tests + 3 new ledger ones)

- [ ] **Step 5: Commit**

```bash
git add research/trader_mining/ingestion.py research/tests/trader_mining/test_ingestion.py
git commit -m "feat(research): add ingest_hyperliquid_ledger"
```

---

### Task 5: Reconciliation in `engine.py`

**Files:**
- Modify: `research/trader_mining/ledger.py` (add `reconciliation_deltas`)
- Modify: `research/trader_mining/engine.py` (`reconcile`/`reconciled_gaps` params,
  `_next_running_position` replacing `_check_position_continuity`)
- Test: `research/tests/trader_mining/test_ledger.py`
- Test: `research/tests/trader_mining/test_engine.py`

**Interfaces:**
- Produces: `research.trader_mining.ledger.reconciliation_deltas(session: Session, trader:
  str, asset: str, window_start: datetime, window_end: datetime) -> Decimal`
- Modifies: `research.trader_mining.engine.reconstruct_trades(trader, symbol, fills,
  reconcile: Callable[[str, datetime, datetime], Decimal] | None = None, reconciled_gaps:
  list[str] | None = None) -> list[ReconstructedTrade]` -- both new params optional,
  default `None`, existing callers/tests unaffected.

- [ ] **Step 1: Write the failing test for `reconciliation_deltas`**

```python
# research/tests/trader_mining/test_ledger.py -- add to the existing file
from datetime import timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as ORMSession

from research.models import Base
from research.trader_mining.ledger import reconciliation_deltas


def _memory_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return ORMSession(engine)


def test_reconciliation_deltas_sums_matching_asset_in_window():
    session = _memory_session()
    session.add(_event("spotTransfer", {
        "type": "spotTransfer", "token": "HYPE", "amount": "62264.0",
        "user": TRADER, "destination": OTHER,
    }))
    session.add(_event("deposit", {"type": "deposit", "usdc": "100.0"}))  # different asset
    session.commit()

    total = reconciliation_deltas(
        session, TRADER, "HYPE",
        datetime(2024, 11, 29, 9, 53, 54, tzinfo=UTC),
        datetime(2024, 11, 29, 10, 7, 13, tzinfo=UTC),
    )

    assert total == Decimal("-62264.0")


def test_reconciliation_deltas_ignores_events_outside_window():
    session = _memory_session()
    session.add(_event("spotTransfer", {
        "type": "spotTransfer", "token": "HYPE", "amount": "62264.0",
        "user": TRADER, "destination": OTHER,
    }))
    session.commit()

    total = reconciliation_deltas(
        session, TRADER, "HYPE",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    )

    assert total == Decimal("0")
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_ledger.py -k reconciliation_deltas -v`
Expected: FAIL with `ImportError: cannot import name 'reconciliation_deltas'`

- [ ] **Step 3: Implement `reconciliation_deltas`**

```python
# research/trader_mining/ledger.py -- add below signed_token_delta

def reconciliation_deltas(
    session: Session, trader: str, asset: str, window_start: datetime, window_end: datetime
) -> Decimal:
    """Sum signed_token_delta for `asset`, over ledger events strictly between
    window_start and window_end -- the two fills whose position mismatch triggered a
    reconciliation check in research.trader_mining.engine."""
    events = (
        session.query(RawLedgerEvent)
        .filter(
            RawLedgerEvent.trader == trader,
            RawLedgerEvent.timestamp > window_start,
            RawLedgerEvent.timestamp < window_end,
        )
        .all()
    )
    total = Decimal(0)
    for event in events:
        parsed = signed_token_delta(event, trader)
        if parsed is not None and parsed[0] == asset:
            total += parsed[1]
    return total
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_ledger.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Write the failing tests for engine.py's reconciliation integration**

```python
# research/tests/trader_mining/test_engine.py -- add to the existing file

def test_reconciled_gap_does_not_raise_and_continues_reconstruction():
    """The exact scenario that motivated this feature: a spotTransfer moved HYPE out of
    the wallet between two fills. reconcile returns -62264.0 (matching the real event),
    which explains the discrepancy exactly."""
    fills = [
        _fill(1, "buy", 100.0, 10.0, position=0.0, ts=T0, symbol="HYPE/USDC"),
        _fill(
            2,
            "sell",
            110.0,
            10.0,
            position=-62254.0,  # 10.0 - 62264.0, the un-reconciled real gap shape
            ts=T0 + timedelta(hours=1),
            closed_pnl=100.0,
            direction="Close Long",
            symbol="HYPE/USDC",
        ),
    ]

    def fake_reconcile(asset, window_start, window_end):
        assert asset == "HYPE"
        return Decimal("-62264.0")

    reconciled_gaps: list[str] = []
    trades = reconstruct_trades(
        TRADER, "HYPE/USDC", fills, reconcile=fake_reconcile, reconciled_gaps=reconciled_gaps
    )

    assert len(trades) == 1
    assert len(reconciled_gaps) == 1
    assert "HYPE/USDC" in reconciled_gaps[0]


def test_unreconciled_gap_still_raises_even_with_reconcile_supplied():
    """reconcile is consulted but doesn't explain the gap -- must still hard-fail, never
    silently proceed."""
    fills = [
        _fill(1, "buy", 100.0, 10.0, position=0.0, ts=T0, symbol="HYPE/USDC"),
        _fill(
            2,
            "sell",
            110.0,
            10.0,
            position=999.0,  # nothing explains this
            ts=T0 + timedelta(hours=1),
            closed_pnl=100.0,
            direction="Close Long",
            symbol="HYPE/USDC",
        ),
    ]

    def fake_reconcile(asset, window_start, window_end):
        return Decimal("0")  # ledger has nothing for this window

    with pytest.raises(ValueError, match="position gap"):
        reconstruct_trades(TRADER, "HYPE/USDC", fills, reconcile=fake_reconcile)
```

(add `from decimal import Decimal` to this test file's imports if not already present)

- [ ] **Step 6: Run tests, verify they fail**

Run: `pytest research/tests/trader_mining/test_engine.py -k reconcil -v`
Expected: FAIL -- `reconstruct_trades() got an unexpected keyword argument 'reconcile'`

- [ ] **Step 7: Implement the reconciliation integration in `engine.py`**

Replace `_check_position_continuity` with `_next_running_position`:

```python
# research/trader_mining/engine.py
from collections.abc import Callable

ReconcileFn = Callable[[str, datetime, datetime], Decimal]
_POSITION_EPSILON = Decimal("1e-8")


def _next_running_position(
    fill: NormalizedFill,
    end_position: Decimal,
    next_fill: NormalizedFill,
    reconcile: ReconcileFn | None,
    reconciled_gaps: list[str] | None,
) -> Decimal:
    """Position to carry forward as running_position for the fill after `fill`.
    Ordinarily just `end_position` (our own computation), confirmed against
    next_fill's own reported position. If they don't match and `reconcile` is
    supplied, checks whether summing ledger deltas for this symbol's base asset
    within (fill.timestamp, next_fill.timestamp) explains the gap -- if so, trusts
    next_fill's own reported position going forward (the ledger event moved the
    balance in a way our own fill-only computation has no way to represent).
    Otherwise raises, exactly as before this feature existed."""
    next_position = Decimal(str(next_fill.position))
    discrepancy = next_position - end_position
    if abs(discrepancy) <= _POSITION_EPSILON:
        return end_position

    if reconcile is not None:
        asset = base_asset_of(fill.symbol)
        if asset is not None:
            ledger_delta = reconcile(asset, fill.timestamp, next_fill.timestamp)
            if abs((end_position + ledger_delta) - next_position) <= _POSITION_EPSILON:
                if reconciled_gaps is not None:
                    reconciled_gaps.append(
                        f"{fill.symbol}: reconciled a {discrepancy} position gap "
                        f"between fill tid={fill.tid} and tid={next_fill.tid} via "
                        f"ingested ledger events ({ledger_delta} {asset})"
                    )
                return next_position

    ledger_note = "" if reconcile is None else ", and the ingested ledger does not explain it"
    raise ValueError(
        f"position gap: fill tid={fill.tid} ends at position {end_position} but next "
        f"fill tid={next_fill.tid} starts at position {next_position} -- likely missing "
        f"fills between them (ingestion gap or provider truncation){ledger_note}"
    )
```

Update `reconstruct_trades`' signature and its two `running_position = end_position`
assignment sites:

```python
def reconstruct_trades(
    trader: str,
    symbol: str,
    fills: list[NormalizedFill],
    reconcile: ReconcileFn | None = None,
    reconciled_gaps: list[str] | None = None,
) -> list[ReconstructedTrade]:
    """... (existing docstring unchanged) ..."""
    if not fills:
        return []

    for fill in fills:
        if fill.quantity <= 0:
            raise ValueError(f"fill tid={fill.tid} has non-positive quantity {fill.quantity}")

    trades: list[ReconstructedTrade] = []
    running_position = Decimal(str(fills[0].position))
    trade: _TradeState | None = None

    if running_position != 0:
        trade = _TradeState(
            is_truncated_start=True,
            fallback_price=Decimal(str(fills[0].price)),
            fallback_timestamp=fills[0].timestamp,
            direction="long" if running_position > 0 else "short",
        )

    for i, fill in enumerate(fills):
        qty = Decimal(str(fill.quantity))
        end_position = _end_position(running_position, fill)

        next_running_position = end_position
        if i + 1 < len(fills):
            next_running_position = _next_running_position(
                fill, end_position, fills[i + 1], reconcile, reconciled_gaps
            )

        if trade is None:
            trade = _TradeState(
                is_truncated_start=False,
                fallback_price=Decimal(str(fill.price)),
                fallback_timestamp=fill.timestamp,
                direction="long" if end_position > 0 else "short",
            )

        is_reversal = (
            running_position != 0
            and end_position != 0
            and (end_position > 0) != (running_position > 0)
        )

        if is_reversal:
            close_qty = abs(running_position)
            open_qty = qty - close_qty
            fee = _fee_in_quote_currency(fill, Decimal(str(fill.fee)))
            close_fee = fee * (close_qty / qty)
            trade.add_exit(fill, close_qty, Decimal(str(fill.closed_pnl)), close_fee)
            trades.append(trade.finalize(trader, symbol))

            trade = _TradeState(
                is_truncated_start=False,
                fallback_price=Decimal(str(fill.price)),
                fallback_timestamp=fill.timestamp,
                direction="long" if end_position > 0 else "short",
            )
            trade.add_entry(fill, open_qty, fee - close_fee)
            running_position = next_running_position
            continue

        fee = _fee_in_quote_currency(fill, Decimal(str(fill.fee)))
        if abs(end_position) > abs(running_position):
            trade.add_entry(fill, qty, fee)
        else:
            trade.add_exit(fill, qty, Decimal(str(fill.closed_pnl)), fee)

        running_position = next_running_position

        if running_position == 0:
            trades.append(trade.finalize(trader, symbol))
            trade = None

    return trades
```

Delete `_check_position_continuity` entirely (replaced by `_next_running_position`).

- [ ] **Step 8: Run tests, verify they pass**

Run: `pytest research/tests/trader_mining/test_engine.py -v`
Expected: PASS, all tests (34 total: existing 32 + 2 new)

- [ ] **Step 9: Commit**

```bash
git add research/trader_mining/ledger.py research/trader_mining/engine.py research/tests/trader_mining/test_ledger.py research/tests/trader_mining/test_engine.py
git commit -m "feat(research): reconcile position gaps against the ingested ledger"
```

---

### Task 6: Wire `cli.py` + real-data validation

**Files:**
- Modify: `research/trader_mining/engine.py` (`ReconstructResult.reconciled_gaps` field)
- Modify: `research/cli.py`
- Test: `research/tests/test_cli.py`

**Interfaces:**
- Modifies: `research.trader_mining.engine.ReconstructResult` gains
  `reconciled_gaps: list[str] = field(default_factory=list)`
- Modifies: `research.trader_mining.engine.reconstruct_and_persist_trades` -- wires
  `functools.partial(reconciliation_deltas, session, trader)` as `reconcile`
- Modifies: `research.cli.py` `trader-import` also calls `ingest_hyperliquid_ledger` and
  prints its counts; `trader-analyze` prints `reconciled_gaps` if any

- [ ] **Step 1: Write the failing test for `ReconstructResult.reconciled_gaps`**

```python
# research/tests/trader_mining/test_engine.py -- add to the existing file
def test_reconstruct_and_persist_trades_reports_reconciled_gaps():
    session = _memory_session()
    # HYPE fills mirroring the real gap, plus the ledger event that explains it
    from research.models import RawLedgerEvent
    import json
    from datetime import UTC

    _add_normalized_fill(
        session, tid=1, symbol="HYPE/USDC", side="buy", position=0.0, timestamp=T0
    )
    _add_normalized_fill(
        session,
        tid=2,
        symbol="HYPE/USDC",
        side="sell",
        position=-62254.0,
        timestamp=T0 + timedelta(hours=1),
        closed_pnl=100.0,
        direction="Close Long",
    )
    session.add(
        RawLedgerEvent(
            trader=TRADER,
            event_id="0xdeadbeef",
            event_type="spotTransfer",
            timestamp=T0 + timedelta(minutes=30),
            info_json=json.dumps(
                {"type": "spotTransfer", "token": "HYPE", "amount": "62264.0",
                 "user": TRADER, "destination": "0xother"}
            ),
            retrieved_at=datetime.now(UTC),
        )
    )
    session.commit()

    result = reconstruct_and_persist_trades(session, TRADER)

    assert result.n_trades == 1
    assert len(result.reconciled_gaps) == 1
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest research/tests/trader_mining/test_engine.py -k reconciled_gaps -v`
Expected: FAIL -- `TypeError: ReconstructResult.__init__() got an unexpected keyword
argument` or `AttributeError: 'ReconstructResult' object has no attribute
'reconciled_gaps'` (raised without the fix, since the gap isn't reconciled yet)

- [ ] **Step 3: Wire reconciliation into `reconstruct_and_persist_trades`**

```python
# research/trader_mining/engine.py
from functools import partial

from research.trader_mining.ledger import reconciliation_deltas


@dataclass
class ReconstructResult:
    n_trades: int
    symbols: list[str]
    reconciled_gaps: list[str] = field(default_factory=list)


def reconstruct_and_persist_trades(
    session: Session, trader: str, symbol: str | None = None
) -> ReconstructResult:
    """... (existing docstring unchanged) ..."""
    symbols_query = session.query(NormalizedFill.symbol).filter(NormalizedFill.trader == trader)
    if symbol is not None:
        symbols_query = symbols_query.filter(NormalizedFill.symbol == symbol)
    symbols = sorted({s for (s,) in symbols_query.distinct().all()})

    reconcile = partial(reconciliation_deltas, session, trader)
    reconciled_gaps: list[str] = []
    total_trades = 0
    try:
        for sym in symbols:
            fills = (
                session.query(NormalizedFill)
                .filter(NormalizedFill.trader == trader, NormalizedFill.symbol == sym)
                .order_by(
                    NormalizedFill.timestamp,
                    func.abs(NormalizedFill.position),
                    NormalizedFill.tid,
                )
                .all()
            )
            session.query(ReconstructedTrade).filter(
                ReconstructedTrade.trader == trader, ReconstructedTrade.symbol == sym
            ).delete()

            new_trades = reconstruct_trades(
                trader, sym, fills, reconcile=reconcile, reconciled_gaps=reconciled_gaps
            )
            for t in new_trades:
                session.add(t)
            total_trades += len(new_trades)
    except Exception:
        session.rollback()
        raise

    session.commit()
    return ReconstructResult(
        n_trades=total_trades, symbols=symbols, reconciled_gaps=reconciled_gaps
    )
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest research/tests/trader_mining/test_engine.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Write the failing CLI tests**

```python
# research/tests/test_cli.py -- add to the existing file
def test_trader_import_command_also_ingests_ledger(mocker, capsys):
    from research.trader_mining.ingestion import IngestResult, LedgerIngestResult

    mocker.patch(
        "research.cli.ingest_hyperliquid_fills",
        return_value=IngestResult(n_fetched=5, n_new=3, history_completeness="complete"),
    )
    mock_ledger = mocker.patch(
        "research.cli.ingest_hyperliquid_ledger",
        return_value=LedgerIngestResult(n_fetched=2, n_new=2),
    )
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    exit_code = main(["trader-import", "--trader", "0x0000000000000000000000000000000000000000"])

    assert mock_ledger.called
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "n_ledger_events_new: 2" in captured.out


def test_trader_analyze_command_prints_reconciled_gaps_when_present(mocker, capsys):
    from research.trader_mining.engine import ReconstructResult

    mocker.patch(
        "research.cli.reconstruct_and_persist_trades",
        return_value=ReconstructResult(
            n_trades=1,
            symbols=["HYPE/USDC"],
            reconciled_gaps=["HYPE/USDC: reconciled a -62264.0 position gap ..."],
        ),
    )
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    exit_code = main(["trader-analyze", "--trader", "0x0000000000000000000000000000000000000000"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "reconciled_gaps" in captured.out
    assert "HYPE/USDC" in captured.out
```

- [ ] **Step 6: Run tests, verify they fail**

Run: `pytest research/tests/test_cli.py -k "ledger or reconciled_gaps" -v`
Expected: FAIL -- `ImportError` (`ingest_hyperliquid_ledger`/`LedgerIngestResult` not
imported in `cli.py`) and missing output lines

- [ ] **Step 7: Wire `cli.py`**

```python
# research/cli.py -- add import
from research.trader_mining.ingestion import ingest_hyperliquid_fills, ingest_hyperliquid_ledger
```

In the `trader-import` branch:

```python
    elif args.command == "trader-import":
        engine = get_engine(args.db_path)
        session = get_session(engine)
        since = datetime.fromisoformat(args.since).replace(tzinfo=UTC) if args.since else None
        ingest_result = ingest_hyperliquid_fills(session, trader=args.trader, since=since)
        print(f"n_fetched: {ingest_result.n_fetched}")
        print(f"n_new: {ingest_result.n_new}")
        print(f"history_completeness: {ingest_result.history_completeness}")
        if ingest_result.history_completeness == "truncated_by_provider_limit":
            print(
                "WARNING: history_completeness=truncated_by_provider_limit -- "
                "Hyperliquid's 10,000-fill ceiling was reached; earlier fills may exist "
                "but are not retrievable via this endpoint."
            )
        ledger_result = ingest_hyperliquid_ledger(session, trader=args.trader)
        print(f"n_ledger_events_fetched: {ledger_result.n_fetched}")
        print(f"n_ledger_events_new: {ledger_result.n_new}")
        return 0
```

In the `trader-analyze` branch:

```python
    elif args.command == "trader-analyze":
        engine = get_engine(args.db_path)
        session = get_session(engine)
        analyze_result = reconstruct_and_persist_trades(
            session, trader=args.trader, symbol=args.symbol
        )
        print(f"n_trades: {analyze_result.n_trades}")
        print(f"symbols: {', '.join(analyze_result.symbols)}")
        if analyze_result.reconciled_gaps:
            print(f"reconciled_gaps ({len(analyze_result.reconciled_gaps)}):")
            for gap in analyze_result.reconciled_gaps:
                print(f"  {gap}")
        return 0
```

- [ ] **Step 8: Run tests, verify they pass**

Run: `pytest research/tests/test_cli.py -v`
Expected: PASS, all tests

- [ ] **Step 9: Run the full targeted suite**

Run: `pytest research/tests/trader_mining/ research/tests/test_models.py research/tests/test_cli.py -q`
Expected: PASS, no regressions vs. the count going into this task

- [ ] **Step 10: Lint and typecheck**

Run: `ruff check research/ && ruff format --check research/ && mypy research/`
Expected: clean

- [ ] **Step 11: Commit**

```bash
git add research/trader_mining/engine.py research/cli.py research/tests/trader_mining/test_engine.py research/tests/test_cli.py
git commit -m "feat(research): wire ledger ingestion and reconciliation reporting into the CLI"
```

- [ ] **Step 12: Real-data validation (the acceptance test for this whole plan)**

Against a scratch db path (not the repo's own `user_data/research.sqlite`):

```bash
python -m research.cli trader-import --trader 0x9794bbbc222b6b93c1417d01aa1ff06d42e5333b --db-path /tmp/ledger_validation.sqlite
python -m research.cli trader-analyze --trader 0x9794bbbc222b6b93c1417d01aa1ff06d42e5333b --symbol HYPE/USDC --db-path /tmp/ledger_validation.sqlite
```

Expected: no `ValueError: position gap` (previously raised on this exact wallet/symbol
before this plan), `reconciled_gaps` output includes a line mentioning the 62264.0 HYPE
`spotTransfer`. If it does NOT reconcile, do not weaken the epsilon or guess -- debug via
`reconciliation_deltas` directly against the stored `RawLedgerEvent` rows for that window
first (matches this plan's real-data-first debugging precedent from Release 2's own tid-
sort-order bug).

---

## Self-review notes (for the implementer)

- Every task's new/changed function keeps the file's existing comment density -- match it
  when writing the real implementation, not just the code shown here (this plan's code
  blocks are complete and copy-pasteable, but double check against the actual current file
  state before pasting, in case Task N's file changed since this plan was written).
- Run `superpowers:requesting-code-review` after Task 6, same pattern as Release 1/2 and
  the two prior bugfix PRs -- this plan does not replace that step.
