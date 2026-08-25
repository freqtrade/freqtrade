# Trader/Wallet Mining Release 1 (Hyperliquid Ingestion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read-only ingestion of one public Hyperliquid wallet's fill history into local
storage -- reproducibly, idempotently, with the provider's real limits (10k-fill ceiling,
weighted rate limit) surfaced honestly rather than silently truncated.

**Architecture:** Three new pieces -- `research/trader_mining/provider.py` (thin
`ccxt.async_support.hyperliquid` wrapper, paginated), `research/trader_mining/ingestion.py`
(idempotent persistence into two new tables on the existing shared `Base`), and a
`trader-import` subcommand on the existing `research/cli.py`. No new database, no new
storage module, no provider abstraction (concrete Hyperliquid-only).

**Tech Stack:** Python, `ccxt.async_support` (already a dependency), SQLAlchemy 2.0
declarative (`research/models.py`'s existing `Base`), `asyncio.run` as the one sync/async
boundary (matches nothing else in `research/` going async beyond this), pytest with a real
captured Hyperliquid response as a frozen fixture (no mocking of ccxt's own parser).

**Spec:** `docs/superpowers/specs/2026-08-25-trader-mining-release-1-design.md`

## Global Constraints

- `FetchFillsResult` (`research/trader_mining/provider.py`): `trades: list[dict]` (ccxt
  unified trade dicts, each `trade["info"]` is the raw Hyperliquid payload),
  `history_completeness: str` (`"complete"` or `"truncated_by_provider_limit"`).
- `fetch_hyperliquid_fills(trader: str, since: datetime | None = None) -> FetchFillsResult`
  -- async. Constructs `ccxt.async_support.hyperliquid()` fresh, no credentials.
- **Pagination correctness (load-bearing, do not "simplify"):** advance `since` to the
  **last returned fill's own timestamp**, not `+1ms` -- an earlier `+1ms` design was
  corrected during spec review because a page boundary can have multiple fills sharing one
  timestamp, and `+1ms` risks silently skipping one of them. The resulting duplicate
  fill(s) refetched at each boundary are expected and harmless; `ingestion.py`'s `tid`-based
  dedup discards them before insert. Stop paginating when a page returns fewer fills than
  requested (`"complete"`) or the running total reaches 10,000 (`"truncated_by_provider_limit"`).
- **`fetch_my_trades` calls `load_markets()` internally if `exchange.markets is None`,
  which itself calls `publicPostInfo` for a *different* request type than fills** --
  confirmed by spiking against the real API. Any test that mocks `publicPostInfo` to return
  fills data must first set `exchange.markets = {}` (an empty, non-`None` dict) so
  `load_markets()` is skipped entirely -- verified working during spec research; omitting
  this makes the mock intercept the markets call too and crash with `KeyError: 'name'` deep
  inside ccxt's HIP-3 market parsing.
- `IngestResult` (`research/trader_mining/ingestion.py`): `n_fetched: int`, `n_new: int`,
  `history_completeness: str`.
- `ingest_hyperliquid_fills(session: Session, trader: str, since: datetime | None = None) ->
  IngestResult` -- calls `provider.fetch_hyperliquid_fills` via `asyncio.run`, dedupes
  against existing `RawFill.tid` for this run, inserts only new fills into both `RawFill`
  and `NormalizedFill` in one transaction.
- `research/models.py` gains `RawFill` and `NormalizedFill` on the existing `Base` --
  `research/db.py` needs **zero changes**; its `Base.metadata.create_all(engine)` already
  picks up any table registered on `Base` before `get_engine()` runs.
- `research/cli.py` gains a `trader-import` subcommand: `--trader` (required), `--since`
  (optional, `YYYY-MM-DD`), `--db-path` (default `user_data/research.sqlite`, matching
  `gate`'s own flag). Always exits `0` -- ingestion has no pass/fail verdict.
- No provider abstraction / `TraderDataProvider` Protocol -- concrete Hyperliquid module
  only, per spec.
- Test fixture: `research/tests/fixtures/hyperliquid_user_fills_raw.json` -- a real captured
  8-fill response (Hyperliquid's zero address, `0x000...000`; every fill is `dir:
  "Settlement"` on an exotic pair -- sufficient for schema fidelity, not representative of a
  normal trader; do not treat as evidence about typical trading behavior).
- Frozen-fixture tests patch `publicPostInfo` (the raw HTTP boundary ccxt itself calls) --
  **never** `fetch_my_trades` directly, which would skip ccxt's real parser and collapse the
  test into a duplicate of the provider-unit-test layer.
- Live canary test is gated by `if not os.environ.get("HYPERLIQUID_LIVE_TEST"):
  pytest.skip(...)` at the top of the test body -- **not** a `pytest.mark`, so it requires no
  change to `pyproject.toml`/CI workflow config and never runs in normal CI.

---

### Task 1: `research/models.py` — `RawFill` and `NormalizedFill` tables

**Files:**
- Modify: `research/models.py` (append two new classes)
- Test: `research/tests/test_models.py` (new)

**Interfaces:**
- Consumes: the existing `Base` (`research/models.py`, already defined).
- Produces: `RawFill`, `NormalizedFill` ORM classes, consumed by Task 3
  (`ingestion.py`).

- [ ] **Step 1: Write the failing test**

Create `research/tests/test_models.py`:

```python
# research/tests/test_models.py
from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from research.models import Base, NormalizedFill, RawFill


def _memory_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_raw_fill_round_trips():
    session = _memory_session()
    session.add(
        RawFill(
            source="hyperliquid",
            trader="0x0000000000000000000000000000000000000000",
            tid=802647614388392,
            payload_json='{"coin": "hyna:XMR", "dir": "Settlement"}',
            retrieved_at=datetime(2026, 8, 25, tzinfo=UTC),
        )
    )
    session.commit()

    row = session.query(RawFill).one()
    assert row.source == "hyperliquid"
    assert row.trader == "0x0000000000000000000000000000000000000000"
    assert row.tid == 802647614388392
    assert row.payload_json == '{"coin": "hyna:XMR", "dir": "Settlement"}'
    assert row.retrieved_at == datetime(2026, 8, 25, tzinfo=UTC)


def test_raw_fill_tid_is_unique():
    session = _memory_session()
    session.add(
        RawFill(
            source="hyperliquid",
            trader="0xAAA",
            tid=1,
            payload_json="{}",
            retrieved_at=datetime(2026, 8, 25, tzinfo=UTC),
        )
    )
    session.commit()
    session.add(
        RawFill(
            source="hyperliquid",
            trader="0xBBB",  # different trader, same tid -- still a conflict
            tid=1,
            payload_json="{}",
            retrieved_at=datetime(2026, 8, 25, tzinfo=UTC),
        )
    )
    with pytest.raises(Exception, match="UNIQUE"):
        session.commit()


def test_normalized_fill_round_trips():
    session = _memory_session()
    session.add(
        NormalizedFill(
            trader="0x0000000000000000000000000000000000000000",
            tid=802647614388392,
            timestamp=datetime(2026, 7, 15, 15, 31, 43, tzinfo=UTC),
            symbol="HYNA-XMR/USDE:USDE",
            side="sell",
            price=331.21,
            quantity=2.387,
            notional=790.59827,
            position=2.387,
            closed_pnl=0.0,
            direction="Settlement",
            crossed=False,
            fee=0.0,
            fee_currency="USDE",
            order_id="496459510818",
        )
    )
    session.commit()

    row = session.query(NormalizedFill).one()
    assert row.trader == "0x0000000000000000000000000000000000000000"
    assert row.tid == 802647614388392
    assert row.symbol == "HYNA-XMR/USDE:USDE"
    assert row.side == "sell"
    assert row.price == 331.21
    assert row.direction == "Settlement"
    assert row.crossed is False
```

Add `import pytest` at the top alongside the other imports (needed for
`test_raw_fill_tid_is_unique`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'RawFill' from 'research.models'`

- [ ] **Step 3: Implement `RawFill` and `NormalizedFill`**

Append to `research/models.py` (after the existing `HealthCheck` class):

```python
class RawFill(Base):
    """One row per raw fill exactly as a provider returned it -- payload_json preserves
    the full original payload verbatim (research.trader_mining.provider's
    trade["info"]), so normalization bugs in NormalizedFill can be investigated against
    real captured data later, per the proposal's explicit requirement."""

    __tablename__ = "raw_fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(40))
    trader: Mapped[str] = mapped_column(String(120), index=True)
    tid: Mapped[int] = mapped_column(Integer, unique=True)
    payload_json: Mapped[str] = mapped_column(String)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime)


class NormalizedFill(Base):
    """One row per fill, mapped to a stable internal shape independent of the upstream
    provider. tid matches a RawFill.tid -- indexed, not a formal ForeignKey, matching
    this file's existing PromotionRecord/HealthCheck convention of a plain indexed
    column rather than a hard constraint."""

    __tablename__ = "normalized_fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trader: Mapped[str] = mapped_column(String(120), index=True)
    tid: Mapped[int] = mapped_column(Integer, index=True, unique=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    symbol: Mapped[str] = mapped_column(String(80))
    side: Mapped[str] = mapped_column(String(10))
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float)
    notional: Mapped[float] = mapped_column(Float)
    position: Mapped[float] = mapped_column(Float)
    closed_pnl: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(40))
    crossed: Mapped[bool] = mapped_column(Boolean)
    fee: Mapped[float] = mapped_column(Float)
    fee_currency: Mapped[str] = mapped_column(String(20))
    order_id: Mapped[str] = mapped_column(String(60))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/test_models.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint and format**

Run: `ruff check research/models.py research/tests/test_models.py` and
`ruff format --check research/models.py research/tests/test_models.py`
Expected: no errors (fix and re-run Step 4 if needed)

- [ ] **Step 6: Commit**

```bash
git add research/models.py research/tests/test_models.py
git commit -m "feat(research): add RawFill/NormalizedFill tables for trader-mining ingestion"
```

---

### Task 2: `research/trader_mining/provider.py` — Hyperliquid fill fetching

**Files:**
- Create: `research/trader_mining/__init__.py` (empty)
- Create: `research/trader_mining/provider.py`
- Create: `research/tests/trader_mining/__init__.py` (empty)
- Test: `research/tests/trader_mining/test_provider.py`

**Interfaces:**
- Consumes: `ccxt.async_support.hyperliquid` (external dependency, already installed).
- Produces: `FetchFillsResult`, `fetch_hyperliquid_fills(trader, since=None)` -- consumed
  by Task 3 (`ingestion.py`).

- [ ] **Step 1: Write the failing tests (provider-unit layer)**

Create `research/trader_mining/__init__.py` and `research/tests/trader_mining/__init__.py`
as empty files.

Create `research/tests/trader_mining/test_provider.py`:

```python
# research/tests/trader_mining/test_provider.py
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from research.trader_mining.provider import fetch_hyperliquid_fills


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hyperliquid_user_fills_raw.json"
)
TRADER = "0x0000000000000000000000000000000000000000"


def _fake_trade(tid: int, timestamp_ms: int) -> dict:
    """A minimal ccxt-shaped unified trade dict, enough for provider.py's own pagination
    logic to operate on -- NOT meant to be schema-realistic (that's the frozen-fixture
    layer's job below)."""
    return {"id": str(tid), "timestamp": timestamp_ms, "info": {"tid": tid}}


async def test_forwards_trader_as_user_param(mocker):
    mock_fetch = mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.fetch_my_trades",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.close", new=AsyncMock()
    )

    await fetch_hyperliquid_fills(TRADER)

    _, kwargs = mock_fetch.call_args
    assert kwargs["params"]["user"] == TRADER


async def test_stops_pagination_on_short_page_reports_complete(mocker):
    # First page: fewer than the 2000-fill page size -- end of history.
    mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.fetch_my_trades",
        new=AsyncMock(return_value=[_fake_trade(1, 1_700_000_000_000)]),
    )
    mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.close", new=AsyncMock()
    )

    result = await fetch_hyperliquid_fills(TRADER)

    assert len(result.trades) == 1
    assert result.history_completeness == "complete"


async def test_paginates_on_full_page_until_short_page(mocker):
    full_page = [_fake_trade(i, 1_700_000_000_000 + i) for i in range(2000)]
    short_page = [_fake_trade(9999, 1_700_002_000_000)]
    mock_fetch = mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.fetch_my_trades",
        new=AsyncMock(side_effect=[full_page, short_page]),
    )
    mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.close", new=AsyncMock()
    )

    result = await fetch_hyperliquid_fills(TRADER)

    assert mock_fetch.await_count == 2
    # second call's `since` is the LAST fill of the first page's own timestamp, not +1ms
    _, second_call_kwargs = mock_fetch.await_args_list[1]
    assert second_call_kwargs["since"] == full_page[-1]["timestamp"]
    assert result.history_completeness == "complete"
    # 2000 (full page) + 1 (short page) fills total -- no fills lost or duplicated in
    # this synthetic scenario (real duplicate-at-boundary handling is ingestion.py's job)
    assert len(result.trades) == 2001


async def test_reports_truncated_at_ten_thousand_fill_ceiling(mocker):
    # Five full 2000-fill pages = 10,000 -- ceiling reached without ever seeing a short page.
    pages = [
        [_fake_trade(p * 2000 + i, 1_700_000_000_000 + p * 2000 + i) for i in range(2000)]
        for p in range(5)
    ]
    mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.fetch_my_trades",
        new=AsyncMock(side_effect=pages),
    )
    mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.close", new=AsyncMock()
    )

    result = await fetch_hyperliquid_fills(TRADER)

    assert len(result.trades) == 10_000
    assert result.history_completeness == "truncated_by_provider_limit"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/trader_mining/test_provider.py -v -k "not fixture and not live"`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'research.trader_mining.provider'`

- [ ] **Step 3: Implement `provider.py`'s core fetch + pagination**

Create `research/trader_mining/provider.py`:

```python
# research/trader_mining/provider.py
"""Hyperliquid research provider: a thin, read-only wrapper around
ccxt.async_support.hyperliquid's fetch_my_trades, using its params.user override to pull
ANY wallet's public, unauthenticated fill history -- not just an authenticated account's.
Confirmed live against the real API before this was written: no apiKey/secret needed at
all. See docs/superpowers/specs/2026-08-25-trader-mining-release-1-design.md for the full
spike writeup, including the exact raw HTTP boundary (publicPostInfo) this module's own
tests patch, and why pagination overlaps by one timestamp instant rather than risking a
skipped fill at a page boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import ccxt.async_support as ccxt_async


PAGE_SIZE = 2000
FILL_CEILING = 10_000


@dataclass
class FetchFillsResult:
    """trades: ccxt unified trade dicts; each trade["info"] is the raw Hyperliquid
    payload, verbatim. history_completeness is "complete" (a page returned fewer than
    PAGE_SIZE fills -- end of available history) or "truncated_by_provider_limit"
    (FILL_CEILING reached without ever seeing a short page -- Hyperliquid's own
    documented per-wallet fill-history ceiling, not a bug in this fetch loop)."""

    trades: list[dict]
    history_completeness: str


async def fetch_hyperliquid_fills(
    trader: str, since: datetime | None = None
) -> FetchFillsResult:
    """Fetch trader's full available fill history from Hyperliquid's public info
    endpoint, paginating until either the provider's own history ends or its 10,000-fill
    ceiling is reached. `since`, when given, is a lower bound in UTC; omit it to fetch
    from the earliest available fill."""
    exchange = ccxt_async.hyperliquid()
    try:
        all_trades: list[dict] = []
        since_ms = int(since.timestamp() * 1000) if since is not None else None
        while True:
            page = await exchange.fetch_my_trades(
                symbol=None, since=since_ms, limit=PAGE_SIZE, params={"user": trader}
            )
            all_trades.extend(page)
            if len(page) < PAGE_SIZE:
                return FetchFillsResult(trades=all_trades, history_completeness="complete")
            if len(all_trades) >= FILL_CEILING:
                return FetchFillsResult(
                    trades=all_trades, history_completeness="truncated_by_provider_limit"
                )
            # Overlap by one timestamp instant rather than +1ms -- a page boundary can
            # have multiple fills sharing the exact same timestamp; +1ms risks silently
            # skipping one. The resulting re-fetched duplicate(s) are harmless here;
            # research.trader_mining.ingestion dedupes by tid before persisting.
            since_ms = page[-1]["timestamp"]
    finally:
        await exchange.close()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/trader_mining/test_provider.py -v -k "not fixture and not live"`
Expected: PASS (4 tests)

- [ ] **Step 5: Write the failing frozen-fixture contract test**

Append to `research/tests/trader_mining/test_provider.py`:

```python
async def test_fixture_frozen_response_parses_through_real_ccxt_parser(mocker):
    """Patches publicPostInfo -- the raw HTTP boundary ccxt itself calls internally
    (research/trader_mining/provider.py's docstring cites where this was confirmed) --
    NOT fetch_my_trades. This exercises ccxt's REAL Hyperliquid parser against a real
    captured response, so it actually catches ccxt/Hyperliquid schema drift; mocking
    fetch_my_trades directly would not."""
    import ccxt.async_support as ccxt_async_module

    with open(FIXTURE_PATH, encoding="utf-8") as f:
        frozen = json.load(f)

    exchange = ccxt_async_module.hyperliquid()
    # fetch_my_trades() calls load_markets() if exchange.markets is None, which itself
    # calls publicPostInfo for a DIFFERENT request type -- pre-seed an empty markets
    # dict so it's skipped (symbol=None below means no real market data is needed).
    # Confirmed necessary by spiking directly against the real API during spec research.
    exchange.markets = {}
    exchange.publicPostInfo = AsyncMock(return_value=frozen)
    try:
        trades = await exchange.fetch_my_trades(
            symbol=None, params={"user": TRADER}
        )
    finally:
        await exchange.close()

    assert len(trades) == len(frozen)
    first = trades[0]
    assert first["id"] == str(frozen[0]["tid"])
    assert first["order"] == str(frozen[0]["oid"])
    assert first["info"]["dir"] == frozen[0]["dir"]
    assert first["info"] == frozen[0]  # raw payload preserved verbatim
```

- [ ] **Step 6: Run the test to verify it fails, then implement (already implemented)**

Run: `pytest research/tests/trader_mining/test_provider.py -v -k fixture`
Expected: at this point it should already PASS, since this test exercises `ccxt`'s own
parser directly (not `provider.py`'s code) -- it's a contract test proving the fixture and
the patching technique work, not a test of new production code. If it fails, the fixture
file or the patching approach has a real problem; do not "fix" this test by weakening its
assertions.

- [ ] **Step 7: Write and add the live-canary test (skipped by default)**

Append to `research/tests/trader_mining/test_provider.py`:

```python
async def test_live_hyperliquid_schema_still_matches_expectations():
    """Real network call against the real Hyperliquid API. Skipped by default -- set
    HYPERLIQUID_LIVE_TEST=1 to run it manually (e.g. before relying on this module after
    a ccxt upgrade). Deliberately NOT a pytest.mark, so it needs no change to this
    repo's shared pyproject.toml/CI config and can never make an unrelated PR's CI run
    flaky by hitting a real external service."""
    if not os.environ.get("HYPERLIQUID_LIVE_TEST"):
        pytest.skip("set HYPERLIQUID_LIVE_TEST=1 to run this live-network test")

    result = await fetch_hyperliquid_fills(TRADER)

    assert isinstance(result.trades, list)
    assert result.history_completeness in ("complete", "truncated_by_provider_limit")
    if result.trades:
        trade = result.trades[0]
        assert "info" in trade
        assert isinstance(trade["timestamp"], int)
        assert isinstance(trade["price"], (int, float))
        assert isinstance(trade["amount"], (int, float))
```

- [ ] **Step 8: Run the full test file to confirm the default (non-live) run is green**

Run: `pytest research/tests/trader_mining/test_provider.py -v`
Expected: PASS (6 tests total -- the live canary SKIPPED, not failed, since
`HYPERLIQUID_LIVE_TEST` is unset)

- [ ] **Step 9: Manually confirm the live canary itself works (not part of the automated
  suite's required pass, but confirm once before considering this task done)**

Run: `HYPERLIQUID_LIVE_TEST=1 pytest research/tests/trader_mining/test_provider.py -v -k live`
Expected: PASS (1 test, real network call)

- [ ] **Step 10: Lint and format**

Run: `ruff check research/trader_mining/ research/tests/trader_mining/` and
`ruff format --check research/trader_mining/ research/tests/trader_mining/`
Expected: no errors (fix and re-run Step 8 if needed)

- [ ] **Step 11: Commit**

```bash
git add research/trader_mining/__init__.py research/trader_mining/provider.py \
  research/tests/trader_mining/__init__.py research/tests/trader_mining/test_provider.py
git commit -m "feat(research): add Hyperliquid fill provider (read-only, unauthenticated, paginated)"
```

---

### Task 3: `research/trader_mining/ingestion.py` — idempotent persistence

**Files:**
- Create: `research/trader_mining/ingestion.py`
- Test: `research/tests/trader_mining/test_ingestion.py`

**Interfaces:**
- Consumes: `research.trader_mining.provider.fetch_hyperliquid_fills`,
  `FetchFillsResult` (Task 2); `research.models.RawFill`, `NormalizedFill` (Task 1).
- Produces: `IngestResult`, `ingest_hyperliquid_fills(session, trader, since=None)` --
  consumed by Task 4 (`cli.py`).

- [ ] **Step 1: Write the failing tests**

Create `research/tests/trader_mining/test_ingestion.py`:

```python
# research/tests/trader_mining/test_ingestion.py
from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from research.models import Base, NormalizedFill, RawFill
from research.trader_mining.ingestion import ingest_hyperliquid_fills
from research.trader_mining.provider import FetchFillsResult


TRADER = "0x0000000000000000000000000000000000000000"


def _memory_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _trade(tid: int, ts_ms: int = 1_700_000_000_000) -> dict:
    return {
        "id": str(tid),
        "order": str(tid + 1_000_000),
        "timestamp": ts_ms,
        "datetime": "2026-01-01T00:00:00.000Z",
        "symbol": "BTC/USDC:USDC",
        "side": "buy",
        "price": 100.0,
        "amount": 1.0,
        "cost": 100.0,
        "fee": {"currency": "USDC", "cost": 0.1},
        "info": {
            "tid": tid,
            "oid": tid + 1_000_000,
            "coin": "BTC",
            "px": "100.0",
            "sz": "1.0",
            "side": "B",
            "time": ts_ms,
            "startPosition": "1.0",
            "dir": "Open Long",
            "closedPnl": "0.0",
            "crossed": True,
            "fee": "0.1",
            "feeToken": "USDC",
        },
    }


def test_first_import_populates_both_tables(mocker):
    session = _memory_session()
    mocker.patch(
        "research.trader_mining.ingestion.asyncio.run",
        return_value=FetchFillsResult(trades=[_trade(1), _trade(2)], history_completeness="complete"),
    )

    result = ingest_hyperliquid_fills(session, TRADER)

    assert result.n_fetched == 2
    assert result.n_new == 2
    assert result.history_completeness == "complete"
    assert session.query(RawFill).count() == 2
    assert session.query(NormalizedFill).count() == 2


def test_rerunning_same_import_is_a_no_op(mocker):
    session = _memory_session()
    mocker.patch(
        "research.trader_mining.ingestion.asyncio.run",
        return_value=FetchFillsResult(trades=[_trade(1), _trade(2)], history_completeness="complete"),
    )
    ingest_hyperliquid_fills(session, TRADER)

    result = ingest_hyperliquid_fills(session, TRADER)

    assert result.n_fetched == 2
    assert result.n_new == 0
    assert session.query(RawFill).count() == 2
    assert session.query(NormalizedFill).count() == 2


def test_new_fill_on_top_of_existing_only_inserts_the_new_one(mocker):
    session = _memory_session()
    mocker.patch(
        "research.trader_mining.ingestion.asyncio.run",
        return_value=FetchFillsResult(trades=[_trade(1)], history_completeness="complete"),
    )
    ingest_hyperliquid_fills(session, TRADER)
    mocker.patch(
        "research.trader_mining.ingestion.asyncio.run",
        return_value=FetchFillsResult(
            trades=[_trade(1), _trade(2)], history_completeness="complete"
        ),
    )

    result = ingest_hyperliquid_fills(session, TRADER)

    assert result.n_fetched == 2
    assert result.n_new == 1
    assert session.query(RawFill).count() == 2


def test_normalized_fields_mapped_correctly(mocker):
    session = _memory_session()
    mocker.patch(
        "research.trader_mining.ingestion.asyncio.run",
        return_value=FetchFillsResult(trades=[_trade(1)], history_completeness="complete"),
    )

    ingest_hyperliquid_fills(session, TRADER)

    row = session.query(NormalizedFill).one()
    assert row.trader == TRADER
    assert row.tid == 1
    assert row.symbol == "BTC/USDC:USDC"
    assert row.side == "buy"
    assert row.price == 100.0
    assert row.quantity == 1.0
    assert row.notional == 100.0
    assert row.position == 1.0
    assert row.closed_pnl == 0.0
    assert row.direction == "Open Long"
    assert row.crossed is True
    assert row.fee == 0.1
    assert row.fee_currency == "USDC"
    assert row.order_id == str(1 + 1_000_000)


def test_history_completeness_passes_through(mocker):
    session = _memory_session()
    mocker.patch(
        "research.trader_mining.ingestion.asyncio.run",
        return_value=FetchFillsResult(trades=[], history_completeness="truncated_by_provider_limit"),
    )

    result = ingest_hyperliquid_fills(session, TRADER)

    assert result.history_completeness == "truncated_by_provider_limit"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/trader_mining/test_ingestion.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'research.trader_mining.ingestion'`

- [ ] **Step 3: Implement `ingestion.py`**

Create `research/trader_mining/ingestion.py`:

```python
# research/trader_mining/ingestion.py
"""Idempotent persistence of research.trader_mining.provider's fetched fills into
research.models.RawFill/NormalizedFill. Re-running the same import for the same trader
is always safe: already-seen fills (by tid) are skipped, never duplicated or
re-inserted, per the proposal's explicit "idempotent, resumable, tolerant of duplicate
fills" requirement."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from research.models import NormalizedFill, RawFill
from research.trader_mining.provider import fetch_hyperliquid_fills


@dataclass
class IngestResult:
    n_fetched: int
    n_new: int
    history_completeness: str


def ingest_hyperliquid_fills(
    session: Session, trader: str, since: datetime | None = None
) -> IngestResult:
    """Fetch trader's fills (via research.trader_mining.provider) and persist any not
    already stored, in one transaction. Returns counts and the provider's own
    history_completeness verdict, unchanged."""
    result = asyncio.run(fetch_hyperliquid_fills(trader, since))
    retrieved_at = datetime.now(UTC)

    existing_tids = {
        tid
        for (tid,) in session.query(RawFill.tid).filter(RawFill.trader == trader).all()
    }

    n_new = 0
    for trade in result.trades:
        tid = trade["info"]["tid"]
        if tid in existing_tids:
            continue
        n_new += 1
        existing_tids.add(tid)

        session.add(
            RawFill(
                source="hyperliquid",
                trader=trader,
                tid=tid,
                payload_json=json.dumps(trade["info"]),
                retrieved_at=retrieved_at,
            )
        )
        session.add(
            NormalizedFill(
                trader=trader,
                tid=tid,
                timestamp=datetime.fromtimestamp(trade["timestamp"] / 1000, tz=UTC),
                symbol=trade["symbol"],
                side=trade["side"],
                price=trade["price"],
                quantity=trade["amount"],
                notional=trade["cost"],
                position=float(trade["info"]["startPosition"]),
                closed_pnl=float(trade["info"]["closedPnl"]),
                direction=trade["info"]["dir"],
                crossed=trade["info"]["crossed"],
                fee=trade["fee"]["cost"],
                fee_currency=trade["fee"]["currency"],
                order_id=trade["order"],
            )
        )

    session.commit()
    return IngestResult(
        n_fetched=len(result.trades),
        n_new=n_new,
        history_completeness=result.history_completeness,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/trader_mining/test_ingestion.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint and format**

Run: `ruff check research/trader_mining/ingestion.py research/tests/trader_mining/test_ingestion.py`
and `ruff format --check research/trader_mining/ingestion.py research/tests/trader_mining/test_ingestion.py`
Expected: no errors (fix and re-run Step 4 if needed)

- [ ] **Step 6: Commit**

```bash
git add research/trader_mining/ingestion.py research/tests/trader_mining/test_ingestion.py
git commit -m "feat(research): add idempotent Hyperliquid fill ingestion"
```

---

### Task 4: `research/cli.py` — `trader-import` subcommand

**Files:**
- Modify: `research/cli.py`
- Test: `research/tests/test_cli.py`

**Interfaces:**
- Consumes: `research.trader_mining.ingestion.ingest_hyperliquid_fills`, `IngestResult`
  (Task 3); `research.db.get_engine`, `get_session` (existing).
- Produces: nothing further downstream -- final task in this plan.

- [ ] **Step 1: Write the failing test**

Append to `research/tests/test_cli.py`:

```python
def test_trader_import_command_forwards_args_and_prints_result(mocker, capsys):
    from research.trader_mining.ingestion import IngestResult

    mock_ingest = mocker.patch(
        "research.cli.ingest_hyperliquid_fills",
        return_value=IngestResult(n_fetched=5, n_new=3, history_completeness="complete"),
    )
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    exit_code = main(
        [
            "trader-import",
            "--trader",
            "0x0000000000000000000000000000000000000000",
            "--since",
            "2026-01-01",
            "--db-path",
            "user_data/research.sqlite",
        ]
    )

    _, kwargs = mock_ingest.call_args
    assert kwargs["trader"] == "0x0000000000000000000000000000000000000000"
    assert kwargs["since"].year == 2026

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "n_fetched: 5" in captured.out
    assert "n_new: 3" in captured.out
    assert "complete" in captured.out


def test_trader_import_command_warns_on_truncated_history(mocker, capsys):
    from research.trader_mining.ingestion import IngestResult

    mocker.patch(
        "research.cli.ingest_hyperliquid_fills",
        return_value=IngestResult(
            n_fetched=10_000, n_new=10_000, history_completeness="truncated_by_provider_limit"
        ),
    )
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    exit_code = main(
        ["trader-import", "--trader", "0x0000000000000000000000000000000000000000"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0  # not a failure -- an honest, informational result
    assert "truncated_by_provider_limit" in captured.out
    assert "WARNING" in captured.out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest research/tests/test_cli.py -v -k trader_import`
Expected: FAIL with `error: argument command: invalid choice: 'trader-import'`

- [ ] **Step 3: Add the `trader-import` subcommand**

In `research/cli.py`, add the new imports (alongside the existing ones):

```python
from research.db import get_engine, get_session
from research.trader_mining.ingestion import ingest_hyperliquid_fills
```

After the existing `gate = sub.add_parser(...)` block and its arguments, before
`args = parser.parse_args(argv)`, add:

```python
    trader_import = sub.add_parser(
        "trader-import", help="Import one wallet's fill history from Hyperliquid"
    )
    trader_import.add_argument("--trader", required=True, help="Wallet address")
    trader_import.add_argument("--since", help="YYYY-MM-DD, earliest fill to fetch")
    trader_import.add_argument("--db-path", default="user_data/research.sqlite")
```

Add the new dispatch branch after the existing `if args.command == "gate":` block (as an
`elif`):

```python
    elif args.command == "trader-import":
        engine = get_engine(args.db_path)
        session = get_session(engine)
        since = (
            datetime.fromisoformat(args.since).replace(tzinfo=UTC) if args.since else None
        )
        result = ingest_hyperliquid_fills(session, trader=args.trader, since=since)
        print(f"n_fetched: {result.n_fetched}")
        print(f"n_new: {result.n_new}")
        print(f"history_completeness: {result.history_completeness}")
        if result.history_completeness == "truncated_by_provider_limit":
            print(
                "WARNING: history_completeness=truncated_by_provider_limit -- "
                "Hyperliquid's 10,000-fill ceiling was reached; earlier fills may exist "
                "but are not retrievable via this endpoint."
            )
        return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest research/tests/test_cli.py -v`
Expected: PASS (all tests in the file, including the 2 new ones and every pre-existing one)

- [ ] **Step 5: Lint and format**

Run: `ruff check research/cli.py research/tests/test_cli.py` and
`ruff format --check research/cli.py research/tests/test_cli.py`
Expected: no errors (fix and re-run Step 4 if needed)

- [ ] **Step 6: Run the full research test suite**

Run: `pytest research/ -v`
Expected: PASS (every test in `research/`, confirming Tasks 1-4 compose cleanly; the live
canary from Task 2 SKIPPED, not failed)

- [ ] **Step 7: Commit**

```bash
git add research/cli.py research/tests/test_cli.py
git commit -m "feat(research): add trader-import CLI subcommand"
```
