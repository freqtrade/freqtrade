# research/tests/trader_mining/test_provider.py
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from research.trader_mining.provider import fetch_hyperliquid_fills


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "hyperliquid_user_fills_raw.json"
FROZEN_FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
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
    mocker.patch("research.trader_mining.provider.ccxt_async.hyperliquid.close", new=AsyncMock())

    await fetch_hyperliquid_fills(TRADER)

    _, kwargs = mock_fetch.call_args
    assert kwargs["params"]["user"] == TRADER


async def test_stops_pagination_on_short_page_reports_complete(mocker):
    # First page: fewer than the 2000-fill page size -- end of history.
    mocker.patch(
        "research.trader_mining.provider.ccxt_async.hyperliquid.fetch_my_trades",
        new=AsyncMock(return_value=[_fake_trade(1, 1_700_000_000_000)]),
    )
    mocker.patch("research.trader_mining.provider.ccxt_async.hyperliquid.close", new=AsyncMock())

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
    mocker.patch("research.trader_mining.provider.ccxt_async.hyperliquid.close", new=AsyncMock())

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
    mocker.patch("research.trader_mining.provider.ccxt_async.hyperliquid.close", new=AsyncMock())

    result = await fetch_hyperliquid_fills(TRADER)

    assert len(result.trades) == 10_000
    assert result.history_completeness == "truncated_by_provider_limit"


async def test_fixture_frozen_response_parses_through_real_ccxt_parser(mocker):
    """Patches publicPostInfo -- the raw HTTP boundary ccxt itself calls internally
    (research/trader_mining/provider.py's docstring cites where this was confirmed) --
    NOT fetch_my_trades. This exercises ccxt's REAL Hyperliquid parser against a real
    captured response, so it actually catches ccxt/Hyperliquid schema drift; mocking
    fetch_my_trades directly would not."""
    import ccxt.async_support as ccxt_async_module

    frozen = FROZEN_FIXTURE

    exchange = ccxt_async_module.hyperliquid()
    # fetch_my_trades() calls load_markets() if exchange.markets is None, which itself
    # calls publicPostInfo for a DIFFERENT request type -- pre-seed an empty markets
    # dict so it's skipped (symbol=None below means no real market data is needed).
    # Confirmed necessary by spiking directly against the real API during spec research.
    exchange.markets = {}
    exchange.publicPostInfo = AsyncMock(return_value=frozen)
    try:
        trades = await exchange.fetch_my_trades(symbol=None, params={"user": TRADER})
    finally:
        await exchange.close()

    assert len(trades) == len(frozen)
    # All 8 fills in this fixture share one exact millisecond timestamp (a real batch
    # settlement) -- ccxt's own sort doesn't promise raw-array order for tied
    # timestamps, so match by tid rather than by position.
    frozen_by_tid = {f["tid"]: f for f in frozen}
    for trade in trades:
        raw = frozen_by_tid[int(trade["id"])]
        assert trade["order"] == str(raw["oid"])
        assert trade["info"]["dir"] == raw["dir"]
        assert trade["info"] == raw  # raw payload preserved verbatim


def _raw_fill(tid: int, ts_ms: int) -> dict:
    """A schema-realistic raw Hyperliquid fill (matches the frozen fixture's own field
    set), for the synthetic large dataset below -- the real captured fixture only has 8
    fills, too small to exercise multi-page pagination at all."""
    return {
        "coin": "BTC",
        "px": "100.0",
        "sz": "1.0",
        "side": "B",
        "time": ts_ms,
        "startPosition": "1.0",
        "dir": "Open Long",
        "closedPnl": "0.0",
        "hash": "0x0",
        "oid": tid,
        "crossed": True,
        "fee": "0.0",
        "tid": tid,
        "feeToken": "USDC",
        "twapId": None,
    }


async def test_omitting_since_still_fetches_full_history_not_just_recent_slice(mocker):
    """Regression test for a real bug found in code review: ccxt's fetch_my_trades takes
    a DIFFERENT code path when since=None (request type "userFills", no time bound) than
    when since is given (type "userFillsByTime", time-bound) -- and the no-time-bound
    path returns only the MOST RECENT slice of a wallet's history, not the earliest
    fills. provider.py used to pass since_ms=None straight through when the caller
    omitted `since`, silently dropping all history older than the most recent ~PAGE_SIZE
    fills while still reporting history_completeness="complete".

    Exercises the REAL ccxt fetch_my_trades/parse_trades/publicPostInfo chain (not a
    mocked fetch_my_trades, which would skip this endpoint-selection logic entirely and
    is exactly why the provider-unit-test layer above didn't catch this) against a
    synthetic 2500-fill dataset -- larger than PAGE_SIZE, so full multi-page pagination
    from the true start of history is actually exercised."""
    import ccxt.async_support as ccxt_async_module

    base_ts = 1_700_000_000_000
    all_fills = [_raw_fill(i, base_ts + i) for i in range(2500)]

    calls: list[dict] = []

    async def fake_server(request):
        calls.append(dict(request))
        if request.get("type") == "userFillsByTime":
            start = request["startTime"]
            return [f for f in all_fills if f["time"] >= start][:2000]
        # type "userFills" (no time bound): real Hyperliquid/ccxt semantics -- only the
        # most recent slice, exactly the trap this test guards against.
        return all_fills[-2000:]

    exchange = ccxt_async_module.hyperliquid()
    exchange.markets = {}  # skip load_markets() -- see the fixture test above for why
    exchange.publicPostInfo = fake_server
    mocker.patch("research.trader_mining.provider.ccxt_async.hyperliquid", return_value=exchange)

    result = await fetch_hyperliquid_fills(TRADER)  # since=None -- the buggy path

    # Every request must go through userFillsByTime -- proves the fix forces that
    # endpoint even with since=None, never the most-recent-only "userFills" branch.
    assert calls
    assert all(c.get("type") == "userFillsByTime" for c in calls)
    fetched_tids = {t["info"]["tid"] for t in result.trades}
    assert 0 in fetched_tids  # the EARLIEST fill must be present
    assert 2499 in fetched_tids  # and the latest
    assert len(fetched_tids) == len(all_fills)
    assert result.history_completeness == "complete"


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
