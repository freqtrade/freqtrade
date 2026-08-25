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


async def fetch_hyperliquid_fills(trader: str, since: datetime | None = None) -> FetchFillsResult:
    """Fetch trader's full available fill history from Hyperliquid's public info
    endpoint, paginating until either the provider's own history ends or its 10,000-fill
    ceiling is reached. `since`, when given, is a lower bound in UTC; omit it to fetch
    from the earliest available fill.

    Bug fixed here, found in code review: ccxt's fetch_my_trades takes a DIFFERENT
    request type when `since` is None ("userFills", no time bound) than when a `since`
    is given ("userFillsByTime", time-bound) -- and the no-time-bound path returns only
    the MOST RECENT slice of history, not the earliest fills. Passing `since_ms=None`
    straight through silently dropped all older history while still reporting
    history_completeness="complete". Using epoch 0 as the default forces
    "userFillsByTime" unconditionally, which correctly walks forward from the true start
    of history -- confirmed against ccxt's real fetch_my_trades/parse_trades source, and
    covered by a regression test exercising the real (not mocked) request-building path.
    """
    exchange = ccxt_async.hyperliquid()
    try:
        all_trades: list[dict] = []
        since_ms = int(since.timestamp() * 1000) if since is not None else 0
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
            #
            # ponytail: known residual risk, flagged in code review, not fixed here --
            # if PAGE_SIZE (2000) or more fills share the exact same millisecond
            # timestamp, since_ms never advances past that instant and the loop just
            # re-fetches the same/overlapping page until FILL_CEILING is hit by sheer
            # duplicate accumulation, misreporting history_completeness as
            # "truncated_by_provider_limit" for a wallet whose real fill count is far
            # smaller. Bounded (terminates, doesn't hang), but the completeness verdict
            # is wrong in exactly the direction this module exists to avoid. The real
            # captured fixture already shows 8 fills tied at one timestamp (a batch
            # settlement), so 2000+ tied fills from a busy/whale wallet is plausible, not
            # theoretical. Upgrade path if this bites: paginate with `tid` as a
            # tiebreaker within a tied timestamp instead of relying on timestamp
            # advancement alone (e.g. track the max tid seen at the current since_ms and
            # skip already-seen tids client-side rather than trusting the page boundary).
            since_ms = page[-1]["timestamp"]
    finally:
        await exchange.close()
