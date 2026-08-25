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


def reconciliation_deltas(
    session: Session, trader: str, asset: str, window_start: datetime, window_end: datetime
) -> Decimal:
    """Sum signed_token_delta for `asset`, over ledger events between window_start and
    window_end (inclusive of both bounds -- the two fills whose position mismatch
    triggered a reconciliation check in research.trader_mining.engine). Inclusive, not
    strict: a ledger event sharing an exact timestamp with one of those two fills must
    not be silently excluded, which would contradict this feature's own "never silently
    drop" philosophy (found in code review)."""
    events = (
        session.query(RawLedgerEvent)
        .filter(
            RawLedgerEvent.trader == trader,
            RawLedgerEvent.timestamp >= window_start,
            RawLedgerEvent.timestamp <= window_end,
        )
        .all()
    )
    total = Decimal(0)
    for event in events:
        parsed = signed_token_delta(event, trader)
        if parsed is not None and parsed[0] == asset:
            total += parsed[1]
    return total
