# research/trader_mining/ingestion.py
"""Idempotent persistence of research.trader_mining.provider's fetched fills into
research.models.RawFill/NormalizedFill. Re-running the same import for the same trader
is always safe: already-seen fills (by tid) are skipped, never duplicated or
re-inserted, per the proposal's explicit "idempotent, resumable, tolerant of duplicate
fills" requirement.
"""

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

    # Scoped to this trader, not global -- relies on Hyperliquid's tid being globally
    # unique (documented, per the spec), not just per-wallet. If that ever proves wrong,
    # a colliding tid from a different trader would slip past this check and hit
    # RawFill.tid's global UNIQUE constraint at commit, aborting the whole batch rather
    # than degrading gracefully -- a real but low-probability risk given the documented
    # guarantee, flagged in code review.
    existing_tids = {
        tid for (tid,) in session.query(RawFill.tid).filter(RawFill.trader == trader).all()
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
