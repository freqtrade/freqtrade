# research/tests/trader_mining/test_ingestion.py
from unittest.mock import AsyncMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from research.models import Base, NormalizedFill, RawFill, RawLedgerEvent
from research.trader_mining.ingestion import ingest_hyperliquid_fills, ingest_hyperliquid_ledger
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
        "research.trader_mining.ingestion.fetch_hyperliquid_fills",
        new=AsyncMock(
            return_value=FetchFillsResult(
                trades=[_trade(1), _trade(2)], history_completeness="complete"
            )
        ),
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
        "research.trader_mining.ingestion.fetch_hyperliquid_fills",
        new=AsyncMock(
            return_value=FetchFillsResult(
                trades=[_trade(1), _trade(2)], history_completeness="complete"
            )
        ),
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
        "research.trader_mining.ingestion.fetch_hyperliquid_fills",
        new=AsyncMock(
            return_value=FetchFillsResult(trades=[_trade(1)], history_completeness="complete")
        ),
    )
    ingest_hyperliquid_fills(session, TRADER)
    mocker.patch(
        "research.trader_mining.ingestion.fetch_hyperliquid_fills",
        new=AsyncMock(
            return_value=FetchFillsResult(
                trades=[_trade(1), _trade(2)], history_completeness="complete"
            )
        ),
    )

    result = ingest_hyperliquid_fills(session, TRADER)

    assert result.n_fetched == 2
    assert result.n_new == 1
    assert session.query(RawFill).count() == 2


def test_normalized_fields_mapped_correctly(mocker):
    session = _memory_session()
    mocker.patch(
        "research.trader_mining.ingestion.fetch_hyperliquid_fills",
        new=AsyncMock(
            return_value=FetchFillsResult(trades=[_trade(1)], history_completeness="complete")
        ),
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
        "research.trader_mining.ingestion.fetch_hyperliquid_fills",
        new=AsyncMock(
            return_value=FetchFillsResult(
                trades=[], history_completeness="truncated_by_provider_limit"
            )
        ),
    )

    result = ingest_hyperliquid_fills(session, TRADER)

    assert result.history_completeness == "truncated_by_provider_limit"


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


_ZERO_HASH = "0x0000000000000000000000000000000000000000000000000000000000000000"


def _c_staking_entry(ts_ms: int, amount: str) -> dict:
    """Real Hyperliquid shape: cStakingTransfer events don't get a real transaction
    hash -- ccxt's `id` (and info["hash"]) is this same zero-sentinel for EVERY such
    event, confirmed live against a real wallet with 7 of these, all sharing this
    identical id. A dedup key that trusts `id` alone silently drops all but the first
    real event and, worse, collides across different traders (event_id was globally
    unique)."""
    return {
        "id": _ZERO_HASH,
        "timestamp": ts_ms,
        "type": "cStakingTransfer",
        "info": {
            "time": ts_ms,
            "hash": _ZERO_HASH,
            "delta": {
                "type": "cStakingTransfer",
                "token": "HYPE",
                "amount": amount,
                "isDeposit": False,
            },
        },
    }


def test_ledger_same_id_different_content_events_are_not_deduped_away(mocker):
    session = _memory_session()
    mocker.patch(
        "research.trader_mining.ingestion.fetch_hyperliquid_ledger",
        new=AsyncMock(
            return_value=[
                _c_staking_entry(1_700_000_000_000, "500000.0"),
                _c_staking_entry(1_700_001_000_000, "12345.0"),
            ]
        ),
    )

    result = ingest_hyperliquid_ledger(session, TRADER)

    assert result.n_fetched == 2
    assert result.n_new == 2
    assert session.query(RawLedgerEvent).count() == 2


def test_ledger_rerun_with_shared_id_events_stays_idempotent(mocker):
    session = _memory_session()
    entries = [
        _c_staking_entry(1_700_000_000_000, "500000.0"),
        _c_staking_entry(1_700_001_000_000, "12345.0"),
    ]
    mocker.patch(
        "research.trader_mining.ingestion.fetch_hyperliquid_ledger",
        new=AsyncMock(return_value=entries),
    )
    ingest_hyperliquid_ledger(session, TRADER)

    result = ingest_hyperliquid_ledger(session, TRADER)

    assert result.n_new == 0
    assert session.query(RawLedgerEvent).count() == 2


def test_ledger_shared_id_across_two_traders_does_not_crash(mocker):
    session = _memory_session()
    entry = _c_staking_entry(1_700_000_000_000, "500000.0")
    mocker.patch(
        "research.trader_mining.ingestion.fetch_hyperliquid_ledger",
        new=AsyncMock(return_value=[entry]),
    )

    ingest_hyperliquid_ledger(session, TRADER)
    other_trader = "0x1111111111111111111111111111111111111111"
    result = ingest_hyperliquid_ledger(session, other_trader)

    assert result.n_new == 1
    assert session.query(RawLedgerEvent).count() == 2


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
