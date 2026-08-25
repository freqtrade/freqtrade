# research/tests/trader_mining/test_ingestion.py
from unittest.mock import AsyncMock

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
