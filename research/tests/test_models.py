# research/tests/test_models.py
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from research.models import Base, NormalizedFill, RawFill, RawLedgerEvent, ReconstructedTrade


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
    # SQLite's DateTime column has no native timezone support -- SQLAlchemy round-trips
    # it as naive, stripping tzinfo. Expected, not a bug: compare against the naive form.
    assert row.retrieved_at == datetime(2026, 8, 25, tzinfo=UTC).replace(tzinfo=None)


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


def test_reconstructed_trade_round_trips():
    session = _memory_session()
    session.add(
        ReconstructedTrade(
            trader="0xAAA",
            symbol="BTC/USDC:USDC",
            direction="long",
            entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            entry_price=100.0,
            exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC),
            exit_price=110.0,
            quantity=5.0,
            gross_pnl=50.0,
            fees=1.0,
            net_pnl=49.0,
            holding_time_seconds=86400.0,
            n_fills=3,
            is_truncated_start=False,
            was_liquidated=False,
        )
    )
    session.commit()

    row = session.query(ReconstructedTrade).one()
    assert row.trader == "0xAAA"
    assert row.symbol == "BTC/USDC:USDC"
    assert row.direction == "long"
    assert row.entry_price == 100.0
    assert row.exit_price == 110.0
    assert row.quantity == 5.0
    assert row.net_pnl == 49.0
    assert row.n_fills == 3
    assert row.is_truncated_start is False
    assert row.was_liquidated is False


def test_raw_ledger_event_round_trips():
    session = _memory_session()
    session.add(
        RawLedgerEvent(
            trader="0xAAA",
            event_id="0xdeadbeef",
            dedup_key="0xAAA:0xdeadbeef:1732875752000:spotTransfer:abc123",
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
