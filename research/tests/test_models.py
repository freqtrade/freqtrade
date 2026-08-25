# research/tests/test_models.py
from datetime import UTC, datetime

import pytest
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
