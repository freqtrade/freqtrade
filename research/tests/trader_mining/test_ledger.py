import json
from datetime import UTC, datetime
from decimal import Decimal

from research.models import RawLedgerEvent
from research.trader_mining.ledger import signed_token_delta


TRADER = "0x9794bbbc222b6b93c1417d01aa1ff06d42e5333b"
OTHER = "0xead210997055781f27eeab816cc548673bf6e500"


def _event(
    event_type: str,
    delta: dict,
    event_id: str = "0xtest",
    timestamp: datetime = datetime(2024, 11, 29, tzinfo=UTC),
) -> RawLedgerEvent:
    """`delta` is the real payload's info["delta"] sub-dict -- real captured ledger
    entries always nest the actual event fields one level down under "delta", e.g.
    {"time": ..., "hash": ..., "delta": {"type": "deposit", "usdc": "288888.0"}}."""
    return RawLedgerEvent(
        trader=TRADER,
        event_id=event_id,
        event_type=event_type,
        timestamp=timestamp,
        info_json=json.dumps({"delta": delta}),
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
    entry = _event("transfer", {"type": "accountClassTransfer", "usdc": "288888.0", "toPerp": True})
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


def _memory_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from research.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_reconciliation_deltas_sums_matching_asset_in_window():
    from research.trader_mining.ledger import reconciliation_deltas

    session = _memory_session()
    session.add(
        _event(
            "spotTransfer",
            {
                "type": "spotTransfer",
                "token": "HYPE",
                "amount": "62264.0",
                "user": TRADER,
                "destination": OTHER,
            },
            event_id="0xa",
            timestamp=datetime(2024, 11, 29, 10, 2, 32, tzinfo=UTC),
        )
    )
    session.add(
        _event(
            "deposit",
            {"type": "deposit", "usdc": "100.0"},  # different asset -- must not count
            event_id="0xb",
            timestamp=datetime(2024, 11, 29, 10, 2, 33, tzinfo=UTC),
        )
    )
    session.commit()

    total = reconciliation_deltas(
        session,
        TRADER,
        "HYPE",
        datetime(2024, 11, 29, 9, 53, 54, tzinfo=UTC),
        datetime(2024, 11, 29, 10, 7, 13, tzinfo=UTC),
    )

    assert total == Decimal("-62264.0")


def test_reconciliation_deltas_ignores_events_outside_window():
    from research.trader_mining.ledger import reconciliation_deltas

    session = _memory_session()
    session.add(
        _event(
            "spotTransfer",
            {
                "type": "spotTransfer",
                "token": "HYPE",
                "amount": "62264.0",
                "user": TRADER,
                "destination": OTHER,
            },
            event_id="0xa",
            timestamp=datetime(2024, 11, 29, 10, 2, 32, tzinfo=UTC),
        )
    )
    session.commit()

    total = reconciliation_deltas(
        session,
        TRADER,
        "HYPE",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    )

    assert total == Decimal(0)
