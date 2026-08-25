import json
from datetime import UTC, datetime
from decimal import Decimal

from research.models import RawLedgerEvent
from research.trader_mining.ledger import signed_token_delta


TRADER = "0x9794bbbc222b6b93c1417d01aa1ff06d42e5333b"
OTHER = "0xead210997055781f27eeab816cc548673bf6e500"


def _event(event_type: str, delta: dict) -> RawLedgerEvent:
    """`delta` is the real payload's info["delta"] sub-dict -- real captured ledger
    entries always nest the actual event fields one level down under "delta", e.g.
    {"time": ..., "hash": ..., "delta": {"type": "deposit", "usdc": "288888.0"}}."""
    return RawLedgerEvent(
        trader=TRADER,
        event_id="0xtest",
        event_type=event_type,
        timestamp=datetime(2024, 11, 29, tzinfo=UTC),
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
