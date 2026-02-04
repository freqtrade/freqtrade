import pytest
from pydantic import ValidationError

from freqtrade.rpc.api_server.api_schemas import (
    BlacklistPayload,
    DeleteLockRequest,
    ExchangeModePayloadMixin,
    PairListsPayload,
)


def test_blacklist_payload_validation():
    # Valid input
    payload = BlacklistPayload(blacklist=["XRP/BTC", "ETH/USDT"])
    assert payload.blacklist == ["XRP/BTC", "ETH/USDT"]

    # Invalid input (ReDoS attempt)
    with pytest.raises(ValidationError) as excinfo:
        BlacklistPayload(blacklist=["(a+)+$"])
    assert "Invalid pair name" in str(excinfo.value)


def test_delete_lock_request_validation():
    # Valid input
    payload = DeleteLockRequest(pair="XRP/BTC")
    assert payload.pair == "XRP/BTC"

    # Invalid input
    with pytest.raises(ValidationError) as excinfo:
        DeleteLockRequest(pair="INVALID(PAIR)")
    assert "Invalid pair name" in str(excinfo.value)


def test_exchange_mode_payload_validation():
    # Valid input
    payload = ExchangeModePayloadMixin(exchange="binance")
    assert payload.exchange == "binance"

    # Invalid input
    with pytest.raises(ValidationError) as excinfo:
        ExchangeModePayloadMixin(exchange="binance;DROP TABLE")
    assert "Invalid exchange name" in str(excinfo.value)


def test_pairlists_payload_validation():
    # Valid input
    payload = PairListsPayload(
        pairlists=[{"method": "StaticPairList"}], blacklist=["XRP/BTC"], stake_currency="USDT"
    )
    assert payload.blacklist == ["XRP/BTC"]
    assert payload.stake_currency == "USDT"

    # Invalid blacklist
    with pytest.raises(ValidationError) as excinfo:
        PairListsPayload(
            pairlists=[{"method": "StaticPairList"}], blacklist=["(a+)+$"], stake_currency="USDT"
        )
    assert "Invalid pair name" in str(excinfo.value)

    # Invalid stake currency
    with pytest.raises(ValidationError) as excinfo:
        PairListsPayload(
            pairlists=[{"method": "StaticPairList"}], blacklist=["XRP/BTC"], stake_currency="USDT;"
        )
    assert "Invalid stake currency" in str(excinfo.value)
