import pytest

from freqtrade.persistence.user_pair import UserPair, is_valid_user_pair


@pytest.mark.usefixtures("init_persistence")
def test_user_pair_roundtrip():
    assert UserPair.list_pairs() == []

    assert UserPair.add_pair("ETH/BTC") is True
    assert UserPair.add_pair("ETH/BTC") is False
    assert UserPair.add_pair("BTC/USDT:USDT") is True
    assert UserPair.list_pairs() == ["ETH/BTC", "BTC/USDT:USDT"]

    # A later session on the same database still sees the committed rows.
    UserPair.session.remove()
    assert UserPair.list_pairs() == ["ETH/BTC", "BTC/USDT:USDT"]

    assert UserPair.delete_pair("ETH/BTC") is True
    assert UserPair.delete_pair("ETH/BTC") is False
    assert UserPair.list_pairs() == ["BTC/USDT:USDT"]


def test_is_valid_user_pair():
    assert is_valid_user_pair("ETH/BTC")
    assert is_valid_user_pair("BTC/USDT:USDT")
    assert is_valid_user_pair("1000PEPE/USDT")
    assert not is_valid_user_pair("ETHBTC")
    assert not is_valid_user_pair(".*/BTC")
    assert not is_valid_user_pair("")
    assert not is_valid_user_pair("ETH/BTC ")
    assert not is_valid_user_pair("eth/btc extra")
