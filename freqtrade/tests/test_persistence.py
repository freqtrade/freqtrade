# pragma pylint: disable=missing-docstring
import pytest

from freqtrade.exchange import Exchanges
from freqtrade.persistence import Trade


def test_update(limit_buy_order, limit_sell_order):
    trade = Trade(
        pair='BTC_ETH',
        stake_amount=1.00,
        fee=0.1,
        exchange=Exchanges.BITTREX,
    )
    assert trade.open_order_id is None
    assert trade.open_rate is None
    assert trade.close_profit is None
    assert trade.close_date is None

    trade.open_order_id = 'something'
    trade.update(limit_buy_order)
    assert trade.open_order_id is None
    assert trade.open_rate == 0.07256061
    assert trade.close_profit is None
    assert trade.close_date is None

    trade.open_order_id = 'something'
    trade.update(limit_sell_order)
    assert trade.open_order_id is None
    assert trade.open_rate == 0.07256061
    assert trade.close_profit == 0.00546755
    assert trade.close_date is not None


def test_update_open_order(limit_buy_order):
    trade = Trade(
        pair='BTC_ETH',
        stake_amount=1.00,
        fee=0.1,
        exchange=Exchanges.BITTREX,
    )

    assert trade.open_order_id is None
    assert trade.open_rate is None
    assert trade.close_profit is None
    assert trade.close_date is None

    limit_buy_order['closed'] = False
    trade.update(limit_buy_order)

    assert trade.open_order_id is None
    assert trade.open_rate is None
    assert trade.close_profit is None
    assert trade.close_date is None


def test_update_invalid_order(limit_buy_order):
    trade = Trade(
        pair='BTC_ETH',
        stake_amount=1.00,
        fee=0.1,
        exchange=Exchanges.BITTREX,
    )
    limit_buy_order['type'] = 'invalid'
    with pytest.raises(ValueError, match=r'Unknown order type'):
        trade.update(limit_buy_order)
