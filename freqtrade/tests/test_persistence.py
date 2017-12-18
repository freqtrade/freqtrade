# pragma pylint: disable=missing-docstring
import pytest

from freqtrade.exchange import Exchanges
from freqtrade.persistence import Trade


def test_update_with_bittrex(limit_buy_order, limit_sell_order):
    """
    On this test we will buy and sell a crypto currency.

    Buy
    - Buy: 90.99181073 Crypto at 0.00001099 BTC (90.99181073*0.00001099 = 0.0009999 BTC)
    - Buying fee: 0.25%
    - Total cost of buy trade: 0.001002500 BTC ((90.99181073*0.00001099) + ((90.99181073*0.00001099)*0.0025))

    Sell
    - Sell: 90.99181073 Crypto at 0.00001173 BTC (90.99181073*0.00001173 = 0,00106733394 BTC)
    - Selling fee: 0.25%
    - Total cost of sell trade: 0.001064666 BTC ((90.99181073*0.00001173) - ((90.99181073*0.00001173)*0.0025))

    Profit/Loss: +0.000062166 BTC (Sell:0.001064666 - Buy:0.001002500)
    Profit/Loss percentage: 0.0620  ((0.001064666/0.001002500)-1 = 6.20%)

    :param limit_buy_order:
    :param limit_sell_order:
    :return:
    """

    trade = Trade(
        pair='BTC_ETH',
        stake_amount=0.001,
        fee=0.0025,
        exchange=Exchanges.BITTREX,
    )
    assert trade.open_order_id is None
    assert trade.open_rate is None
    assert trade.close_profit is None
    assert trade.close_date is None

    trade.open_order_id = 'something'
    trade.update(limit_buy_order)
    assert trade.open_order_id is None
    assert trade.open_rate == 0.00001099
    assert trade.close_profit is None
    assert trade.close_date is None

    trade.open_order_id = 'something'
    trade.update(limit_sell_order)
    assert trade.open_order_id is None
    assert trade.close_rate == 0.00001173
    assert trade.close_profit == 0.06201057
    assert trade.close_date is not None


def test_calc_open_close_trade_price(limit_buy_order, limit_sell_order):
    trade = Trade(
        pair='BTC_ETH',
        stake_amount=0.001,
        fee=0.0025,
        exchange=Exchanges.BITTREX,
    )

    trade.open_order_id = 'something'
    trade.update(limit_buy_order)
    assert trade.calc_open_trade_price() == 0.001002500

    trade.update(limit_sell_order)
    assert trade.calc_close_trade_price() == 0.0010646656

    # Profit in BTC
    assert trade.calc_profit() == 0.00006217

    # Profit in percent
    assert trade.calc_profit_percent() == 0.06201057


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


def test_calc_open_trade_price(limit_buy_order):
    trade = Trade(
        pair='BTC_ETH',
        stake_amount=0.001,
        fee=0.0025,
        exchange=Exchanges.BITTREX,
    )
    trade.open_order_id = 'open_trade'
    trade.update(limit_buy_order)  # Buy @ 0.00001099

    # Get the open rate price with the standard fee rate
    assert trade.calc_open_trade_price() == 0.001002500

    # Get the open rate price with a custom fee rate
    assert trade.calc_open_trade_price(fee=0.003) == 0.001003000


def test_calc_close_trade_price(limit_buy_order, limit_sell_order):
    trade = Trade(
        pair='BTC_ETH',
        stake_amount=0.001,
        fee=0.0025,
        exchange=Exchanges.BITTREX,
    )
    trade.open_order_id = 'close_trade'
    trade.update(limit_buy_order)  # Buy @ 0.00001099

    # Get the close rate price with a custom close rate and a regular fee rate
    assert trade.calc_close_trade_price(rate=0.00001234) == 0.0011200318

    # Get the close rate price with a custom close rate and a custom fee rate
    assert trade.calc_close_trade_price(rate=0.00001234, fee=0.003) == 0.0011194704

    # Test when we apply a Sell order, and ask price with a custom fee rate
    trade.update(limit_sell_order)
    assert trade.calc_close_trade_price(fee=0.005) == 0.0010619972


def test_calc_profit(limit_buy_order, limit_sell_order):
    trade = Trade(
        pair='BTC_ETH',
        stake_amount=0.001,
        fee=0.0025,
        exchange=Exchanges.BITTREX,
    )
    trade.open_order_id = 'profit_percent'
    trade.update(limit_buy_order)  # Buy @ 0.00001099

    # Custom closing rate and regular fee rate
    # Higher than open rate
    assert trade.calc_profit(rate=0.00001234) == 0.00011753
    # Lower than open rate
    assert trade.calc_profit(rate=0.00000123) == -0.00089086

    # Custom closing rate and custom fee rate
    # Higher than open rate
    assert trade.calc_profit(rate=0.00001234, fee=0.003) == 0.00011697
    # Lower than open rate
    assert trade.calc_profit(rate=0.00000123, fee=0.003) == -0.00089092

    # Only custom fee without sell order applied
    with pytest.raises(TypeError):
        trade.calc_profit(fee=0.003)

    # Test when we apply a Sell order. Sell higher than open rate @ 0.00001173
    trade.update(limit_sell_order)
    assert trade.calc_profit() == 0.00006217

    # Test with a custom fee rate on the close trade
    assert trade.calc_profit(fee=0.003) == 0.00006163


def test_calc_profit_percent(limit_buy_order, limit_sell_order):
    trade = Trade(
        pair='BTC_ETH',
        stake_amount=0.001,
        fee=0.0025,
        exchange=Exchanges.BITTREX,
    )
    trade.open_order_id = 'profit_percent'
    trade.update(limit_buy_order)  # Buy @ 0.00001099

    # Get percent of profit with a custom rate (Higher than open rate)
    assert trade.calc_profit_percent(rate=0.00001234) == 0.1172387

    # Get percent of profit with a custom rate (Lower than open rate)
    assert trade.calc_profit_percent(rate=0.00000123) == -0.88863827

    # Only custom fee without sell order applied
    with pytest.raises(TypeError):
        trade.calc_profit_percent(fee=0.003)

    # Test when we apply a Sell order. Sell higher than open rate @ 0.00001173
    trade.update(limit_sell_order)
    assert trade.calc_profit_percent() == 0.06201057

    # Test with a custom fee rate on the close trade
    assert trade.calc_profit_percent(fee=0.003) == 0.0614782
