# pragma pylint: disable=invalid-sequence-index, invalid-name, too-many-arguments

"""
Unit test file for rpc/rpc.py
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade
from freqtrade.rpc.rpc import RPC, RPCException
from freqtrade.state import State
from freqtrade.tests.test_freqtradebot import (patch_coinmarketcap,
                                               patch_get_signal)


# Functions for recurrent object patching
def prec_satoshi(a, b) -> float:
    """
    :return: True if A and B differs less than one satoshi.
    """
    return abs(a - b) < 0.00000001


# Unit tests
def test_rpc_trade_status(default_conf, ticker, fee, markets, mocker) -> None:
    """
    Test rpc_trade_status() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=fee,
        get_markets=markets
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.STOPPED
    with pytest.raises(RPCException, match=r'.*trader is not running*'):
        rpc._rpc_trade_status()

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*no active trade*'):
        rpc._rpc_trade_status()

    freqtradebot.create_trade()
    results = rpc._rpc_trade_status()

    assert {
        'trade_id': 1,
        'pair': 'ETH/BTC',
        'market_url': 'https://bittrex.com/Market/Index?MarketName=BTC-ETH',
        'date': 'just now',
        'open_rate': 1.099e-05,
        'close_rate': None,
        'current_rate': 1.098e-05,
        'amount': 90.99181074,
        'close_profit': None,
        'current_profit': -0.59,
        'open_order': '(limit buy rem=0.00000000)'
    } == results[0]


def test_rpc_status_table(default_conf, ticker, fee, markets, mocker) -> None:
    """
    Test rpc_status_table() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=fee,
        get_markets=markets
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.STOPPED
    with pytest.raises(RPCException, match=r'.*trader is not running*'):
        rpc._rpc_status_table()

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*no active order*'):
        rpc._rpc_status_table()

    freqtradebot.create_trade()
    result = rpc._rpc_status_table()
    assert 'just now' in result['Since'].all()
    assert 'ETH/BTC' in result['Pair'].all()
    assert '-0.59%' in result['Profit'].all()


def test_rpc_daily_profit(default_conf, update, ticker, fee,
                          limit_buy_order, limit_sell_order, markets, mocker) -> None:
    """
    Test rpc_daily_profit() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=fee,
        get_markets=markets
    )

    freqtradebot = FreqtradeBot(default_conf)
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    rpc = RPC(freqtradebot)

    # Create some test data
    freqtradebot.create_trade()
    trade = Trade.query.first()
    assert trade

    # Simulate buy & sell
    trade.update(limit_buy_order)
    trade.update(limit_sell_order)
    trade.close_date = datetime.utcnow()
    trade.is_open = False

    # Try valid data
    update.message.text = '/daily 2'
    days = rpc._rpc_daily_profit(7, stake_currency, fiat_display_currency)
    assert len(days) == 7
    for day in days:
        # [datetime.date(2018, 1, 11), '0.00000000 BTC', '0.000 USD']
        assert (day[1] == '0.00000000 BTC' or
                day[1] == '0.00006217 BTC')

        assert (day[2] == '0.000 USD' or
                day[2] == '0.933 USD')
    # ensure first day is current date
    assert str(days[0][0]) == str(datetime.utcnow().date())

    # Try invalid data
    with pytest.raises(RPCException, match=r'.*must be an integer greater than 0*'):
        rpc._rpc_daily_profit(0, stake_currency, fiat_display_currency)


def test_rpc_trade_statistics(default_conf, ticker, ticker_sell_up, fee,
                              limit_buy_order, limit_sell_order, markets, mocker) -> None:
    """
    Test rpc_trade_statistics() method
    """
    patch_get_signal(mocker, (True, False))
    mocker.patch.multiple(
        'freqtrade.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=fee,
        get_markets=markets
    )

    freqtradebot = FreqtradeBot(default_conf)
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    rpc = RPC(freqtradebot)

    with pytest.raises(RPCException, match=r'.*no closed trade*'):
        rpc._rpc_trade_statistics(stake_currency, fiat_display_currency)

    # Create some test data
    freqtradebot.create_trade()
    trade = Trade.query.first()
    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Update the ticker with a market going up
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker_sell_up
    )
    trade.update(limit_sell_order)
    trade.close_date = datetime.utcnow()
    trade.is_open = False

    freqtradebot.create_trade()
    trade = Trade.query.first()
    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Update the ticker with a market going up
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker_sell_up
    )
    trade.update(limit_sell_order)
    trade.close_date = datetime.utcnow()
    trade.is_open = False

    stats = rpc._rpc_trade_statistics(stake_currency, fiat_display_currency)
    assert prec_satoshi(stats['profit_closed_coin'], 6.217e-05)
    assert prec_satoshi(stats['profit_closed_percent'], 6.2)
    assert prec_satoshi(stats['profit_closed_fiat'], 0.93255)
    assert prec_satoshi(stats['profit_all_coin'], 5.632e-05)
    assert prec_satoshi(stats['profit_all_percent'], 2.81)
    assert prec_satoshi(stats['profit_all_fiat'], 0.8448)
    assert stats['trade_count'] == 2
    assert stats['first_trade_date'] == 'just now'
    assert stats['latest_trade_date'] == 'just now'
    assert stats['avg_duration'] == '0:00:00'
    assert stats['best_pair'] == 'ETH/BTC'
    assert prec_satoshi(stats['best_rate'], 6.2)


# Test that rpc_trade_statistics can handle trades that lacks
# trade.open_rate (it is set to None)
def test_rpc_trade_statistics_closed(mocker, default_conf, ticker, fee, markets,
                                     ticker_sell_up, limit_buy_order, limit_sell_order):
    """
    Test rpc_trade_statistics() method
    """
    patch_get_signal(mocker, (True, False))
    mocker.patch.multiple(
        'freqtrade.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=fee,
        get_markets=markets
    )

    freqtradebot = FreqtradeBot(default_conf)
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    rpc = RPC(freqtradebot)

    # Create some test data
    freqtradebot.create_trade()
    trade = Trade.query.first()
    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)
    # Update the ticker with a market going up
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker_sell_up,
        get_fee=fee
    )
    trade.update(limit_sell_order)
    trade.close_date = datetime.utcnow()
    trade.is_open = False

    for trade in Trade.query.order_by(Trade.id).all():
        trade.open_rate = None

    stats = rpc._rpc_trade_statistics(stake_currency, fiat_display_currency)
    assert prec_satoshi(stats['profit_closed_coin'], 0)
    assert prec_satoshi(stats['profit_closed_percent'], 0)
    assert prec_satoshi(stats['profit_closed_fiat'], 0)
    assert prec_satoshi(stats['profit_all_coin'], 0)
    assert prec_satoshi(stats['profit_all_percent'], 0)
    assert prec_satoshi(stats['profit_all_fiat'], 0)
    assert stats['trade_count'] == 1
    assert stats['first_trade_date'] == 'just now'
    assert stats['latest_trade_date'] == 'just now'
    assert stats['avg_duration'] == '0:00:00'
    assert stats['best_pair'] == 'ETH/BTC'
    assert prec_satoshi(stats['best_rate'], 6.2)


def test_rpc_balance_handle(default_conf, mocker):
    """
    Test rpc_balance() method
    """
    mock_balance = {
        'BTC': {
            'free': 10.0,
            'total': 12.0,
            'used': 2.0,
        },
        'ETH': {
            'free': 0.0,
            'total': 0.0,
            'used': 0.0,
        }
    }

    patch_get_signal(mocker, (True, False))
    mocker.patch.multiple(
        'freqtrade.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_balances=MagicMock(return_value=mock_balance)
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)

    result = rpc._rpc_balance(default_conf['fiat_display_currency'])
    assert prec_satoshi(result['total'], 12)
    assert prec_satoshi(result['value'], 180000)
    assert 'USD' == result['symbol']
    assert result['currencies'] == [{
        'currency': 'BTC',
        'available': 10.0,
        'balance': 12.0,
        'pending': 2.0,
        'est_btc': 12.0,
    }]


def test_rpc_start(mocker, default_conf) -> None:
    """
    Test rpc_start() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=MagicMock()
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.STOPPED

    result = rpc._rpc_start()
    assert {'status': 'starting trader ...'} == result
    assert freqtradebot.state == State.RUNNING

    result = rpc._rpc_start()
    assert {'status': 'already running'} == result
    assert freqtradebot.state == State.RUNNING


def test_rpc_stop(mocker, default_conf) -> None:
    """
    Test rpc_stop() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=MagicMock()
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.RUNNING

    result = rpc._rpc_stop()
    assert {'status': 'stopping trader ...'} == result
    assert freqtradebot.state == State.STOPPED

    result = rpc._rpc_stop()

    assert {'status': 'already stopped'} == result
    assert freqtradebot.state == State.STOPPED


def test_rpc_forcesell(default_conf, ticker, fee, mocker, markets) -> None:
    """
    Test rpc_forcesell() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        cancel_order=cancel_order_mock,
        get_order=MagicMock(
            return_value={
                'status': 'closed',
                'type': 'limit',
                'side': 'buy'
            }
        ),
        get_fee=fee,
        get_markets=markets
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.STOPPED
    with pytest.raises(RPCException, match=r'.*trader is not running*'):
        rpc._rpc_forcesell(None)

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*invalid argument*'):
        rpc._rpc_forcesell(None)

    rpc._rpc_forcesell('all')

    freqtradebot.create_trade()
    rpc._rpc_forcesell('all')

    rpc._rpc_forcesell('1')

    freqtradebot.state = State.STOPPED
    with pytest.raises(RPCException, match=r'.*trader is not running*'):
        rpc._rpc_forcesell(None)

    with pytest.raises(RPCException, match=r'.*trader is not running*'):
        rpc._rpc_forcesell('all')

    freqtradebot.state = State.RUNNING
    assert cancel_order_mock.call_count == 0
    # make an limit-buy open trade
    trade = Trade.query.filter(Trade.id == '1').first()
    filled_amount = trade.amount / 2
    mocker.patch(
        'freqtrade.exchange.Exchange.get_order',
        return_value={
            'status': 'open',
            'type': 'limit',
            'side': 'buy',
            'filled': filled_amount
        }
    )
    # check that the trade is called, which is done by ensuring exchange.cancel_order is called
    # and trade amount is updated
    rpc._rpc_forcesell('1')
    assert cancel_order_mock.call_count == 1
    assert trade.amount == filled_amount

    freqtradebot.create_trade()
    trade = Trade.query.filter(Trade.id == '2').first()
    amount = trade.amount
    # make an limit-buy open trade, if there is no 'filled', don't sell it
    mocker.patch(
        'freqtrade.exchange.Exchange.get_order',
        return_value={
            'status': 'open',
            'type': 'limit',
            'side': 'buy',
            'filled': None
        }
    )
    # check that the trade is called, which is done by ensuring exchange.cancel_order is called
    rpc._rpc_forcesell('2')
    assert cancel_order_mock.call_count == 2
    assert trade.amount == amount

    freqtradebot.create_trade()
    # make an limit-sell open trade
    mocker.patch(
        'freqtrade.exchange.Exchange.get_order',
        return_value={
            'status': 'open',
            'type': 'limit',
            'side': 'sell'
        }
    )
    rpc._rpc_forcesell('3')
    # status quo, no exchange calls
    assert cancel_order_mock.call_count == 2


def test_performance_handle(default_conf, ticker, limit_buy_order, fee,
                            limit_sell_order, markets, mocker) -> None:
    """
    Test rpc_performance() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        get_markets=markets
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)

    # Create some test data
    freqtradebot.create_trade()
    trade = Trade.query.first()
    assert trade

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False
    res = rpc._rpc_performance()
    assert len(res) == 1
    assert res[0]['pair'] == 'ETH/BTC'
    assert res[0]['count'] == 1
    assert prec_satoshi(res[0]['profit'], 6.2)


def test_rpc_count(mocker, default_conf, ticker, fee, markets) -> None:
    """
    Test rpc_count() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        validate_pairs=MagicMock(),
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        get_markets=markets
    )

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)

    trades = rpc._rpc_count()
    nb_trades = len(trades)
    assert nb_trades == 0

    # Create some test data
    freqtradebot.create_trade()
    trades = rpc._rpc_count()
    nb_trades = len(trades)
    assert nb_trades == 1
