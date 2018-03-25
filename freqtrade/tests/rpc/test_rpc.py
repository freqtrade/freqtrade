# pragma pylint: disable=invalid-sequence-index, invalid-name, too-many-arguments

"""
Unit test file for rpc/rpc.py
"""

from datetime import datetime
from unittest.mock import MagicMock

from sqlalchemy import create_engine

from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade
from freqtrade.rpc.rpc import RPC
from freqtrade.state import State
from freqtrade.tests.test_freqtradebot import patch_get_signal, patch_coinmarketcap


# Functions for recurrent object patching
def prec_satoshi(a, b) -> float:
    """
    :return: True if A and B differs less than one satoshi.
    """
    return abs(a - b) < 0.00000001


# Unit tests
def test_rpc_trade_status(default_conf, ticker, mocker) -> None:
    """
    Test rpc_trade_status() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=MagicMock(return_value=0.0025)
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.STOPPED
    (error, result) = rpc.rpc_trade_status()
    assert error
    assert 'trader is not running' in result

    freqtradebot.state = State.RUNNING
    (error, result) = rpc.rpc_trade_status()
    assert error
    assert 'no active trade' in result

    freqtradebot.create_trade()
    (error, result) = rpc.rpc_trade_status()
    assert not error
    trade = result[0]

    result_message = [
        '*Trade ID:* `1`\n'
        '*Current Pair:* '
        '[ETH/BTC](https://bittrex.com/Market/Index?MarketName=BTC-ETH)\n'
        '*Open Since:* `just now`\n'
        '*Amount:* `90.99181074`\n'
        '*Open Rate:* `0.00001099`\n'
        '*Close Rate:* `None`\n'
        '*Current Rate:* `0.00001098`\n'
        '*Close Profit:* `None`\n'
        '*Current Profit:* `-0.59%`\n'
        '*Open Order:* `(limit buy rem=0.00000000)`'
    ]
    assert result == result_message
    assert trade.find('[ETH/BTC]') >= 0


def test_rpc_status_table(default_conf, ticker, mocker) -> None:
    """
    Test rpc_status_table() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=MagicMock(return_value=0.0025)
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.STOPPED
    (error, result) = rpc.rpc_status_table()
    assert error
    assert '*Status:* `trader is not running`' in result

    freqtradebot.state = State.RUNNING
    (error, result) = rpc.rpc_status_table()
    assert error
    assert '*Status:* `no active order`' in result

    freqtradebot.create_trade()
    (error, result) = rpc.rpc_status_table()
    assert 'just now' in result['Since'].all()
    assert 'ETH/BTC' in result['Pair'].all()
    assert '-0.59%' in result['Profit'].all()


def test_rpc_daily_profit(default_conf, update, ticker, limit_buy_order, limit_sell_order, mocker)\
        -> None:
    """
    Test rpc_daily_profit() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker, value={'price_usd': 15000.0})
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=MagicMock(return_value=0.0025)
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
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
    (error, days) = rpc.rpc_daily_profit(7, stake_currency, fiat_display_currency)
    assert not error
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
    (error, days) = rpc.rpc_daily_profit(0, stake_currency, fiat_display_currency)
    assert error
    assert days.find('must be an integer greater than 0') >= 0


def test_rpc_trade_statistics(
        default_conf, ticker, ticker_sell_up, limit_buy_order, limit_sell_order, mocker) -> None:
    """
    Test rpc_trade_statistics() method
    """
    patch_get_signal(mocker, (True, False))
    mocker.patch.multiple(
        'freqtrade.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=MagicMock(return_value=0.0025)
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    rpc = RPC(freqtradebot)

    (error, stats) = rpc.rpc_trade_statistics(stake_currency, fiat_display_currency)
    assert error
    assert stats.find('no closed trade') >= 0

    # Create some test data
    freqtradebot.create_trade()
    trade = Trade.query.first()
    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Update the ticker with a market going up
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker_sell_up
    )
    trade.update(limit_sell_order)
    trade.close_date = datetime.utcnow()
    trade.is_open = False

    (error, stats) = rpc.rpc_trade_statistics(stake_currency, fiat_display_currency)
    assert not error
    assert prec_satoshi(stats['profit_closed_coin'], 6.217e-05)
    assert prec_satoshi(stats['profit_closed_percent'], 6.2)
    assert prec_satoshi(stats['profit_closed_fiat'], 0.93255)
    assert prec_satoshi(stats['profit_all_coin'], 6.217e-05)
    assert prec_satoshi(stats['profit_all_percent'], 6.2)
    assert prec_satoshi(stats['profit_all_fiat'], 0.93255)
    assert stats['trade_count'] == 1
    assert stats['first_trade_date'] == 'just now'
    assert stats['latest_trade_date'] == 'just now'
    assert stats['avg_duration'] == '0:00:00'
    assert stats['best_pair'] == 'ETH/BTC'
    assert prec_satoshi(stats['best_rate'], 6.2)


# Test that rpc_trade_statistics can handle trades that lacks
# trade.open_rate (it is set to None)
def test_rpc_trade_statistics_closed(mocker, default_conf, ticker, ticker_sell_up, limit_buy_order,
                                     limit_sell_order):
    """
    Test rpc_trade_statistics() method
    """
    patch_get_signal(mocker, (True, False))
    mocker.patch.multiple(
        'freqtrade.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        get_fee=MagicMock(return_value=0.0025)
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
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
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker_sell_up,
        get_fee=MagicMock(return_value=0.0025)
    )
    trade.update(limit_sell_order)
    trade.close_date = datetime.utcnow()
    trade.is_open = False

    for trade in Trade.query.order_by(Trade.id).all():
        trade.open_rate = None

    (error, stats) = rpc.rpc_trade_statistics(stake_currency, fiat_display_currency)
    assert not error
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
    mock_balance = [
        {
            'Currency': 'BTC',
            'Balance': 10.0,
            'Available': 12.0,
            'Pending': 0.0,
            'CryptoAddress': 'XXXX',
        },
        {
            'Currency': 'ETH',
            'Balance': 0.0,
            'Available': 0.0,
            'Pending': 0.0,
            'CryptoAddress': 'XXXX',
        }
    ]

    patch_get_signal(mocker, (True, False))
    mocker.patch.multiple(
        'freqtrade.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_balances=MagicMock(return_value=mock_balance)
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    rpc = RPC(freqtradebot)

    (error, res) = rpc.rpc_balance(default_conf['fiat_display_currency'])
    assert not error
    (trade, x, y, z) = res
    assert prec_satoshi(x, 10)
    assert prec_satoshi(z, 150000)
    assert 'USD' in y
    assert len(trade) == 1
    assert 'BTC' in trade[0]['currency']
    assert prec_satoshi(trade[0]['available'], 12)
    assert prec_satoshi(trade[0]['balance'], 10)
    assert prec_satoshi(trade[0]['pending'], 0)
    assert prec_satoshi(trade[0]['est_btc'], 10)


def test_rpc_start(mocker, default_conf) -> None:
    """
    Test rpc_start() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=MagicMock()
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.STOPPED

    (error, result) = rpc.rpc_start()
    assert not error
    assert '`Starting trader ...`' in result
    assert freqtradebot.state == State.RUNNING

    (error, result) = rpc.rpc_start()
    assert error
    assert '*Status:* `already running`' in result
    assert freqtradebot.state == State.RUNNING


def test_rpc_stop(mocker, default_conf) -> None:
    """
    Test rpc_stop() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=MagicMock()
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.RUNNING

    (error, result) = rpc.rpc_stop()
    assert not error
    assert '`Stopping trader ...`' in result
    assert freqtradebot.state == State.STOPPED

    (error, result) = rpc.rpc_stop()
    assert error
    assert '*Status:* `already stopped`' in result
    assert freqtradebot.state == State.STOPPED


def test_rpc_forcesell(default_conf, ticker, mocker) -> None:
    """
    Test rpc_forcesell() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())

    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_ticker=ticker,
        cancel_order=cancel_order_mock,
        get_order=MagicMock(
            return_value={
                'status': 'closed',
                'type': 'limit',
                'side': 'buy'
            }
        )
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.STOPPED
    (error, res) = rpc.rpc_forcesell(None)
    assert error
    assert res == '`trader is not running`'

    freqtradebot.state = State.RUNNING
    (error, res) = rpc.rpc_forcesell(None)
    assert error
    assert res == 'Invalid argument.'

    (error, res) = rpc.rpc_forcesell('all')
    assert not error
    assert res == ''

    freqtradebot.create_trade()
    (error, res) = rpc.rpc_forcesell('all')
    assert not error
    assert res == ''

    (error, res) = rpc.rpc_forcesell('1')
    assert not error
    assert res == ''

    freqtradebot.state = State.STOPPED
    (error, res) = rpc.rpc_forcesell(None)
    assert error
    assert res == '`trader is not running`'

    (error, res) = rpc.rpc_forcesell('all')
    assert error
    assert res == '`trader is not running`'

    freqtradebot.state = State.RUNNING
    assert cancel_order_mock.call_count == 0
    # make an limit-buy open trade
    mocker.patch(
        'freqtrade.freqtradebot.exchange.get_order',
        return_value={
            'status': 'open',
            'type': 'limit',
            'side': 'buy'
        }
    )
    # check that the trade is called, which is done
    # by ensuring exchange.cancel_order is called
    (error, res) = rpc.rpc_forcesell('1')
    assert not error
    assert res == ''
    assert cancel_order_mock.call_count == 1

    freqtradebot.create_trade()
    # make an limit-sell open trade
    mocker.patch(
        'freqtrade.freqtradebot.exchange.get_order',
        return_value={
            'status': 'open',
            'type': 'limit',
            'side': 'sell'
        }
    )
    (error, res) = rpc.rpc_forcesell('2')
    assert not error
    assert res == ''
    # status quo, no exchange calls
    assert cancel_order_mock.call_count == 1


def test_performance_handle(default_conf, ticker, limit_buy_order,
                            limit_sell_order, mocker) -> None:
    """
    Test rpc_performance() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=MagicMock(return_value=0.0025)
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
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
    (error, res) = rpc.rpc_performance()
    assert not error
    assert len(res) == 1
    assert res[0]['pair'] == 'ETH/BTC'
    assert res[0]['count'] == 1
    assert prec_satoshi(res[0]['profit'], 6.2)


def test_rpc_count(mocker, default_conf, ticker) -> None:
    """
    Test rpc_count() method
    """
    patch_get_signal(mocker, (True, False))
    patch_coinmarketcap(mocker)
    mocker.patch('freqtrade.rpc.rpc_manager.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.freqtradebot.exchange',
        validate_pairs=MagicMock(),
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker
    )

    freqtradebot = FreqtradeBot(default_conf, create_engine('sqlite://'))
    rpc = RPC(freqtradebot)

    (error, trades) = rpc.rpc_count()
    nb_trades = len(trades)
    assert not error
    assert nb_trades == 0

    # Create some test data
    freqtradebot.create_trade()
    (error, trades) = rpc.rpc_count()
    nb_trades = len(trades)
    assert not error
    assert nb_trades == 1
