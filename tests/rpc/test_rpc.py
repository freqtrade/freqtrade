# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=invalid-sequence-index, invalid-name, too-many-arguments

from datetime import datetime
from unittest.mock import ANY, MagicMock, PropertyMock

import pytest
from numpy import isnan

from freqtrade import DependencyException, TemporaryError
from freqtrade.edge import PairInfo
from freqtrade.persistence import Trade
from freqtrade.rpc import RPC, RPCException
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from freqtrade.state import State
from tests.conftest import patch_get_signal, get_patched_freqtradebot


# Functions for recurrent object patching
def prec_satoshi(a, b) -> float:
    """
    :return: True if A and B differs less than one satoshi.
    """
    return abs(a - b) < 0.00000001


# Unit tests
def test_rpc_trade_status(default_conf, ticker, fee, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*no active trade*'):
        rpc._rpc_trade_status()

    freqtradebot.process_maybe_execute_buys()
    results = rpc._rpc_trade_status()
    assert {
        'trade_id': 1,
        'pair': 'ETH/BTC',
        'base_currency': 'BTC',
        'open_date': ANY,
        'open_date_hum': ANY,
        'close_date': None,
        'close_date_hum': None,
        'open_rate': 1.099e-05,
        'close_rate': None,
        'current_rate': 1.098e-05,
        'amount': 90.99181074,
        'stake_amount': 0.001,
        'close_profit': None,
        'current_profit': -0.59,
        'stop_loss': 0.0,
        'initial_stop_loss': 0.0,
        'initial_stop_loss_pct': None,
        'stop_loss_pct': None,
        'open_order': '(limit buy rem=0.00000000)'
    } == results[0]

    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(side_effect=DependencyException(f"Pair 'ETH/BTC' not available")))
    # invalidate ticker cache
    rpc._freqtrade.exchange._cached_ticker = {}
    results = rpc._rpc_trade_status()
    assert isnan(results[0]['current_profit'])
    assert isnan(results[0]['current_rate'])
    assert {
        'trade_id': 1,
        'pair': 'ETH/BTC',
        'base_currency': 'BTC',
        'open_date': ANY,
        'open_date_hum': ANY,
        'close_date': None,
        'close_date_hum': None,
        'open_rate': 1.099e-05,
        'close_rate': None,
        'current_rate': ANY,
        'amount': 90.99181074,
        'stake_amount': 0.001,
        'close_profit': None,
        'current_profit': ANY,
        'stop_loss': 0.0,
        'initial_stop_loss': 0.0,
        'initial_stop_loss_pct': None,
        'stop_loss_pct': None,
        'open_order': '(limit buy rem=0.00000000)'
    } == results[0]


def test_rpc_status_table(default_conf, ticker, fee, mocker) -> None:
    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*no active order*'):
        rpc._rpc_status_table(default_conf['stake_currency'], 'USD')

    freqtradebot.process_maybe_execute_buys()

    result, headers = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert "Since" in headers
    assert "Pair" in headers
    assert 'instantly' == result[0][2]
    assert 'ETH/BTC' == result[0][1]
    assert '-0.59%' == result[0][3]
    # Test with fiatconvert

    rpc._fiat_converter = CryptoToFiatConverter()
    result, headers = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert "Since" in headers
    assert "Pair" in headers
    assert 'instantly' == result[0][2]
    assert 'ETH/BTC' == result[0][1]
    assert '-0.59% (-0.09)' == result[0][3]

    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(side_effect=DependencyException(f"Pair 'ETH/BTC' not available")))
    # invalidate ticker cache
    rpc._freqtrade.exchange._cached_ticker = {}
    result, headers = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert 'instantly' == result[0][2]
    assert 'ETH/BTC' == result[0][1]
    assert 'nan%' == result[0][3]


def test_rpc_daily_profit(default_conf, update, ticker, fee,
                          limit_buy_order, limit_sell_order, markets, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()
    # Create some test data
    freqtradebot.process_maybe_execute_buys()
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
                              limit_buy_order, limit_sell_order, mocker) -> None:
    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()

    with pytest.raises(RPCException, match=r'.*no closed trade*'):
        rpc._rpc_trade_statistics(stake_currency, fiat_display_currency)

    # Create some test data
    freqtradebot.process_maybe_execute_buys()
    trade = Trade.query.first()
    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Update the ticker with a market going up
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_up
    )
    trade.update(limit_sell_order)
    trade.close_date = datetime.utcnow()
    trade.is_open = False

    freqtradebot.process_maybe_execute_buys()
    trade = Trade.query.first()
    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Update the ticker with a market going up
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_up
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

    # Test non-available pair
    mocker.patch('freqtrade.exchange.Exchange.fetch_ticker',
                 MagicMock(side_effect=DependencyException(f"Pair 'ETH/BTC' not available")))
    # invalidate ticker cache
    rpc._freqtrade.exchange._cached_ticker = {}
    stats = rpc._rpc_trade_statistics(stake_currency, fiat_display_currency)
    assert stats['trade_count'] == 2
    assert stats['first_trade_date'] == 'just now'
    assert stats['latest_trade_date'] == 'just now'
    assert stats['avg_duration'] == '0:00:00'
    assert stats['best_pair'] == 'ETH/BTC'
    assert prec_satoshi(stats['best_rate'], 6.2)
    assert isnan(stats['profit_all_coin'])


# Test that rpc_trade_statistics can handle trades that lacks
# trade.open_rate (it is set to None)
def test_rpc_trade_statistics_closed(mocker, default_conf, ticker, fee,
                                     ticker_sell_up, limit_buy_order, limit_sell_order):
    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    rpc = RPC(freqtradebot)

    # Create some test data
    freqtradebot.process_maybe_execute_buys()
    trade = Trade.query.first()
    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)
    # Update the ticker with a market going up
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker_sell_up,
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


def test_rpc_balance_handle_error(default_conf, mocker):
    mock_balance = {
        'BTC': {
            'free': 10.0,
            'total': 12.0,
            'used': 2.0,
        },
        'ETH': {
            'free': 1.0,
            'total': 5.0,
            'used': 4.0,
        }
    }
    # ETH will be skipped due to mocked Error below

    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=mock_balance),
        get_tickers=MagicMock(side_effect=TemporaryError('Could not load ticker due to xxx'))
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()
    with pytest.raises(RPCException, match="Error getting current tickers."):
        rpc._rpc_balance(default_conf['stake_currency'], default_conf['fiat_display_currency'])


def test_rpc_balance_handle(default_conf, mocker, tickers):
    mock_balance = {
        'BTC': {
            'free': 10.0,
            'total': 12.0,
            'used': 2.0,
        },
        'ETH': {
            'free': 1.0,
            'total': 5.0,
            'used': 4.0,
        },
        'USDT': {
            'free': 5.0,
            'total': 10.0,
            'used': 5.0,
        }
    }

    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=mock_balance),
        get_tickers=tickers,
        get_valid_pair_combination=MagicMock(
            side_effect=lambda a, b: f"{b}/{a}" if a == "USDT" else f"{a}/{b}")
    )
    default_conf['dry_run'] = False
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()

    result = rpc._rpc_balance(default_conf['stake_currency'], default_conf['fiat_display_currency'])
    assert prec_satoshi(result['total'], 12.309096315)
    assert prec_satoshi(result['value'], 184636.44472997)
    assert 'USD' == result['symbol']
    assert result['currencies'] == [
        {'currency': 'BTC',
         'free': 10.0,
         'balance': 12.0,
         'used': 2.0,
         'est_stake': 12.0,
         'stake': 'BTC',
         },
        {'free': 1.0,
         'balance': 5.0,
         'currency': 'ETH',
         'est_stake': 0.30794,
         'used': 4.0,
         'stake': 'BTC',

         },
        {'free': 5.0,
         'balance': 10.0,
         'currency': 'USDT',
         'est_stake': 0.0011563153318162476,
         'used': 5.0,
         'stake': 'BTC',
         }
    ]
    assert result['total'] == 12.309096315331816


def test_rpc_start(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock()
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.STOPPED

    result = rpc._rpc_start()
    assert {'status': 'starting trader ...'} == result
    assert freqtradebot.state == State.RUNNING

    result = rpc._rpc_start()
    assert {'status': 'already running'} == result
    assert freqtradebot.state == State.RUNNING


def test_rpc_stop(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock()
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.RUNNING

    result = rpc._rpc_stop()
    assert {'status': 'stopping trader ...'} == result
    assert freqtradebot.state == State.STOPPED

    result = rpc._rpc_stop()

    assert {'status': 'already stopped'} == result
    assert freqtradebot.state == State.STOPPED


def test_rpc_stopbuy(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=MagicMock()
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.RUNNING

    assert freqtradebot.config['max_open_trades'] != 0
    result = rpc._rpc_stopbuy()
    assert {'status': 'No more buy will occur from now. Run /reload_conf to reset.'} == result
    assert freqtradebot.config['max_open_trades'] == 0


def test_rpc_forcesell(default_conf, ticker, fee, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        fetch_ticker=ticker,
        cancel_order=cancel_order_mock,
        get_order=MagicMock(
            return_value={
                'status': 'closed',
                'type': 'limit',
                'side': 'buy'
            }
        ),
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.STOPPED
    with pytest.raises(RPCException, match=r'.*trader is not running*'):
        rpc._rpc_forcesell(None)

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*invalid argument*'):
        rpc._rpc_forcesell(None)

    msg = rpc._rpc_forcesell('all')
    assert msg == {'result': 'Created sell orders for all open trades.'}

    freqtradebot.process_maybe_execute_buys()
    msg = rpc._rpc_forcesell('all')
    assert msg == {'result': 'Created sell orders for all open trades.'}

    msg = rpc._rpc_forcesell('1')
    assert msg == {'result': 'Created sell order for trade 1.'}

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

    freqtradebot.process_maybe_execute_buys()
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
    msg = rpc._rpc_forcesell('2')
    assert msg == {'result': 'Created sell order for trade 2.'}
    assert cancel_order_mock.call_count == 2
    assert trade.amount == amount

    freqtradebot.process_maybe_execute_buys()
    # make an limit-sell open trade
    mocker.patch(
        'freqtrade.exchange.Exchange.get_order',
        return_value={
            'status': 'open',
            'type': 'limit',
            'side': 'sell'
        }
    )
    msg = rpc._rpc_forcesell('3')
    assert msg == {'result': 'Created sell order for trade 3.'}
    # status quo, no exchange calls
    assert cancel_order_mock.call_count == 2


def test_performance_handle(default_conf, ticker, limit_buy_order, fee,
                            limit_sell_order, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)

    # Create some test data
    freqtradebot.process_maybe_execute_buys()
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


def test_rpc_count(mocker, default_conf, ticker, fee) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)

    counts = rpc._rpc_count()
    assert counts["current"] == 0

    # Create some test data
    freqtradebot.process_maybe_execute_buys()
    counts = rpc._rpc_count()
    assert counts["current"] == 1


def test_rpcforcebuy(mocker, default_conf, ticker, fee, limit_buy_order) -> None:
    default_conf['forcebuy_enable'] = True
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    buy_mm = MagicMock(return_value={'id': limit_buy_order['id']})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        fetch_ticker=ticker,
        get_fee=fee,
        buy=buy_mm
    )

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    pair = 'ETH/BTC'
    trade = rpc._rpc_forcebuy(pair, None)
    assert isinstance(trade, Trade)
    assert trade.pair == pair
    assert trade.open_rate == ticker()['ask']

    # Test buy duplicate
    with pytest.raises(RPCException, match=r'position for ETH/BTC already open - id: 1'):
        rpc._rpc_forcebuy(pair, 0.0001)
    pair = 'XRP/BTC'
    trade = rpc._rpc_forcebuy(pair, 0.0001)
    assert isinstance(trade, Trade)
    assert trade.pair == pair
    assert trade.open_rate == 0.0001

    # Test buy pair not with stakes
    with pytest.raises(RPCException, match=r'Wrong pair selected. Please pairs with stake.*'):
        rpc._rpc_forcebuy('XRP/ETH', 0.0001)
    pair = 'XRP/BTC'

    # Test not buying
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    freqtradebot.config['stake_amount'] = 0.0000001
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    pair = 'TKN/BTC'
    trade = rpc._rpc_forcebuy(pair, None)
    assert trade is None


def test_rpcforcebuy_stopped(mocker, default_conf) -> None:
    default_conf['forcebuy_enable'] = True
    default_conf['initial_state'] = 'stopped'
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    pair = 'ETH/BTC'
    with pytest.raises(RPCException, match=r'trader is not running'):
        rpc._rpc_forcebuy(pair, None)


def test_rpcforcebuy_disabled(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    pair = 'ETH/BTC'
    with pytest.raises(RPCException, match=r'Forcebuy not enabled.'):
        rpc._rpc_forcebuy(pair, None)


def test_rpc_whitelist(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)
    ret = rpc._rpc_whitelist()
    assert len(ret['method']) == 1
    assert 'StaticPairList' in ret['method']
    assert ret['whitelist'] == default_conf['exchange']['pair_whitelist']


def test_rpc_whitelist_dynamic(mocker, default_conf) -> None:
    default_conf['pairlists'] = [{'method': 'VolumePairList',
                                  'number_assets': 4,
                                  }]
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)
    ret = rpc._rpc_whitelist()
    assert len(ret['method']) == 1
    assert 'VolumePairList' in ret['method']
    assert ret['length'] == 4
    assert ret['whitelist'] == default_conf['exchange']['pair_whitelist']


def test_rpc_blacklist(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)
    ret = rpc._rpc_blacklist(None)
    assert len(ret['method']) == 1
    assert 'StaticPairList' in ret['method']
    assert len(ret['blacklist']) == 2
    assert ret['blacklist'] == default_conf['exchange']['pair_blacklist']
    assert ret['blacklist'] == ['DOGE/BTC', 'HOT/BTC']

    ret = rpc._rpc_blacklist(["ETH/BTC"])
    assert 'StaticPairList' in ret['method']
    assert len(ret['blacklist']) == 3
    assert ret['blacklist'] == default_conf['exchange']['pair_blacklist']
    assert ret['blacklist'] == ['DOGE/BTC', 'HOT/BTC', 'ETH/BTC']


def test_rpc_edge_disabled(mocker, default_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtradebot)
    with pytest.raises(RPCException, match=r'Edge is not enabled.'):
        rpc._rpc_edge()


def test_rpc_edge_enabled(mocker, edge_conf) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value={
            'E/F': PairInfo(-0.02, 0.66, 3.71, 0.50, 1.71, 10, 60),
        }
    ))
    freqtradebot = get_patched_freqtradebot(mocker, edge_conf)

    rpc = RPC(freqtradebot)
    ret = rpc._rpc_edge()

    assert len(ret) == 1
    assert ret[0]['Pair'] == 'E/F'
    assert ret[0]['Winrate'] == 0.66
    assert ret[0]['Expectancy'] == 1.71
    assert ret[0]['Stoploss'] == -0.02
