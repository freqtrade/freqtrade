# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=invalid-sequence-index, invalid-name, too-many-arguments

from datetime import datetime
from unittest.mock import ANY, MagicMock, PropertyMock

import pytest
from numpy import isnan

from freqtrade import DependencyException, TemporaryError
from freqtrade.edge import PairInfo
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Trade
from freqtrade.rpc import RPC, RPCException
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from freqtrade.state import State
from freqtrade.tests.conftest import patch_exchange
from freqtrade.tests.test_freqtradebot import patch_get_signal


# Functions for recurrent object patching
def prec_satoshi(a, b) -> float:
    """
    :return: True if A and B differs less than one satoshi.
    """
    return abs(a - b) < 0.00000001


# Unit tests
def test_rpc_trade_status(default_conf, ticker, fee, markets, mocker) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        _load_markets=MagicMock(return_value={}),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*no active trade*'):
        rpc._rpc_trade_status()

    freqtradebot.create_trade()
    results = rpc._rpc_trade_status()

    assert {
        'trade_id': 1,
        'pair': 'ETH/BTC',
        'date': ANY,
        'open_rate': 1.099e-05,
        'close_rate': None,
        'current_rate': 1.098e-05,
        'amount': 90.99181074,
        'close_profit': None,
        'current_profit': -0.59,
        'stop_loss': 0.0,
        'initial_stop_loss': 0.0,
        'initial_stop_loss_pct': None,
        'stop_loss_pct': None,
        'open_order': '(limit buy rem=0.00000000)'
    } == results[0]

    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(side_effect=DependencyException(f"Pair 'ETH/BTC' not available")))
    # invalidate ticker cache
    rpc._freqtrade.exchange._cached_ticker = {}
    results = rpc._rpc_trade_status()
    assert isnan(results[0]['current_profit'])
    assert isnan(results[0]['current_rate'])
    assert {
        'trade_id': 1,
        'pair': 'ETH/BTC',
        'date': ANY,
        'open_rate': 1.099e-05,
        'close_rate': None,
        'current_rate': ANY,
        'amount': 90.99181074,
        'close_profit': None,
        'current_profit': ANY,
        'stop_loss': 0.0,
        'initial_stop_loss': 0.0,
        'initial_stop_loss_pct': None,
        'stop_loss_pct': None,
        'open_order': '(limit buy rem=0.00000000)'
    } == results[0]


def test_rpc_status_table(default_conf, ticker, fee, markets, mocker) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)

    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match=r'.*no active order*'):
        rpc._rpc_status_table()

    freqtradebot.create_trade()
    result = rpc._rpc_status_table()
    assert 'just now' in result['Since'].all()
    assert 'ETH/BTC' in result['Pair'].all()
    assert '-0.59%' in result['Profit'].all()

    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
                 MagicMock(side_effect=DependencyException(f"Pair 'ETH/BTC' not available")))
    # invalidate ticker cache
    rpc._freqtrade.exchange._cached_ticker = {}
    result = rpc._rpc_status_table()
    assert 'just now' in result['Since'].all()
    assert 'ETH/BTC' in result['Pair'].all()
    assert 'nan%' in result['Profit'].all()


def test_rpc_daily_profit(default_conf, update, ticker, fee,
                          limit_buy_order, limit_sell_order, markets, mocker) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()
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
    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()

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

    # Test non-available pair
    mocker.patch('freqtrade.exchange.Exchange.get_ticker',
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
def test_rpc_trade_statistics_closed(mocker, default_conf, ticker, fee, markets,
                                     ticker_sell_up, limit_buy_order, limit_sell_order):
    patch_exchange(mocker)
    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.Market',
        ticker=MagicMock(return_value={'price_usd': 15000.0}),
    )
    mocker.patch('freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price',
                 return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
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
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=mock_balance),
        get_ticker=MagicMock(side_effect=TemporaryError('Could not load ticker due to xxx'))
    )

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter()

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
    assert result['total'] == 12.0


def test_rpc_start(mocker, default_conf) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock()
    )

    freqtradebot = FreqtradeBot(config=default_conf)

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
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock()
    )

    freqtradebot = FreqtradeBot(config=default_conf)

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
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_ticker=MagicMock()
    )

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.RUNNING

    assert freqtradebot.config['max_open_trades'] != 0
    result = rpc._rpc_stopbuy()
    assert {'status': 'No more buy will occur from now. Run /reload_conf to reset.'} == result
    assert freqtradebot.config['max_open_trades'] == 0


def test_rpc_forcesell(default_conf, ticker, fee, mocker, markets) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    cancel_order_mock = MagicMock()
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
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
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
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
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
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
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets)
    )

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)

    trades = rpc._rpc_count()
    nb_trades = len(trades)
    assert nb_trades == 0

    # Create some test data
    freqtradebot.create_trade()
    trades = rpc._rpc_count()
    nb_trades = len(trades)
    assert nb_trades == 1


def test_rpcforcebuy(mocker, default_conf, ticker, fee, markets, limit_buy_order) -> None:
    default_conf['forcebuy_enable'] = True
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    buy_mm = MagicMock(return_value={'id': limit_buy_order['id']})
    mocker.patch.multiple(
        'freqtrade.exchange.Exchange',
        get_balances=MagicMock(return_value=ticker),
        get_ticker=ticker,
        get_fee=fee,
        markets=PropertyMock(return_value=markets),
        buy=buy_mm
    )

    freqtradebot = FreqtradeBot(config=default_conf)

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
    default_conf['stake_amount'] = 0.0000001

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    pair = 'TKN/BTC'
    trade = rpc._rpc_forcebuy(pair, None)
    assert trade is None


def test_rpcforcebuy_stopped(mocker, default_conf) -> None:
    default_conf['forcebuy_enable'] = True
    default_conf['initial_state'] = 'stopped'
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    pair = 'ETH/BTC'
    with pytest.raises(RPCException, match=r'trader is not running'):
        rpc._rpc_forcebuy(pair, None)


def test_rpcforcebuy_disabled(mocker, default_conf) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = FreqtradeBot(config=default_conf)

    patch_get_signal(freqtradebot, (True, False))
    rpc = RPC(freqtradebot)
    pair = 'ETH/BTC'
    with pytest.raises(RPCException, match=r'Forcebuy not enabled.'):
        rpc._rpc_forcebuy(pair, None)


def test_rpc_whitelist(mocker, default_conf) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = FreqtradeBot(config=default_conf)

    rpc = RPC(freqtradebot)
    ret = rpc._rpc_whitelist()
    assert ret['method'] == 'StaticPairList'
    assert ret['whitelist'] == default_conf['exchange']['pair_whitelist']


def test_rpc_whitelist_dynamic(mocker, default_conf) -> None:
    patch_exchange(mocker)
    default_conf['pairlist'] = {'method': 'VolumePairList',
                                'config': {'number_assets': 4}
                                }
    mocker.patch('freqtrade.exchange.Exchange.exchange_has', MagicMock(return_value=True))
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = FreqtradeBot(config=default_conf)

    rpc = RPC(freqtradebot)
    ret = rpc._rpc_whitelist()
    assert ret['method'] == 'VolumePairList'
    assert ret['length'] == 4
    assert ret['whitelist'] == default_conf['exchange']['pair_whitelist']


def test_rpc_blacklist(mocker, default_conf) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())

    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    ret = rpc._rpc_blacklist(None)
    assert ret['method'] == 'StaticPairList'
    assert len(ret['blacklist']) == 2
    assert ret['blacklist'] == default_conf['exchange']['pair_blacklist']
    assert ret['blacklist'] == ['DOGE/BTC', 'HOT/BTC']

    ret = rpc._rpc_blacklist(["ETH/BTC"])
    assert ret['method'] == 'StaticPairList'
    assert len(ret['blacklist']) == 3
    assert ret['blacklist'] == default_conf['exchange']['pair_blacklist']
    assert ret['blacklist'] == ['DOGE/BTC', 'HOT/BTC', 'ETH/BTC']


def test_rpc_edge_disabled(mocker, default_conf) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    freqtradebot = FreqtradeBot(default_conf)
    rpc = RPC(freqtradebot)
    with pytest.raises(RPCException, match=r'Edge is not enabled.'):
        rpc._rpc_edge()


def test_rpc_edge_enabled(mocker, edge_conf) -> None:
    patch_exchange(mocker)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(
        return_value={
            'E/F': PairInfo(-0.02, 0.66, 3.71, 0.50, 1.71, 10, 60),
        }
    ))
    freqtradebot = FreqtradeBot(edge_conf)

    rpc = RPC(freqtradebot)
    ret = rpc._rpc_edge()

    assert len(ret) == 1
    assert ret[0]['Pair'] == 'E/F'
    assert ret[0]['Winrate'] == 0.66
    assert ret[0]['Expectancy'] == 1.71
    assert ret[0]['Stoploss'] == -0.02
