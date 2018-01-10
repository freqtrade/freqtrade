# pragma pylint: disable=missing-docstring,C0103
import copy
from unittest.mock import MagicMock
import freqtrade.tests.conftest as tt  # test tools

import pytest
import requests
import logging
import arrow
from sqlalchemy import create_engine

from freqtrade import DependencyException, OperationalException
from freqtrade.analyze import SignalType
from freqtrade.exchange import Exchanges
from freqtrade.main import create_trade, handle_trade, init, \
    get_target_bid, _process, execute_sell, check_handle_timedout
from freqtrade.misc import get_state, State
from freqtrade.persistence import Trade
import freqtrade.main as main


# Test that main() can start backtesting or hyperopt.
# and also ensure we can pass some specific arguments
# argument parsing is done in test_misc.py

def test_parse_args_backtesting(mocker):
    backtesting_mock = mocker.patch(
        'freqtrade.optimize.backtesting.start', MagicMock())
    with pytest.raises(SystemExit, match=r'0'):
        main.main(['backtesting'])
    assert backtesting_mock.call_count == 1
    call_args = backtesting_mock.call_args[0][0]
    assert call_args.config == 'config.json'
    assert call_args.live is False
    assert call_args.loglevel == 20
    assert call_args.subparser == 'backtesting'
    assert call_args.func is not None
    assert call_args.ticker_interval == 5


def test_main_start_hyperopt(mocker):
    hyperopt_mock = mocker.patch(
        'freqtrade.optimize.hyperopt.start', MagicMock())
    with pytest.raises(SystemExit, match=r'0'):
        main.main(['hyperopt'])
    assert hyperopt_mock.call_count == 1
    call_args = hyperopt_mock.call_args[0][0]
    assert call_args.config == 'config.json'
    assert call_args.loglevel == 20
    assert call_args.subparser == 'hyperopt'
    assert call_args.func is not None


def test_process_maybe_execute_buy(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.create_trade', return_value=True)
    assert main.process_maybe_execute_buy(default_conf)
    mocker.patch('freqtrade.main.create_trade', return_value=False)
    assert not main.process_maybe_execute_buy(default_conf)


def test_process_maybe_execute_sell(default_conf, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.handle_trade', return_value=True)
    mocker.patch('freqtrade.exchange.get_order', return_value=1)
    trade = MagicMock()
    trade.open_order_id = '123'
    assert not main.process_maybe_execute_sell(trade)
    trade.is_open = True
    trade.open_order_id = None
    # Assert we call handle_trade() if trade is feasible for execution
    assert main.process_maybe_execute_sell(trade)


def test_process_maybe_execute_buy_exception(default_conf, mocker, caplog):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.create_trade', MagicMock(side_effect=DependencyException))
    main.process_maybe_execute_buy(default_conf)
    tt.log_has('Unable to create trade:', caplog.record_tuples)


def test_process_trade_creation(default_conf, ticker, limit_buy_order, health, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_wallet_health=health,
                          buy=MagicMock(return_value='mocked_limit_buy'),
                          get_order=MagicMock(return_value=limit_buy_order))
    init(default_conf, create_engine('sqlite://'))

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert not trades

    result = _process()
    assert result is True

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1
    trade = trades[0]
    assert trade is not None
    assert trade.stake_amount == default_conf['stake_amount']
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == Exchanges.BITTREX.name
    assert trade.open_rate == 0.00001099
    assert trade.amount == 90.99181073703367


def test_process_exchange_failures(default_conf, ticker, health, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    sleep_mock = mocker.patch('time.sleep', side_effect=lambda _: None)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_wallet_health=health,
                          buy=MagicMock(side_effect=requests.exceptions.RequestException))
    init(default_conf, create_engine('sqlite://'))
    result = _process()
    assert result is False
    assert sleep_mock.has_calls()


def test_process_operational_exception(default_conf, ticker, health, mocker):
    msg_mock = MagicMock()
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=msg_mock)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_wallet_health=health,
                          buy=MagicMock(side_effect=OperationalException))
    init(default_conf, create_engine('sqlite://'))
    assert get_state() == State.RUNNING

    result = _process()
    assert result is False
    assert get_state() == State.STOPPED
    assert 'OperationalException' in msg_mock.call_args_list[-1][0][0]


def test_process_trade_handling(default_conf, ticker, limit_buy_order, health, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch('freqtrade.main.get_signal',
                 side_effect=lambda *args: False if args[1] == SignalType.SELL else True)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_wallet_health=health,
                          buy=MagicMock(return_value='mocked_limit_buy'),
                          get_order=MagicMock(return_value=limit_buy_order))
    init(default_conf, create_engine('sqlite://'))

    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert not trades
    result = _process()
    assert result is True
    trades = Trade.query.filter(Trade.is_open.is_(True)).all()
    assert len(trades) == 1

    result = _process()
    assert result is False


def test_create_trade(default_conf, ticker, limit_buy_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'))
    # Save state of current whitelist
    whitelist = copy.deepcopy(default_conf['exchange']['pair_whitelist'])

    init(default_conf, create_engine('sqlite://'))
    create_trade(0.001)

    trade = Trade.query.first()
    assert trade is not None
    assert trade.stake_amount == 0.001
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == Exchanges.BITTREX.name

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    assert trade.open_rate == 0.00001099
    assert trade.amount == 90.99181073

    assert whitelist == default_conf['exchange']['pair_whitelist']


def test_create_trade_minimal_amount(default_conf, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    buy_mock = mocker.patch(
        'freqtrade.main.exchange.buy', MagicMock(return_value='mocked_limit_buy')
    )
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    init(default_conf, create_engine('sqlite://'))
    min_stake_amount = 0.0005
    create_trade(min_stake_amount)
    rate, amount = buy_mock.call_args[0][1], buy_mock.call_args[0][2]
    assert rate * amount >= min_stake_amount


def test_create_trade_no_stake_amount(default_conf, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'),
                          get_balance=MagicMock(return_value=default_conf['stake_amount'] * 0.5))
    with pytest.raises(DependencyException, match=r'.*stake amount.*'):
        create_trade(default_conf['stake_amount'])


def test_create_trade_no_pairs(default_conf, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'))

    with pytest.raises(DependencyException, match=r'.*No pair in whitelist.*'):
        conf = copy.deepcopy(default_conf)
        conf['exchange']['pair_whitelist'] = []
        mocker.patch.dict('freqtrade.main._CONF', conf)
        create_trade(default_conf['stake_amount'])


def test_create_trade_no_pairs_after_blacklist(default_conf, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'))

    with pytest.raises(DependencyException, match=r'.*No pair in whitelist.*'):
        conf = copy.deepcopy(default_conf)
        conf['exchange']['pair_whitelist'] = ["BTC_ETH"]
        conf['exchange']['pair_blacklist'] = ["BTC_ETH"]
        mocker.patch.dict('freqtrade.main._CONF', conf)
        create_trade(default_conf['stake_amount'])


def test_handle_trade(default_conf, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.00001172,
                              'ask': 0.00001173,
                              'last': 0.00001172
                          }),
                          buy=MagicMock(return_value='mocked_limit_buy'),
                          sell=MagicMock(return_value='mocked_limit_sell'))
    mocker.patch.multiple('freqtrade.fiat_convert.Pymarketcap',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}),
                          _cache_symbols=MagicMock(return_value={'BTC': 1}))
    init(default_conf, create_engine('sqlite://'))
    create_trade(0.001)

    trade = Trade.query.first()
    assert trade

    trade.update(limit_buy_order)
    assert trade.is_open is True

    handle_trade(trade)
    assert trade.open_order_id == 'mocked_limit_sell'

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    assert trade.close_rate == 0.00001173
    assert trade.close_profit == 0.06201057
    assert trade.calc_profit() == 0.00006217
    assert trade.close_date is not None


def test_handle_trade_roi(default_conf, ticker, limit_buy_order, mocker, caplog):
    default_conf.update({'experimental': {'use_sell_signal': True}})
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'))
    mocker.patch('freqtrade.main.min_roi_reached', return_value=True)

    init(default_conf, create_engine('sqlite://'))
    create_trade(0.001)

    trade = Trade.query.first()
    trade.is_open = True

    # FIX: sniffing logs, suggest handle_trade should not execute_sell
    #      instead that responsibility should be moved out of handle_trade(),
    #      we might just want to check if we are in a sell condition without
    #      executing
    # if ROI is reached we must sell
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: False)
    assert handle_trade(trade)
    assert ('freqtrade', logging.DEBUG, 'Executing sell due to ROI ...') in caplog.record_tuples
    # if ROI is reached we must sell even if sell-signal is not signalled
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    assert handle_trade(trade)
    assert ('freqtrade', logging.DEBUG, 'Executing sell due to ROI ...') in caplog.record_tuples


def test_handle_trade_experimental(default_conf, ticker, limit_buy_order, mocker, caplog):
    default_conf.update({'experimental': {'use_sell_signal': True}})
    mocker.patch.dict('freqtrade.main._CONF', default_conf)

    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'))
    mocker.patch('freqtrade.main.min_roi_reached', return_value=False)

    init(default_conf, create_engine('sqlite://'))
    create_trade(0.001)

    trade = Trade.query.first()
    trade.is_open = True

    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: False)
    value_returned = handle_trade(trade)
    assert ('freqtrade', logging.DEBUG, 'Checking sell_signal ...') in caplog.record_tuples
    assert value_returned is False
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    assert handle_trade(trade)
    s = 'Executing sell due to sell signal ...'
    assert ('freqtrade', logging.DEBUG, s) in caplog.record_tuples


def test_close_trade(default_conf, ticker, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          buy=MagicMock(return_value='mocked_limit_buy'))

    # Create trade and sell it
    init(default_conf, create_engine('sqlite://'))
    create_trade(0.001)

    trade = Trade.query.first()
    assert trade

    trade.update(limit_buy_order)
    trade.update(limit_sell_order)
    assert trade.is_open is False

    with pytest.raises(ValueError, match=r'.*closed trade.*'):
        handle_trade(trade)


def test_check_handle_timedout_buy(default_conf, ticker, health, limit_buy_order_old, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_order=MagicMock(return_value=limit_buy_order_old),
                          cancel_order=cancel_order_mock)
    init(default_conf, create_engine('sqlite://'))

    trade_buy = Trade(
        pair='BTC_ETH',
        open_rate=0.00001099,
        exchange='BITTREX',
        open_order_id='123456789',
        amount=90.99181073,
        fee=0.0,
        stake_amount=1,
        open_date=arrow.utcnow().shift(minutes=-601).datetime,
        is_open=True
    )

    Trade.session.add(trade_buy)

    # check it does cancel buy orders over the time limit
    check_handle_timedout(600)
    assert cancel_order_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(trade_buy.open_order_id)).all()
    assert len(trades) == 0


def test_check_handle_timedout_sell(default_conf, ticker, health, limit_sell_order_old, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_order=MagicMock(return_value=limit_sell_order_old),
                          cancel_order=cancel_order_mock)
    init(default_conf, create_engine('sqlite://'))

    trade_sell = Trade(
        pair='BTC_ETH',
        open_rate=0.00001099,
        exchange='BITTREX',
        open_order_id='123456789',
        amount=90.99181073,
        fee=0.0,
        stake_amount=1,
        open_date=arrow.utcnow().shift(hours=-5).datetime,
        close_date=arrow.utcnow().shift(minutes=-601).datetime,
        is_open=False
    )

    Trade.session.add(trade_sell)

    # check it does cancel sell orders over the time limit
    check_handle_timedout(600)
    assert cancel_order_mock.call_count == 1
    assert trade_sell.is_open is True


def test_check_handle_timedout_partial(default_conf, ticker, limit_buy_order_old_partial,
                                       health, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          get_order=MagicMock(return_value=limit_buy_order_old_partial),
                          cancel_order=cancel_order_mock)
    init(default_conf, create_engine('sqlite://'))

    trade_buy = Trade(
        pair='BTC_ETH',
        open_rate=0.00001099,
        exchange='BITTREX',
        open_order_id='123456789',
        amount=90.99181073,
        fee=0.0,
        stake_amount=1,
        open_date=arrow.utcnow().shift(minutes=-601).datetime,
        is_open=True
    )

    Trade.session.add(trade_buy)

    # check it does cancel buy orders over the time limit
    # note this is for a partially-complete buy order
    check_handle_timedout(600)
    assert cancel_order_mock.call_count == 1
    trades = Trade.query.filter(Trade.open_order_id.is_(trade_buy.open_order_id)).all()
    assert len(trades) == 1
    assert trades[0].amount == 23.0
    assert trades[0].stake_amount == trade_buy.open_rate * trades[0].amount


def test_balance_fully_ask_side(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 0.0}})
    assert get_target_bid({'ask': 20, 'last': 10}) == 20


def test_balance_fully_last_side(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 1.0}})
    assert get_target_bid({'ask': 20, 'last': 10}) == 10


def test_balance_bigger_last_ask(mocker):
    mocker.patch.dict('freqtrade.main._CONF', {'bid_strategy': {'ask_last_balance': 1.0}})
    assert get_target_bid({'ask': 5, 'last': 10}) == 5


def test_execute_sell_up(default_conf, ticker, ticker_sell_up, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch('freqtrade.rpc.init', MagicMock())
    rpc_mock = mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch.multiple('freqtrade.fiat_convert.Pymarketcap',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}),
                          _cache_symbols=MagicMock(return_value={'BTC': 1}))
    init(default_conf, create_engine('sqlite://'))

    # Create some test data
    create_trade(0.001)

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker_sell_up)

    execute_sell(trade=trade, limit=ticker_sell_up()['bid'])

    assert rpc_mock.call_count == 2
    assert 'Selling [BTC/ETH]' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001172' in rpc_mock.call_args_list[-1][0][0]
    assert 'profit: 6.11%, 0.00006126' in rpc_mock.call_args_list[-1][0][0]
    assert '0.919 USD' in rpc_mock.call_args_list[-1][0][0]


def test_execute_sell_down(default_conf, ticker, ticker_sell_down, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch('freqtrade.rpc.init', MagicMock())
    rpc_mock = mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch.multiple('freqtrade.fiat_convert.Pymarketcap',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}),
                          _cache_symbols=MagicMock(return_value={'BTC': 1}))
    init(default_conf, create_engine('sqlite://'))

    # Create some test data
    create_trade(0.001)

    trade = Trade.query.first()
    assert trade

    # Decrease the price and sell it
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker_sell_down)

    execute_sell(trade=trade, limit=ticker_sell_down()['bid'])

    assert rpc_mock.call_count == 2
    assert 'Selling [BTC/ETH]' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001044' in rpc_mock.call_args_list[-1][0][0]
    assert 'loss: -5.48%, -0.00005492' in rpc_mock.call_args_list[-1][0][0]
    assert '-0.824 USD' in rpc_mock.call_args_list[-1][0][0]


def test_execute_sell_without_conf(default_conf, ticker, ticker_sell_up, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch('freqtrade.rpc.init', MagicMock())
    rpc_mock = mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    init(default_conf, create_engine('sqlite://'))

    # Create some test data
    create_trade(0.001)

    trade = Trade.query.first()
    assert trade

    # Increase the price and sell it
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker_sell_up)
    mocker.patch('freqtrade.main._CONF', {})

    execute_sell(trade=trade, limit=ticker_sell_up()['bid'])

    assert rpc_mock.call_count == 2
    assert 'Selling [BTC/ETH]' in rpc_mock.call_args_list[-1][0][0]
    assert '0.00001172' in rpc_mock.call_args_list[-1][0][0]
    assert '(profit: 6.11%, 0.00006126)' in rpc_mock.call_args_list[-1][0][0]
    assert 'USD' not in rpc_mock.call_args_list[-1][0][0]


def test_sell_profit_only_enable_profit(default_conf, limit_buy_order, mocker):
    default_conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': True,
    }

    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.min_roi_reached', return_value=False)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.00002172,
                              'ask': 0.00002173,
                              'last': 0.00002172
                          }),
                          buy=MagicMock(return_value='mocked_limit_buy'))

    init(default_conf, create_engine('sqlite://'))
    create_trade(0.001)

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    assert handle_trade(trade) is True


def test_sell_profit_only_disable_profit(default_conf, limit_buy_order, mocker):
    default_conf['experimental'] = {
        'use_sell_signal': True,
        'sell_profit_only': False,
    }

    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.min_roi_reached', return_value=False)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
    mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=MagicMock(return_value={
                              'bid': 0.00002172,
                              'ask': 0.00002173,
                              'last': 0.00002172
                          }),
                          buy=MagicMock(return_value='mocked_limit_buy'))

    init(default_conf, create_engine('sqlite://'))
    create_trade(0.001)

    trade = Trade.query.first()
    trade.update(limit_buy_order)
    assert handle_trade(trade) is True


def test_sell_profit_only_enable_loss(default_conf, limit_buy_order, mocker):
        default_conf['experimental'] = {
            'use_sell_signal': True,
            'sell_profit_only': True,
        }

        mocker.patch.dict('freqtrade.main._CONF', default_conf)
        mocker.patch('freqtrade.main.min_roi_reached', return_value=False)
        mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
        mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
        mocker.patch.multiple('freqtrade.main.exchange',
                              validate_pairs=MagicMock(),
                              get_ticker=MagicMock(return_value={
                                  'bid': 0.00000172,
                                  'ask': 0.00000173,
                                  'last': 0.00000172
                              }),
                              buy=MagicMock(return_value='mocked_limit_buy'))

        init(default_conf, create_engine('sqlite://'))
        create_trade(0.001)

        trade = Trade.query.first()
        trade.update(limit_buy_order)
        assert handle_trade(trade) is False


def test_sell_profit_only_disable_loss(default_conf, limit_buy_order, mocker):
        default_conf['experimental'] = {
            'use_sell_signal': True,
            'sell_profit_only': False,
        }

        mocker.patch.dict('freqtrade.main._CONF', default_conf)
        mocker.patch('freqtrade.main.min_roi_reached', return_value=False)
        mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
        mocker.patch.multiple('freqtrade.rpc', init=MagicMock(), send_msg=MagicMock())
        mocker.patch.multiple('freqtrade.main.exchange',
                              validate_pairs=MagicMock(),
                              get_ticker=MagicMock(return_value={
                                  'bid': 0.00000172,
                                  'ask': 0.00000173,
                                  'last': 0.00000172
                              }),
                              buy=MagicMock(return_value='mocked_limit_buy'))

        init(default_conf, create_engine('sqlite://'))
        create_trade(0.001)

        trade = Trade.query.first()
        trade.update(limit_buy_order)
        assert handle_trade(trade) is True
