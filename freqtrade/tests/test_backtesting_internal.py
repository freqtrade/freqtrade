# pragma pylint: disable=missing-docstring,protected-access
import os
from datetime import datetime
from unittest.mock import MagicMock

import arrow
import pytest
from jsonschema import validate

from freqtrade.exchange import backtesting
from freqtrade.exchange.backtesting import Backtesting
from freqtrade.main import create_trade, init
from freqtrade.misc import CONF_SCHEMA
from freqtrade.persistence import Trade


@pytest.fixture
def conf():
    configuration = {
        'max_open_trades': 3,
        'stake_currency': 'BTC',
        'stake_amount': 0.05,
        'dry_run': True,
        'minimal_roi': {
            '60': 0.0,
            '40': 0.01,
            '20': 0.02,
            '0': 0.03
        },
        'stoploss': -0.40,
        'bid_strategy': {
            'ask_last_balance': 0.0
        },
        'exchange': {
            'name': 'bittrex',
            'key': 'key',
            'secret': 'secret',
            'pair_whitelist': [
                'BTC_RLC'
            ]
        },
        'telegram': {
            'enabled': False,
            'token': 'token',
            'chat_id': 'chat_id'
        }
    }
    validate(configuration, CONF_SCHEMA)
    return configuration


FILES = [
    os.path.join('freqtrade', 'tests', 'testdata', 'btc-edg.json'),
    os.path.join('freqtrade', 'tests', 'testdata', 'btc-etc.json')
]
PAIRS = ['BTC_EDG', 'BTC_ETC']
TESTDATA = {
    PAIRS[0]: {
        'success': True,
        'message': '',
        'result': [
            {'O': 0.00014469, 'H': 0.00014469, 'L': 0.00014469, 'C': 0.00014469, 'V': 10.66173857, 'T': '2017-09-05T18:55:00', 'BV': 0.00154264},
            {'O': 0.00014469, 'H': 0.00014477, 'L': 0.00014469, 'C': 0.00014477, 'V': 410.54795113, 'T': '2017-09-05T19:00:00', 'BV': 0.05942728},
            {'O': 0.00014477, 'H': 0.00014477, 'L': 0.00014477, 'C': 0.00014477, 'V': 69.10850034, 'T': '2017-09-05T19:05:00', 'BV': 0.01000482},
            {'O': 0.00014470, 'H': 0.00014474, 'L': 0.00014400, 'C': 0.00014473, 'V': 7612.36224582, 'T': '2017-09-05T19:10:00', 'BV': 1.09730748},
        ]
    },
    PAIRS[1]: {
        'success': True,
        'message': '',
        'result': [
            {'O': 0.00391500, 'H': 0.00392700, 'L': 0.00391500, 'C': 0.00392000, 'V': 29.90264260, 'T': '2017-09-05T18:55:00', 'BV': 0.11712504},
            {'O': 0.00392680, 'H': 0.00392749, 'L': 0.00391500, 'C': 0.00391500, 'V': 329.35043009, 'T': '2017-09-05T19:00:00', 'BV': 1.29065913},
            {'O': 0.00391500, 'H': 0.00392733, 'L': 0.00391500, 'C': 0.00392300, 'V': 186.96019741, 'T': '2017-09-05T19:05:00', 'BV': 0.73332203},
            {'O': 0.00391500, 'H': 0.00391500, 'L': 0.00390007, 'C': 0.00390007, 'V': 298.06457786, 'T': '2017-09-05T19:10:00', 'BV': 1.16560055},
            {'O': 0.00391490, 'H': 0.00391490, 'L': 0.00389126, 'C': 0.00389126, 'V': 1007.91208513, 'T': '2017-09-05T19:15:00', 'BV': 3.92491826}
        ]
    }
}


@pytest.fixture()
def len_rows():
    ticker_history = TESTDATA[PAIRS[0]]
    return len(ticker_history['result'])


def _json_load(file):
    pair = os.path.splitext(os.path.basename(file.name))[0].replace('-', '_').upper()
    return TESTDATA[pair]


def test_init(conf, mocker):
    mocker.patch('json.load', side_effect=_json_load)
    mocker.patch('glob.glob', return_value=FILES)
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch('freqtrade.main.get_args', backtesting=True)
    mocker.patch('freqtrade.exchange', init=MagicMock())
    mocker.patch('freqtrade.main.backtesting.TICKER_HISTORY_INTERVAL_H', 1/6)
    init(conf)
    assert backtesting._TESTDATA == TESTDATA
    assert backtesting._LEN_ROWS == 4
    # 10 minutes (1/6 hours) should correspond to 2 rows
    assert backtesting._ROW_INDEX == 2
    assert backtesting._ROW_INTERVAL == 2

    # Available test data should not cover the interval
    mocker.patch('freqtrade.main.backtesting.TICKER_HISTORY_INTERVAL_H', 24)
    with pytest.raises(RuntimeError):
        assert init(conf)


def test_time_step(len_rows, mocker):
    mocker.patch('freqtrade.exchange.backtesting._LEN_ROWS', len_rows)
    for index in range(4):
        assert backtesting._ROW_INDEX == index
        assert backtesting.time_step() is True
    assert backtesting.time_step() is False


def test_returns_minimum_date(mocker):
    mocker.patch('freqtrade.exchange.backtesting._TESTDATA', TESTDATA)
    mocker.patch('freqtrade.exchange.backtesting._ROW_INDEX', 3)
    mocker.patch('freqtrade.exchange.backtesting._ROW_INTERVAL', 2)
    assert backtesting.get_minimum_date(PAIRS[0]) == '2017-09-05T19:00:00'


def test_returns_ticker(conf, mocker):
    mocker.patch('freqtrade.main.get_args', backtesting=True)
    mocker.patch('freqtrade.exchange.backtesting._get_testdata', return_value=(TESTDATA, 4))
    mocker.patch('freqtrade.exchange.backtesting.TICKER_HISTORY_INTERVAL_H', 1/6)
    first_pair_close = 0.00014477
    backtesting = Backtesting(conf)
    mocker.patch('freqtrade.exchange.backtesting._ROW_INDEX', 2)
    assert backtesting.get_ticker(PAIRS[0]) == \
           {'bid': first_pair_close, 'ask': first_pair_close, 'last': first_pair_close}


def test_returns_ticker_history(conf, mocker):
    mocker.patch('freqtrade.exchange.backtesting._get_testdata', return_value=(TESTDATA, 4))
    mocker.patch('freqtrade.main.backtesting.TICKER_HISTORY_INTERVAL_H', 1/6)
    backtesting = Backtesting(conf)
    mocker.patch('freqtrade.exchange.backtesting._ROW_INDEX', 3)
    mocker.patch('freqtrade.exchange.backtesting._ROW_INTERVAL', 2)
    assert backtesting.get_ticker_history(PAIRS[0]) == \
           {'success': True,
            'message': '',
            'result': [
                {'O': 0.00014469, 'H': 0.00014477, 'L': 0.00014469, 'C': 0.00014477, 'V': 410.54795113, 'T': '2017-09-05T19:00:00', 'BV': 0.05942728},
                {'O': 0.00014477, 'H': 0.00014477, 'L': 0.00014477, 'C': 0.00014477, 'V': 69.10850034, 'T': '2017-09-05T19:05:00', 'BV': 0.01000482}
            ]}


def test_returns_current_time(mocker):
    mocker.patch('freqtrade.exchange.backtesting._TESTDATA', TESTDATA)
    mocker.patch('freqtrade.exchange.backtesting._ROW_INDEX', 1)
    assert backtesting.current_time(PAIRS[0]) == \
           arrow.get('2017-09-05T19:00:00').datetime.replace(tzinfo=None)


def test_results(conf, mocker):
    current_time = datetime.utcnow()
    logger_mock = MagicMock()
    mocker.patch.dict('freqtrade.main._CONF', conf)
    mocker.patch('freqtrade.main.get_args', backtesting=True)
    mocker.patch('freqtrade.main.backtesting', init=MagicMock())
    mocker.patch('freqtrade.main.telegram', init=MagicMock())
    mocker.patch('freqtrade.main.backtesting.current_time', return_value=current_time)
    mocker.patch('freqtrade.main.get_buy_signal', return_value=True)
    mocker.patch('freqtrade.exchange.backtesting.logger', info=logger_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_ticker=MagicMock(return_value={
                              'bid': 0.07256061,
                              'ask': 0.072661,
                              'last': 0.07256061
                          }),
                          buy=MagicMock(return_value='mocked_order_id'))
    init(conf)

    # Create some test data
    trade = create_trade(15.0)
    assert trade
    trade.close_rate = 0.07256061
    trade.close_profit = 100.00
    trade.close_date = current_time
    trade.open_order_id = None
    trade.is_open = False
    Trade.session.add(trade)
    Trade.session.flush()

    backtesting.print_results()
    assert logger_mock.call_count == 1
    assert '(100.00%)' in logger_mock.call_args_list[-1][0][0]

    # Trade should not be closed yet
    Trade.session.delete(trade)
    trade.close_rate = None
    trade.close_profit = None
    trade.close_date = None
    trade.is_open = True
    Trade.session.add(trade)
    Trade.session.flush()

    backtesting.print_results()
    assert logger_mock.call_count == 2
    assert 'No closed trade' in logger_mock.call_args_list[-1][0][0]
