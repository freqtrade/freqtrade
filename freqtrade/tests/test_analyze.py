# pragma pylint: disable=missing-docstring
from datetime import datetime
import json
import pytest
from pandas import DataFrame

from freqtrade.analyze import parse_ticker_dataframe, populate_buy_trend, populate_indicators, \
    get_signal, SignalType


@pytest.fixture
def result():
    with open('freqtrade/tests/testdata/btc-eth.json') as data_file:
        return parse_ticker_dataframe(json.load(data_file))


def test_dataframe_correct_columns(result):
    assert result.columns.tolist() == \
        ['close', 'high', 'low', 'open', 'date', 'volume']


def test_dataframe_correct_length(result):
    assert len(result.index) == 5751


def test_populates_buy_trend(result):
    dataframe = populate_buy_trend(populate_indicators(result))
    assert 'buy' in dataframe.columns
    assert 'buy_price' in dataframe.columns


def test_returns_latest_buy_signal(mocker):
    buydf = DataFrame([{'buy': 1, 'date': datetime.today()}])
    mocker.patch('freqtrade.analyze.analyze_ticker', return_value=buydf)
    assert get_signal('BTC-ETH', SignalType.BUY)

    buydf = DataFrame([{'buy': 0, 'date': datetime.today()}])
    mocker.patch('freqtrade.analyze.analyze_ticker', return_value=buydf)
    assert not get_signal('BTC-ETH', SignalType.BUY)


def test_returns_latest_sell_signal(mocker):
    selldf = DataFrame([{'sell': 1, 'date': datetime.today()}])
    mocker.patch('freqtrade.analyze.analyze_ticker', return_value=selldf)
    assert get_signal('BTC-ETH', SignalType.SELL)

    selldf = DataFrame([{'sell': 0, 'date': datetime.today()}])
    mocker.patch('freqtrade.analyze.analyze_ticker', return_value=selldf)
    assert not get_signal('BTC-ETH', SignalType.SELL)
