# pragma pylint: disable=missing-docstring
import json
import pytest
import arrow
from pandas import DataFrame

from freqtrade.analyze import parse_ticker_dataframe, populate_buy_trend, populate_indicators, \
    get_buy_signal

@pytest.fixture
def result():
    with open('freqtrade/tests/testdata/btc-eth.json') as data_file:
        data = json.load(data_file)

    return parse_ticker_dataframe(data['result'], arrow.get('2017-08-30T10:00:00'))

def test_dataframe_has_correct_columns(result):
    assert result.columns.tolist() == \
                        ['close', 'high', 'low', 'open', 'date', 'volume']

def test_dataframe_has_correct_length(result):
    assert len(result.index) == 5751

def test_populates_buy_trend(result):
    dataframe = populate_buy_trend(populate_indicators(result))
    assert 'buy' in dataframe.columns
    assert 'buy_price' in dataframe.columns

def test_returns_latest_buy_signal(mocker):
    buydf = DataFrame([{'buy': 1, 'date': arrow.utcnow()}])
    mocker.patch('freqtrade.analyze.analyze_ticker', return_value=buydf)
    assert get_buy_signal('BTC-ETH')

    buydf = DataFrame([{'buy': 0, 'date': arrow.utcnow()}])
    mocker.patch('freqtrade.analyze.analyze_ticker', return_value=buydf)
    assert not get_buy_signal('BTC-ETH')
