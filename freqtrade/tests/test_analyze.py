# pragma pylint: disable=missing-docstring
import pytest
import arrow
from pandas import DataFrame

from freqtrade.analyze import parse_ticker_dataframe, populate_buy_trend, populate_indicators, \
    get_buy_signal

RESULT_BITTREX = {
    'success': True,
    'message': '',
    'result': [
        {'O': 0.00065311, 'H': 0.00065311, 'L': 0.00065311, 'C': 0.00065311, 'V': 22.17210568, 'T': '2017-08-30T10:40:00', 'BV': 0.01448082},
        {'O': 0.00066194, 'H': 0.00066195, 'L': 0.00066194, 'C': 0.00066195, 'V': 33.4727437, 'T': '2017-08-30T10:34:00', 'BV': 0.02215696},
        {'O': 0.00065311, 'H': 0.00065311, 'L': 0.00065311, 'C': 0.00065311, 'V': 53.85127609, 'T': '2017-08-30T10:37:00', 'BV': 0.0351708},
        {'O': 0.00066194, 'H': 0.00066194, 'L': 0.00065311, 'C': 0.00065311, 'V': 46.29210665, 'T': '2017-08-30T10:42:00', 'BV': 0.03063118},
    ]
}

@pytest.fixture
def result():
    return parse_ticker_dataframe(RESULT_BITTREX['result'], arrow.get('2017-08-30T10:00:00'))

def test_dataframe_has_correct_columns(result):
    assert result.columns.tolist() == \
                        ['close', 'high', 'low', 'open', 'date', 'volume']

def test_orders_by_date(result):
    assert result['date'].tolist() == \
                        ['2017-08-30T10:34:00',
                        '2017-08-30T10:37:00',
                        '2017-08-30T10:40:00',
                        '2017-08-30T10:42:00']

def test_populates_buy_trend(result):
    dataframe = populate_buy_trend(populate_indicators(result))
    assert 'buy' in dataframe.columns
    assert 'buy_price' in dataframe.columns

def test_returns_latest_buy_signal(mocker):
    buydf = DataFrame([{'buy': 1, 'date': arrow.utcnow()}])
    mocker.patch('freqtrade.analyze.analyze_ticker', return_value=buydf)
    assert get_buy_signal('BTC-ETH') == True

    buydf = DataFrame([{'buy': 0, 'date': arrow.utcnow()}])
    mocker.patch('freqtrade.analyze.analyze_ticker', return_value=buydf)
    assert get_buy_signal('BTC-ETH') == False
