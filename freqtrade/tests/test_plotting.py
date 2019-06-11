
from unittest.mock import MagicMock

import plotly.graph_objs as go

from freqtrade.arguments import Arguments, TimeRange
from freqtrade.data import history
from freqtrade.plot.plotting import (generate_graph, generate_plot_file,
                                     generate_row, plot_trades)
from freqtrade.strategy.default_strategy import DefaultStrategy
from freqtrade.tests.conftest import log_has, log_has_re

def fig_generating_mock(fig, *args, **kwargs):
    """ Return Fig - used to mock generate_row and plot_trades"""
    return fig


def find_trace_in_fig_data(data, search_string: str):
    matches = filter(lambda x: x.name == search_string, data)
    return next(matches)


def test_generate_row():
    # TODO: implement me
    pass


def test_plot_trades():
    # TODO: implement me
    pass


def test_generate_graph_no_signals_no_trades(default_conf, mocker, caplog):
    row_mock = mocker.patch('freqtrade.plot.plotting.generate_row',
                            MagicMock(side_effect=fig_generating_mock))
    trades_mock = mocker.patch('freqtrade.plot.plotting.plot_trades',
                               MagicMock(side_effect=fig_generating_mock))

    pair = "UNITTEST/BTC"
    timerange = TimeRange(None, 'line', 0, -1000)
    data = history.load_pair_history(pair=pair, ticker_interval='1m',
                                     datadir=None, timerange=timerange)
    data['buy'] = 0
    data['sell'] = 0

    indicators1 = []
    indicators2 = []
    fig = generate_graph(pair=pair, data=data, trades=None,
                         indicators1=indicators1, indicators2=indicators2)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == pair
    figure = fig.layout.figure

    assert len(figure.data) == 2
    # Candlesticks are plotted first
    candles = find_trace_in_fig_data(figure.data, "Price")
    assert isinstance(candles, go.Candlestick)

    volume = find_trace_in_fig_data(figure.data, "Volume")
    assert isinstance(volume, go.Bar)

    assert row_mock.call_count == 2
    assert trades_mock.call_count == 1

    assert log_has("No buy-signals found.", caplog.record_tuples)
    assert log_has("No sell-signals found.", caplog.record_tuples)


def test_generate_graph_no_trades(default_conf, mocker):
    row_mock = mocker.patch('freqtrade.plot.plotting.generate_row',
                            MagicMock(side_effect=fig_generating_mock))
    trades_mock = mocker.patch('freqtrade.plot.plotting.plot_trades',
                               MagicMock(side_effect=fig_generating_mock))
    pair = 'UNITTEST/BTC'
    timerange = TimeRange(None, 'line', 0, -1000)
    data = history.load_pair_history(pair=pair, ticker_interval='1m',
                                     datadir=None, timerange=timerange)

    # Generate buy/sell signals and indicators
    strat = DefaultStrategy(default_conf)
    data = strat.analyze_ticker(data, {'pair': pair})

    indicators1 = []
    indicators2 = []
    fig = generate_graph(pair=pair, data=data, trades=None,
                         indicators1=indicators1, indicators2=indicators2)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == pair
    figure = fig.layout.figure

    assert len(figure.data) == 6
    # Candlesticks are plotted first
    candles = find_trace_in_fig_data(figure.data, "Price")
    assert isinstance(candles, go.Candlestick)

    volume = find_trace_in_fig_data(figure.data, "Volume")
    assert isinstance(volume, go.Bar)

    buy = find_trace_in_fig_data(figure.data, "buy")
    assert isinstance(buy, go.Scatter)
    # All buy-signals should be plotted
    assert int(data.buy.sum()) == len(buy.x)

    sell = find_trace_in_fig_data(figure.data, "sell")
    assert isinstance(sell, go.Scatter)
    # All buy-signals should be plotted
    assert int(data.sell.sum()) == len(sell.x)

    assert find_trace_in_fig_data(figure.data, "BB lower")
    assert find_trace_in_fig_data(figure.data, "BB upper")

    assert row_mock.call_count == 2
    assert trades_mock.call_count == 1
