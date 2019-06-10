
from unittest.mock import MagicMock

import plotly.graph_objs as go

from freqtrade.arguments import Arguments, TimeRange
from freqtrade.data import history
from freqtrade.plot.plotting import (generate_graph, generate_plot_file,
                                     generate_row, plot_trades)


def fig_generating_mock(fig, *args, **kwargs):
    """ Return Fig - used to mock generate_row and plot_trades"""
    return fig


def test_generate_row():
    # TODO: implement me
    pass


def test_plot_trades():
    # TODO: implement me
    pass


def test_generate_graph(default_conf, mocker):
    row_mock = mocker.patch('freqtrade.plot.plotting.generate_row',
                            MagicMock(side_effect=fig_generating_mock))
    trades_mock = mocker.patch('freqtrade.plot.plotting.plot_trades',
                               MagicMock(side_effect=fig_generating_mock))

    timerange = TimeRange(None, 'line', 0, -100)
    data = history.load_pair_history(pair='UNITTEST/BTC', ticker_interval='1m',
                                     datadir=None, timerange=timerange)

    indicators1 = []
    indicators2 = []
    fig = generate_graph(pair="UNITTEST/BTC", data=data, trades=None,
                         indicators1=indicators1, indicators2=indicators2)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "UNITTEST/BTC"
    figure = fig.layout.figure
    # Candlesticks are plotted first
    assert isinstance(figure.data[0], go.Candlestick)
    assert figure.data[0].name == "Price"

    assert isinstance(figure.data[1], go.Bar)
    assert figure.data[1].name == "Volume"

    assert row_mock.call_count == 2
    assert trades_mock.call_count == 1
