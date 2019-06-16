
from unittest.mock import MagicMock

from plotly import tools
import plotly.graph_objs as go
from copy import deepcopy

from freqtrade.arguments import TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import load_backtest_data
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


def generage_empty_figure():
    return tools.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_width=[1, 1, 4],
        vertical_spacing=0.0001,
    )


def test_generate_row(default_conf, caplog):
    pair = "UNITTEST/BTC"
    timerange = TimeRange(None, 'line', 0, -1000)

    data = history.load_pair_history(pair=pair, ticker_interval='1m',
                                     datadir=None, timerange=timerange)
    indicators1 = ["ema10"]
    indicators2 = ["macd"]

    # Generate buy/sell signals and indicators
    strat = DefaultStrategy(default_conf)
    data = strat.analyze_ticker(data, {'pair': pair})
    fig = generage_empty_figure()

    # Row 1
    fig1 = generate_row(fig=deepcopy(fig), row=1, indicators=indicators1, data=data)
    figure = fig1.layout.figure
    ema10 = find_trace_in_fig_data(figure.data, "ema10")
    assert isinstance(ema10, go.Scatter)
    assert ema10.yaxis == "y"

    fig2 = generate_row(fig=deepcopy(fig), row=3, indicators=indicators2, data=data)
    figure = fig2.layout.figure
    macd = find_trace_in_fig_data(figure.data, "macd")
    assert isinstance(macd, go.Scatter)
    assert macd.yaxis == "y3"

    # No indicator found
    fig3 = generate_row(fig=deepcopy(fig), row=3, indicators=['no_indicator'], data=data)
    assert fig == fig3
    assert log_has_re(r'Indicator "no_indicator" ignored\..*', caplog.record_tuples)


def test_plot_trades():
    fig1 = generage_empty_figure()
    # nothing happens when no trades are available
    fig = plot_trades(fig1, None)
    assert fig == fig1
    pair = "ADA/BTC"
    filename = history.make_testdata_path(None) / "backtest-result_test.json"
    trades = load_backtest_data(filename)
    trades = trades.loc[trades['pair'] == pair]

    fig = plot_trades(fig, trades)
    figure = fig1.layout.figure

    # Check buys - color, should be in first graph, ...
    trade_buy = find_trace_in_fig_data(figure.data, "trade_buy")
    assert isinstance(trade_buy, go.Scatter)
    assert trade_buy.yaxis == 'y'
    assert len(trades) == len(trade_buy.x)
    assert trade_buy.marker.color == 'green'

    trade_sell = find_trace_in_fig_data(figure.data, "trade_sell")
    assert isinstance(trade_sell, go.Scatter)
    assert trade_sell.yaxis == 'y'
    assert len(trades) == len(trade_sell.x)
    assert trade_sell.marker.color == 'red'


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


def test_generate_plot_file(mocker, caplog):
    fig = generage_empty_figure()
    plot_mock = mocker.patch("freqtrade.plot.plotting.plot", MagicMock())
    generate_plot_file(fig, "UNITTEST/BTC", "5m")

    assert plot_mock.call_count == 1
    assert plot_mock.call_args[0][0] == fig
    assert (plot_mock.call_args_list[0][1]['filename']
            == "user_data/plots/freqtrade-plot-UNITTEST_BTC-5m.html")
