
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import create_cum_profit, load_backtest_data
from freqtrade.plot.plot_utils import start_plot_dataframe
from freqtrade.plot.plotting import (add_indicators, add_profit,
                                     analyse_and_plot_pairs,
                                     generate_candlestick_graph,
                                     generate_plot_filename,
                                     generate_profit_graph, init_plotscript,
                                     plot_trades, store_plot_file)
from freqtrade.strategy.default_strategy import DefaultStrategy
from freqtrade.tests.conftest import get_args, log_has, log_has_re


def fig_generating_mock(fig, *args, **kwargs):
    """ Return Fig - used to mock add_indicators and plot_trades"""
    return fig


def find_trace_in_fig_data(data, search_string: str):
    matches = (d for d in data if d.name == search_string)
    return next(matches)


def generage_empty_figure():
    return make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_width=[1, 1, 4],
        vertical_spacing=0.0001,
    )


def test_init_plotscript(default_conf, mocker):
    default_conf['timerange'] = "20180110-20180112"
    default_conf['trade_source'] = "file"
    default_conf['ticker_interval'] = "5m"
    default_conf["datadir"] = history.make_testdata_path(None)
    default_conf['exportfilename'] = str(
        history.make_testdata_path(None) / "backtest-result_test.json")
    ret = init_plotscript(default_conf)
    assert "tickers" in ret
    assert "trades" in ret
    assert "pairs" in ret
    assert "strategy" in ret

    default_conf['pairs'] = ["POWR/BTC", "XLM/BTC"]
    ret = init_plotscript(default_conf)
    assert "tickers" in ret
    assert "POWR/BTC" in ret["tickers"]
    assert "XLM/BTC" in ret["tickers"]


def test_add_indicators(default_conf, caplog):
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
    fig1 = add_indicators(fig=deepcopy(fig), row=1, indicators=indicators1, data=data)
    figure = fig1.layout.figure
    ema10 = find_trace_in_fig_data(figure.data, "ema10")
    assert isinstance(ema10, go.Scatter)
    assert ema10.yaxis == "y"

    fig2 = add_indicators(fig=deepcopy(fig), row=3, indicators=indicators2, data=data)
    figure = fig2.layout.figure
    macd = find_trace_in_fig_data(figure.data, "macd")
    assert isinstance(macd, go.Scatter)
    assert macd.yaxis == "y3"

    # No indicator found
    fig3 = add_indicators(fig=deepcopy(fig), row=3, indicators=['no_indicator'], data=data)
    assert fig == fig3
    assert log_has_re(r'Indicator "no_indicator" ignored\..*', caplog)


def test_plot_trades(caplog):
    fig1 = generage_empty_figure()
    # nothing happens when no trades are available
    fig = plot_trades(fig1, None)
    assert fig == fig1
    assert log_has("No trades found.", caplog)
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


def test_generate_candlestick_graph_no_signals_no_trades(default_conf, mocker, caplog):
    row_mock = mocker.patch('freqtrade.plot.plotting.add_indicators',
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
    fig = generate_candlestick_graph(pair=pair, data=data, trades=None,
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

    assert log_has("No buy-signals found.", caplog)
    assert log_has("No sell-signals found.", caplog)


def test_generate_candlestick_graph_no_trades(default_conf, mocker):
    row_mock = mocker.patch('freqtrade.plot.plotting.add_indicators',
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
    fig = generate_candlestick_graph(pair=pair, data=data, trades=None,
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


def test_generate_Plot_filename():
    fn = generate_plot_filename("UNITTEST/BTC", "5m")
    assert fn == "freqtrade-plot-UNITTEST_BTC-5m.html"


def test_generate_plot_file(mocker, caplog):
    fig = generage_empty_figure()
    plot_mock = mocker.patch("freqtrade.plot.plotting.plot", MagicMock())
    store_plot_file(fig, filename="freqtrade-plot-UNITTEST_BTC-5m.html",
                    directory=Path("user_data/plots"))

    assert plot_mock.call_count == 1
    assert plot_mock.call_args[0][0] == fig
    assert (plot_mock.call_args_list[0][1]['filename']
            == "user_data/plots/freqtrade-plot-UNITTEST_BTC-5m.html")
    assert log_has("Stored plot as user_data/plots/freqtrade-plot-UNITTEST_BTC-5m.html",
                   caplog)


def test_add_profit():
    filename = history.make_testdata_path(None) / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)
    timerange = TimeRange.parse_timerange("20180110-20180112")

    df = history.load_pair_history(pair="POWR/BTC", ticker_interval='5m',
                                   datadir=None, timerange=timerange)
    fig = generage_empty_figure()

    cum_profits = create_cum_profit(df.set_index('date'),
                                    bt_data[bt_data["pair"] == 'POWR/BTC'],
                                    "cum_profits")

    fig1 = add_profit(fig, row=2, data=cum_profits, column='cum_profits', name='Profits')
    figure = fig1.layout.figure
    profits = find_trace_in_fig_data(figure.data, "Profits")
    assert isinstance(profits, go.Scattergl)
    assert profits.yaxis == "y2"


def test_generate_profit_graph():
    filename = history.make_testdata_path(None) / "backtest-result_test.json"
    trades = load_backtest_data(filename)
    timerange = TimeRange.parse_timerange("20180110-20180112")
    pairs = ["POWR/BTC", "XLM/BTC"]

    tickers = history.load_data(datadir=None,
                                pairs=pairs,
                                ticker_interval='5m',
                                timerange=timerange
                                )
    trades = trades[trades['pair'].isin(pairs)]

    fig = generate_profit_graph(pairs, tickers, trades)
    assert isinstance(fig, go.Figure)

    assert fig.layout.title.text == "Profit plot"
    figure = fig.layout.figure
    assert len(figure.data) == 4

    avgclose = find_trace_in_fig_data(figure.data, "Avg close price")
    assert isinstance(avgclose, go.Scattergl)

    profit = find_trace_in_fig_data(figure.data, "Profit")
    assert isinstance(profit, go.Scattergl)

    for pair in pairs:
        profit_pair = find_trace_in_fig_data(figure.data, f"Profit {pair}")
        assert isinstance(profit_pair, go.Scattergl)


def test_start_plot_dataframe(mocker):
    aup = mocker.patch("freqtrade.plot.plotting.analyse_and_plot_pairs", MagicMock())
    args = [
        "--config", "config.json.example",
        "plot-dataframe",
        "--pairs", "ETH/BTC"
    ]
    start_plot_dataframe(get_args(args))

    assert aup.call_count == 1
    called_config = aup.call_args_list[0][0][0]
    assert "pairs" in called_config
    assert called_config['pairs'] == ["ETH/BTC"]


def test_analyse_and_plot_pairs(default_conf, mocker, caplog):
    default_conf['trade_source'] = 'file'
    default_conf["datadir"] = history.make_testdata_path(None)
    default_conf['exportfilename'] = str(
        history.make_testdata_path(None) / "backtest-result_test.json")
    default_conf['indicators1'] = "sma5,ema10"
    default_conf['indicators2'] = "macd"
    default_conf['pairs'] = ["ETH/BTC", "LTC/BTC"]

    candle_mock = MagicMock()
    store_mock = MagicMock()
    mocker.patch.multiple(
        "freqtrade.plot.plotting",
        generate_candlestick_graph=candle_mock,
        store_plot_file=store_mock
        )
    analyse_and_plot_pairs(default_conf)

    # Both mocks should be called once per pair
    assert candle_mock.call_count == 2
    assert store_mock.call_count == 2

    assert candle_mock.call_args_list[0][1]['indicators1'] == ['sma5', 'ema10']
    assert candle_mock.call_args_list[0][1]['indicators2'] == ['macd']

    assert log_has("End of plotting process. 2 plots generated", caplog)
