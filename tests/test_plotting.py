
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import create_cum_profit, load_backtest_data
from freqtrade.exceptions import OperationalException
from freqtrade.plot.plot_utils import start_plot_dataframe, start_plot_profit
from freqtrade.plot.plotting import (add_indicators, add_profit,
                                     generate_candlestick_graph,
                                     generate_plot_filename,
                                     generate_profit_graph, init_plotscript,
                                     load_and_plot_trades, plot_profit,
                                     plot_trades, store_plot_file)
from freqtrade.strategy.default_strategy import DefaultStrategy
from tests.conftest import get_args, log_has, log_has_re


def fig_generating_mock(fig, *args, **kwargs):
    """ Return Fig - used to mock add_indicators and plot_trades"""
    return fig


def find_trace_in_fig_data(data, search_string: str):
    matches = (d for d in data if d.name == search_string)
    return next(matches)


def generate_empty_figure():
    return make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_width=[1, 1, 4],
        vertical_spacing=0.0001,
    )


def test_init_plotscript(default_conf, mocker, testdatadir):
    default_conf['timerange'] = "20180110-20180112"
    default_conf['trade_source'] = "file"
    default_conf['ticker_interval'] = "5m"
    default_conf["datadir"] = testdatadir
    default_conf['exportfilename'] = str(testdatadir / "backtest-result_test.json")
    ret = init_plotscript(default_conf)
    assert "tickers" in ret
    assert "trades" in ret
    assert "pairs" in ret

    default_conf['pairs'] = ["TRX/BTC", "ADA/BTC"]
    ret = init_plotscript(default_conf)
    assert "tickers" in ret
    assert "TRX/BTC" in ret["tickers"]
    assert "ADA/BTC" in ret["tickers"]


def test_add_indicators(default_conf, testdatadir, caplog):
    pair = "UNITTEST/BTC"
    timerange = TimeRange(None, 'line', 0, -1000)

    data = history.load_pair_history(pair=pair, timeframe='1m',
                                     datadir=testdatadir, timerange=timerange)
    indicators1 = {"ema10": {}}
    indicators2 = {"macd": {"color": "red"}}

    # Generate buy/sell signals and indicators
    strat = DefaultStrategy(default_conf)
    data = strat.analyze_ticker(data, {'pair': pair})
    fig = generate_empty_figure()

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
    assert macd.line.color == "red"

    # No indicator found
    fig3 = add_indicators(fig=deepcopy(fig), row=3, indicators={'no_indicator': {}}, data=data)
    assert fig == fig3
    assert log_has_re(r'Indicator "no_indicator" ignored\..*', caplog)


def test_plot_trades(testdatadir, caplog):
    fig1 = generate_empty_figure()
    # nothing happens when no trades are available
    fig = plot_trades(fig1, None)
    assert fig == fig1
    assert log_has("No trades found.", caplog)
    pair = "ADA/BTC"
    filename = testdatadir / "backtest-result_test.json"
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
    assert trade_sell.text[0] == "4.0%, roi, 15 min"


def test_generate_candlestick_graph_no_signals_no_trades(default_conf, mocker, testdatadir, caplog):
    row_mock = mocker.patch('freqtrade.plot.plotting.add_indicators',
                            MagicMock(side_effect=fig_generating_mock))
    trades_mock = mocker.patch('freqtrade.plot.plotting.plot_trades',
                               MagicMock(side_effect=fig_generating_mock))

    pair = "UNITTEST/BTC"
    timerange = TimeRange(None, 'line', 0, -1000)
    data = history.load_pair_history(pair=pair, timeframe='1m',
                                     datadir=testdatadir, timerange=timerange)
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


def test_generate_candlestick_graph_no_trades(default_conf, mocker, testdatadir):
    row_mock = mocker.patch('freqtrade.plot.plotting.add_indicators',
                            MagicMock(side_effect=fig_generating_mock))
    trades_mock = mocker.patch('freqtrade.plot.plotting.plot_trades',
                               MagicMock(side_effect=fig_generating_mock))
    pair = 'UNITTEST/BTC'
    timerange = TimeRange(None, 'line', 0, -1000)
    data = history.load_pair_history(pair=pair, timeframe='1m',
                                     datadir=testdatadir, timerange=timerange)

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

    assert find_trace_in_fig_data(figure.data, "Bollinger Band")

    assert row_mock.call_count == 2
    assert trades_mock.call_count == 1


def test_generate_Plot_filename():
    fn = generate_plot_filename("UNITTEST/BTC", "5m")
    assert fn == "freqtrade-plot-UNITTEST_BTC-5m.html"


def test_generate_plot_file(mocker, caplog):
    fig = generate_empty_figure()
    plot_mock = mocker.patch("freqtrade.plot.plotting.plot", MagicMock())
    store_plot_file(fig, filename="freqtrade-plot-UNITTEST_BTC-5m.html",
                    directory=Path("user_data/plot"))

    expected_fn = str(Path("user_data/plot/freqtrade-plot-UNITTEST_BTC-5m.html"))
    assert plot_mock.call_count == 1
    assert plot_mock.call_args[0][0] == fig
    assert (plot_mock.call_args_list[0][1]['filename']
            == expected_fn)
    assert log_has(f"Stored plot as {expected_fn}",
                   caplog)


def test_add_profit(testdatadir):
    filename = testdatadir / "backtest-result_test.json"
    bt_data = load_backtest_data(filename)
    timerange = TimeRange.parse_timerange("20180110-20180112")

    df = history.load_pair_history(pair="TRX/BTC", timeframe='5m',
                                   datadir=testdatadir, timerange=timerange)
    fig = generate_empty_figure()

    cum_profits = create_cum_profit(df.set_index('date'),
                                    bt_data[bt_data["pair"] == 'TRX/BTC'],
                                    "cum_profits", timeframe="5m")

    fig1 = add_profit(fig, row=2, data=cum_profits, column='cum_profits', name='Profits')
    figure = fig1.layout.figure
    profits = find_trace_in_fig_data(figure.data, "Profits")
    assert isinstance(profits, go.Scatter)
    assert profits.yaxis == "y2"


def test_generate_profit_graph(testdatadir):
    filename = testdatadir / "backtest-result_test.json"
    trades = load_backtest_data(filename)
    timerange = TimeRange.parse_timerange("20180110-20180112")
    pairs = ["TRX/BTC", "ADA/BTC"]

    tickers = history.load_data(datadir=testdatadir,
                                pairs=pairs,
                                timeframe='5m',
                                timerange=timerange
                                )
    trades = trades[trades['pair'].isin(pairs)]

    fig = generate_profit_graph(pairs, tickers, trades, timeframe="5m")
    assert isinstance(fig, go.Figure)

    assert fig.layout.title.text == "Freqtrade Profit plot"
    assert fig.layout.yaxis.title.text == "Price"
    assert fig.layout.yaxis2.title.text == "Profit"
    assert fig.layout.yaxis3.title.text == "Profit"

    figure = fig.layout.figure
    assert len(figure.data) == 4

    avgclose = find_trace_in_fig_data(figure.data, "Avg close price")
    assert isinstance(avgclose, go.Scatter)

    profit = find_trace_in_fig_data(figure.data, "Profit")
    assert isinstance(profit, go.Scatter)

    for pair in pairs:
        profit_pair = find_trace_in_fig_data(figure.data, f"Profit {pair}")
        assert isinstance(profit_pair, go.Scatter)


def test_start_plot_dataframe(mocker):
    aup = mocker.patch("freqtrade.plot.plotting.load_and_plot_trades", MagicMock())
    args = [
        "plot-dataframe",
        "--config", "config.json.example",
        "--pairs", "ETH/BTC"
    ]
    start_plot_dataframe(get_args(args))

    assert aup.call_count == 1
    called_config = aup.call_args_list[0][0][0]
    assert "pairs" in called_config
    assert called_config['pairs'] == ["ETH/BTC"]


def test_load_and_plot_trades(default_conf, mocker, caplog, testdatadir):
    default_conf['trade_source'] = 'file'
    default_conf["datadir"] = testdatadir
    default_conf['exportfilename'] = str(testdatadir / "backtest-result_test.json")
    default_conf['indicators1'] = ["sma5", "ema10"]
    default_conf['indicators2'] = ["macd"]
    default_conf['pairs'] = ["ETH/BTC", "LTC/BTC"]

    candle_mock = MagicMock()
    store_mock = MagicMock()
    mocker.patch.multiple(
        "freqtrade.plot.plotting",
        generate_candlestick_graph=candle_mock,
        store_plot_file=store_mock
        )
    load_and_plot_trades(default_conf)

    # Both mocks should be called once per pair
    assert candle_mock.call_count == 2
    assert store_mock.call_count == 2

    assert candle_mock.call_args_list[0][1]['indicators1'] == ['sma5', 'ema10']
    assert candle_mock.call_args_list[0][1]['indicators2'] == ['macd']

    assert log_has("End of plotting process. 2 plots generated", caplog)


def test_start_plot_profit(mocker):
    aup = mocker.patch("freqtrade.plot.plotting.plot_profit", MagicMock())
    args = [
        "plot-profit",
        "--config", "config.json.example",
        "--pairs", "ETH/BTC"
    ]
    start_plot_profit(get_args(args))

    assert aup.call_count == 1
    called_config = aup.call_args_list[0][0][0]
    assert "pairs" in called_config
    assert called_config['pairs'] == ["ETH/BTC"]


def test_start_plot_profit_error(mocker):

    args = [
        "plot-profit",
        "--pairs", "ETH/BTC"
    ]
    argsp = get_args(args)
    # Make sure we use no config. Details: #2241
    # not resetting config causes random failures if config.json exists
    argsp["config"] = []
    with pytest.raises(OperationalException):
        start_plot_profit(argsp)


def test_plot_profit(default_conf, mocker, testdatadir, caplog):
    default_conf['trade_source'] = 'file'
    default_conf["datadir"] = testdatadir
    default_conf['exportfilename'] = str(testdatadir / "backtest-result_test.json")
    default_conf['pairs'] = ["ETH/BTC", "LTC/BTC"]

    profit_mock = MagicMock()
    store_mock = MagicMock()
    mocker.patch.multiple(
        "freqtrade.plot.plotting",
        generate_profit_graph=profit_mock,
        store_plot_file=store_mock
    )
    plot_profit(default_conf)

    # Plot-profit generates one combined plot
    assert profit_mock.call_count == 1
    assert store_mock.call_count == 1

    assert profit_mock.call_args_list[0][0][0] == default_conf['pairs']
    assert store_mock.call_args_list[0][1]['auto_open'] is True
