# Plotting

This page explains how to plot prices, indicators and profits.

## Installation

Plotting scripts use Plotly library. Install/upgrade it with:

``` bash
pip install -U -r requirements-plot.txt
```

## Plot price and indicators

Usage for the price plotter:

``` bash
python3 script/plot_dataframe.py [-h] [-p pairs] [--live]
```

Example

``` bash
python3 scripts/plot_dataframe.py -p BTC/ETH
```

The `-p` pairs argument can be used to specify pairs you would like to plot.

Specify custom indicators.
Use `--indicators1` for the main plot and `--indicators2` for the subplot below (if values are in a different range than prices).

``` bash
python3 scripts/plot_dataframe.py -p BTC/ETH --indicators1 sma,ema --indicators2 macd
```

### Advanced use

To plot multiple pairs, separate them with a comma:

``` bash
python3 scripts/plot_dataframe.py -p BTC/ETH,XRP/ETH
```

To plot the current live price use the `--live` flag:

``` bash
python3 scripts/plot_dataframe.py -p BTC/ETH --live
```

To plot a timerange (to zoom in):

``` bash
python3 scripts/plot_dataframe.py -p BTC/ETH --timerange=100-200
```

Timerange doesn't work with live data.

To plot trades stored in a database use `--db-url` argument:

``` bash
python3 scripts/plot_dataframe.py --db-url sqlite:///tradesv3.dry_run.sqlite -p BTC/ETH --trade-source DB
```

To plot trades from a backtesting result, use `--export-filename <filename>`

``` bash
python3 scripts/plot_dataframe.py --export-filename user_data/backtest_data/backtest-result.json -p BTC/ETH
```

To plot a custom strategy the strategy should have first be backtested.
The results may then be plotted with the -s argument:

``` bash
python3 scripts/plot_dataframe.py -s Strategy_Name -p BTC/ETH --datadir user_data/data/<exchange_name>/
```

## Plot profit

The profit plotter shows a picture with three plots:

1) Average closing price for all pairs
2) The summarized profit made by backtesting.
   Note that this is not the real-world profit, but
   more of an estimate.
3) Each pair individually profit

The first graph is good to get a grip of how the overall market
progresses.

The second graph will show how your algorithm works or doesn't.
Perhaps you want an algorithm that steadily makes small profits,
or one that acts less seldom, but makes big swings.

The third graph can be useful to spot outliers, events in pairs
that makes profit spikes.

Usage for the profit plotter:

``` bash
python3 script/plot_profit.py [-h] [-p pair] [--datadir directory] [--ticker_interval num]
```

The `-p` pair argument, can be used to plot a single pair

Example

``` bash
python3 scripts/plot_profit.py --datadir ../freqtrade/freqtrade/tests/testdata-20171221/ -p LTC/BTC
```
