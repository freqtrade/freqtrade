# Plotting
This page explains how to plot prices, indicator, profits.

## Table of Contents
- [Plot price and indicators](#plot-price-and-indicators)
- [Plot profit](#plot-profit)

## Installation

Plotting scripts use Plotly library. Install/upgrade it with:

```
pip install --upgrade plotly
```

At least version 2.3.0 is required.

## Plot price and indicators
Usage for the price plotter:

```
script/plot_dataframe.py [-h] [-p pair] [--live]
```

Example
```
python scripts/plot_dataframe.py -p BTC/ETH
```

The `-p` pair argument, can be used to specify what
pair you would like to plot.

**Advanced use**

To plot the current live price use the `--live` flag:
```
python scripts/plot_dataframe.py -p BTC/ETH --live
```

To plot a timerange (to zoom in):
```
python scripts/plot_dataframe.py -p BTC/ETH --timerange=100-200
```
Timerange doesn't work with live data.

To plot trades stored in a database use `--db-url` argument:
```
python scripts/plot_dataframe.py --db-url sqlite:///tradesv3.dry_run.sqlite -p BTC/ETH
```

To plot a test strategy the strategy should have first be backtested. 
The results may then be plotted with the -s argument:
```
python scripts/plot_dataframe.py -s Strategy_Name -p BTC/ETH --datadir user_data/data/<exchange_name>/
```

## Plot profit

The profit plotter show a picture with three plots:
1) Average closing price for all pairs
2) The summarized profit made by backtesting.
   Note that this is not the real-world profit, but
   more of an estimate.
3) Each pair individually profit

The first graph is good to get a grip of how the overall market
progresses.

The second graph will show how you algorithm works or doesnt.
Perhaps you want an algorithm that steadily makes small profits,
or one that acts less seldom, but makes big swings.

The third graph can be useful to spot outliers, events in pairs
that makes profit spikes.

Usage for the profit plotter:

```
script/plot_profit.py [-h] [-p pair] [--datadir directory] [--ticker_interval num]
```

The `-p` pair argument, can be used to plot a single pair

Example
```
python3 scripts/plot_profit.py --datadir ../freqtrade/freqtrade/tests/testdata-20171221/ -p BTC_LTC
```
