# Plotting

This page explains how to plot prices, indicators and profits.

## Installation

Plotting scripts use Plotly library. Install/upgrade it with:

``` bash
pip install -U -r requirements-plot.txt
```

## Plot price and indicators

Usage for the candlestick plotting:

```
usage: freqtrade plot-dataframe [-h] [-p PAIRS [PAIRS ...]]
                                [--indicators1 INDICATORS1 [INDICATORS1 ...]]
                                [--indicators2 INDICATORS2 [INDICATORS2 ...]]
                                [--plot-limit INT] [--db-url PATH]
                                [--trade-source {DB,file}] [--export EXPORT]
                                [--export-filename PATH]
                                [--timerange TIMERANGE]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Show profits for only these pairs. Pairs are space-
                        separated.
  --indicators1 INDICATORS1 [INDICATORS1 ...]
                        Set indicators from your strategy you want in the
                        first row of the graph. Space-separated list. Example:
                        `ema3 ema5`. Default: `['sma', 'ema3', 'ema5']`.
  --indicators2 INDICATORS2 [INDICATORS2 ...]
                        Set indicators from your strategy you want in the
                        third row of the graph. Space-separated list. Example:
                        `fastd fastk`. Default: `['macd', 'macdsignal']`.
  --plot-limit INT      Specify tick limit for plotting. Notice: too high
                        values cause huge files. Default: 750.
  --db-url PATH         Override trades database URL, this is useful in custom
                        deployments (default: `sqlite:///tradesv3.sqlite` for
                        Live Run mode, `sqlite://` for Dry Run).
  --trade-source {DB,file}
                        Specify the source for trades (Can be DB or file
                        (backtest file)) Default: file
  --export EXPORT       Export backtest results, argument are: trades.
                        Example: `--export=trades`
  --export-filename PATH
                        Save backtest results to the file with this filename
                        (default: `user_data/backtest_results/backtest-
                        result.json`). Requires `--export` to be set as well.
                        Example: `--export-filename=user_data/backtest_results
                        /backtest_today.json`
  --timerange TIMERANGE
                        Specify what timerange of data to use.

```

Example

``` bash
freqtrade plot-dataframe -p BTC/ETH
```

The `--pairs` argument can be used to specify pairs you would like to plot.

!!! Note
    Generates one plot-file per pair.

Specify custom indicators.
Use `--indicators1` for the main plot and `--indicators2` for the subplot below (if values are in a different range than prices).

``` bash
freqtrade plot-dataframe -p BTC/ETH --indicators1 sma ema --indicators2 macd
```

### Advanced use

To plot multiple pairs, separate them with a comma:

``` bash
freqtrade plot-dataframe -p BTC/ETH XRP/ETH
```

To plot a timerange (to zoom in):

``` bash
freqtrade plot-dataframe -p BTC/ETH --timerange=20180801-20180805
```

To plot trades stored in a database use `--db-url` argument:

``` bash
freqtrade plot-dataframe --db-url sqlite:///tradesv3.dry_run.sqlite -p BTC/ETH --trade-source DB
```

To plot trades from a backtesting result, use `--export-filename <filename>`

``` bash
freqtrade plot-dataframe --export-filename user_data/backtest_results/backtest-result.json -p BTC/ETH
```

To plot a custom strategy the strategy should have first be backtested.
The results may then be plotted with the -s argument:

``` bash
freqtrade plot-dataframe -s Strategy_Name -p BTC/ETH --datadir user_data/data/<exchange_name>/
```

## Plot profit

The profit plotter shows a picture with three plots:

1) Average closing price for all pairs
2) The summarized profit made by backtesting.
   Note that this is not the real-world profit, but
   more of an estimate.
3) Each pair individually profit

The first graph is good to get a grip of how the overall market progresses.

The second graph will show how your algorithm works or doesn't.
Perhaps you want an algorithm that steadily makes small profits,
or one that acts less seldom, but makes big swings.

The third graph can be useful to spot outliers, events in pairs that makes profit spikes.

Usage for the profit plotter:

```
usage: freqtrade plot-profit [-h] [-p PAIRS [PAIRS ...]]
                             [--timerange TIMERANGE] [--export EXPORT]
                             [--export-filename PATH] [--db-url PATH]
                             [--trade-source {DB,file}]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Show profits for only these pairs. Pairs are space-
                        separated.
  --timerange TIMERANGE
                        Specify what timerange of data to use.
  --export EXPORT       Export backtest results, argument are: trades.
                        Example: `--export=trades`
  --export-filename PATH
                        Save backtest results to the file with this filename
                        (default: `user_data/backtest_results/backtest-
                        result.json`). Requires `--export` to be set as well.
                        Example: `--export-filename=user_data/backtest_results
                        /backtest_today.json`
  --db-url PATH         Override trades database URL, this is useful in custom
                        deployments (default: `sqlite:///tradesv3.sqlite` for
                        Live Run mode, `sqlite://` for Dry Run).
  --trade-source {DB,file}
                        Specify the source for trades (Can be DB or file
                        (backtest file)) Default: file

```

The `--pairs`  argument, can be used to limit the pairs that are considered for this calculation.

Example

``` bash
freqtrade plot-profit --datadir ../freqtrade/freqtrade/tests/testdata-20171221/ -p LTC/BTC
```
