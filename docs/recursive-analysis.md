# Recursive analysis

This page explains how to validate your strategy for inaccuracies due to recursive issues with certain indicators.

A recursive formula defines any term of a sequence relative to its preceding term(s). An example of a recursive formula is a<sub>n</sub> = a<sub>n-1</sub> + b.

Why does this matter for Freqtrade? In backtesting, the bot will get full data of the pairs according to the timerange specified. But in a dry/live run, the bot will be limited by the amount of data each exchanges gives.

For example, to calculate a very basic indicator called `steps`, the first row's value is always 0, while the following rows' values are equal to the value of the previous row plus 1. If I were to calculate it using the latest 1000 candles, then the `steps` value of the first row is 0, and the `steps` value at the last closed candle is 999.

What happens if the calculation is using only the latest 500 candles? Then instead of 999, the `steps` value at last closed candle is 499. The difference of the value means your backtest result can differ from your dry/live run result.

The `recursive-analysis` command requires historic data to be available. To learn how to get data for the pairs and exchange you're interested in,
head over to the [Data Downloading](data-download.md) section of the documentation.

This command is built upon backtesting since it internally chains backtests to prepare different lengths of data and calculates indicators based on the downloaded data.
This does not run the strategy itself, but rather uses the indicators it contains. After multiple backtests are done to calculate the indicators of different startup candle values (`startup_candle_count`), the values of last rows across all backtests are compared to see how much variance they show compared to the base backtest.

Command settings:
- Use the `-p` option to set your desired pair to analyse. Since we are only looking at indicator values, using more than one pair is redundant. Preferably use a pair with a relatively high price and at least moderate volatility, such as BTC or ETH, to avoid rounding issues that can make the results inaccurate. If no pair is set on the command, the pair used for this analysis is the first pair in the whitelist.
- It is recommended to set a long timerange (at least 5000 candles) so that the initial backtest that is going to be used as a benchmark has very small or no recursive issues itself. For example, for a 5m timeframe, a timerange of 5000 candles would be equal to 18 days.
- `--cache` is forced to "none" to avoid loading previous backtest results automatically.

In addition to the recursive formula check, this command also carries out a simple lookahead bias check on the indicator values only. For a full lookahead check, use [Lookahead-analysis](lookahead-analysis.md).

## Recursive-analysis command reference

```
usage: freqtrade recursive-analysis [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                    [-d PATH] [--userdir PATH] [-s NAME]
                                    [--strategy-path PATH]
                                    [--recursive-strategy-search]
                                    [--freqaimodel NAME]
                                    [--freqaimodel-path PATH] [-i TIMEFRAME]
                                    [--timerange TIMERANGE]
                                    [--data-format-ohlcv {json,jsongz,hdf5,feather,parquet}]
                                    [-p PAIR]
                                    [--freqai-backtest-live-models]
                                    [--startup-candle STARTUP_CANDLES [STARTUP_CANDLES ...]]

optional arguments:
-p PAIR, --pairs PAIR
                        Limit command to this pair.
--startup-candle STARTUP_CANDLE [STARTUP_CANDLE ...]
                        Provide a space-separated list of startup_candle_count to
                        be checked. Default : `199 399 499 999 1999`.
```

### Summary

Checks a given strategy for recursive formula issue via `recursive-analysis`.

### Why are odd-numbered default startup candles used?

The default value for startup candles are odd numbers. When the bot fetches candle data from the exchange's API, the last candle is the one being checked by the bot and the rest of the data are the "startup candles".

For example, Binance allows 1000 candles per API call. When the bot receives 1000 candles, the last candle is the "current candle", and the preceding 999 candles are the "startup candles". By setting the startup candle count as 1000 instead of 999, the bot will try to fetch 1001 candles instead. The exchange API will then send candle data in a paginated form, i.e. in case of the Binance API, this will be two groups- one of length 1000 and another of length 1. This results in the bot thinking the strategy needs 1001 candles of data, and so it will download 2000 candles worth of data instead, which means there will be 1 "current candle" and 1999 "startup candles".

Furthermore, exchanges limit the number of consecutive bulk API calls, e.g. Binance allows 5 calls. In this case, only 5000 candles can be downloaded from Binance API without hitting the API rate limit, which means the max `startup_candle_count` you can have is 4999.

Please note that this candle limit may be changed in the future by the exchanges without any prior notice.

### How does the command work?

- Firstly an initial backtest is carried out using the supplied timerange to generate a benchmark for indicator values.
- After setting the benchmark it will then carry out additional runs for each different startup candle count.
- It will then compare the indicator values at the last candle rows and report the differences in a table.

## Understand the recursive analysis output

This is an example of how the output will look like when at least one indicator have recursive formula issue

```
| indicators   | 20      | 40      | 80     | 100    | 150     | 300     | 999    |
|--------------+---------+---------+--------+--------+---------+---------+--------|
| rsi_30       | nan%    | -6.025% | 0.612% | 0.828% | -0.140% | 0.000%  | 0.000% |
| rsi_14       | 24.141% | -0.876% | 0.070% | 0.007% | -0.000% | -0.000% | -      |
```

The numbers at the header indicates different `startup_candle_count` used in the analysis. The numbers in the table indicates how much varied are they compared to the benchmark value.

`nan%` means the value of that indicator can't be calculated due to lack of data. In this example, you can't calculate rsi with length of 30 with just 21 (1 current candle + 20 startup candles) data.

Important thing to note, we can't tell you which `startup_candle_count` to use because it depends on each users' preference on how much variance is small enough in their opinion to not have any effect on entries and/or exits.

Aiming for zero variance (shown by `-` value) might not be the best option, because some indicators might requires you to use a very long startup to have zero variance.

## Caveats

- `recursive-analysis` will only calculate and compare the indicator values at the last row. The output table reports the percentage differences between the different startup candle count backtests and the original benchmark backtest. Whether it has any actual impact on your entries and exits is not included.
- The ideal scenario is that indicators will have no variance (or at least very close to 0%) despite the startup candle being varied. In reality, indicators such as EMA are using a recursive formula to calculate indicator values, so the goal is not necessarily to have zero percentage variance, but to have the variance low enough (and the `startup_candle_count` high enough) that the recursion inherent in the indicator will not have any real impact on trading decisions.
