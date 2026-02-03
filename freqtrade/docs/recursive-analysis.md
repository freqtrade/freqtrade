# Recursive analysis

This page explains how to validate your strategy for inaccuracies due to recursive issues with certain indicators.

A recursive formula defines any term of a sequence relative to its preceding term(s). An example of a recursive formula is a<sub>n</sub> = a<sub>n-1</sub> + b.

Why does this matter for Freqtrade? In backtesting, the bot will get full data of the pairs according to the timerange specified. But in a dry/live run, the bot will be limited by the amount of data each exchanges gives.

For example, to calculate a very basic indicator called `steps`, the first row's value is always 0, while the following rows' values are equal to the value of the previous row plus 1. If I were to calculate it using the latest 1000 candles, then the `steps` value of the first row is 0, and the `steps` value at the last closed candle is 999.

What happens if the calculation is using only the latest 500 candles? Then instead of 999, the `steps` value at last closed candle is 499. The difference of the value means your backtest result can differ from your dry/live run result.

The `recursive-analysis` command requires historic data to be available. To learn how to get data for the pairs and exchange you're interested in,
head over to the [Data Downloading](data-download.md) section of the documentation.

This command is built upon preparing different lengths of data and calculates indicators based on them.
This does not backtest the strategy itself, but rather only calculates the indicators. After calculating the indicators of different startup candle values (`startup_candle_count`) are done, the values of last rows across all specified `startup_candle_count` are compared to see how much variance they show compared to the base calculation.

Command settings:

- Use the `-p` option to set your desired pair to analyze. Since we are only looking at indicator values, using more than one pair is redundant. Preferably use a pair with a relatively high price and at least moderate volatility, such as BTC or ETH, to avoid rounding issues that can make the results inaccurate. If no pair is set on the command, the pair used for this analysis is the first pair in the whitelist.
- It is recommended to set a long timerange (at least 5000 candles) so that the initial indicators' calculation that is going to be used as a benchmark has very small or no recursive issues itself. For example, for a 5m timeframe, a timerange of 5000 candles would be equal to 18 days.
- `--cache` is forced to "none" to avoid loading previous indicators calculation automatically.

In addition to the recursive formula check, this command also carries out a simple lookahead bias check on the indicator values only. For a full lookahead check, use [Lookahead-analysis](lookahead-analysis.md).

## Recursive-analysis command reference

--8<-- "commands/recursive-analysis.md"

### Why are odd-numbered default startup candles used?

The default value for startup candles are odd numbers. When the bot fetches candle data from the exchange's API, the last candle is the one being checked by the bot and the rest of the data are the "startup candles".

For example, Binance allows 1000 candles per API call. When the bot receives 1000 candles, the last candle is the "current candle", and the preceding 999 candles are the "startup candles". By setting the startup candle count as 1000 instead of 999, the bot will try to fetch 1001 candles instead. The exchange API will then send candle data in a paginated form, i.e. in case of the Binance API, this will be two groups- one of length 1000 and another of length 1. This results in the bot thinking the strategy needs 1001 candles of data, and so it will download 2000 candles worth of data instead, which means there will be 1 "current candle" and 1999 "startup candles".

Furthermore, exchanges limit the number of consecutive bulk API calls, e.g. Binance allows 5 calls. In this case, only 5000 candles can be downloaded from Binance API without hitting the API rate limit, which means the max `startup_candle_count` you can have is 4999.

Please note that this candle limit may be changed in the future by the exchanges without any prior notice.

### How does the command work?

- Firstly an initial indicator calculation is carried out using the supplied timerange to generate a benchmark for indicator values.
- After setting the benchmark it will then carry out additional runs for each of the different startup candle count values.
- The command will then compare the indicator values at the last candle rows and report the differences in a table.

## Understanding the recursive-analysis output

This is an example of an output results table where at least one indicator has a recursive formula issue:

```
| indicators   | 20      | 40      | 80     | 100    | 150     | 300     | 999    |
|--------------+---------+---------+--------+--------+---------+---------+--------|
| rsi_30       | nan%    | -6.025% | 0.612% | 0.828% | -0.140% | 0.000%  | 0.000% |
| rsi_14       | 24.141% | -0.876% | 0.070% | 0.007% | -0.000% | -0.000% | -      |
```

The column headers indicate the different `startup_candle_count` used in the analysis. The values in the table indicate the variance of the calculated indicators compared to the benchmark value.

`nan%` means the value of that indicator cannot be calculated due to lack of data. In this example, you cannot calculate RSI with length 30 with just 21 candles (1 current candle + 20 startup candles).

Users should assess the table per indicator to decide if the specified `startup_candle_count` results in a sufficiently small variance so that the indicator does not have any effect on entries and/or exits.

As such, aiming for absolute zero variance (shown by `-` value) might not be the best option, because some indicators might require you to use such a long `startup_candle_count` to have zero variance.

## Caveats

- `recursive-analysis` will only calculate and compare the indicator values at the last row. The output table reports the percentage differences between the different startup candle count calculations and the original benchmark calculation. Whether it has any actual impact on your entries and exits is not included.
- The ideal scenario is that indicators will have no variance (or at least very close to 0%) despite the startup candle being varied. In reality, indicators such as EMA are using a recursive formula to calculate indicator values, so the goal is not necessarily to have zero percentage variance, but to have the variance low enough (and therefore `startup_candle_count` high enough) that the recursion inherent in the indicator will not have any real impact on trading decisions.
- `recursive-analysis` will only run calculations on `populate_indicators` and `@informative` decorator(s). If you put any indicator calculation on `populate_entry_trend` or `populate_exit_trend`, it won't be calculated.
