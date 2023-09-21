# Recursive analysis

This page explains how to validate your strategy in terms of recursive formula issue.

First of all, what is recursive formula? Recursive formula is a formula that defines any term of a sequence in terms of its preceding term(s). Example of a recursive formula is a<sub>n</sub> = a<sub>n-1</sub> + b.

Second question is why is it matter for Freqtrade? It matters because in backtesting, the bot will get full data of the pairs according to the timerange specified. But in dry/live run, the bot will have limited amounts of data, limited by what each exchanges gives.

For example, let's say that I want to calculate a very basic indicator called `steps`. The first row's value is always 0, while the following rows' values are equal to the value of the previous row's plus 1. If I were to calculate it using latest 1000 candles, then the `steps` value of first row is 0, and the `steps` value at last closed candle is 999.

But what if I only calculate based of latest 500 candles? Then instead of 999, the `steps` value at last closed candle is 499. The difference of the value means your backtest result can differ from your dry/live run result.

Recursive-analysis requires historic data to be available. To learn how to get data for the pairs and exchange you're interested in,
head over to the [Data Downloading](data-download.md) section of the documentation.

This command is built upon backtesting since it internally chains backtests to prepare different lenghts of data and calculate indicators based of each of the prepared data.
This is done by not looking at the strategy itself - but at the value of the indicators it returned. After multiple backtests are done to calculate the indicators of different startup candles value, the values of last rows are compared to see hoe much differences are they compared to the base backtest.

- `--cache` is forced to "none".
- Since we are only looking at indicators' value, using more than one pair is redundant. It is recommended to set the pair used in the command using `-p` flag, preferably using pair with high price, such as BTC or ETH, to avoid having rounding issue that can make the results inaccurate. If no pair is set on the command, the pair used for this analysis is the first pair in the whitelist.
- It's recommended to set a long timerange (at least consist of 5000 candles), so that the initial backtest that going to be used as benchmark have very small or no recursive issue at all. For example, for a 5m timeframe, timerange of 5000 candles would be equal to 18 days.

Beside recursive formula check, this command also going to do a simple lookahead bias check on the indicators' value only. It won't replace [Lookahead-analysis](lookahead-analysis.md), since this check won't check the difference in trades' entries and exits, which is the important effect of lookahead bias. It will only check whether there is any lookahead bias in indicators if the end of the data are moved.

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

Checks a given strategy for recursive formula issue via recursive-analysis.
Recursive formula issue means that the indicator's calculation don't have enough data for its calculation to produce correct value.

### How does the command work?

It will start with a backtest using the supplied timerange to generate a baseline for indicators' value.
After setting the baseline it will then do additional runs for each different startup candles.
When the additional runs are done, it will compare the indicators at the last rows and report the differences in a table.

### Caveats

- `recursive-analysis` will only calculate and compare the indicators' value at the last row. If there are any differences, the table will only tell you the percentage differences. Whether it has any real impact on your entries and exits isn't checked.
- The ideal scenario is to have your indicators have no difference at all despite the startup candle being varied. But in reality, some of publicly-available formulas are using recursive formula. So the goal isn't to have zero differences, but to have the differences low enough to make sure they won't have any real impact on trading decisions.
