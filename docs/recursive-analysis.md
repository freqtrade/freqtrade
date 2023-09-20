# Lookahead analysis

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
- Since we are only looking at indicators' value, using more than one pair is redundant. It is recommended to set the pair used in the command using `-p` flag, preferably using pair with high price, such as BTC or ETH, to avoid having rounding issue that can make the results inaccurate. If no pair is set on the command, the pair used for this analysis the first pair in the whitelist.
- It's recommended to set a long timerange (at least consist of 5000 candles), so that the initial backtest that going to be used as benchmark have very small or no recursive issue at all. For example, for a 5m timeframe, timerange of 5000 candles would be equal to 18 days.

Beside recursive formula check, this command also going to do a simple lookahead bias check on the indicators' value only. It won't replace [Lookahead-analysis](lookahead-analysis.md), since this check won't check the difference in trades' entries and exits. It will only check whether there is any difference in indicators' value if the end of the data are moved.

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
                                    [-p PAIRS [PAIRS ...]]
                                    [--freqai-backtest-live-models]

```

### Summary

Checks a given strategy for look ahead bias via lookahead-analysis
Look ahead bias means that the backtest uses data from future candles thereby not making it viable beyond backtesting
and producing false hopes for the one backtesting.

### Introduction

Many strategies - without the programmer knowing - have fallen prey to look ahead bias.

Any backtest will populate the full dataframe including all time stamps at the beginning.
If the programmer is not careful or oblivious how things work internally
(which sometimes can be really hard to find out) then it will just look into the future making the strategy amazing
but not realistic.

This command is made to try to verify the validity in the form of the aforementioned look ahead bias.

### How does the command work?

It will start with a backtest of all pairs to generate a baseline for indicators and entries/exits.
After the backtest ran, it will look if the `minimum-trade-amount` is met
and if not cancel the lookahead-analysis for this strategy.

After setting the baseline it will then do additional runs for every entry and exit separately.
When a verification-backtest is done, it will compare the indicators as the signal (either entry or exit) and report the bias.
After all signals have been verified or falsified a result-table will be generated for the user to see.

### Caveats

- `lookahead-analysis` can only verify / falsify the trades it calculated and verified.
If the strategy has many different signals / signal types, it's up to you to select appropriate parameters to ensure that all signals have triggered at least once. Not triggered signals will not have been verified.
This could lead to a false-negative (the strategy will then be reported as non-biased).
- `lookahead-analysis` has access to everything that backtesting has too.
Please don't provoke any configs like enabling position stacking.
If you decide to do so, then make doubly sure that you won't ever run out of `max_open_trades` amount and neither leftover money in your wallet.
