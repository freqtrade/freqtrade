# Lookahead analysis
This page explains how to validate your strategy in terms of look ahead bias.

Checking look ahead bias is the bane of any strategy since it is sometimes very easy to introduce backtest bias -
but very hard to detect.

Backtesting initializes all timestamps at once and calculates all indicators in the beginning.
This means that if you are allowing your indicators (or the libraries that get used) then you would 
look into the future and falsify your backtest.

Lookahead-analysis requires historic data to be available.
To learn how to get data for the pairs and exchange you're interested in,
head over to the [Data Downloading](data-download.md) section of the documentation.

This command is built upon backtesting
since it internally chains backtests and pokes at the strategy to provoke it to show look ahead bias.
This is done by looking not at the strategy itself - but at the results it returned.
The results are things like changed indicator-values and moved entries/exits compared to the full backtest. 

You can use commands of [Backtesting](backtesting.md).
It also supports the lookahead-analysis of freqai strategies.

--cache is enforced to be "none"

## Backtesting command reference

```
usage: freqtrade lookahead-analysis [-h] [-v] [-V] 
                             [--minimum-trade-amount INT]
                             [--targeted-trade-amount INT]
                             [--lookahead-analysis-exportfilename PATH]

optional arguments:
  -h, --help            show this help message and exit
  --minimum-trade-amount INT
                        Override the value of the `minimum_trade_amount` configuration
                        setting
                        Requires `--targeted-trade-amount` to be larger or equal to --minimum-trade-amount.
                        (default: 10)
  --targeted-trade-amount INT
                        Override the value of the `minimum_trade_amount` configuration
                        (default: 20)
  --lookahead-analysis-exportfilename PATH
                        Use this filename to save your lookahead-analysis-results to a csv file
```


#### Summary
Checks a given strategy for look ahead bias via backtest-analysis
Look ahead bias means that the backtest uses data from future candles thereby not making it viable beyond backtesting
and producing false hopes for the one backtesting.

#### Introduction:
Many strategies - without the programmer knowing - have fallen prey to look ahead bias.

Any backtest will populate the full dataframe including all time stamps at the beginning.
If the programmer is not careful or oblivious how things work internally
(which sometimes can be really hard to find out) then it will just look into the future making the strategy amazing
but not realistic.

This command is made to try to verify the validity in the form of the aforementioned look ahead bias.

#### How does the command work?
It will not look at the strategy or any contents itself but instead will run multiple backtests 
by using precisely cut timeranges and analyzing the results each time, comparing to the full timerange.

At first, it starts a backtest over the whole duration
and then repeats backtests from the same starting point to the respective points to watch.
In addition, it analyzes the dataframes form the overall backtest to the cut ones.

At the end it will return a result-table in terminal.

Hint:
If an entry or exit condition is only triggered rarely or the timerange was chosen
so only a few entry conditions are met
then the bias checker is unable to catch the biased entry or exit condition.
In the end it only checks which entry and exit signals have been triggered.

---Flow chart here for better understanding---
