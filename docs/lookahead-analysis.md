# Lookahead analysis
This page explains how to validate your strategy in terms of look ahead bias.

Checking look ahead bias is the bane of any strategy since it is sometimes very easy to introduce backtest bias -
but very hard to detect.

Backtesting initializes all timestamps at once and calculates all indicators in the beginning.
This means that if your indicators or entry/exit signals could look into future candles and falsify your backtest.

Lookahead-analysis requires historic data to be available.
To learn how to get data for the pairs and exchange you're interested in,
head over to the [Data Downloading](data-download.md) section of the documentation.

This command is built upon backtesting
since it internally chains backtests and pokes at the strategy to provoke it to show look ahead bias.
This is done by not looking at the strategy itself - but at the results it returned.
The results are things like changed indicator-values and moved entries/exits compared to the full backtest. 

You can use commands of [Backtesting](backtesting.md).
It also supports the lookahead-analysis of freqai strategies.

- --cache is forced to "none"
- --max_open_trades is forced to be at least equal to the number of pairs 
- --dry_run_wallet is forced to be basically infinite

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
Checks a given strategy for look ahead bias via lookahead-analysis
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
It will start with a backtest of all pairs to generate a baseline for indicators and entries/exits.
After the backtest ran, it will look if the minimum-trade-amount is met
and if not cancel the lookahead-analysis for this strategy.

After setting the baseline it will then do additional runs for every entry and exit separately.
When a verification-backtest is done, it will compare the indicators as the signal (either entry or exit) 
and report the bias.
After all signals have been verified or falsified a result-table will be generated for the user to see.

#### Caveats:
- The lookahead-analysis can only verify / falsify the trades it calculated through.
If there was a strategy with signals that were not triggered in the lookahead-analysis
then it will not have it verified that entry/exit signal either.
This could then lead to a false-negative (the strategy will then be reported as non-biased).
- lookahead-analysis has access to everything that backtesting has too. 
Please don't provoke any configs like enabling position stacking.
If you decide to do so,
then make doubly sure that you won't ever run out of max_open_trades
amount and neither leftover money in your wallet.
