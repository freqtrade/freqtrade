# Advanced Backtesting Analysis

## Analyze the buy/entry and sell/exit tags

It can be helpful to understand how a strategy behaves according to the buy/entry tags used to
mark up different buy conditions. You might want to see more complex statistics about each buy and
sell condition above those provided by the default backtesting output. You may also want to
determine indicator values on the signal candle that resulted in a trade opening.

!!! Note
    The following buy reason analysis is only available for backtesting, *not hyperopt*.

We need to run backtesting with the `--export` option set to `signals` to enable the exporting of
signals **and** trades:

``` bash
freqtrade backtesting -c <config.json> --timeframe <tf> --strategy <strategy_name> --timerange=<timerange> --export=signals
```

This will tell freqtrade to output a pickled dictionary of strategy, pairs and corresponding
DataFrame of the candles that resulted in buy signals. Depending on how many buys your strategy
makes, this file may get quite large, so periodically check your `user_data/backtest_results`
folder to delete old exports.

To analyze the buy tags, we need to use the `buy_reasons.py` script from
[froggleston's repo](https://github.com/froggleston/freqtrade-buyreasons). Follow the instructions
in their README to copy the script into your `freqtrade/scripts/` folder.

Before running your next backtest, make sure you either delete your old backtest results or run
backtesting with the `--cache none` option to make sure no cached results are used.

If all goes well, you should now see a `backtest-result-{timestamp}_signals.pkl` file in the
`user_data/backtest_results` folder.

Now run the `buy_reasons.py` script, supplying a few options:

``` bash
python3 scripts/buy_reasons.py -c <config.json> -s <strategy_name> -t <timerange> -g0,1,2,3,4
```

The `-g` option is used to specify the various tabular outputs, ranging from the simplest (0)
to the most detailed per pair, per buy and per sell tag (4). More options are available by
running with the `-h` option.

### Tuning the buy tags and sell tags to display

To show only certain buy and sell tags in the displayed output, use the following two options:

```
--enter_reason_list : Comma separated list of enter signals to analyse. Default: "all"
--exit_reason_list : Comma separated list of exit signals to analyse. Default: "stop_loss,trailing_stop_loss"
```

For example:

```bash
python3 scripts/buy_reasons.py -c <config.json> -s <strategy_name> -t <timerange> -g0,1,2,3,4 --enter_reason_list "enter_tag_a,enter_tag_b" --exit_reason_list "roi,custom_exit_tag_a,stop_loss"
```

### Outputting signal candle indicators

The real power of the buy_reasons.py script comes from the ability to print out the indicator
values present on signal candles to allow fine-grained investigation and tuning of buy signal
indicators. To print out a column for a given set of indicators, use the `--indicator-list`
option:

```bash
python3 scripts/buy_reasons.py -c <config.json> -s <strategy_name> -t <timerange> -g0,1,2,3,4 --enter_reason_list "enter_tag_a,enter_tag_b" --exit_reason_list "roi,custom_exit_tag_a,stop_loss" --indicator_list "rsi,rsi_1h,bb_lowerband,ema_9,macd,macdsignal"
```

The indicators have to be present in your strategy's main DataFrame (either for your main
timeframe or for informative timeframes) otherwise they will simply be ignored in the script
output.
