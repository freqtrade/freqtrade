# Advanced Backtesting Analysis

## Analyse the buy/entry and sell/exit tags

It can be helpful to understand how a strategy behaves according to the buy/entry tags used to
mark up different buy conditions. You might want to see more complex statistics about each buy and
sell condition above those provided by the default backtesting output. You may also want to
determine indicator values on the signal candle that resulted in a trade opening.

!!! Note
    The following buy reason analysis is only available for backtesting, *not hyperopt*.

We first need to enable the exporting of trades from backtesting:

```bash
freqtrade backtesting -c <config.json> --timeframe <tf> --strategy <strategy_name> --timerange=<timerange> --export=trades --export-filename=user_data/backtest_results/<name>-<timerange>
```

To analyse the buy tags, we need to use the `freqtrade tag-analysis` command. We need the signal
candles for each opened trade so add the following option to your config file:

```
'backtest_signal_candle_export_enable': true,
```

This will tell freqtrade to output a pickled dictionary of strategy, pairs and corresponding
DataFrame of the candles that resulted in buy signals. Depending on how many buys your strategy
makes, this file may get quite large, so periodically check your `user_data/backtest_results`
folder to delete old exports.

Before running your next backtest, make sure you either delete your old backtest results or run
backtesting with the `--cache none` option to make sure no cached results are used.

If all goes well, you should now see a `backtest-result-{timestamp}_signals.pkl` file in the
`user_data/backtest_results` folder.

Now run the buy_reasons.py script, supplying a few options:

```bash
freqtrade tag-analysis -c <config.json> -s <strategy_name> -t <timerange> -g0,1,2,3,4
```

The `-g` option is used to specify the various tabular outputs, ranging from the simplest (0)
to the most detailed per pair, per buy and per sell tag (4). More options are available by
running with the `-h` option.

### Tuning the buy tags and sell tags to display

To show only certain buy and sell tags in the displayed output, use the following two options:

```
--buy_reason_list : Comma separated list of buy signals to analyse. Default: "all"
--sell_reason_list : Comma separated list of sell signals to analyse. Default: "stop_loss,trailing_stop_loss"
```

For example:

```bash
freqtrade tag-analysis -c <config.json> -s <strategy_name> -t <timerange> -g0,1,2,3,4 --buy_reason_list "buy_tag_a,buy_tag_b" --sell_reason_list "roi,custom_sell_tag_a,stop_loss"
```

### Outputting signal candle indicators

The real power of the buy_reasons.py script comes from the ability to print out the indicator
values present on signal candles to allow fine-grained investigation and tuning of buy signal
indicators. To print out a column for a given set of indicators, use the `--indicator-list`
option:

```bash
freqtrade tag-analysis -c <config.json> -s <strategy_name> -t <timerange> -g0,1,2,3,4 --buy_reason_list "buy_tag_a,buy_tag_b" --sell_reason_list "roi,custom_sell_tag_a,stop_loss" --indicator_list "rsi,rsi_1h,bb_lowerband,ema_9,macd,macdsignal"
```

The indicators have to be present in your strategy's main DataFrame (either for your main
timeframe or for informatives) otherwise they will simply be ignored in the script
output.
