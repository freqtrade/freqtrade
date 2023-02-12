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

Before running your next backtest, make sure you either delete your old backtest results or run
backtesting with the `--cache none` option to make sure no cached results are used.

If all goes well, you should now see a `backtest-result-{timestamp}_signals.pkl` file in the
`user_data/backtest_results` folder.

To analyze the entry/exit tags, we now need to use the `freqtrade backtesting-analysis` command
with `--analysis-groups` option provided with space-separated arguments (default `0 1 2`):

``` bash
freqtrade backtesting-analysis -c <config.json> --analysis-groups 0 1 2 3 4 5
```

This command will read from the last backtesting results. The `--analysis-groups` option is
used to specify the various tabular outputs showing the profit fo each group or trade,
ranging from the simplest (0) to the most detailed per pair, per buy and per sell tag (4):

* 1: profit summaries grouped by enter_tag
* 2: profit summaries grouped by enter_tag and exit_tag
* 3: profit summaries grouped by pair and enter_tag
* 4: profit summaries grouped by pair, enter_ and exit_tag (this can get quite large)
* 5: profit summaries grouped by exit_tag

More options are available by running with the `-h` option.

### Using export-filename

Normally, `backtesting-analysis` uses the latest backtest results, but if you wanted to go
back to a previous backtest output, you need to supply the `--export-filename` option.
You can supply the same parameter to `backtest-analysis` with the name of the final backtest
output file. This allows you to keep historical versions of backtest results and re-analyse
them at a later date:

``` bash
freqtrade backtesting -c <config.json> --timeframe <tf> --strategy <strategy_name> --timerange=<timerange> --export=signals --export-filename=/tmp/mystrat_backtest.json
```

You should see some output similar to below in the logs with the name of the timestamped
filename that was exported:

```
2022-06-14 16:28:32,698 - freqtrade.misc - INFO - dumping json to "/tmp/mystrat_backtest-2022-06-14_16-28-32.json"
```

You can then use that filename in `backtesting-analysis`:

```
freqtrade backtesting-analysis -c <config.json> --export-filename=/tmp/mystrat_backtest-2022-06-14_16-28-32.json
```

### Tuning the buy tags and sell tags to display

To show only certain buy and sell tags in the displayed output, use the following two options:

```
--enter-reason-list : Space-separated list of enter signals to analyse. Default: "all"
--exit-reason-list : Space-separated list of exit signals to analyse. Default: "all"
```

For example:

```bash
freqtrade backtesting-analysis -c <config.json> --analysis-groups 0 2 --enter-reason-list enter_tag_a enter_tag_b --exit-reason-list roi custom_exit_tag_a stop_loss
```

### Outputting signal candle indicators

The real power of `freqtrade backtesting-analysis` comes from the ability to print out the indicator
values present on signal candles to allow fine-grained investigation and tuning of buy signal
indicators. To print out a column for a given set of indicators, use the `--indicator-list`
option:

```bash
freqtrade backtesting-analysis -c <config.json> --analysis-groups 0 2 --enter-reason-list enter_tag_a enter_tag_b --exit-reason-list roi custom_exit_tag_a stop_loss --indicator-list rsi rsi_1h bb_lowerband ema_9 macd macdsignal
```

The indicators have to be present in your strategy's main DataFrame (either for your main
timeframe or for informative timeframes) otherwise they will simply be ignored in the script
output.

### Filtering the trade output by date

To show only trades between dates within your backtested timerange, supply the usual `timerange` option in `YYYYMMDD-[YYYYMMDD]` format:

```
--timerange : Timerange to filter output trades, start date inclusive, end date exclusive. e.g. 20220101-20221231
```

For example, if your backtest timerange was `20220101-20221231` but you only want to output trades in January:

```bash
freqtrade backtesting-analysis -c <config.json> --timerange 20220101-20220201
```
