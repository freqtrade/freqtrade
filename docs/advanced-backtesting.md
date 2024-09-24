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
DataFrame of the candles that resulted in entry and exit signals.
Depending on how many entries your strategy makes, this file may get quite large, so periodically check your `user_data/backtest_results` folder to delete old exports.

Before running your next backtest, make sure you either delete your old backtest results or run
backtesting with the `--cache none` option to make sure no cached results are used.

If all goes well, you should now see a `backtest-result-{timestamp}_signals.pkl` and `backtest-result-{timestamp}_exited.pkl` files in the `user_data/backtest_results` folder.

To analyze the entry/exit tags, we now need to use the `freqtrade backtesting-analysis` command
with `--analysis-groups` option provided with space-separated arguments:

``` bash
freqtrade backtesting-analysis -c <config.json> --analysis-groups 0 1 2 3 4 5
```

This command will read from the last backtesting results. The `--analysis-groups` option is
used to specify the various tabular outputs showing the profit of each group or trade,
ranging from the simplest (0) to the most detailed per pair, per buy and per sell tag (4):

* 0: overall winrate and profit summary by enter_tag
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

!!! Note "Indicator List"
    The indicator values will be displayed for both entry and exit points. If `--indicator-list all` is specified, 
    only the indicators at the entry point will be shown to avoid excessively large lists, which could occur depending on the strategy.

There are a range of candle and trade-related fields that are included in the analysis so are 
automatically accessible by including them on the indicator-list, and these include:

- **open_date     :** trade open datetime
- **close_date    :** trade close datetime
- **min_rate      :** minimum price seen throughout the position
- **max_rate      :** maximum price seen throughout the position
- **open          :** signal candle open price
- **close         :** signal candle close price
- **high          :** signal candle high price
- **low           :** signal candle low price
- **volume        :** signal candle volume
- **profit_ratio  :** trade profit ratio
- **profit_abs    :** absolute profit return of the trade 

#### Sample Output for Indicator Values

```bash
freqtrade backtesting-analysis -c user_data/config.json --analysis-groups 0 --indicator-list chikou_span tenkan_sen 
```

In this example,
we aim to display the `chikou_span` and `tenkan_sen` indicator values at both the entry and exit points of trades.

A sample output for indicators might look like this:

| pair      | open_date                 | enter_reason | exit_reason | chikou_span (entry) | tenkan_sen (entry) | chikou_span (exit) | tenkan_sen (exit) |
|-----------|---------------------------|--------------|-------------|---------------------|--------------------|--------------------|-------------------|
| DOGE/USDT | 2024-07-06 00:35:00+00:00 |              | exit_signal | 0.105               | 0.106              | 0.105              | 0.107             |
| BTC/USDT  | 2024-08-05 14:20:00+00:00 |              | roi         | 54643.440           | 51696.400          | 54386.000          | 52072.010         |

As shown in the table, `chikou_span (entry)` represents the indicator value at the time of trade entry, 
while `chikou_span (exit)` reflects its value at the time of exit. 
This detailed view of indicator values enhances the analysis.

The `(entry)` and `(exit)` suffixes are added to indicators
to distinguish the values at the entry and exit points of the trade.

!!! Note "Trade-wide Indicators"
    Certain trade-wide indicators do not have the `(entry)` or `(exit)` suffix. These indicators include: `pair`, `stake_amount`, 
    `max_stake_amount`, `amount`, `open_date`, `close_date`, `open_rate`, `close_rate`, `fee_open`, `fee_close`, `trade_duration`, 
    `profit_ratio`, `profit_abs`, `exit_reason`,`initial_stop_loss_abs`, `initial_stop_loss_ratio`, `stop_loss_abs`, `stop_loss_ratio`, 
    `min_rate`, `max_rate`, `is_open`, `enter_tag`, `leverage`, `is_short`, `open_timestamp`, `close_timestamp` and `orders`

#### Filtering Indicators Based on Entry or Exit Signals

The `--indicator-list` option, by default, displays indicator values for both entry and exit signals. To filter the indicator values exclusively for entry signals, you can use the `--entry-only` argument. Similarly, to display indicator values only at exit signals, use the `--exit-only` argument.

Example: Display indicator values at entry signals:

```bash
freqtrade backtesting-analysis -c user_data/config.json --analysis-groups 0 --indicator-list chikou_span tenkan_sen --entry-only
```

Example: Display indicator values at exit signals:

```bash
freqtrade backtesting-analysis -c user_data/config.json --analysis-groups 0 --indicator-list chikou_span tenkan_sen --exit-only
```

!!! note 
    When using these filters, the indicator names will not be suffixed with `(entry)` or `(exit)`.

### Filtering the trade output by date

To show only trades between dates within your backtested timerange, supply the usual `timerange` option in `YYYYMMDD-[YYYYMMDD]` format:

```
--timerange : Timerange to filter output trades, start date inclusive, end date exclusive. e.g. 20220101-20221231
```

For example, if your backtest timerange was `20220101-20221231` but you only want to output trades in January:

```bash
freqtrade backtesting-analysis -c <config.json> --timerange 20220101-20220201
```

### Printing out rejected signals

Use the `--rejected-signals` option to print out rejected signals.

```bash
freqtrade backtesting-analysis -c <config.json> --rejected-signals
```

### Writing tables to CSV

Some of the tabular outputs can become large, so printing them out to the terminal is not preferable.
Use the `--analysis-to-csv` option to disable printing out of tables to standard out and write them to CSV files.

```bash
freqtrade backtesting-analysis -c <config.json> --analysis-to-csv
```

By default this will write one file per output table you specified in the `backtesting-analysis` command, e.g.

```bash
freqtrade backtesting-analysis -c <config.json> --analysis-to-csv --rejected-signals --analysis-groups 0 1
```

This will write to `user_data/backtest_results`:

* rejected_signals.csv
* group_0.csv
* group_1.csv

To override where the files will be written, also specify the `--analysis-csv-path` option.

```bash
freqtrade backtesting-analysis -c <config.json> --analysis-to-csv --analysis-csv-path another/data/path/
```
