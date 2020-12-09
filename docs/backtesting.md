# Backtesting

This page explains how to validate your strategy performance by using Backtesting.

Backtesting requires historic data to be available.
To learn how to get data for the pairs and exchange you're interested in, head over to the [Data Downloading](data-download.md) section of the documentation.

## Test your strategy with Backtesting

Now you have good Buy and Sell strategies and some historic data, you want to test it against
real data. This is what we call
[backtesting](https://en.wikipedia.org/wiki/Backtesting).

Backtesting will use the crypto-currencies (pairs) from your config file and load historical candle (OHCLV) data from `user_data/data/<exchange>` by default.
If no data is available for the exchange / pair / timeframe combination, backtesting will ask you to download them first using `freqtrade download-data`.
For details on downloading, please refer to the [Data Downloading](data-download.md) section in the documentation.

The result of backtesting will confirm if your bot has better odds of making a profit than a loss.

!!! Warning "Using dynamic pairlists for backtesting"
    Using dynamic pairlists is possible, however it relies on the current market conditions - which will not reflect the historic status of the pairlist.
    Also, when using pairlists other than StaticPairlist, reproducability of backtesting-results cannot be guaranteed.
    Please read the [pairlists documentation](configuration.md#pairlists) for more information.

    To achieve reproducible results, best generate a pairlist via the [`test-pairlist`](utils.md#test-pairlist) command and use that as static pairlist.

### Run a backtesting against the currencies listed in your config file

#### With 5 min candle (OHLCV) data (per default)

```bash
freqtrade backtesting
```

#### With 1 min candle (OHLCV) data

```bash
freqtrade backtesting --timeframe 1m
```

#### Using a different on-disk historical candle (OHLCV) data source

Assume you downloaded the history data from the Bittrex exchange and kept it in the `user_data/data/bittrex-20180101` directory. 
You can then use this data for backtesting as follows:

```bash
freqtrade --datadir user_data/data/bittrex-20180101 backtesting
```

#### With a (custom) strategy file

```bash
freqtrade backtesting -s SampleStrategy
```

Where `-s SampleStrategy` refers to the class name within the strategy file `sample_strategy.py` found in the `freqtrade/user_data/strategies` directory.

#### Comparing multiple Strategies

```bash
freqtrade backtesting --strategy-list SampleStrategy1 AwesomeStrategy --timeframe 5m
```

Where `SampleStrategy1` and `AwesomeStrategy` refer to class names of strategies.

#### Exporting trades to file

```bash
freqtrade backtesting --export trades --config config.json --strategy SampleStrategy
```

The exported trades can be used for [further analysis](#further-backtest-result-analysis), or can be used by the plotting script `plot_dataframe.py` in the scripts directory.

#### Exporting trades to file specifying a custom filename

```bash
freqtrade backtesting --export trades --export-filename=backtest_samplestrategy.json
```

Please also read about the [strategy startup period](strategy-customization.md#strategy-startup-period).

#### Supplying custom fee value

Sometimes your account has certain fee rebates (fee reductions starting with a certain account size or monthly volume), which are not visible to ccxt.
To account for this in backtesting, you can use the `--fee` command line option to supply this value to backtesting.
This fee must be a ratio, and will be applied twice (once for trade entry, and once for trade exit).

For example, if the buying and selling commission fee is 0.1% (i.e., 0.001 written as ratio), then you would run backtesting as the following:

```bash
freqtrade backtesting --fee 0.001
```

!!! Note
    Only supply this option (or the corresponding configuration parameter) if you want to experiment with different fee values. By default, Backtesting fetches the default fee from the exchange pair/market info.

#### Running backtest with smaller testset by using timerange

Use the `--timerange` argument to change how much of the testset you want to use.


For example, running backtesting with the `--timerange=20190501-` option will use all available data starting with May 1st, 2019 from your inputdata.

```bash
freqtrade backtesting --timerange=20190501-
```

You can also specify particular dates or a range span indexed by start and stop.

The full timerange specification:

- Use tickframes till 2018/01/31: `--timerange=-20180131`
- Use tickframes since 2018/01/31: `--timerange=20180131-`
- Use tickframes since 2018/01/31 till 2018/03/01 : `--timerange=20180131-20180301`
- Use tickframes between POSIX timestamps 1527595200 1527618600:
                                                `--timerange=1527595200-1527618600`

## Understand the backtesting result

The most important in the backtesting is to understand the result.

A backtesting result will look like that:

```
========================================================= BACKTESTING REPORT ========================================================
| Pair     |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit BTC |   Tot Profit % | Avg Duration   |  Wins |  Draws |  Losses |
|:---------|-------:|---------------:|---------------:|-----------------:|---------------:|:---------------|------:|-------:|--------:|
| ADA/BTC  |     35 |          -0.11 |          -3.88 |      -0.00019428 |          -1.94 | 4:35:00        |    14 |      0 |      21 |
| ARK/BTC  |     11 |          -0.41 |          -4.52 |      -0.00022647 |          -2.26 | 2:03:00        |     3 |      0 |       8 |
| BTS/BTC  |     32 |           0.31 |           9.78 |       0.00048938 |           4.89 | 5:05:00        |    18 |      0 |      14 |
| DASH/BTC |     13 |          -0.08 |          -1.07 |      -0.00005343 |          -0.53 | 4:39:00        |     6 |      0 |       7 |
| ENG/BTC  |     18 |           1.36 |          24.54 |       0.00122807 |          12.27 | 2:50:00        |     8 |      0 |      10 |
| EOS/BTC  |     36 |           0.08 |           3.06 |       0.00015304 |           1.53 | 3:34:00        |    16 |      0 |      20 |
| ETC/BTC  |     26 |           0.37 |           9.51 |       0.00047576 |           4.75 | 6:14:00        |    11 |      0 |      15 |
| ETH/BTC  |     33 |           0.30 |           9.96 |       0.00049856 |           4.98 | 7:31:00        |    16 |      0 |      17 |
| IOTA/BTC |     32 |           0.03 |           1.09 |       0.00005444 |           0.54 | 3:12:00        |    14 |      0 |      18 |
| LSK/BTC  |     15 |           1.75 |          26.26 |       0.00131413 |          13.13 | 2:58:00        |     6 |      0 |       9 |
| LTC/BTC  |     32 |          -0.04 |          -1.38 |      -0.00006886 |          -0.69 | 4:49:00        |    11 |      0 |      21 |
| NANO/BTC |     17 |           1.26 |          21.39 |       0.00107058 |          10.70 | 1:55:00        |    10 |      0 |       7 |
| NEO/BTC  |     23 |           0.82 |          18.97 |       0.00094936 |           9.48 | 2:59:00        |    10 |      0 |      13 |
| REQ/BTC  |      9 |           1.17 |          10.54 |       0.00052734 |           5.27 | 3:47:00        |     4 |      0 |       5 |
| XLM/BTC  |     16 |           1.22 |          19.54 |       0.00097800 |           9.77 | 3:15:00        |     7 |      0 |       9 |
| XMR/BTC  |     23 |          -0.18 |          -4.13 |      -0.00020696 |          -2.07 | 5:30:00        |    12 |      0 |      11 |
| XRP/BTC  |     35 |           0.66 |          22.96 |       0.00114897 |          11.48 | 3:49:00        |    12 |      0 |      23 |
| ZEC/BTC  |     22 |          -0.46 |         -10.18 |      -0.00050971 |          -5.09 | 2:22:00        |     7 |      0 |      15 |
| TOTAL    |    429 |           0.36 |         152.41 |       0.00762792 |          76.20 | 4:12:00        |   186 |      0 |     243 |
========================================================= SELL REASON STATS =========================================================
| Sell Reason        |   Sells |  Wins |  Draws |  Losses |
|:-------------------|--------:|------:|-------:|--------:|
| trailing_stop_loss |     205 |   150 |      0 |      55 |
| stop_loss          |     166 |     0 |      0 |     166 |
| sell_signal        |      56 |    36 |      0 |      20 |
| force_sell         |       2 |     0 |      0 |       2 |
====================================================== LEFT OPEN TRADES REPORT ======================================================
| Pair     |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit BTC |   Tot Profit % | Avg Duration   |  Wins |  Draws |  Losses |
|:---------|-------:|---------------:|---------------:|-----------------:|---------------:|:---------------|------:|-------:|--------:|
| ADA/BTC  |      1 |           0.89 |           0.89 |       0.00004434 |           0.44 | 6:00:00        |     1 |      0 |       0 |
| LTC/BTC  |      1 |           0.68 |           0.68 |       0.00003421 |           0.34 | 2:00:00        |     1 |      0 |       0 |
| TOTAL    |      2 |           0.78 |           1.57 |       0.00007855 |           0.78 | 4:00:00        |     2 |      0 |       0 |
=============== SUMMARY METRICS ===============
| Metric                | Value               |
|-----------------------+---------------------|
| Backtesting from      | 2019-01-01 00:00:00 |
| Backtesting to        | 2019-05-01 00:00:00 |
| Max open trades       | 3                   |
|                       |                     |
| Total trades          | 429                 |
| Total Profit %        | 152.41%             |
| Trades per day        | 3.575               |
|                       |                     |
| Best Pair             | LSK/BTC 26.26%      |
| Worst Pair            | ZEC/BTC -10.18%     |
| Best Trade            | LSK/BTC 4.25%       |
| Worst Trade           | ZEC/BTC -10.25%     |
| Best day              | 25.27%              |
| Worst day             | -30.67%             |
| Avg. Duration Winners | 4:23:00             |
| Avg. Duration Loser   | 6:55:00             |
|                       |                     |
| Max Drawdown          | 50.63%              |
| Drawdown Start        | 2019-02-15 14:10:00 |
| Drawdown End          | 2019-04-11 18:15:00 |
| Market change         | -5.88%              |
===============================================
```

### Backtesting report table

The 1st table contains all trades the bot made, including "left open trades".

The last line will give you the overall performance of your strategy,
here:

```
| TOTAL    |         429 |           0.36 |         152.41 |       0.00762792 |          76.20 | 4:12:00        |      186 |    243 |
```

The bot has made `429` trades for an average duration of `4:12:00`, with a performance of `76.20%` (profit), that means it has
earned a total of `0.00762792 BTC` starting with a capital of 0.01 BTC.

The column `avg profit %` shows the average profit for all trades made while the column `cum profit %` sums up all the profits/losses.
The column `tot profit %` shows instead the total profit % in relation to allocated capital (`max_open_trades * stake_amount`).
In the above results we have `max_open_trades=2` and `stake_amount=0.005` in config  so `tot_profit %` will be `(76.20/100) * (0.005 * 2) =~ 0.00762792 BTC`.

Your strategy performance is influenced by your buy strategy, your sell strategy, and also by the `minimal_roi` and `stop_loss` you have set.

For example, if your `minimal_roi` is only `"0":  0.01` you cannot expect the bot to make more profit than 1% (because it will sell every time a trade reaches 1%).

```json
"minimal_roi": {
    "0":  0.01
},
```

On the other hand, if you set a too high `minimal_roi` like `"0":  0.55`
(55%), there is almost no chance that the bot will ever reach this profit.
Hence, keep in mind that your performance is an integral mix of all different elements of the strategy, your configuration, and the crypto-currency pairs you have set up.

### Sell reasons table

The 2nd table contains a recap of sell reasons.
This table can tell you which area needs some additional work (e.g. all or many of the `sell_signal` trades are losses, so you should work on improving the sell signal, or consider disabling it).

### Left open trades table

The 3rd table contains all trades the bot had to `forcesell` at the end of the backtesting period to present you the full picture.
This is necessary to simulate realistic behavior, since the backtest period has to end at some point, while realistically, you could leave the bot running forever.
These trades are also included in the first table, but are also shown separately in this table for clarity.

### Summary metrics

The last element of the backtest report is the summary metrics table.
It contains some useful key metrics about performance of your strategy on backtesting data.

```
=============== SUMMARY METRICS ===============
| Metric                | Value               |
|-----------------------+---------------------|
| Backtesting from      | 2019-01-01 00:00:00 |
| Backtesting to        | 2019-05-01 00:00:00 |
| Max open trades       | 3                   |
|                       |                     |
| Total trades          | 429                 |
| Total Profit %        | 152.41%             |
| Trades per day        | 3.575               |
|                       |                     |
| Best Pair             | LSK/BTC 26.26%      |
| Worst Pair            | ZEC/BTC -10.18%     |
| Best Trade            | LSK/BTC 4.25%       |
| Worst Trade           | ZEC/BTC -10.25%     |
| Best day              | 25.27%              |
| Worst day             | -30.67%             |
| Avg. Duration Winners | 4:23:00             |
| Avg. Duration Loser   | 6:55:00             |
|                       |                     |
| Max Drawdown          | 50.63%              |
| Drawdown Start        | 2019-02-15 14:10:00 |
| Drawdown End          | 2019-04-11 18:15:00 |
| Market change         | -5.88%              |
===============================================

```

- `Backtesting from` / `Backtesting to`: Backtesting range (usually defined with the `--timerange` option).
- `Max open trades`: Setting of `max_open_trades` (or `--max-open-trades`) - to clearly see settings for this.
- `Total trades`: Identical to the total trades of the backtest output table.
- `Total Profit %`: Total profit per stake amount. Aligned to the TOTAL column of the first table.
- `Trades per day`: Total trades divided by the backtesting duration in days (this will give you information about how many trades to expect from the strategy).
- `Best Pair` / `Worst Pair`: Best and worst performing pair, and it's corresponding `Cum Profit %`.
- `Best Trade` / `Worst Trade`: Biggest winning trade and biggest losing trade
- `Best day` / `Worst day`: Best and worst day based on daily profit.
- `Avg. Duration Winners` / `Avg. Duration Loser`: Average durations for winning and losing trades.
- `Max Drawdown`: Maximum drawdown experienced. For example, the value of 50% means that from highest to subsequent lowest point, a 50% drop was experienced).
- `Drawdown Start` / `Drawdown End`: Start and end datetime for this largest drawdown (can also be visualized via the `plot-dataframe` sub-command).
- `Market change`: Change of the market during the backtest period. Calculated as average of all pairs changes from the first to the last candle using the "close" column.

### Assumptions made by backtesting

Since backtesting lacks some detailed information about what happens within a candle, it needs to take a few assumptions:

- Buys happen at open-price
- Sell-signal sells happen at open-price of the consecutive candle
- Sell-signal is favored over Stoploss, because sell-signals are assumed to trigger on candle's open
- ROI
  - sells are compared to high - but the ROI value is used (e.g. ROI = 2%, high=5% - so the sell will be at 2%)
  - sells are never "below the candle", so a ROI of 2% may result in a sell at 2.4% if low was at 2.4% profit
  - Forcesells caused by `<N>=-1` ROI entries use low as sell value, unless N falls on the candle open (e.g. `120: -1` for 1h candles)
- Stoploss sells happen exactly at stoploss price, even if low was lower, but the loss will be 0.32% lower than the stoploss price
- Stoploss is evaluated before ROI within one candle. So you can often see more trades with the `stoploss` sell reason comparing to the results obtained with the same strategy in the Dry Run/Live Trade modes
- Low happens before high for stoploss, protecting capital first
- Trailing stoploss
  - High happens first - adjusting stoploss
  - Low uses the adjusted stoploss (so sells with large high-low difference are backtested correctly)
  - ROI applies before trailing-stop, ensuring profits are "top-capped" at ROI if both ROI and trailing stop applies
- Sell-reason does not explain if a trade was positive or negative, just what triggered the sell (this can look odd if negative ROI values are used)
- Evaluation sequence (if multiple signals happen on the same candle)
  - ROI (if not stoploss)
  - Sell-signal
  - Stoploss

Taking these assumptions, backtesting tries to mirror real trading as closely as possible. However, backtesting will **never** replace running a strategy in dry-run mode.
Also, keep in mind that past results don't guarantee future success.

In addition to the above assumptions, strategy authors should carefully read the [Common Mistakes](strategy-customization.md#common-mistakes-when-developing-strategies) section, to avoid using data in backtesting which is not available in real market conditions.

### Further backtest-result analysis

To further analyze your backtest results, you can [export the trades](#exporting-trades-to-file).
You can then load the trades to perform further analysis as shown in our [data analysis](data-analysis.md#backtesting) backtesting section.

## Backtesting multiple strategies

To compare multiple strategies, a list of Strategies can be provided to backtesting.

This is limited to 1 timeframe value per run. However, data is only loaded once from disk so if you have multiple
strategies you'd like to compare, this will give a nice runtime boost.

All listed Strategies need to be in the same directory.

``` bash
freqtrade backtesting --timerange 20180401-20180410 --timeframe 5m --strategy-list Strategy001 Strategy002 --export trades
```

This will save the results to `user_data/backtest_results/backtest-result-<strategy>.json`, injecting the strategy-name into the target filename.
There will be an additional table comparing win/losses of the different strategies (identical to the "Total" row in the first table).
Detailed output for all strategies one after the other will be available, so make sure to scroll up to see the details per strategy.

```
=========================================================== STRATEGY SUMMARY ===========================================================
| Strategy    |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit BTC |   Tot Profit % | Avg Duration   |  Wins |  Draws | Losses |
|:------------|-------:|---------------:|---------------:|-----------------:|---------------:|:---------------|------:|-------:|-------:|
| Strategy1   |    429 |           0.36 |         152.41 |       0.00762792 |          76.20 | 4:12:00        |   186 |      0 |    243 |
| Strategy2   |   1487 |          -0.13 |        -197.58 |      -0.00988917 |         -98.79 | 4:43:00        |   662 |      0 |    825 |
```

## Next step

Great, your strategy is profitable. What if the bot can give your the
optimal parameters to use for your strategy?
Your next step is to learn [how to find optimal parameters with Hyperopt](hyperopt.md)
