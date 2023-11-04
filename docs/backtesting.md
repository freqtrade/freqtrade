# Backtesting

This page explains how to validate your strategy performance by using Backtesting.

Backtesting requires historic data to be available.
To learn how to get data for the pairs and exchange you're interested in, head over to the [Data Downloading](data-download.md) section of the documentation.

## Backtesting command reference

```
usage: freqtrade backtesting [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                             [-d PATH] [--userdir PATH] [-s NAME]
                             [--strategy-path PATH] [-i TIMEFRAME]
                             [--timerange TIMERANGE]
                             [--data-format-ohlcv {json,jsongz,hdf5}]
                             [--max-open-trades INT]
                             [--stake-amount STAKE_AMOUNT] [--fee FLOAT]
                             [-p PAIRS [PAIRS ...]] [--eps] [--dmmp]
                             [--enable-protections]
                             [--dry-run-wallet DRY_RUN_WALLET]
                             [--timeframe-detail TIMEFRAME_DETAIL]
                             [--strategy-list STRATEGY_LIST [STRATEGY_LIST ...]]
                             [--export {none,trades,signals}]
                             [--export-filename PATH]
                             [--breakdown {day,week,month} [{day,week,month} ...]]
                             [--cache {none,day,week,month}]

optional arguments:
  -h, --help            show this help message and exit
  -i TIMEFRAME, --timeframe TIMEFRAME
                        Specify timeframe (`1m`, `5m`, `30m`, `1h`, `1d`).
  --timerange TIMERANGE
                        Specify what timerange of data to use.
  --data-format-ohlcv {json,jsongz,hdf5,feather,parquet}
                        Storage format for downloaded candle (OHLCV) data.
                        (default: `feather`).
  --max-open-trades INT
                        Override the value of the `max_open_trades`
                        configuration setting.
  --stake-amount STAKE_AMOUNT
                        Override the value of the `stake_amount` configuration
                        setting.
  --fee FLOAT           Specify fee ratio. Will be applied twice (on trade
                        entry and exit).
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  --eps, --enable-position-stacking
                        Allow buying the same pair multiple times (position
                        stacking).
  --dmmp, --disable-max-market-positions
                        Disable applying `max_open_trades` during backtest
                        (same as setting `max_open_trades` to a very high
                        number).
  --enable-protections, --enableprotections
                        Enable protections for backtesting.Will slow
                        backtesting down by a considerable amount, but will
                        include configured protections
  --dry-run-wallet DRY_RUN_WALLET, --starting-balance DRY_RUN_WALLET
                        Starting balance, used for backtesting / hyperopt and
                        dry-runs.
  --timeframe-detail TIMEFRAME_DETAIL
                        Specify detail timeframe for backtesting (`1m`, `5m`,
                        `30m`, `1h`, `1d`).
  --strategy-list STRATEGY_LIST [STRATEGY_LIST ...]
                        Provide a space-separated list of strategies to
                        backtest. Please note that timeframe needs to be set
                        either in config or via command line. When using this
                        together with `--export trades`, the strategy-name is
                        injected into the filename (so `backtest-data.json`
                        becomes `backtest-data-SampleStrategy.json`
  --export {none,trades,signals}
                        Export backtest results (default: trades).
  --export-filename PATH, --backtest-filename PATH
                        Use this filename for backtest results.Requires
                        `--export` to be set as well. Example: `--export-filen
                        ame=user_data/backtest_results/backtest_today.json`
  --breakdown {day,week,month} [{day,week,month} ...]
                        Show backtesting breakdown per [day, week, month].
  --cache {none,day,week,month}
                        Load a cached backtest result no older than specified
                        age (default: day).

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --logfile FILE        Log to the file specified. Special values are:
                        'syslog', 'journald'. See the documentation for more
                        details.
  -V, --version         show program's version number and exit
  -c PATH, --config PATH
                        Specify configuration file (default:
                        `userdir/config.json` or `config.json` whichever
                        exists). Multiple --config options may be used. Can be
                        set to `-` to read config from stdin.
  -d PATH, --datadir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.

Strategy arguments:
  -s NAME, --strategy NAME
                        Specify strategy class name which will be used by the
                        bot.
  --strategy-path PATH  Specify additional strategy lookup path.

```

## Test your strategy with Backtesting

Now you have good Entry and exit strategies and some historic data, you want to test it against
real data. This is what we call [backtesting](https://en.wikipedia.org/wiki/Backtesting).

Backtesting will use the crypto-currencies (pairs) from your config file and load historical candle (OHLCV) data from `user_data/data/<exchange>` by default.
If no data is available for the exchange / pair / timeframe combination, backtesting will ask you to download them first using `freqtrade download-data`.
For details on downloading, please refer to the [Data Downloading](data-download.md) section in the documentation.

The result of backtesting will confirm if your bot has better odds of making a profit than a loss.

All profit calculations include fees, and freqtrade will use the exchange's default fees for the calculation.

!!! Warning "Using dynamic pairlists for backtesting"
    Using dynamic pairlists is possible (not all of the handlers are allowed to be used in backtest mode), however it relies on the current market conditions - which will not reflect the historic status of the pairlist.
    Also, when using pairlists other than StaticPairlist, reproducibility of backtesting-results cannot be guaranteed.
    Please read the [pairlists documentation](plugins.md#pairlists) for more information.

    To achieve reproducible results, best generate a pairlist via the [`test-pairlist`](utils.md#test-pairlist) command and use that as static pairlist.

!!! Note
    By default, Freqtrade will export backtesting results to `user_data/backtest_results`.
    The exported trades can be used for [further analysis](#further-backtest-result-analysis) or can be used by the [plotting sub-command](plotting.md#plot-price-and-indicators) (`freqtrade plot-dataframe`) in the scripts directory.


### Starting balance

Backtesting will require a starting balance, which can be provided as `--dry-run-wallet <balance>` or `--starting-balance <balance>` command line argument, or via `dry_run_wallet` configuration setting.
This amount must be higher than `stake_amount`, otherwise the bot will not be able to simulate any trade.

### Dynamic stake amount

Backtesting supports [dynamic stake amount](configuration.md#dynamic-stake-amount) by configuring `stake_amount` as `"unlimited"`, which will split the starting balance into `max_open_trades` pieces.
Profits from early trades will result in subsequent higher stake amounts, resulting in compounding of profits over the backtesting period.

### Example backtesting commands

With 5 min candle (OHLCV) data (per default)

```bash
freqtrade backtesting --strategy AwesomeStrategy
```

Where `--strategy AwesomeStrategy` / `-s AwesomeStrategy` refers to the class name of the strategy, which is within a python file in the `user_data/strategies` directory.

---

With 1 min candle (OHLCV) data

```bash
freqtrade backtesting --strategy AwesomeStrategy --timeframe 1m
```

---

Providing a custom starting balance of 1000 (in stake currency)

```bash
freqtrade backtesting --strategy AwesomeStrategy --dry-run-wallet 1000
```

---

Using a different on-disk historical candle (OHLCV) data source

Assume you downloaded the history data from the Bittrex exchange and kept it in the `user_data/data/bittrex-20180101` directory. 
You can then use this data for backtesting as follows:

```bash
freqtrade backtesting --strategy AwesomeStrategy --datadir user_data/data/bittrex-20180101 
```

---

Comparing multiple Strategies

```bash
freqtrade backtesting --strategy-list SampleStrategy1 AwesomeStrategy --timeframe 5m
```

Where `SampleStrategy1` and `AwesomeStrategy` refer to class names of strategies.

---

Prevent exporting trades to file

```bash
freqtrade backtesting --strategy backtesting --export none --config config.json 
```

Only use this if you're sure you'll not want to plot or analyze your results further.

---

Exporting trades to file specifying a custom filename

```bash
freqtrade backtesting --strategy backtesting --export trades --export-filename=backtest_samplestrategy.json
```

Please also read about the [strategy startup period](strategy-customization.md#strategy-startup-period).

---

Supplying custom fee value

Sometimes your account has certain fee rebates (fee reductions starting with a certain account size or monthly volume), which are not visible to ccxt.
To account for this in backtesting, you can use the `--fee` command line option to supply this value to backtesting.
This fee must be a ratio, and will be applied twice (once for trade entry, and once for trade exit).

For example, if the commission fee per order is 0.1% (i.e., 0.001 written as ratio), then you would run backtesting as the following:

```bash
freqtrade backtesting --fee 0.001
```

!!! Note
    Only supply this option (or the corresponding configuration parameter) if you want to experiment with different fee values. By default, Backtesting fetches the default fee from the exchange pair/market info.

---

Running backtest with smaller test-set by using timerange

Use the `--timerange` argument to change how much of the test-set you want to use.

For example, running backtesting with the `--timerange=20190501-` option will use all available data starting with May 1st, 2019 from your input data.

```bash
freqtrade backtesting --timerange=20190501-
```

You can also specify particular date ranges.

The full timerange specification:

- Use data until 2018/01/31: `--timerange=-20180131`
- Use data since 2018/01/31: `--timerange=20180131-`
- Use data since 2018/01/31 till 2018/03/01 : `--timerange=20180131-20180301`
- Use data between POSIX / epoch timestamps 1527595200 1527618600: `--timerange=1527595200-1527618600`

## Understand the backtesting result

The most important in the backtesting is to understand the result.

A backtesting result will look like that:

```
========================================================= BACKTESTING REPORT =========================================================
| Pair     | Entries |   Avg Profit % |   Cum Profit % |   Tot Profit BTC |   Tot Profit % | Avg Duration |  Wins Draws Loss   Win%  |
|:---------|--------:|---------------:|---------------:|-----------------:|---------------:|:-------------|-------------------------:|
| ADA/BTC  |      35 |          -0.11 |          -3.88 |      -0.00019428 |          -1.94 | 4:35:00      |    14     0    21   40.0 |
| ARK/BTC  |      11 |          -0.41 |          -4.52 |      -0.00022647 |          -2.26 | 2:03:00      |     3     0     8   27.3 |
| BTS/BTC  |      32 |           0.31 |           9.78 |       0.00048938 |           4.89 | 5:05:00      |    18     0    14   56.2 |
| DASH/BTC |      13 |          -0.08 |          -1.07 |      -0.00005343 |          -0.53 | 4:39:00      |     6     0     7   46.2 |
| ENG/BTC  |      18 |           1.36 |          24.54 |       0.00122807 |          12.27 | 2:50:00      |     8     0    10   44.4 |
| EOS/BTC  |      36 |           0.08 |           3.06 |       0.00015304 |           1.53 | 3:34:00      |    16     0    20   44.4 |
| ETC/BTC  |      26 |           0.37 |           9.51 |       0.00047576 |           4.75 | 6:14:00      |    11     0    15   42.3 |
| ETH/BTC  |      33 |           0.30 |           9.96 |       0.00049856 |           4.98 | 7:31:00      |    16     0    17   48.5 |
| IOTA/BTC |      32 |           0.03 |           1.09 |       0.00005444 |           0.54 | 3:12:00      |    14     0    18   43.8 |
| LSK/BTC  |      15 |           1.75 |          26.26 |       0.00131413 |          13.13 | 2:58:00      |     6     0     9   40.0 |
| LTC/BTC  |      32 |          -0.04 |          -1.38 |      -0.00006886 |          -0.69 | 4:49:00      |    11     0    21   34.4 |
| NANO/BTC |      17 |           1.26 |          21.39 |       0.00107058 |          10.70 | 1:55:00      |    10     0     7   58.5 |
| NEO/BTC  |      23 |           0.82 |          18.97 |       0.00094936 |           9.48 | 2:59:00      |    10     0    13   43.5 |
| REQ/BTC  |       9 |           1.17 |          10.54 |       0.00052734 |           5.27 | 3:47:00      |     4     0     5   44.4 |
| XLM/BTC  |      16 |           1.22 |          19.54 |       0.00097800 |           9.77 | 3:15:00      |     7     0     9   43.8 |
| XMR/BTC  |      23 |          -0.18 |          -4.13 |      -0.00020696 |          -2.07 | 5:30:00      |    12     0    11   52.2 |
| XRP/BTC  |      35 |           0.66 |          22.96 |       0.00114897 |          11.48 | 3:49:00      |    12     0    23   34.3 |
| ZEC/BTC  |      22 |          -0.46 |         -10.18 |      -0.00050971 |          -5.09 | 2:22:00      |     7     0    15   31.8 |
| TOTAL    |     429 |           0.36 |         152.41 |       0.00762792 |          76.20 | 4:12:00      |   186     0   243   43.4 |
====================================================== LEFT OPEN TRADES REPORT ======================================================
| Pair     |  Entries |   Avg Profit % |   Cum Profit % |   Tot Profit BTC |   Tot Profit % | Avg Duration   |  Win Draw Loss Win% |
|:---------|---------:|---------------:|---------------:|-----------------:|---------------:|:---------------|--------------------:|
| ADA/BTC  |        1 |           0.89 |           0.89 |       0.00004434 |           0.44 | 6:00:00        |    1    0    0  100 |
| LTC/BTC  |        1 |           0.68 |           0.68 |       0.00003421 |           0.34 | 2:00:00        |    1    0    0  100 |
| TOTAL    |        2 |           0.78 |           1.57 |       0.00007855 |           0.78 | 4:00:00        |    2    0    0  100 |
==================== EXIT REASON STATS ====================
| Exit Reason        |   Exits |  Wins |  Draws |  Losses |
|:-------------------|--------:|------:|-------:|--------:|
| trailing_stop_loss |     205 |   150 |      0 |      55 |
| stop_loss          |     166 |     0 |      0 |     166 |
| exit_signal        |      56 |    36 |      0 |      20 |
| force_exit         |       2 |     0 |      0 |       2 |

================== SUMMARY METRICS ==================
| Metric                      | Value               |
|-----------------------------+---------------------|
| Backtesting from            | 2019-01-01 00:00:00 |
| Backtesting to              | 2019-05-01 00:00:00 |
| Max open trades             | 3                   |
|                             |                     |
| Total/Daily Avg Trades      | 429 / 3.575         |
| Starting balance            | 0.01000000 BTC      |
| Final balance               | 0.01762792 BTC      |
| Absolute profit             | 0.00762792 BTC      |
| Total profit %              | 76.2%               |
| CAGR %                      | 460.87%             |
| Sortino                     | 1.88                |
| Sharpe                      | 2.97                |
| Calmar                      | 6.29                |
| Profit factor               | 1.11                |
| Expectancy (Ratio)          | -0.15 (-0.05)       |
| Avg. stake amount           | 0.001      BTC      |
| Total trade volume          | 0.429      BTC      |
|                             |                     |
| Long / Short                | 352 / 77            |
| Total profit Long %         | 1250.58%            |
| Total profit Short %        | -15.02%             |
| Absolute profit Long        | 0.00838792 BTC      |
| Absolute profit Short       | -0.00076 BTC        |
|                             |                     |
| Best Pair                   | LSK/BTC 26.26%      |
| Worst Pair                  | ZEC/BTC -10.18%     |
| Best Trade                  | LSK/BTC 4.25%       |
| Worst Trade                 | ZEC/BTC -10.25%     |
| Best day                    | 0.00076 BTC         |
| Worst day                   | -0.00036 BTC        |
| Days win/draw/lose          | 12 / 82 / 25        |
| Avg. Duration Winners       | 4:23:00             |
| Avg. Duration Loser         | 6:55:00             |
| Max Consecutive Wins / Loss | 3 / 4               |
| Rejected Entry signals      | 3089                |
| Entry/Exit Timeouts         | 0 / 0               |
| Canceled Trade Entries      | 34                  |
| Canceled Entry Orders       | 123                 |
| Replaced Entry Orders       | 89                  |
|                             |                     |
| Min balance                 | 0.00945123 BTC      |
| Max balance                 | 0.01846651 BTC      |
| Max % of account underwater | 25.19%              |
| Absolute Drawdown (Account) | 13.33%              |
| Drawdown                    | 0.0015 BTC          |
| Drawdown high               | 0.0013 BTC          |
| Drawdown low                | -0.0002 BTC         |
| Drawdown Start              | 2019-02-15 14:10:00 |
| Drawdown End                | 2019-04-11 18:15:00 |
| Market change               | -5.88%              |
=====================================================
```

### Backtesting report table

The 1st table contains all trades the bot made, including "left open trades".

The last line will give you the overall performance of your strategy,
here:

```
| TOTAL    |    429 |           0.36 |         152.41 |       0.00762792 |          76.20 | 4:12:00      |   186     0   243   43.4 |
```

The bot has made `429` trades for an average duration of `4:12:00`, with a performance of `76.20%` (profit), that means it has
earned a total of `0.00762792 BTC` starting with a capital of 0.01 BTC.

The column `Avg Profit %` shows the average profit for all trades made while the column `Cum Profit %` sums up all the profits/losses.
The column `Tot Profit %` shows instead the total profit % in relation to the starting balance.
In the above results, we have a starting balance of 0.01 BTC and an absolute profit of 0.00762792 BTC - so the `Tot Profit %` will be `(0.00762792 / 0.01) * 100 ~= 76.2%`.

Your strategy performance is influenced by your entry strategy, your exit strategy, and also by the `minimal_roi` and `stop_loss` you have set.

For example, if your `minimal_roi` is only `"0":  0.01` you cannot expect the bot to make more profit than 1% (because it will exit every time a trade reaches 1%).

```json
"minimal_roi": {
    "0":  0.01
},
```

On the other hand, if you set a too high `minimal_roi` like `"0":  0.55`
(55%), there is almost no chance that the bot will ever reach this profit.
Hence, keep in mind that your performance is an integral mix of all different elements of the strategy, your configuration, and the crypto-currency pairs you have set up.

### Exit reasons table

The 2nd table contains a recap of exit reasons.
This table can tell you which area needs some additional work (e.g. all or many of the `exit_signal` trades are losses, so you should work on improving the exit signal, or consider disabling it).

### Left open trades table

The 3rd table contains all trades the bot had to `force_exit` at the end of the backtesting period to present you the full picture.
This is necessary to simulate realistic behavior, since the backtest period has to end at some point, while realistically, you could leave the bot running forever.
These trades are also included in the first table, but are also shown separately in this table for clarity.

### Summary metrics

The last element of the backtest report is the summary metrics table.
It contains some useful key metrics about performance of your strategy on backtesting data.

```
================== SUMMARY METRICS ==================
| Metric                      | Value               |
|-----------------------------+---------------------|
| Backtesting from            | 2019-01-01 00:00:00 |
| Backtesting to              | 2019-05-01 00:00:00 |
| Max open trades             | 3                   |
|                             |                     |
| Total/Daily Avg Trades      | 429 / 3.575         |
| Starting balance            | 0.01000000 BTC      |
| Final balance               | 0.01762792 BTC      |
| Absolute profit             | 0.00762792 BTC      |
| Total profit %              | 76.2%               |
| CAGR %                      | 460.87%             |
| Sortino                     | 1.88                |
| Sharpe                      | 2.97                |
| Calmar                      | 6.29                |
| Profit factor               | 1.11                |
| Expectancy (Ratio)          | -0.15 (-0.05)       |
| Avg. stake amount           | 0.001      BTC      |
| Total trade volume          | 0.429      BTC      |
|                             |                     |
| Long / Short                | 352 / 77            |
| Total profit Long %         | 1250.58%            |
| Total profit Short %        | -15.02%             |
| Absolute profit Long        | 0.00838792 BTC      |
| Absolute profit Short       | -0.00076 BTC        |
|                             |                     |
| Best Pair                   | LSK/BTC 26.26%      |
| Worst Pair                  | ZEC/BTC -10.18%     |
| Best Trade                  | LSK/BTC 4.25%       |
| Worst Trade                 | ZEC/BTC -10.25%     |
| Best day                    | 0.00076 BTC         |
| Worst day                   | -0.00036 BTC        |
| Days win/draw/lose          | 12 / 82 / 25        |
| Avg. Duration Winners       | 4:23:00             |
| Avg. Duration Loser         | 6:55:00             |
| Max Consecutive Wins / Loss | 3 / 4               |
| Rejected Entry signals      | 3089                |
| Entry/Exit Timeouts         | 0 / 0               |
| Canceled Trade Entries      | 34                  |
| Canceled Entry Orders       | 123                 |
| Replaced Entry Orders       | 89                  |
|                             |                     |
| Min balance                 | 0.00945123 BTC      |
| Max balance                 | 0.01846651 BTC      |
| Max % of account underwater | 25.19%              |
| Absolute Drawdown (Account) | 13.33%              |
| Drawdown                    | 0.0015 BTC          |
| Drawdown high               | 0.0013 BTC          |
| Drawdown low                | -0.0002 BTC         |
| Drawdown Start              | 2019-02-15 14:10:00 |
| Drawdown End                | 2019-04-11 18:15:00 |
| Market change               | -5.88%              |
=====================================================

```

- `Backtesting from` / `Backtesting to`: Backtesting range (usually defined with the `--timerange` option).
- `Max open trades`: Setting of `max_open_trades` (or `--max-open-trades`) - or number of pairs in the pairlist (whatever is lower).
- `Total/Daily Avg Trades`: Identical to the total trades of the backtest output table / Total trades divided by the backtesting duration in days (this will give you information about how many trades to expect from the strategy).
- `Starting balance`: Start balance - as given by dry-run-wallet (config or command line).
- `Final balance`: Final balance - starting balance + absolute profit.
- `Absolute profit`: Profit made in stake currency.
- `Total profit %`: Total profit. Aligned to the `TOTAL` row's `Tot Profit %` from the first table. Calculated as `(End capital − Starting capital) / Starting capital`.
- `CAGR %`: Compound annual growth rate.
- `Sortino`: Annualized Sortino ratio.
- `Sharpe`: Annualized Sharpe ratio.
- `Calmar`: Annualized Calmar ratio.
- `Profit factor`: profit / loss.
- `Avg. stake amount`: Average stake amount, either `stake_amount` or the average when using dynamic stake amount.
- `Total trade volume`: Volume generated on the exchange to reach the above profit.
- `Best Pair` / `Worst Pair`: Best and worst performing pair, and it's corresponding `Cum Profit %`.
- `Best Trade` / `Worst Trade`: Biggest single winning trade and biggest single losing trade.
- `Best day` / `Worst day`: Best and worst day based on daily profit.
- `Days win/draw/lose`: Winning / Losing days (draws are usually days without closed trade).
- `Avg. Duration Winners` / `Avg. Duration Loser`: Average durations for winning and losing trades.
- `Max Consecutive Wins / Loss`: Maximum consecutive wins/losses in a row.
- `Rejected Entry signals`: Trade entry signals that could not be acted upon due to `max_open_trades` being reached.
- `Entry/Exit Timeouts`: Entry/exit orders which did not fill (only applicable if custom pricing is used).
- `Canceled Trade Entries`: Number of trades that have been canceled by user request via `adjust_entry_price`.
- `Canceled Entry Orders`: Number of entry orders that have been canceled by user request via `adjust_entry_price`.
- `Replaced Entry Orders`: Number of entry orders that have been replaced by user request via `adjust_entry_price`.
- `Min balance` / `Max balance`: Lowest and Highest Wallet balance during the backtest period.
- `Max % of account underwater`: Maximum percentage your account has decreased from the top since the simulation started.
Calculated as the maximum of `(Max Balance - Current Balance) / (Max Balance)`.
- `Absolute Drawdown (Account)`: Maximum Account Drawdown experienced. Calculated as `(Absolute Drawdown) / (DrawdownHigh + startingBalance)`.
- `Drawdown`: Maximum, absolute drawdown experienced. Difference between Drawdown High and Subsequent Low point.
- `Drawdown high` / `Drawdown low`: Profit at the beginning and end of the largest drawdown period. A negative low value means initial capital lost.
- `Drawdown Start` / `Drawdown End`: Start and end datetime for this largest drawdown (can also be visualized via the `plot-dataframe` sub-command).
- `Market change`: Change of the market during the backtest period. Calculated as average of all pairs changes from the first to the last candle using the "close" column.
- `Long / Short`: Split long/short values (Only shown when short trades were made).
- `Total profit Long %` / `Absolute profit Long`: Profit long trades only (Only shown when short trades were made).
- `Total profit Short %` / `Absolute profit Short`: Profit short trades only (Only shown when short trades were made).

### Daily / Weekly / Monthly breakdown

You can get an overview over daily / weekly or monthly results by using the `--breakdown <>` switch.

To visualize daily and weekly breakdowns, you can use the following:

``` bash
freqtrade backtesting --strategy MyAwesomeStrategy --breakdown day week
```

``` output
======================== DAY BREAKDOWN =========================
|        Day |   Tot Profit USDT |   Wins |   Draws |   Losses |
|------------+-------------------+--------+---------+----------|
| 03/07/2021 |           200.0   |      2 |       0 |        0 |
| 04/07/2021 |           -50.31  |      0 |       0 |        2 |
| 05/07/2021 |           220.611 |      3 |       2 |        0 |
| 06/07/2021 |           150.974 |      3 |       0 |        2 |
| 07/07/2021 |           -70.193 |      1 |       0 |        2 |
| 08/07/2021 |           212.413 |      2 |       0 |        3 |

```

The output will show a table containing the realized absolute Profit (in stake currency) for the given timeperiod, as well as wins, draws and losses that materialized (closed) on this day. Below that there will be a second table for the summarized values of weeks indicated by the date of the closing Sunday. The same would apply to a monthly breakdown indicated by the last day of the month.

### Backtest result caching

To save time, by default backtest will reuse a cached result from within the last day when the backtested strategy and config match that of a previous backtest. To force a new backtest despite existing result for an identical run specify `--cache none` parameter.

!!! Warning
    Caching is automatically disabled for open-ended timeranges (`--timerange 20210101-`), as freqtrade cannot ensure reliably that the underlying data didn't change. It can also use cached results where it shouldn't if the original backtest had missing data at the end, which was fixed by downloading more data.
    In this instance, please use `--cache none` once to force a fresh backtest.

### Further backtest-result analysis

To further analyze your backtest results, you can [export the trades](#exporting-trades-to-file).
You can then load the trades to perform further analysis as shown in the [data analysis](data-analysis.md#backtesting) backtesting section.

## Assumptions made by backtesting

Since backtesting lacks some detailed information about what happens within a candle, it needs to take a few assumptions:

- Exchange [trading limits](#trading-limits-in-backtesting) are respected
- Entries happen at open-price
- All orders are filled at the requested price (no slippage, no unfilled orders)
- Exit-signal exits happen at open-price of the consecutive candle
- Exit-signal is favored over Stoploss, because exit-signals are assumed to trigger on candle's open
- ROI
  - exits are compared to high - but the ROI value is used (e.g. ROI = 2%, high=5% - so the exit will be at 2%)
  - exits are never "below the candle", so a ROI of 2% may result in a exit at 2.4% if low was at 2.4% profit
  - ROI entries which came into effect on the triggering candle (e.g. `120: 0.02` for 1h candles, from `60: 0.05`) will use the candle's open as exit rate
  - Force-exits caused by `<N>=-1` ROI entries use low as exit value, unless N falls on the candle open (e.g. `120: -1` for 1h candles)
- Stoploss exits happen exactly at stoploss price, even if low was lower, but the loss will be `2 * fees` higher than the stoploss price
- Stoploss is evaluated before ROI within one candle. So you can often see more trades with the `stoploss` exit reason comparing to the results obtained with the same strategy in the Dry Run/Live Trade modes
- Low happens before high for stoploss, protecting capital first
- Trailing stoploss
  - Trailing Stoploss is only adjusted if it's below the candle's low (otherwise it would be triggered)
  - On trade entry candles that trigger trailing stoploss, the "minimum offset" (`stop_positive_offset`) is assumed (instead of high) - and the stop is calculated from this point. This rule is NOT applicable to custom-stoploss scenarios, since there's no information about the stoploss logic available.
  - High happens first - adjusting stoploss
  - Low uses the adjusted stoploss (so exits with large high-low difference are backtested correctly)
  - ROI applies before trailing-stop, ensuring profits are "top-capped" at ROI if both ROI and trailing stop applies
- Exit-reason does not explain if a trade was positive or negative, just what triggered the exit (this can look odd if negative ROI values are used)
- Evaluation sequence (if multiple signals happen on the same candle)
  - Exit-signal
  - Stoploss
  - ROI
  - Trailing stoploss

Taking these assumptions, backtesting tries to mirror real trading as closely as possible. However, backtesting will **never** replace running a strategy in dry-run mode.
Also, keep in mind that past results don't guarantee future success.

In addition to the above assumptions, strategy authors should carefully read the [Common Mistakes](strategy-customization.md#common-mistakes-when-developing-strategies) section, to avoid using data in backtesting which is not available in real market conditions.

### Trading limits in backtesting

Exchanges have certain trading limits, like minimum (and maximum) base currency, or minimum/maximum stake (quote) currency.
These limits are usually listed in the exchange documentation as "trading rules" or similar and can be quite different between different pairs.

Backtesting (as well as live and dry-run) does honor these limits, and will ensure that a stoploss can be placed below this value - so the value will be slightly higher than what the exchange specifies.
Freqtrade has however no information about historic limits.

This can lead to situations where trading-limits are inflated by using a historic price, resulting in minimum amounts > 50$.

For example:

BTC minimum tradable amount is 0.001.
BTC trades at 22.000\$ today (0.001 BTC is related to this) - but the backtesting period includes prices as high as 50.000\$.
Today's minimum would be `0.001 * 22_000` - or 22\$.  
However the limit could also be 50$ - based on `0.001 * 50_000` in some historic setting.

#### Trading precision limits

Most exchanges pose precision limits on both price and amounts, so you cannot buy 1.0020401 of a pair, or at a price of 1.24567123123.  
Instead, these prices and amounts will be rounded or truncated (based on the exchange definition) to the defined trading precision.
The above values may for example be rounded to an amount of 1.002, and a price of 1.24567.

These precision values are based on current exchange limits (as described in the [above section](#trading-limits-in-backtesting)), as historic precision limits are not available.

## Improved backtest accuracy

One big limitation of backtesting is it's inability to know how prices moved intra-candle (was high before close, or viceversa?).
So assuming you run backtesting with a 1h timeframe, there will be 4 prices for that candle (Open, High, Low, Close).

While backtesting does take some assumptions (read above) about this - this can never be perfect, and will always be biased in one way or the other.
To mitigate this, freqtrade can use a lower (faster) timeframe to simulate intra-candle movements.

To utilize this, you can append `--timeframe-detail 5m` to your regular backtesting command.

``` bash
freqtrade backtesting --strategy AwesomeStrategy --timeframe 1h --timeframe-detail 5m
```

This will load 1h data as well as 5m data for the timeframe. The strategy will be analyzed with the 1h timeframe, and Entry orders will only be placed at the main timeframe, however Order fills and exit signals will be evaluated at the 5m candle, simulating intra-candle movements.

All callback functions (`custom_exit()`, `custom_stoploss()`, ... ) will be running for each 5m candle once the trade is opened (so 12 times in the above example of 1h timeframe, and 5m detailed timeframe).

`--timeframe-detail` must be smaller than the original timeframe, otherwise backtesting will fail to start.

Obviously this will require more memory (5m data is bigger than 1h data), and will also impact runtime (depending on the amount of trades and trade durations).
Also, data must be available / downloaded already.

!!! Tip
    You can use this function as the last part of strategy development, to ensure your strategy is not exploiting one of the [backtesting assumptions](#assumptions-made-by-backtesting). Strategies that perform similarly well with this mode have a good chance to perform well in dry/live modes too (although only forward-testing (dry-mode) can really confirm a strategy).

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
=========================================================== STRATEGY SUMMARY ===========================================================================
| Strategy    |  Entries |   Avg Profit % |   Cum Profit % |   Tot Profit BTC |   Tot Profit % | Avg Duration   |  Wins |  Draws | Losses | Drawdown % |
|:------------|---------:|---------------:|---------------:|-----------------:|---------------:|:---------------|------:|-------:|-------:|-----------:|
| Strategy1   |      429 |           0.36 |         152.41 |       0.00762792 |          76.20 | 4:12:00        |   186 |      0 |    243 |       45.2 |
| Strategy2   |     1487 |          -0.13 |        -197.58 |      -0.00988917 |         -98.79 | 4:43:00        |   662 |      0 |    825 |     241.68 |
```

## Next step

Great, your strategy is profitable. What if the bot can give your the optimal parameters to use for your strategy?
Your next step is to learn [how to find optimal parameters with Hyperopt](hyperopt.md)
