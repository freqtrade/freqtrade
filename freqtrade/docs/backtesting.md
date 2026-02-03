# Backtesting

This page explains how to validate your strategy performance by using Backtesting.

Backtesting requires historic data to be available.
To learn how to get data for the pairs and exchange you're interested in, head over to the [Data Downloading](data-download.md) section of the documentation.

Backtesting is also available in [webserver mode](freq-ui.md#backtesting), which allows you to run backtests via the web interface.

## Backtesting command reference

--8<-- "commands/backtesting.md"

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

Assume you downloaded the history data from the Binance exchange and kept it in the `user_data/data/binance-20180101` directory. 
You can then use this data for backtesting as follows:

```bash
freqtrade backtesting --strategy AwesomeStrategy --datadir user_data/data/binance-20180101 
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

Exporting trades to file specifying a custom directory

```bash
freqtrade backtesting --strategy backtesting --export trades --backtest-directory=user_data/custom-backtest-results
```

---

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
                                                 BACKTESTING REPORT                                                  
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Pair ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ LTC/USDT:USDT │     16 │          1.0 │          56.176 │         5.62 │        16:16:00 │   16     0     0   100 │
│ ETC/USDT:USDT │     12 │         0.72 │          30.936 │         3.09 │         9:55:00 │   11     0     1  91.7 │
│ ETH/USDT:USDT │      8 │         0.66 │          17.864 │         1.79 │ 1 day, 13:55:00 │    7     0     1  87.5 │
│ XLM/USDT:USDT │     10 │         0.31 │          11.054 │         1.11 │        12:08:00 │    9     0     1  90.0 │
│ BTC/USDT:USDT │      8 │         0.21 │           7.289 │         0.73 │ 3 days, 1:24:00 │    6     0     2  75.0 │
│ XRP/USDT:USDT │      9 │        -0.14 │          -7.261 │        -0.73 │        21:18:00 │    8     0     1  88.9 │
│ DOT/USDT:USDT │      6 │         -0.4 │          -9.187 │        -0.92 │         5:35:00 │    4     0     2  66.7 │
│ ADA/USDT:USDT │      8 │        -1.76 │         -52.098 │        -5.21 │        11:38:00 │    6     0     2  75.0 │
│         TOTAL │     77 │         0.22 │          54.774 │         5.48 │        22:12:00 │   67     0    10  87.0 │
└───────────────┴────────┴──────────────┴─────────────────┴──────────────┴─────────────────┴────────────────────────┘
                                               LEFT OPEN TRADES REPORT                                                
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃          Pair ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃     Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ BTC/USDT:USDT │      1 │        -4.14 │          -9.930 │        -0.99 │ 17 days, 8:00:00 │    0     0     1     0 │
│ ETC/USDT:USDT │      1 │        -4.24 │         -15.365 │        -1.54 │         10:40:00 │    0     0     1     0 │
│ DOT/USDT:USDT │      1 │        -5.29 │         -19.125 │        -1.91 │         11:30:00 │    0     0     1     0 │
│         TOTAL │      3 │        -4.56 │         -44.420 │        -4.44 │  6 days, 2:03:00 │    0     0     3     0 │
└───────────────┴────────┴──────────────┴─────────────────┴──────────────┴──────────────────┴────────────────────────┘
                                                ENTER TAG STATS                                                
┏━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Enter Tag ┃ Entries ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│     OTHER │      77 │         0.22 │          54.774 │         5.48 │     22:12:00 │   67     0    10  87.0 │
│     TOTAL │      77 │         0.22 │          54.774 │         5.48 │     22:12:00 │   67     0    10  87.0 │
└───────────┴─────────┴──────────────┴─────────────────┴──────────────┴──────────────┴────────────────────────┘
                                                EXIT REASON STATS                                                 
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Exit Reason ┃ Exits ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│         roi │    67 │         1.05 │         242.179 │        24.22 │        15:49:00 │   67     0     0   100 │
│ exit_signal │     4 │        -2.23 │         -31.217 │        -3.12 │  1 day, 8:38:00 │    0     0     4     0 │
│  force_exit │     3 │        -4.56 │         -44.420 │        -4.44 │ 6 days, 2:03:00 │    0     0     3     0 │
│   stop_loss │     3 │       -10.14 │        -111.768 │       -11.18 │  1 day, 3:05:00 │    0     0     3     0 │
│       TOTAL │    77 │         0.22 │          54.774 │         5.48 │        22:12:00 │   67     0    10  87.0 │
└─────────────┴───────┴──────────────┴─────────────────┴──────────────┴─────────────────┴────────────────────────┘
                                                        MIXED TAG STATS                                                        
┏━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Enter Tag ┃ Exit Reason ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃    Avg Duration ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│           │         roi │     67 │         1.05 │         242.179 │        24.22 │        15:49:00 │   67     0     0   100 │
│           │ exit_signal │      4 │        -2.23 │         -31.217 │        -3.12 │  1 day, 8:38:00 │    0     0     4     0 │
│           │  force_exit │      3 │        -4.56 │         -44.420 │        -4.44 │ 6 days, 2:03:00 │    0     0     3     0 │
│           │   stop_loss │      3 │       -10.14 │        -111.768 │       -11.18 │  1 day, 3:05:00 │    0     0     3     0 │
│     TOTAL │             │     77 │         0.22 │          54.774 │         5.48 │        22:12:00 │   67     0    10  87.0 │
└───────────┴─────────────┴────────┴──────────────┴─────────────────┴──────────────┴─────────────────┴────────────────────────┘
                          SUMMARY METRICS                          
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                        ┃ Value                           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Backtesting from              │ 2025-07-01 00:00:00             │
│ Backtesting to                │ 2025-08-01 00:00:00             │
│ Trading Mode                  │ Isolated Futures                │
│ Max open trades               │ 3                               │
│                               │                                 │
│ Total/Daily Avg Trades        │ 77 / 2.48                       │
│ Starting balance              │ 1000 USDT                       │
│ Final balance                 │ 1054.774 USDT                   │
│ Absolute profit               │ 54.774 USDT                     │
│ Total profit %                │ 5.48%                           │
│ CAGR %                        │ 87.36%                          │
│ Sortino                       │ 2.48                            │
│ Sharpe                        │ 3.75                            │
│ Calmar                        │ 40.99                           │
│ SQN                           │ 0.69                            │
│ Profit factor                 │ 1.29                            │
│ Expectancy (Ratio)            │ 0.71 (0.04)                     │
│ Avg. daily profit             │ 1.767 USDT                      │
│ Avg. stake amount             │ 345.016 USDT                    │
│ Total trade volume            │ 53316.954 USDT                  │
│                               │                                 │
│ Long / Short trades           │ 67 / 10                         │
│ Long / Short profit %         │ 8.94% / -3.47%                  │
│ Long / Short profit USDT      │ 89.425 / -34.651                │
│                               │                                 │
│ Best Pair                     │ LTC/USDT:USDT 5.62%             │
│ Worst Pair                    │ ADA/USDT:USDT -5.21%            │
│ Best trade                    │ ETC/USDT:USDT 2.00%             │
│ Worst trade                   │ ADA/USDT:USDT -10.17%           │
│ Best day                      │ 26.91 USDT                      │
│ Worst day                     │ -47.741 USDT                    │
│ Days win/draw/lose            │ 20 / 6 / 5                      │
│ Min/Max/Avg. Duration Winners │ 0d 00:35 / 5d 18:15 / 0d 15:49  │
│ Min/Max/Avg. Duration Losers  │ 0d 10:40 / 17d 08:00 / 2d 17:00 │
│ Max Consecutive Wins / Loss   │ 36 / 3                          │
│ Rejected Entry signals        │ 258                             │
│ Entry/Exit Timeouts           │ 0 / 0                           │
│                               │                                 │
│ Min balance                   │ 1003.168 USDT                   │
│ Max balance                   │ 1149.421 USDT                   │
│ Max % of account underwater   │ 8.23%                           │
│ Absolute drawdown             │ 94.647 USDT (8.23%)             │
│ Drawdown duration             │ 9 days 08:50:00                 │
│ Profit at drawdown start      │ 149.421 USDT                    │
│ Profit at drawdown end        │ 54.774 USDT                     │
│ Drawdown start                │ 2025-07-22 15:10:00             │
│ Drawdown end                  │ 2025-08-01 00:00:00             │
│ Market change                 │ 30.51%                          │
└───────────────────────────────┴─────────────────────────────────┘

Backtested 2025-07-01 00:00:00 -> 2025-08-01 00:00:00 | Max open trades : 3
                                                            STRATEGY SUMMARY                                                            
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃       Strategy ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃           Drawdown ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ SampleStrategy │     77 │         0.22 │          54.774 │         5.48 │     22:12:00 │   67     0    10  87.0 │ 94.647 USDT  8.23% │
└────────────────┴────────┴──────────────┴─────────────────┴──────────────┴──────────────┴────────────────────────┴────────────────────┘
```

### Backtesting report table

The first table contains all trades the bot made, including "left open trades".

The last line will give you the overall performance of your strategy,
here:

```
│         TOTAL │     77 │         0.22 │          54.774 │         5.48 │        22:12:00 │   67     0    10  87.0 │
```

The bot has made `77` trades for an average duration of `22:12:00`, with a performance of `5.48%` (profit), that means it has earned a total of `54.774 USDT` starting with a capital of 1000 USDT.

The column `Avg Profit %` shows the average profit for all trades made.
The column `Tot Profit %` shows instead the total profit % in relation to the starting balance.

In the above results, we have a starting balance of 1000 USDT and an absolute profit of 54.774 USDT - so the `Tot Profit %` will be `(54.774 / 1000) * 100 ~= 5.48%`.

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

### Left open trades table

The second table contains all trades the bot had to `force_exit` at the end of the backtesting period to present you the full picture.
This is necessary to simulate realistic behavior, since the backtest period has to end at some point, while realistically, you could leave the bot running forever.
These trades are also included in the first table, but are also shown separately in this table for clarity.

### Enter tag stats table

The third table provides a breakdown of trades by their entry tags (e.g., `enter_long`, `enter_short`), showing the number of entries, average profit percentage, total profit in the stake currency, total profit percentage, average duration, and the number of wins, draws, and losses for each tag.

### Exit reason stats table

The fourth table contains a recap of exit reasons (e.g., `exit_signal`, `roi`, `stop_loss`, `force_exit`). This table can tell you which area needs additional work (e.g., if many `exit_signal` trades are losses, you should work on improving the exit signal or consider disabling it).

### Mixed tag stats table

The fifth table combines entry tags and exit reasons, providing a detailed view of how different entry tags performed with specific exit reasons. This can help identify which combinations of entry and exit strategies are most effective.

### Summary metrics

The last element of the backtest report is the summary metrics table.
It contains key metrics about the performance of your strategy on backtesting data.

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                        ┃ Value                           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Backtesting from              │ 2025-07-01 00:00:00             │
│ Backtesting to                │ 2025-08-01 00:00:00             │
│ Trading Mode                  │ Isolated Futures                │
│ Max open trades               │ 3                               │
│                               │                                 │
│ Total/Daily Avg Trades        │ 72 / 2.32                       │
│ Starting balance              │ 1000 USDT                       │
│ Final balance                 │ 1106.734 USDT                   │
│ Absolute profit               │ 106.734 USDT                    │
│ Total profit %                │ 10.67%                          │
│ CAGR %                        │ 230.04%                         │
│ Sortino                       │ 4.99                            │
│ Sharpe                        │ 8.00                            │
│ Calmar                        │ 77.76                           │
│ SQN                           │ 1.52                            │
│ Profit factor                 │ 1.79                            │
│ Expectancy (Ratio)            │ 1.48 (0.07)                     │
│ Avg. daily profit             │ 3.443 USDT                      │
│ Avg. stake amount             │ 363.133 USDT                    │
│ Total trade volume            │ 52466.174 USDT                  │
│                               │                                 │
│ Best Pair                     │ LTC/USDT:USDT 4.48%             │
│ Worst Pair                    │ ADA/USDT:USDT -1.78%            │
│ Best trade                    │ ETC/USDT:USDT 2.00%             │
│ Worst trade                   │ ADA/USDT:USDT -10.17%           │
│ Best day                      │ 23.535 USDT                     │
│ Worst day                     │ -49.813 USDT                    │
│ Days win/draw/lose            │ 21 / 6 / 4                      │
│ Min/Max/Avg. Duration Winners │ 0d 00:35 / 5d 18:15 / 0d 15:30  │
│ Min/Max/Avg. Duration Losers  │ 0d 12:00 / 17d 08:00 / 3d 23:28 │
│ Max Consecutive Wins / Loss   │ 58 / 4                          │
│ Rejected Entry signals        │ 254                             │
│ Entry/Exit Timeouts           │ 0 / 0                           │
│                               │                                 │
│ Min balance                   │ 1003.168 USDT                   │
│ Max balance                   │ 1209 USDT                       │
│ Max % of account underwater   │ 8.46%                           │
│ Absolute drawdown             │ 102.266 USDT (8.46%)            │
│ Drawdown duration             │ 9 days 08:50:00                 │
│ Profit at drawdown start      │ 209 USDT                        │
│ Profit at drawdown end        │ 106.734 USDT                    │
│ Drawdown start                │ 2025-07-22 15:10:00             │
│ Drawdown end                  │ 2025-08-01 00:00:00             │
│ Market change                 │ 30.51%                          │
└───────────────────────────────┴─────────────────────────────────┘
```

- `Backtesting from` / `Backtesting to`: Backtesting range (usually defined with the `--timerange` option).
- `Trading Mode`: Spot or Futures trading.
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
- `SQN`: System Quality Number (SQN) - by Van Tharp.
- `Profit factor`: Sum of the profits of all winning trades divided by the sum of the losses of all losing trades.
- `Expectancy (Ratio)`: Expectancy ratio, which is the average profit or loss per trade. A negative expectancy ratio means that your strategy is not profitable.
- `Avg. daily profit`: Average profit per day, calculated as `(Total Profit / Backtest Days)`.
- `Avg. stake amount`: Average stake amount, either `stake_amount` or the average when using dynamic stake amount.
- `Total trade volume`: Volume generated on the exchange to reach the above profit.
- `Long / Short trades`: Split long/short trade counts (only shown when short trades were made).
- `Long / Short profit %`: Profit percentage for long and short trades (only shown when short trades were made).
- `Long / Short profit USDT`: Profit in stake currency for long and short trades (only shown when short trades were made).
- `Best Pair` / `Worst Pair`: Best and worst performing pair (based on total profit percentage), and its corresponding `Tot Profit %`.
- `Best trade` / `Worst trade`: Biggest single winning trade and biggest single losing trade.
- `Best day` / `Worst day`: Best and worst day based on daily profit.
- `Days win/draw/lose`: Winning / Losing days (draws are usually days without closed trades).
- `Min/Max/Avg. Duration Winners`: Minimum, maximum, and average durations for winning trades.
- `Min/Max/Avg. Duration Losers`: Minimum, maximum, and average durations for losing trades.
- `Max Consecutive Wins / Loss`: Maximum consecutive wins/losses in a row.
- `Rejected Entry signals`: Trade entry signals that could not be acted upon due to `max_open_trades` being reached.
- `Entry/Exit Timeouts`: Entry/exit orders which did not fill (only applicable if custom pricing is used).
- `Min balance` / `Max balance`: Lowest and Highest Wallet balance during the backtest period.
- `Max % of account underwater`: Maximum percentage your account has decreased from the top since the simulation started. Calculated as the maximum of `(Max Balance - Current Balance) / (Max Balance)`.
- `Absolute drawdown`: Maximum absolute drawdown experienced, including percentage relative to the account calculated as `(Absolute Drawdown) / (DrawdownHigh + startingBalance)`..
- `Drawdown duration`: Duration of the largest drawdown period.
- `Profit at drawdown start` / `Profit at drawdown end`: Profit at the beginning and end of the largest drawdown period.
- `Drawdown start` / `Drawdown end`: Start and end datetime for the largest drawdown (can also be visualized via the `plot-dataframe` sub-command).
- `Market change`: Change of the market during the backtest period. Calculated as the average of all pairs' changes from the first to the last candle using the "close" column.

### Daily / Weekly / Monthly / Yearly breakdown

You can get an overview over daily, weekly, monthly, or yearly results by using the `--breakdown <>` switch.

To visualize monthly and yearly breakdowns, you can use the following:

``` bash
freqtrade backtesting --strategy MyAwesomeStrategy --breakdown month year
```

``` output
                                 MONTH BREAKDOWN
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Month ┃ Trades ┃ Tot Profit USDT ┃ Profit Factor ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 31/01/2020 │     12 │          44.451 │          7.28 │   10     0     2  83.3 │
│ 29/02/2020 │     30 │           45.41 │          2.36 │   17     0    13  56.7 │
│ 31/03/2020 │     35 │         142.024 │          2.42 │   14     0    21  40.0 │
│ 30/04/2020 │     67 │         -23.692 │          0.81 │   24     0    43  35.8 │
...
...
│ 30/04/2025 │    203 │          -63.43 │          0.81 │   73     0   130  36.0 │
│ 31/05/2025 │    142 │         104.675 │          1.28 │   59     0    83  41.5 │
│ 30/06/2025 │    177 │          -1.014 │           1.0 │   85     0    92  48.0 │
│ 31/07/2025 │    155 │         232.762 │           1.6 │   63     0    92  40.6 │
└────────────┴────────┴─────────────────┴───────────────┴────────────────────────┘
                                  YEAR BREAKDOWN
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃       Year ┃ Trades ┃ Tot Profit USDT ┃ Profit Factor ┃  Win  Draw  Loss  Win% ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 31/12/2020 │    896 │         868.889 │          1.46 │  351     0   545  39.2 │
│ 31/12/2021 │   1778 │        4487.163 │          1.93 │  745     0  1033  41.9 │
│ 31/12/2022 │   1736 │          938.27 │          1.27 │  698     0  1038  40.2 │
│ 31/12/2023 │   1712 │        1677.126 │          1.68 │  670     0  1042  39.1 │
│ 31/12/2024 │   1609 │        3198.424 │          2.22 │  773     0   836  48.0 │
│ 31/12/2025 │   1042 │         716.174 │          1.33 │  420     0   622  40.3 │
└────────────┴────────┴─────────────────┴───────────────┴────────────────────────┘
```

The output will display tables containing the realized absolute profit (in stake currency) for the selected period, along with additional statistics such as number of trades, profit factor, and distribution of wins, draws, and losses that materialized (closed) on this period.

### Backtest result caching

To save time, by default backtest will reuse a cached result from within the last day when the backtested strategy and config match that of a previous backtest. To force a new backtest despite existing result for an identical run specify `--cache none` parameter.

!!! Warning
    Caching is automatically disabled for open-ended timeranges (`--timerange 20210101-`), as freqtrade cannot ensure reliably that the underlying data didn't change. It can also use cached results where it shouldn't if the original backtest had missing data at the end, which was fixed by downloading more data.
    In this instance, please use `--cache none` once to force a fresh backtest.

### Further backtest-result analysis

To further analyze your backtest results, freqtrade will export the trades to file by default.
You can then load the trades to perform further analysis as shown in the [data analysis](strategy_analysis_example.md#load-backtest-results-to-pandas-dataframe) backtesting section.

Also, you can use freqtrade in [webserver mode](freq-ui.md#backtesting) to visualize the backtest results in a web interface.
This mode also allows you to load existing backtest results, so you can analyze them without running the backtest again.  
For this mode - `--notes "<notes>"` can be used to add notes to the backtest results, which will be shown in the web interface.

### Backtest output file

The output file freqtrade produces is a zip file containing the following files:

- The backtest report in json format
- The market change data in feather format
- A copy of the strategy file
- A copy of the strategy parameters (if a parameter file was used)
- A sanitized copy of the config file

This will ensure results are reproducible - under the assumption that the same data is available.

Only the strategy file and the config file are included in the zip file, eventual dependencies are not included.

## Assumptions made by backtesting

Since backtesting lacks some detailed information about what happens within a candle, it needs to take a few assumptions:

- Exchange [trading limits](#trading-limits-in-backtesting) are respected
- Entries happen at open-price unless a custom price logic has been specified
- All orders are filled at the requested price (no slippage) as long as the price is within the candle's high/low range
- Exit-signal exits happen at open-price of the consecutive candle
- Exits free their trade slot for a new trade with a different pair
- Exit-signal is favored over Stoploss, because exit-signals are assumed to trigger on candle's open
- ROI
  - Exits are compared to high - but the ROI value is used (e.g. ROI = 2%, high=5% - so the exit will be at 2%)
  - Exits are never "below the candle", so a ROI of 2% may result in an exit at 2.4% if low was at 2.4% profit
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
- Position reversals (futures only) happen if an entry signal in the other direction than the closing trade triggers at the candle the existing trade closes.

Taking these assumptions, backtesting tries to mirror real trading as closely as possible. However, backtesting will **never** replace running a strategy in dry-run mode.
Also, keep in mind that past results don't guarantee future success.

In addition to the above assumptions, strategy authors should carefully read the [Common Mistakes](strategy-customization.md#common-mistakes-when-developing-strategies) section, to avoid using data in backtesting which is not available in real market conditions.

### Trading limits in backtesting

Exchanges have certain trading limits, like minimum (and maximum) base currency, or minimum/maximum stake (quote) currency.
These limits are usually listed in the exchange documentation as "trading rules" or similar and can be quite different between different pairs.

Backtesting (as well as live and dry-run) does honor these limits, and will ensure that a stoploss can be placed below this value - so the value will be slightly higher than what the exchange specifies.
Freqtrade has however no information about historic limits.

This can lead to situations where trading-limits are inflated by using a historic price, resulting in minimum amounts > 50\$.

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

One big limitation of backtesting is it's inability to know how prices moved intra-candle (was high before close, or vice-versa?).
So assuming you run backtesting with a 1h timeframe, there will be 4 prices for that candle (Open, High, Low, Close).

While backtesting does take some assumptions (read above) about this - this can never be perfect, and will always be biased in one way or the other.
To mitigate this, freqtrade can use a lower (faster) timeframe to simulate intra-candle movements.

To utilize this, you can append `--timeframe-detail 5m` to your regular backtesting command.

``` bash
freqtrade backtesting --strategy AwesomeStrategy --timeframe 1h --timeframe-detail 5m
```

This will load 1h data (the main timeframe) as well as 5m data (detail timeframe) for the selected timerange.
The strategy will be analyzed with the 1h timeframe.
Candles where activity may take place (there's an active signal, the pair is in a trade) are evaluated at the 5m timeframe.
This will allow for a more accurate simulation of intra-candle movements - and can lead to different results, especially on higher timeframes.

Entries will generally still happen at the main candle's open, however freed trade slots may be freed earlier (if the exit signal is triggered on the 5m candle), which can then be used for a new trade of a different pair.

All callback functions (`custom_exit()`, `custom_stoploss()`, ... ) will be running for each 5m candle once the trade is opened (so 12 times in the above example of 1h timeframe, and 5m detailed timeframe).

`--timeframe-detail` must be smaller than the original timeframe, otherwise backtesting will fail to start.

Obviously this will require more memory (5m data is bigger than 1h data), and will also impact runtime (depending on the amount of trades and trade durations).
Also, data must be available / downloaded already.

!!! Tip
    You can use this function as the last part of strategy development, to ensure your strategy is not exploiting one of the [backtesting assumptions](#assumptions-made-by-backtesting). Strategies that perform similarly well with this mode have a good chance to perform well in dry/live modes too (although only forward-testing (dry-mode) can really confirm a strategy).

??? Sample "Extreme Difference Example"
    Using `--timeframe-detail` on an extreme example (all below pairs have the 10:00 candle with an entry signal) may lead to the following backtesting Trade sequence with 1 max_open_trades:

    | Pair | Entry Time | Exit Time | Duration |
    |------|------------|-----------| -------- |
    | BTC/USDT | 2024-01-01 10:00:00 | 2021-01-01 10:05:00 | 5m |
    | ETH/USDT | 2024-01-01 10:05:00 | 2021-01-01 10:15:00 | 10m |
    | XRP/USDT | 2024-01-01 10:15:00 | 2021-01-01 10:30:00 | 15m |
    | SOL/USDT | 2024-01-01 10:15:00 | 2021-01-01 11:05:00 | 50m |
    | BTC/USDT | 2024-01-01 11:05:00 | 2021-01-01 12:00:00 | 55m |

    Without timeframe-detail, this would look like:

    | Pair | Entry Time | Exit Time | Duration |
    |------|------------|-----------| -------- |
    | BTC/USDT | 2024-01-01 10:00:00 | 2021-01-01 11:00:00 | 1h |
    | BTC/USDT | 2024-01-01 11:00:00 | 2021-01-01 12:00:00 | 1h |

    The difference is significant, as without detail data, only the first `max_open_trades` signals per candle are evaluated, and the trade slots are only freed at the end of the candle, allowing for a new trade to be opened at the next candle.


## Backtesting multiple strategies

To compare multiple strategies, a list of Strategies can be provided to backtesting.

This is limited to 1 timeframe value per run. However, data is only loaded once from disk so if you have multiple
strategies you'd like to compare, this will give a nice runtime boost.

All listed Strategies need to be in the same directory, unless also `--recursive-strategy-search` is specified, where sub-directories within the strategy directory are also considered.

``` bash
freqtrade backtesting --timerange 20180401-20180410 --timeframe 5m --strategy-list Strategy001 Strategy002 --export trades
```

This will save the results to `user_data/backtest_results/backtest-result-<datetime>.json`, including results for both `Strategy001` and `Strategy002`.
There will be an additional table comparing win/losses of the different strategies (identical to the "Total" row in the first table).
Detailed output for all strategies one after the other will be available, so make sure to scroll up to see the details per strategy.

```
================================================== STRATEGY SUMMARY ===================================================================
| Strategy    |  Trades |   Avg Profit % |   Tot Profit BTC |   Tot Profit % | Avg Duration   |  Wins |  Draws | Losses | Drawdown % |
|-------------+---------+----------------+------------------+----------------+----------------+-------+--------+--------+------------|
| Strategy1   |     429 |           0.36 |       0.00762792 |          76.20 | 4:12:00        |   186 |      0 |    243 |       45.2 |
| Strategy2   |    1487 |          -0.13 |      -0.00988917 |         -98.79 | 4:43:00        |   662 |      0 |    825 |     241.68 |
```

## Next step

Great, your strategy is profitable. What if the bot can give you the optimal parameters to use for your strategy?
Your next step is to learn [how to find optimal parameters with Hyperopt](hyperopt.md)
