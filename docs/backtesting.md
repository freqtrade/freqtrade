# Backtesting

This page explains how to validate your strategy performance by using
Backtesting.

## Getting data for backtesting and hyperopt

To download backtesting data (candles / OHLCV) and hyperoptimization using the `freqtrade download-data` command.

If no additional parameter is specified, freqtrade will download data for `"1m"` and `"5m"` timeframes.
Exchange and pairs will come from `config.json` (if specified using `-c/--config`). Otherwise `--exchange` becomes mandatory.

Alternatively, a `pairs.json` file can be used.

If you are using Binance for example:

- create a directory `user_data/data/binance` and copy `pairs.json` in that directory.
- update the `pairs.json` to contain the currency pairs you are interested in.

```bash
mkdir -p user_data/data/binance
cp freqtrade/tests/testdata/pairs.json user_data/data/binance
```

Then run:

```bash
freqtrade download-data --exchange binance
```

This will download ticker data for all the currency pairs you defined in `pairs.json`.

- To use a different directory than the exchange specific default, use `--datadir user_data/data/some_directory`.
- To change the exchange used to download the tickers, please use a different configuration file (you'll probably need to adjust ratelimits etc.)
- To use `pairs.json` from some other directory, use `--pairs-file some_other_dir/pairs.json`.
- To download ticker data for only 10 days, use `--days 10` (defaults to 30 days).
- Use `--timeframes` to specify which tickers to download. Default is `--timeframes 1m 5m` which will download 1-minute and 5-minute tickers.
- To use exchange, timeframe and list of pairs as defined in your configuration file, use the `-c/--config` option. With this, the script uses the whitelist defined in the config as the list of currency pairs to download data for and does not require the pairs.json file. You can combine `-c/--config` with most other options.

## Test your strategy with Backtesting

Now you have good Buy and Sell strategies and some historic data, you want to test it against
real data. This is what we call
[backtesting](https://en.wikipedia.org/wiki/Backtesting).

Backtesting will use the crypto-currencies (pair) from your config file
and load static tickers located in
[/freqtrade/tests/testdata](https://github.com/freqtrade/freqtrade/tree/develop/freqtrade/tests/testdata).
If the 5 min and 1 min ticker for the crypto-currencies to test is not
already in the `testdata` directory, backtesting will download them
automatically. Testdata files will not be updated until you specify it.

The result of backtesting will confirm you if your bot has better odds of making a profit than a loss.

The backtesting is very easy with freqtrade.

### Run a backtesting against the currencies listed in your config file
#### With 5 min tickers (Per default)

```bash
freqtrade backtesting
```

#### With 1 min tickers

```bash
freqtrade backtesting --ticker-interval 1m
```

#### Using a different on-disk ticker-data source

```bash
freqtrade backtesting --datadir user_data/data/bittrex-20180101
```

#### With a (custom) strategy file

```bash
freqtrade -s TestStrategy backtesting
```

Where `-s TestStrategy` refers to the class name within the strategy file `test_strategy.py` found in the `freqtrade/user_data/strategies` directory.

#### Comparing multiple Strategies

```bash
freqtrade backtesting --strategy-list TestStrategy1 AwesomeStrategy --ticker-interval 5m
```

Where `TestStrategy1` and `AwesomeStrategy` refer to class names of strategies.

#### Exporting trades to file

```bash
freqtrade backtesting --export trades
```

The exported trades can be used for [further analysis](#further-backtest-result-analysis), or can be used by the plotting script `plot_dataframe.py` in the scripts directory.

#### Exporting trades to file specifying a custom filename

```bash
freqtrade backtesting --export trades --export-filename=backtest_teststrategy.json
```

#### Running backtest with smaller testset

Use the `--timerange` argument to change how much of the testset
you want to use. The last N ticks/timeframes will be used.

Example:

```bash
freqtrade backtesting --timerange=-200
```

#### Advanced use of timerange

Doing `--timerange=-200` will get the last 200 timeframes
from your inputdata. You can also specify specific dates,
or a range span indexed by start and stop.

The full timerange specification:

- Use last 123 tickframes of data: `--timerange=-123`
- Use first 123 tickframes of data: `--timerange=123-`
- Use tickframes from line 123 through 456: `--timerange=123-456`
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
| pair     |   buy count |   avg profit % |   cum profit % |   tot profit BTC |   tot profit % | avg duration   |   profit |   loss |
|:---------|------------:|---------------:|---------------:|-----------------:|---------------:|:---------------|---------:|-------:|
| ADA/BTC  |          35 |          -0.11 |          -3.88 |      -0.00019428 |          -1.94 | 4:35:00        |       14 |     21 |
| ARK/BTC  |          11 |          -0.41 |          -4.52 |      -0.00022647 |          -2.26 | 2:03:00        |        3 |      8 |
| BTS/BTC  |          32 |           0.31 |           9.78 |       0.00048938 |           4.89 | 5:05:00        |       18 |     14 |
| DASH/BTC |          13 |          -0.08 |          -1.07 |      -0.00005343 |          -0.53 | 4:39:00        |        6 |      7 |
| ENG/BTC  |          18 |           1.36 |          24.54 |       0.00122807 |          12.27 | 2:50:00        |        8 |     10 |
| EOS/BTC  |          36 |           0.08 |           3.06 |       0.00015304 |           1.53 | 3:34:00        |       16 |     20 |
| ETC/BTC  |          26 |           0.37 |           9.51 |       0.00047576 |           4.75 | 6:14:00        |       11 |     15 |
| ETH/BTC  |          33 |           0.30 |           9.96 |       0.00049856 |           4.98 | 7:31:00        |       16 |     17 |
| IOTA/BTC |          32 |           0.03 |           1.09 |       0.00005444 |           0.54 | 3:12:00        |       14 |     18 |
| LSK/BTC  |          15 |           1.75 |          26.26 |       0.00131413 |          13.13 | 2:58:00        |        6 |      9 |
| LTC/BTC  |          32 |          -0.04 |          -1.38 |      -0.00006886 |          -0.69 | 4:49:00        |       11 |     21 |
| NANO/BTC |          17 |           1.26 |          21.39 |       0.00107058 |          10.70 | 1:55:00        |       10 |      7 |
| NEO/BTC  |          23 |           0.82 |          18.97 |       0.00094936 |           9.48 | 2:59:00        |       10 |     13 |
| REQ/BTC  |           9 |           1.17 |          10.54 |       0.00052734 |           5.27 | 3:47:00        |        4 |      5 |
| XLM/BTC  |          16 |           1.22 |          19.54 |       0.00097800 |           9.77 | 3:15:00        |        7 |      9 |
| XMR/BTC  |          23 |          -0.18 |          -4.13 |      -0.00020696 |          -2.07 | 5:30:00        |       12 |     11 |
| XRP/BTC  |          35 |           0.66 |          22.96 |       0.00114897 |          11.48 | 3:49:00        |       12 |     23 |
| ZEC/BTC  |          22 |          -0.46 |         -10.18 |      -0.00050971 |          -5.09 | 2:22:00        |        7 |     15 |
| TOTAL    |         429 |           0.36 |         152.41 |       0.00762792 |          76.20 | 4:12:00        |      186 |    243 |
========================================================= SELL REASON STATS =========================================================
| Sell Reason        |   Count |
|:-------------------|--------:|
| trailing_stop_loss |     205 |
| stop_loss          |     166 |
| sell_signal        |      56 |
| force_sell         |       2 |
====================================================== LEFT OPEN TRADES REPORT ======================================================
| pair     |   buy count |   avg profit % |   cum profit % |   tot profit BTC |   tot profit % | avg duration   |   profit |   loss |
|:---------|------------:|---------------:|---------------:|-----------------:|---------------:|:---------------|---------:|-------:|
| ADA/BTC  |           1 |           0.89 |           0.89 |       0.00004434 |           0.44 | 6:00:00        |        1 |      0 |
| LTC/BTC  |           1 |           0.68 |           0.68 |       0.00003421 |           0.34 | 2:00:00        |        1 |      0 |
| TOTAL    |           2 |           0.78 |           1.57 |       0.00007855 |           0.78 | 4:00:00        |        2 |      0 |
```

The 1st table will contain all trades the bot made.

The 2nd table will contain a recap of sell reasons.

The 3rd table will contain all trades the bot had to `forcesell` at the end of the backtest period to present a full picture.
These trades are also included in the first table, but are extracted separately for clarity.

The last line will give you the overall performance of your strategy,
here:

```
| TOTAL    |         429 |           0.36 |         152.41 |       0.00762792 |          76.20 | 4:12:00        |      186 |    243 |
```

We understand the bot has made `429` trades for an average duration of
`4:12:00`, with a performance of `76.20%` (profit), that means it has
earned a total of `0.00762792 BTC` starting with a capital of 0.01 BTC.

The column `avg profit %` shows the average profit for all trades made while the column `cum profit %` sums all the profits/losses. 
The column `tot profit %` shows instead the total profit % in relation to allocated capital 
(`max_open_trades * stake_amount`). In the above results we have `max_open_trades=2 stake_amount=0.005` in config 
so `(76.20/100) * (0.005 * 2) =~ 0.00762792 BTC`.

As you will see your strategy performance will be influenced by your buy
strategy, your sell strategy, and also by the `minimal_roi` and
`stop_loss` you have set.

As for an example if your minimal_roi is only `"0":  0.01`. You cannot
expect the bot to make more profit than 1% (because it will sell every
time a trade will reach 1%).

```json
"minimal_roi": {
    "0":  0.01
},
```

On the other hand, if you set a too high `minimal_roi` like `"0":  0.55`
(55%), there is a lot of chance that the bot will never reach this
profit. Hence, keep in mind that your performance is a mix of your
strategies, your configuration, and the crypto-currency you have set up.

### Further backtest-result analysis

To further analyze your backtest results, you can [export the trades](#exporting-trades-to-file).
You can then load the trades to perform further analysis as shown in our [data analysis](data-analysis.md#backtesting) backtesting section.


## Backtesting multiple strategies

To backtest multiple strategies, a list of Strategies can be provided.

This is limited to 1 ticker-interval per run, however, data is only loaded once from disk so if you have multiple
strategies you'd like to compare, this should give a nice runtime boost.

All listed Strategies need to be in the same directory.

``` bash
freqtrade backtesting --timerange 20180401-20180410 --ticker-interval 5m --strategy-list Strategy001 Strategy002 --export trades
```

This will save the results to `user_data/backtest_data/backtest-result-<strategy>.json`, injecting the strategy-name into the target filename.
There will be an additional table comparing win/losses of the different strategies (identical to the "Total" row in the first table).
Detailed output for all strategies one after the other will be available, so make sure to scroll up.

```
=========================================================== Strategy Summary ===========================================================
| Strategy    |   buy count |   avg profit % |   cum profit % |   tot profit BTC |   tot profit % | avg duration   |   profit |   loss |
|:------------|------------:|---------------:|---------------:|-----------------:|---------------:|:---------------|---------:|-------:|
| Strategy1   |         429 |           0.36 |         152.41 |       0.00762792 |          76.20 | 4:12:00        |      186 |    243 |
| Strategy2   |        1487 |          -0.13 |        -197.58 |      -0.00988917 |         -98.79 | 4:43:00        |      662 |    825 |
```

## Next step

Great, your strategy is profitable. What if the bot can give your the
optimal parameters to use for your strategy?
Your next step is to learn [how to find optimal parameters with Hyperopt](hyperopt.md)
