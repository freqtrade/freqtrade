# Backtesting

This page explains how to validate your strategy performance by using
Backtesting.

## Test your strategy with Backtesting

Now you have good Buy and Sell strategies, you want to test it against
real data. This is what we call
[backtesting](https://en.wikipedia.org/wiki/Backtesting).

Backtesting will use the crypto-currencies (pair) from your config file
and load static tickers located in
[/freqtrade/tests/testdata](https://github.com/freqtrade/freqtrade/tree/develop/freqtrade/tests/testdata).
If the 5 min and 1 min ticker for the crypto-currencies to test is not
already in the `testdata` folder, backtesting will download them
automatically. Testdata files will not be updated until you specify it.

The result of backtesting will confirm you if your bot has better odds of making a profit than a loss.

The backtesting is very easy with freqtrade.

### Run a backtesting against the currencies listed in your config file
#### With 5 min tickers (Per default)

```bash
python3 ./freqtrade/main.py backtesting
```

#### With 1 min tickers

```bash
python3 ./freqtrade/main.py backtesting --ticker-interval 1m
```

#### Update cached pairs with the latest data

```bash
python3 ./freqtrade/main.py backtesting --refresh-pairs-cached
```

#### With live data (do not alter your testdata files)

```bash
python3 ./freqtrade/main.py backtesting --live
```

#### Using a different on-disk ticker-data source

```bash
python3 ./freqtrade/main.py backtesting --datadir freqtrade/tests/testdata-20180101
```

#### With a (custom) strategy file

```bash
python3 ./freqtrade/main.py -s TestStrategy backtesting
```

Where `-s TestStrategy` refers to the class name within the strategy file `test_strategy.py` found in the `freqtrade/user_data/strategies` directory

#### Exporting trades to file

```bash
python3 ./freqtrade/main.py backtesting --export trades
```

The exported trades can be read using the following code for manual analysis, or can be used by the plotting script `plot_dataframe.py` in the scripts folder.

``` python
import json
from pathlib import Path
import pandas as pd

filename=Path('user_data/backtest_data/backtest-result.json')

with filename.open() as file:
        data = json.load(file)

columns = ["pair", "profit", "opents", "closets", "index", "duration",
           "open_rate", "close_rate", "open_at_end", "sell_reason"]
df = pd.DataFrame(data, columns=columns)

df['opents'] = pd.to_datetime(df['opents'],
                              unit='s',
                              utc=True,
                              infer_datetime_format=True
                             )
df['closets'] = pd.to_datetime(df['closets'],
                               unit='s',
                               utc=True,
                               infer_datetime_format=True
                              )
```

If you have some ideas for interesting / helpful backtest data analysis, feel free to submit a PR so the community can benefit from it.

#### Exporting trades to file specifying a custom filename

```bash
python3 ./freqtrade/main.py backtesting --export trades --export-filename=backtest_teststrategy.json
```

#### Running backtest with smaller testset

Use the `--timerange` argument to change how much of the testset
you want to use. The last N ticks/timeframes will be used.

Example:

```bash
python3 ./freqtrade/main.py backtesting --timerange=-200
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

#### Downloading new set of ticker data

To download new set of backtesting ticker data, you can use a download script.

If you are using Binance for example:

- create a folder `user_data/data/binance` and copy `pairs.json` in that folder.
- update the `pairs.json` to contain the currency pairs you are interested in.

```bash
mkdir -p user_data/data/binance
cp freqtrade/tests/testdata/pairs.json user_data/data/binance
```

Then run:

```bash
python scripts/download_backtest_data.py --exchange binance
```

This will download ticker data for all the currency pairs you defined in `pairs.json`.

- To use a different folder than the exchange specific default, use `--export user_data/data/some_directory`.
- To change the exchange used to download the tickers, use `--exchange`. Default is `bittrex`.
- To use `pairs.json` from some other folder, use `--pairs-file some_other_dir/pairs.json`.
- To download ticker data for only 10 days, use `--days 10`.
- Use `--timeframes` to specify which tickers to download. Default is `--timeframes 1m 5m` which will download 1-minute and 5-minute tickers.

For help about backtesting usage, please refer to [Backtesting commands](#backtesting-commands).

## Understand the backtesting result

The most important in the backtesting is to understand the result.

A backtesting result will look like that:

```
======================================== BACKTESTING REPORT =========================================
| pair     |   buy count |   avg profit % |   total profit BTC |   avg duration |   profit |   loss |
|:---------|------------:|---------------:|-------------------:|---------------:|---------:|-------:|
| ETH/BTC  |          44 |           0.18 |         0.00159118 |           50.9 |       44 |      0 |
| LTC/BTC  |          27 |           0.10 |         0.00051931 |          103.1 |       26 |      1 |
| ETC/BTC  |          24 |           0.05 |         0.00022434 |          166.0 |       22 |      2 |
| DASH/BTC |          29 |           0.18 |         0.00103223 |          192.2 |       29 |      0 |
| ZEC/BTC  |          65 |          -0.02 |        -0.00020621 |          202.7 |       62 |      3 |
| XLM/BTC  |          35 |           0.02 |         0.00012877 |          242.4 |       32 |      3 |
| BCH/BTC  |          12 |           0.62 |         0.00149284 |           50.0 |       12 |      0 |
| POWR/BTC |          21 |           0.26 |         0.00108215 |          134.8 |       21 |      0 |
| ADA/BTC  |          54 |          -0.19 |        -0.00205202 |          191.3 |       47 |      7 |
| XMR/BTC  |          24 |          -0.43 |        -0.00206013 |          120.6 |       20 |      4 |
| TOTAL    |         335 |           0.03 |         0.00175246 |          157.9 |      315 |     20 |
2018-06-13 06:57:27,347 - freqtrade.optimize.backtesting - INFO -
====================================== LEFT OPEN TRADES REPORT ======================================
| pair     |   buy count |   avg profit % |   total profit BTC |   avg duration |   profit |   loss |
|:---------|------------:|---------------:|-------------------:|---------------:|---------:|-------:|
| ETH/BTC  |           3 |           0.16 |         0.00009619 |           25.0 |        3 |      0 |
| LTC/BTC  |           1 |          -1.00 |        -0.00020118 |         1085.0 |        0 |      1 |
| ETC/BTC  |           2 |          -1.80 |        -0.00071933 |         1092.5 |        0 |      2 |
| DASH/BTC |           0 |         nan    |         0.00000000 |          nan   |        0 |      0 |
| ZEC/BTC  |           3 |          -4.27 |        -0.00256826 |         1301.7 |        0 |      3 |
| XLM/BTC  |           3 |          -1.11 |        -0.00066744 |          965.0 |        0 |      3 |
| BCH/BTC  |           0 |         nan    |         0.00000000 |          nan   |        0 |      0 |
| POWR/BTC |           0 |         nan    |         0.00000000 |          nan   |        0 |      0 |
| ADA/BTC  |           7 |          -3.58 |        -0.00503604 |          850.0 |        0 |      7 |
| XMR/BTC  |           4 |          -3.79 |        -0.00303456 |          291.2 |        0 |      4 |
| TOTAL    |          23 |          -2.63 |        -0.01213062 |          750.4 |        3 |     20 |

```

The 1st table will contain all trades the bot made.

The 2nd table will contain all trades the bot had to `forcesell` at the end of the backtest period to present a full picture.
These trades are also included in the first table, but are extracted separately for clarity.

The last line will give you the overall performance of your strategy,
here:

```
TOTAL             419           -0.41         -0.00348593            52.9
```

We understand the bot has made `419` trades for an average duration of
`52.9` min, with a performance of `-0.41%` (loss), that means it has
lost a total of `-0.00348593 BTC`.

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

## Backtesting multiple strategies

To backtest multiple strategies, a list of Strategies can be provided.

This is limited to 1 ticker-interval per run, however, data is only loaded once from disk so if you have multiple
strategies you'd like to compare, this should give a nice runtime boost.

All listed Strategies need to be in the same folder.

``` bash
freqtrade backtesting --timerange 20180401-20180410 --ticker-interval 5m --strategy-list Strategy001 Strategy002 --export trades
```

This will save the results to `user_data/backtest_data/backtest-result-<strategy>.json`, injecting the strategy-name into the target filename.
There will be an additional table comparing win/losses of the different strategies (identical to the "Total" row in the first table).
Detailed output for all strategies one after the other will be available, so make sure to scroll up.

```
=================================================== Strategy Summary ====================================================
| Strategy   |   buy count |   avg profit % |   cum profit % |   total profit ETH | avg duration    |   profit |   loss |
|:-----------|------------:|---------------:|---------------:|-------------------:|:----------------|---------:|-------:|
| Strategy1  |          19 |          -0.76 |         -14.39 |        -0.01440287 | 15:48:00        |       15 |      4 |
| Strategy2  |           6 |          -2.73 |         -16.40 |        -0.01641299 | 1 day, 14:12:00 |        3 |      3 |
```

## Next step

Great, your strategy is profitable. What if the bot can give your the
optimal parameters to use for your strategy?
Your next step is to learn [how to find optimal parameters with Hyperopt](https://github.com/freqtrade/freqtrade/blob/develop/docs/hyperopt.md)
