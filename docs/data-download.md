# Data Downloading

## Getting data for backtesting and hyperopt

To download data (candles / OHLCV) needed for backtesting and hyperoptimization use the `freqtrade download-data` command.

If no additional parameter is specified, freqtrade will download data for `"1m"` and `"5m"` timeframes for the last 30 days.
Exchange and pairs will come from `config.json` (if specified using `-c/--config`).
Without provided configuration, `--exchange` becomes mandatory.

You can use a relative timerange (`--days 20`) or an absolute starting point (`--timerange 20200101-`). For incremental downloads, the relative approach should be used.

!!! Tip "Tip: Updating existing data"
    If you already have backtesting data available in your data-directory and would like to refresh this data up to today, freqtrade will automatically calculate the data missing for the existing pairs and the download will occur from the latest available point until "now", neither --days or --timerange parameters are required. Freqtrade will keep the available data and only download the missing data.
    If you are updating existing data after inserting new pairs that you have no data for, use `--new-pairs-days xx` parameter. Specified number of days will be downloaded for new pairs while old pairs will be updated with missing data only.
    If you use `--days xx` parameter alone - data for specified number of days will be downloaded for _all_ pairs. Be careful, if specified number of days is smaller than gap between now and last downloaded candle - freqtrade will delete all existing data to avoid gaps in candle data.

### Usage

```
usage: freqtrade download-data [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                               [-d PATH] [--userdir PATH]
                               [-p PAIRS [PAIRS ...]] [--pairs-file FILE]
                               [--days INT] [--new-pairs-days INT]
                               [--include-inactive-pairs]
                               [--timerange TIMERANGE] [--dl-trades]
                               [--exchange EXCHANGE]
                               [-t TIMEFRAMES [TIMEFRAMES ...]] [--erase]
                               [--data-format-ohlcv {json,jsongz,hdf5,feather,parquet}]
                               [--data-format-trades {json,jsongz,hdf5}]
                               [--trading-mode {spot,margin,futures}]
                               [--prepend]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  --pairs-file FILE     File containing a list of pairs. Takes precedence over
                        --pairs or pairs configured in the configuration.
  --days INT            Download data for given number of days.
  --new-pairs-days INT  Download data of new pairs for given number of days.
                        Default: `None`.
  --include-inactive-pairs
                        Also download data from inactive pairs.
  --timerange TIMERANGE
                        Specify what timerange of data to use.
  --dl-trades           Download trades instead of OHLCV data. The bot will
                        resample trades to the desired timeframe as specified
                        as --timeframes/-t.
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no
                        config is provided.
  -t TIMEFRAMES [TIMEFRAMES ...], --timeframes TIMEFRAMES [TIMEFRAMES ...]
                        Specify which tickers to download. Space-separated
                        list. Default: `1m 5m`.
  --erase               Clean all existing data for the selected
                        exchange/pairs/timeframes.
  --data-format-ohlcv {json,jsongz,hdf5,feather,parquet}
                        Storage format for downloaded candle (OHLCV) data.
                        (default: `json`).
  --data-format-trades {json,jsongz,hdf5}
                        Storage format for downloaded trades data. (default:
                        `jsongz`).
  --trading-mode {spot,margin,futures}, --tradingmode {spot,margin,futures}
                        Select Trading mode
  --prepend             Allow data prepending. (Data-appending is disabled)

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
  -d PATH, --datadir PATH, --data-dir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.

```

!!! Tip "Downloading all data for one quote currency"
    Often, you'll want to download data for all pairs of a specific quote-currency. In such cases, you can use the following shorthand:
    `freqtrade download-data --exchange binance --pairs .*/USDT <...>`. The provided "pairs" string will be expanded to contain all active pairs on the exchange.
    To also download data for inactive (delisted) pairs, add `--include-inactive-pairs` to the command.

!!! Note "Startup period"
    `download-data` is a strategy-independent command. The idea is to download a big chunk of data once, and then iteratively increase the amount of data stored.

    For that reason, `download-data` does not care about the "startup-period" defined in a strategy. It's up to the user to download additional days if the backtest should start at a specific point in time (while respecting startup period).

### Start download

A very simple command (assuming an available `config.json` file) can look as follows.

```bash
freqtrade download-data --exchange binance
```

This will download historical candle (OHLCV) data for all the currency pairs defined in the configuration.

Alternatively, specify the pairs directly

```bash
freqtrade download-data --exchange binance --pairs ETH/USDT XRP/USDT BTC/USDT
```

or as regex (in this case, to download all active USDT pairs)

```bash
freqtrade download-data --exchange binance --pairs .*/USDT
```

### Other Notes

* To use a different directory than the exchange specific default, use `--datadir user_data/data/some_directory`.
* To change the exchange used to download the historical data from, please use a different configuration file (you'll probably need to adjust rate limits etc.)
* To use `pairs.json` from some other directory, use `--pairs-file some_other_dir/pairs.json`.
* To download historical candle (OHLCV) data for only 10 days, use `--days 10` (defaults to 30 days).
* To download historical candle (OHLCV) data from a fixed starting point, use `--timerange 20200101-` - which will download all data from January 1st, 2020.
* Use `--timeframes` to specify what timeframe download the historical candle (OHLCV) data for. Default is `--timeframes 1m 5m` which will download 1-minute and 5-minute data.
* To use exchange, timeframe and list of pairs as defined in your configuration file, use the `-c/--config` option. With this, the script uses the whitelist defined in the config as the list of currency pairs to download data for and does not require the pairs.json file. You can combine `-c/--config` with most other options.

??? Note "Permission denied errors"
    If your configuration directory `user_data` was made by docker, you may get the following error:

    ```
    cp: cannot create regular file 'user_data/data/binance/pairs.json': Permission denied
    ```

    You can fix the permissions of your user-data directory as follows:

    ```
    sudo chown -R $UID:$GID user_data
    ```

### Download additional data before the current timerange

Assuming you downloaded all data from 2022 (`--timerange 20220101-`) - but you'd now like to also backtest with earlier data.
You can do so by using the `--prepend` flag, combined with `--timerange` - specifying an end-date.

``` bash
freqtrade download-data --exchange binance --pairs ETH/USDT XRP/USDT BTC/USDT --prepend --timerange 20210101-20220101
```

!!! Note
    Freqtrade will ignore the end-date in this mode if data is available, updating the end-date to the existing data start point.

### Data format

Freqtrade currently supports the following data-formats:

* `json` -  plain "text" json files
* `jsongz` - a gzip-zipped version of json files
* `hdf5` - a high performance datastore
* `feather` - a dataformat based on Apache Arrow (OHLCV only)
* `parquet` - columnar datastore (OHLCV only)

By default, OHLCV data is stored as `json` data, while trades data is stored as `jsongz` data.

This can be changed via the `--data-format-ohlcv` and `--data-format-trades` command line arguments respectively.
To persist this change, you should also add the following snippet to your configuration, so you don't have to insert the above arguments each time:

``` jsonc
    // ...
    "dataformat_ohlcv": "hdf5",
    "dataformat_trades": "hdf5",
    // ...
```

If the default data-format has been changed during download, then the keys `dataformat_ohlcv` and `dataformat_trades` in the configuration file need to be adjusted to the selected dataformat as well.

!!! Note
    You can convert between data-formats using the [convert-data](#sub-command-convert-data) and [convert-trade-data](#sub-command-convert-trade-data) methods.

#### Dataformat comparison

The following comparisons have been made with the following data, and by using the linux `time` command.

```
Found 6 pair / timeframe combinations.
+----------+-------------+--------+---------------------+---------------------+
|     Pair |   Timeframe |   Type |                From |                  To |
|----------+-------------+--------+---------------------+---------------------|
| BTC/USDT |          5m |   spot | 2017-08-17 04:00:00 | 2022-09-13 19:25:00 |
| ETH/USDT |          1m |   spot | 2017-08-17 04:00:00 | 2022-09-13 19:26:00 |
| BTC/USDT |          1m |   spot | 2017-08-17 04:00:00 | 2022-09-13 19:30:00 |
| XRP/USDT |          5m |   spot | 2018-05-04 08:10:00 | 2022-09-13 19:15:00 |
| XRP/USDT |          1m |   spot | 2018-05-04 08:11:00 | 2022-09-13 19:22:00 |
| ETH/USDT |          5m |   spot | 2017-08-17 04:00:00 | 2022-09-13 19:20:00 |
+----------+-------------+--------+---------------------+---------------------+
```

Timings have been taken in a not very scientific way with the following command, which forces reading the data into memory.

``` bash
time freqtrade list-data --show-timerange --data-format-ohlcv <dataformat>
```

|  Format | Size | timing |
|------------|-------------|-------------|
| `json` | 149Mb | 25.6s |
| `jsongz` | 39Mb | 27s |
| `hdf5` | 145Mb | 3.9s |
| `feather` | 72Mb | 3.5s |
| `parquet` | 83Mb | 3.8s |

Size has been taken from the BTC/USDT 1m spot combination for the timerange specified above.

To have a best performance/size mix, we recommend the use of either feather or parquet.

### Pairs file

In alternative to the whitelist from `config.json`, a `pairs.json` file can be used.
If you are using Binance for example:

* create a directory `user_data/data/binance` and copy or create the `pairs.json` file in that directory.
* update the `pairs.json` file to contain the currency pairs you are interested in.

```bash
mkdir -p user_data/data/binance
touch user_data/data/binance/pairs.json
```

The format of the `pairs.json` file is a simple json list.
Mixing different stake-currencies is allowed for this file, since it's only used for downloading.

``` json
[
    "ETH/BTC",
    "ETH/USDT",
    "BTC/USDT",
    "XRP/ETH"
]
```

!!! Note
    The `pairs.json` file is only used when no configuration is loaded (implicitly by naming, or via `--config` flag).
    You can force the usage of this file via `--pairs-file pairs.json` - however we recommend to use the pairlist from within the configuration, either via `exchange.pair_whitelist` or `pairs` setting in the configuration.

## Sub-command convert data

```
usage: freqtrade convert-data [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                              [-d PATH] [--userdir PATH]
                              [-p PAIRS [PAIRS ...]] --format-from
                              {json,jsongz,hdf5,feather,parquet} --format-to
                              {json,jsongz,hdf5,feather,parquet} [--erase]
                              [--exchange EXCHANGE]
                              [-t TIMEFRAMES [TIMEFRAMES ...]]
                              [--trading-mode {spot,margin,futures}]
                              [--candle-types {spot,futures,mark,index,premiumIndex,funding_rate} [{spot,futures,mark,index,premiumIndex,funding_rate} ...]]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  --format-from {json,jsongz,hdf5,feather,parquet}
                        Source format for data conversion.
  --format-to {json,jsongz,hdf5,feather,parquet}
                        Destination format for data conversion.
  --erase               Clean all existing data for the selected
                        exchange/pairs/timeframes.
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no
                        config is provided.
  -t TIMEFRAMES [TIMEFRAMES ...], --timeframes TIMEFRAMES [TIMEFRAMES ...]
                        Specify which tickers to download. Space-separated
                        list. Default: `1m 5m`.
  --trading-mode {spot,margin,futures}, --tradingmode {spot,margin,futures}
                        Select Trading mode
  --candle-types {spot,futures,mark,index,premiumIndex,funding_rate} [{spot,futures,mark,index,premiumIndex,funding_rate} ...]
                        Select candle type to use

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
  -d PATH, --datadir PATH, --data-dir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.

```

### Example converting data

The following command will convert all candle (OHLCV) data available in `~/.freqtrade/data/binance` from json to jsongz, saving diskspace in the process.
It'll also remove original json data files (`--erase` parameter).

``` bash
freqtrade convert-data --format-from json --format-to jsongz --datadir ~/.freqtrade/data/binance -t 5m 15m --erase
```

## Sub-command convert trade data

```
usage: freqtrade convert-trade-data [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                    [-d PATH] [--userdir PATH]
                                    [-p PAIRS [PAIRS ...]] --format-from
                                    {json,jsongz,hdf5,feather,parquet}
                                    --format-to
                                    {json,jsongz,hdf5,feather,parquet}
                                    [--erase] [--exchange EXCHANGE]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  --format-from {json,jsongz,hdf5,feather,parquet}
                        Source format for data conversion.
  --format-to {json,jsongz,hdf5,feather,parquet}
                        Destination format for data conversion.
  --erase               Clean all existing data for the selected
                        exchange/pairs/timeframes.
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no
                        config is provided.

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
  -d PATH, --datadir PATH, --data-dir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.

```

### Example converting trades

The following command will convert all available trade-data in `~/.freqtrade/data/kraken` from jsongz to json.
It'll also remove original jsongz data files (`--erase` parameter).

``` bash
freqtrade convert-trade-data --format-from jsongz --format-to json --datadir ~/.freqtrade/data/kraken --erase
```

## Sub-command trades to ohlcv

When you need to use `--dl-trades` (kraken only) to download data, conversion of trades data to ohlcv data is the last step.
This command will allow you to repeat this last step for additional timeframes without re-downloading the data.

```
usage: freqtrade trades-to-ohlcv [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                 [-d PATH] [--userdir PATH]
                                 [-p PAIRS [PAIRS ...]]
                                 [-t TIMEFRAMES [TIMEFRAMES ...]]
                                 [--exchange EXCHANGE]
                                 [--data-format-ohlcv {json,jsongz,hdf5,feather,parquet}]
                                 [--data-format-trades {json,jsongz,hdf5}]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  -t TIMEFRAMES [TIMEFRAMES ...], --timeframes TIMEFRAMES [TIMEFRAMES ...]
                        Specify which tickers to download. Space-separated
                        list. Default: `1m 5m`.
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no
                        config is provided.
  --data-format-ohlcv {json,jsongz,hdf5,feather,parquet}
                        Storage format for downloaded candle (OHLCV) data.
                        (default: `json`).
  --data-format-trades {json,jsongz,hdf5}
                        Storage format for downloaded trades data. (default:
                        `jsongz`).

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
  -d PATH, --datadir PATH, --data-dir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.

```

### Example trade-to-ohlcv conversion

``` bash
freqtrade trades-to-ohlcv --exchange kraken -t 5m 1h 1d --pairs BTC/EUR ETH/EUR
```

## Sub-command list-data

You can get a list of downloaded data using the `list-data` sub-command.

```
usage: freqtrade list-data [-h] [-v] [--logfile FILE] [-V] [-c PATH] [-d PATH]
                           [--userdir PATH] [--exchange EXCHANGE]
                           [--data-format-ohlcv {json,jsongz,hdf5,feather,parquet}]
                           [-p PAIRS [PAIRS ...]]
                           [--trading-mode {spot,margin,futures}]
                           [--show-timerange]

optional arguments:
  -h, --help            show this help message and exit
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no
                        config is provided.
  --data-format-ohlcv {json,jsongz,hdf5,feather,parquet}
                        Storage format for downloaded candle (OHLCV) data.
                        (default: `json`).
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  --trading-mode {spot,margin,futures}, --tradingmode {spot,margin,futures}
                        Select Trading mode
  --show-timerange      Show timerange available for available data. (May take
                        a while to calculate).

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
  -d PATH, --datadir PATH, --data-dir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.

```

### Example list-data

```bash
> freqtrade list-data --userdir ~/.freqtrade/user_data/

Found 33 pair / timeframe combinations.
pairs       timeframe
----------  -----------------------------------------
ADA/BTC     5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
ADA/ETH     5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
ETH/BTC     5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
ETH/USDT    5m, 15m, 30m, 1h, 2h, 4h
```

## Trades (tick) data

By default, `download-data` sub-command downloads Candles (OHLCV) data. Some exchanges also provide historic trade-data via their API.
This data can be useful if you need many different timeframes, since it is only downloaded once, and then resampled locally to the desired timeframes.

Since this data is large by default, the files use gzip by default. They are stored in your data-directory with the naming convention of `<pair>-trades.json.gz` (`ETH_BTC-trades.json.gz`). Incremental mode is also supported, as for historic OHLCV data, so downloading the data once per week with `--days 8` will create an incremental data-repository.

To use this mode, simply add `--dl-trades` to your call. This will swap the download method to download trades, and resamples the data locally.

!!! Warning "do not use"
    You should not use this unless you're a kraken user. Most other exchanges provide OHLCV data with sufficient history.

Example call:

```bash
freqtrade download-data --exchange kraken --pairs XRP/EUR ETH/EUR --days 20 --dl-trades
```

!!! Note
    While this method uses async calls, it will be slow, since it requires the result of the previous call to generate the next request to the exchange.

!!! Warning
    The historic trades are not available during Freqtrade dry-run and live trade modes because all exchanges tested provide this data with a delay of few 100 candles, so it's not suitable for real-time trading.

!!! Note "Kraken user"
    Kraken users should read [this](exchanges.md#historic-kraken-data) before starting to download data.

## Next step

Great, you now have backtest data downloaded, so you can now start [backtesting](backtesting.md) your strategy.
