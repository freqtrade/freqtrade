# Data Downloading

## Getting data for backtesting and hyperopt

To download data (candles / OHLCV) needed for backtesting and hyperoptimization use the `freqtrade download-data` command.

If no additional parameter is specified, freqtrade will download data for `"1m"` and `"5m"` timeframes for the last 30 days.
Exchange and pairs will come from `config.json` (if specified using `-c/--config`).
Otherwise `--exchange` becomes mandatory.

You can use a relative timerange (`--days 20`) or an absolute starting point (`--timerange 20200101-`). For incremental downloads, the relative approach should be used.

!!! Tip "Tip: Updating existing data"
    If you already have backtesting data available in your data-directory and would like to refresh this data up to today, do not use `--days` or `--timerange` parameters. Freqtrade will keep the available data and only download the missing data.
    If you are updating existing data after inserting new pairs that you have no data for, use `--new-pairs-days xx` parameter. Specified number of days will be downloaded for new pairs while old pairs will be updated with missing data only.
    If you use `--days xx` parameter alone - data for specified number of days will be downloaded for _all_ pairs. Be careful, if specified number of days is smaller than gap between now and last downloaded candle - freqtrade will delete all existing data to avoid gaps in candle data.

### Usage

```
usage: freqtrade download-data [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                               [-d PATH] [--userdir PATH]
                               [-p PAIRS [PAIRS ...]] [--pairs-file FILE]
                               [--days INT] [--new-pairs-days INT]
                               [--timerange TIMERANGE] [--dl-trades]
                               [--exchange EXCHANGE]
                               [-t {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} ...]]
                               [--erase]
                               [--data-format-ohlcv {json,jsongz,hdf5}]
                               [--data-format-trades {json,jsongz,hdf5}]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  --pairs-file FILE     File containing a list of pairs to download.
  --days INT            Download data for given number of days.
  --new-pairs-days INT  Download data of new pairs for given number of days.
                        Default: `None`.
  --timerange TIMERANGE
                        Specify what timerange of data to use.
  --dl-trades           Download trades instead of OHLCV data. The bot will
                        resample trades to the desired timeframe as specified
                        as --timeframes/-t.
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no
                        config is provided.
  -t {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} ...], --timeframes {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} ...]
                        Specify which tickers to download. Space-separated
                        list. Default: `1m 5m`.
  --erase               Clean all existing data for the selected
                        exchange/pairs/timeframes.
  --data-format-ohlcv {json,jsongz,hdf5}
                        Storage format for downloaded candle (OHLCV) data.
                        (default: `None`).
  --data-format-trades {json,jsongz,hdf5}
                        Storage format for downloaded trades data. (default:
                        `None`).

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

```

!!! Note "Startup period"
    `download-data` is a strategy-independent command. The idea is to download a big chunk of data once, and then iteratively increase the amount of data stored.

    For that reason, `download-data` does not care about the "startup-period" defined in a strategy. It's up to the user to download additional days if the backtest should start at a specific point in time (while respecting startup period).

### Data format

Freqtrade currently supports 3 data-formats for both OHLCV and trades data:

* `json` (plain "text" json files)
* `jsongz` (a gzip-zipped version of json files)
* `hdf5` (a high performance datastore)

By default, OHLCV data is stored as `json` data, while trades data is stored as `jsongz` data.

This can be changed via the `--data-format-ohlcv` and `--data-format-trades` command line arguments respectively.
To persist this change, you can should also add the following snippet to your configuration, so you don't have to insert the above arguments each time:

``` jsonc
    // ...
    "dataformat_ohlcv": "hdf5",
    "dataformat_trades": "hdf5",
    // ...
```

If the default data-format has been changed during download, then the keys `dataformat_ohlcv` and `dataformat_trades` in the configuration file need to be adjusted to the selected dataformat as well.

!!! Note
    You can convert between data-formats using the [convert-data](#sub-command-convert-data) and [convert-trade-data](#sub-command-convert-trade-data) methods.

#### Sub-command convert data

```
usage: freqtrade convert-data [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                              [-d PATH] [--userdir PATH]
                              [-p PAIRS [PAIRS ...]] --format-from
                              {json,jsongz,hdf5} --format-to
                              {json,jsongz,hdf5} [--erase]
                              [-t {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} ...]]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Show profits for only these pairs. Pairs are space-
                        separated.
  --format-from {json,jsongz,hdf5}
                        Source format for data conversion.
  --format-to {json,jsongz,hdf5}
                        Destination format for data conversion.
  --erase               Clean all existing data for the selected
                        exchange/pairs/timeframes.
  -t {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} ...], --timeframes {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,2w,1M,1y} ...]
                        Specify which tickers to download. Space-separated
                        list. Default: `1m 5m`.

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
```

##### Example converting data

The following command will convert all candle (OHLCV) data available in `~/.freqtrade/data/binance` from json to jsongz, saving diskspace in the process.
It'll also remove original json data files (`--erase` parameter).

``` bash
freqtrade convert-data --format-from json --format-to jsongz --datadir ~/.freqtrade/data/binance -t 5m 15m --erase
```

#### Sub-command convert trade data

```
usage: freqtrade convert-trade-data [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                    [-d PATH] [--userdir PATH]
                                    [-p PAIRS [PAIRS ...]] --format-from
                                    {json,jsongz,hdf5} --format-to
                                    {json,jsongz,hdf5} [--erase]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Show profits for only these pairs. Pairs are space-
                        separated.
  --format-from {json,jsongz,hdf5}
                        Source format for data conversion.
  --format-to {json,jsongz,hdf5}
                        Destination format for data conversion.
  --erase               Clean all existing data for the selected
                        exchange/pairs/timeframes.

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

```

##### Example converting trades

The following command will convert all available trade-data in `~/.freqtrade/data/kraken` from jsongz to json.
It'll also remove original jsongz data files (`--erase` parameter).

``` bash
freqtrade convert-trade-data --format-from jsongz --format-to json --datadir ~/.freqtrade/data/kraken --erase
```

### Sub-command list-data

You can get a list of downloaded data using the `list-data` sub-command.

```
usage: freqtrade list-data [-h] [-v] [--logfile FILE] [-V] [-c PATH] [-d PATH]
                           [--userdir PATH] [--exchange EXCHANGE]
                           [--data-format-ohlcv {json,jsongz,hdf5}]
                           [-p PAIRS [PAIRS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no
                        config is provided.
  --data-format-ohlcv {json,jsongz,hdf5}
                        Storage format for downloaded candle (OHLCV) data.
                        (default: `json`).
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Show profits for only these pairs. Pairs are space-
                        separated.

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

```

#### Example list-data

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

### Pairs file

In alternative to the whitelist from `config.json`, a `pairs.json` file can be used.

If you are using Binance for example:

- create a directory `user_data/data/binance` and copy or create the `pairs.json` file in that directory.
- update the `pairs.json` file to contain the currency pairs you are interested in.

```bash
mkdir -p user_data/data/binance
cp tests/testdata/pairs.json user_data/data/binance
```

If you your configuration directory `user_data` was made by docker, you may get the following error:

```
cp: cannot create regular file 'user_data/data/binance/pairs.json': Permission denied
```

You can fix the permissions of your user-data directory as follows:

```
sudo chown -R $UID:$GID user_data
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

### Start download

Then run:

```bash
freqtrade download-data --exchange binance
```

This will download historical candle (OHLCV) data for all the currency pairs you defined in `pairs.json`.

### Other Notes

- To use a different directory than the exchange specific default, use `--datadir user_data/data/some_directory`.
- To change the exchange used to download the historical data from, please use a different configuration file (you'll probably need to adjust rate limits etc.)
- To use `pairs.json` from some other directory, use `--pairs-file some_other_dir/pairs.json`.
- To download historical candle (OHLCV) data for only 10 days, use `--days 10` (defaults to 30 days).
- To download historical candle (OHLCV) data from a fixed starting point, use `--timerange 20200101-` - which will download all data from January 1st, 2020. Eventually set end dates are ignored.
- Use `--timeframes` to specify what timeframe download the historical candle (OHLCV) data for. Default is `--timeframes 1m 5m` which will download 1-minute and 5-minute data.
- To use exchange, timeframe and list of pairs as defined in your configuration file, use the `-c/--config` option. With this, the script uses the whitelist defined in the config as the list of currency pairs to download data for and does not require the pairs.json file. You can combine `-c/--config` with most other options.

### Trades (tick) data

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
