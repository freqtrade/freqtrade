# Data Downloading

## Getting data for backtesting and hyperopt

To download data (candles / OHLCV) needed for backtesting and hyperoptimization use the `freqtrade download-data` command.

If no additional parameter is specified, freqtrade will download data for `"1m"` and `"5m"` timeframes for the last 30 days.
Exchange and pairs will come from `config.json` (if specified using `-c/--config`).
Otherwise `--exchange` becomes mandatory.

!!! Tip "Tip: Updating existing data"
    If you already have backtesting data available in your data-directory and would like to refresh this data up to today, use `--days xx` with a number slightly higher than the missing number of days. Freqtrade will keep the available data and only download the missing data.
    Be carefull though: If the number is too small (which would result in a few missing days), the whole dataset will be removed and only xx days will be downloaded.

### Usage

```
usage: freqtrade download-data [-h] [-v] [--logfile FILE] [-V] [-c PATH] [-d PATH] [--userdir PATH] [-p PAIRS [PAIRS ...]]
                               [--pairs-file FILE] [--days INT] [--dl-trades] [--exchange EXCHANGE]
                               [-t {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} ...]]
                               [--erase] [--data-format {json,jsongz}] [--data-format-trades {json,jsongz}]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Show profits for only these pairs. Pairs are space-separated.
  --pairs-file FILE     File containing a list of pairs to download.
  --days INT            Download data for given number of days.
  --dl-trades           Download trades instead of OHLCV data. The bot will resample trades to the desired timeframe as specified as
                        --timeframes/-t.
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no config is provided.
  -t {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} ...], --timeframes {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} ...]
                        Specify which tickers to download. Space-separated list. Default: `1m 5m`.
  --erase               Clean all existing data for the selected exchange/pairs/timeframes.
  --data-format {json,jsongz}
                        Storage format for downloaded ohlcv data. (default: `json`).
  --data-format-trades {json,jsongz}
                        Storage format for downloaded trades data. (default: `jsongz`).

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --logfile FILE        Log to the file specified. Special values are: 'syslog', 'journald'. See the documentation for more details.
  -V, --version         show program's version number and exit
  -c PATH, --config PATH
                        Specify configuration file (default: `config.json`). Multiple --config options may be used. Can be set to `-`
                        to read config from stdin.
  -d PATH, --datadir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
```

### Data format

Freqtrade currently supports 2 dataformats, `json` and `jsongz`, a zipped version of json files.
By default, OHLCV data is stored as json data, while trades data is stored as `jsongz` data.

This can be changed via the `--data-format` and `--data-format-trades` parameters respectivly.

If the default dataformat has been changed during download, then the keys `dataformat_ohlcv` and `dataformat_trades` in the configuration file need to be adjusted to the selected dataformat as well.

!!! Note
    You can convert between data-formats using the [convert-data](#subcommand-convert-data) and [convert-trade-data](#subcommand-convert-trade-data) methods.

#### Subcommand convert data

```
usage: freqtrade convert-data [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                              [-d PATH] [--userdir PATH]
                              [-p PAIRS [PAIRS ...]] --format-from
                              {json,jsongz} --format-to {json,jsongz}
                              [--erase]
                              [-t {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} ...]]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Show profits for only these pairs. Pairs are space-
                        separated.
  --format-from {json,jsongz}
                        Source format for data conversation.
  --format-to {json,jsongz}
                        Destination format for data conversation.
  --erase               Clean all existing data for the selected
                        exchange/pairs/timeframes.
  -t {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} ...], --timeframes {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} [{1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w} ...]
                        Specify which tickers to download. Space-separated
                        list. Default: `1m 5m`.

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --logfile FILE        Log to the file specified. Special values are:
                        'syslog', 'journald'. See the documentation for more
                        details.
  -V, --version         show program's version number and exit
  -c PATH, --config PATH
                        Specify configuration file (default: `config.json`).
                        Multiple --config options may be used. Can be set to
                        `-` to read config from stdin.
  -d PATH, --datadir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
```

##### Example converting data

The following command will convert all Candle data available in `~/.freqtrade/data/binance` from json to jsongz, saving diskspace in the process.
It'll also remove source files (`--erase` parameter).

``` bash
freqtrade convert-data --format-from json --format-to jsongz --data-dir ~/.freqtrade/data/binance -t 5m 15m --erase
```

#### Subcommand convert-trade data

```
usage: freqtrade convert-trade-data [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                    [-d PATH] [--userdir PATH]
                                    [-p PAIRS [PAIRS ...]] --format-from
                                    {json,jsongz} --format-to {json,jsongz}
                                    [--erase]

optional arguments:
  -h, --help            show this help message and exit
  -p PAIRS [PAIRS ...], --pairs PAIRS [PAIRS ...]
                        Show profits for only these pairs. Pairs are space-
                        separated.
  --format-from {json,jsongz}
                        Source format for data conversation.
  --format-to {json,jsongz}
                        Destination format for data conversation.
  --erase               Clean all existing data for the selected
                        exchange/pairs/timeframes.

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --logfile FILE        Log to the file specified. Special values are:
                        'syslog', 'journald'. See the documentation for more
                        details.
  -V, --version         show program's version number and exit
  -c PATH, --config PATH
                        Specify configuration file (default: `config.json`).
                        Multiple --config options may be used. Can be set to
                        `-` to read config from stdin.
  -d PATH, --datadir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
```

##### Example converting trades

The following command will convert all available trade-data in `~/.freqtrade/data/kraken` from json to jsongz, saving diskspace in the process.
It'll also remove source files (`--erase` parameter).

``` bash
freqtrade convert-trade-data --format-from jsongz --format-to json --data-dir ~/.freqtrade/data/kraken --erase
```

### Pairs file

In alternative to the whitelist from `config.json`, a `pairs.json` file can be used.

If you are using Binance for example:

- create a directory `user_data/data/binance` and copy or create the `pairs.json` file in that directory.
- update the `pairs.json` file to contain the currency pairs you are interested in.

```bash
mkdir -p user_data/data/binance
cp freqtrade/tests/testdata/pairs.json user_data/data/binance
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

This will download ticker data for all the currency pairs you defined in `pairs.json`.

### Other Notes

- To use a different directory than the exchange specific default, use `--datadir user_data/data/some_directory`.
- To change the exchange used to download the tickers, please use a different configuration file (you'll probably need to adjust ratelimits etc.)
- To use `pairs.json` from some other directory, use `--pairs-file some_other_dir/pairs.json`.
- To download ticker data for only 10 days, use `--days 10` (defaults to 30 days).
- Use `--timeframes` to specify which tickers to download. Default is `--timeframes 1m 5m` which will download 1-minute and 5-minute tickers.
- To use exchange, timeframe and list of pairs as defined in your configuration file, use the `-c/--config` option. With this, the script uses the whitelist defined in the config as the list of currency pairs to download data for and does not require the pairs.json file. You can combine `-c/--config` with most other options.

### Trades (tick) data

By default, `download-data` subcommand downloads Candles (OHLCV) data. Some exchanges also provide historic trade-data via their API.
This data can be useful if you need many different timeframes, since it is only downloaded once, and then resampled locally to the desired timeframes.

Since this data is large by default, the files use gzip by default. They are stored in your data-directory with the naming convention of `<pair>-trades.json.gz` (`ETH_BTC-trades.json.gz`). Incremental mode is also supported, as for historic OHLCV data, so downloading the data once per week with `--days 8` will create an incremental data-repository.

To use this mode, simply add `--dl-trades` to your call. This will swap the download method to download trades, and resamples the data locally.

Example call:

```bash
freqtrade download-data --exchange binance --pairs XRP/ETH ETH/BTC --days 20 --dl-trades
```

!!! Note
    While this method uses async calls, it will be slow, since it requires the result of the previous call to generate the next request to the exchange.

!!! Warning
    The historic trades are not available during Freqtrade dry-run and live trade modes because all exchanges tested provide this data with a delay of few 100 candles, so it's not suitable for real-time trading.

!!! Note "Kraken user"
    Kraken users should read [this](exchanges.md#historic-kraken-data) before starting to download data.

## Next step

Great, you now have backtest data downloaded, so you can now start [backtesting](backtesting.md) your strategy.
