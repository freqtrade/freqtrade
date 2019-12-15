# Data Downloading

## Getting data for backtesting and hyperopt

To download data (candles / OHLCV) needed for backtesting and hyperoptimization use the `freqtrade download-data` command.

If no additional parameter is specified, freqtrade will download data for `"1m"` and `"5m"` timeframes for the last 30 days.
Exchange and pairs will come from `config.json` (if specified using `-c/--config`).
Otherwise `--exchange` becomes mandatory.

!!! Tip "Tip: Updating existing data"
    If you already have backtesting data available in your data-directory and would like to refresh this data up to today, use `--days xx` with a number slightly higher than the missing number of days. Freqtrade will keep the available data and only download the missing data.
    Be carefull though: If the number is too small (which would result in a few missing days), the whole dataset will be removed and only xx days will be downloaded.

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
