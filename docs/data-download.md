# Data download

## Getting data for backtesting and hyperopt

To download data (candles / OHLCV) needed for backtesting and hyperoptimization use the `freqtrade download-data` command.

If no additional parameter is specified, freqtrade will download data for `"1m"` and `"5m"` timeframes for the last 30 days.
Exchange and pairs will come from `config.json` (if specified using `-c/--config`).
Otherwise `--exchange` becomes mandatory.

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

### start download

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

## Next step

Great, you now have backtest data downloaded, so you can now start [backtesting](backtesting.md) your strategy.
