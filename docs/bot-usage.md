# Bot usage
This page explains the difference parameters of the bot and how to run 
it.

## Table of Contents
- [Bot commands](#bot-commands)
- [Backtesting commands](#backtesting-commands)
- [Hyperopt commands](#hyperopt-commands)

## Bot commands
```
usage: freqtrade [-h] [-v] [--version] [-c PATH] [-d PATH] [-s NAME]
                 [--strategy-path PATH] [--dynamic-whitelist [INT]]
                 [--db-url PATH]
                 {backtesting,hyperopt} ...

Simple High Frequency Trading Bot for crypto currencies

positional arguments:
  {backtesting,hyperopt}
    backtesting         backtesting module
    hyperopt            hyperopt module

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         be verbose
  --version             show program's version number and exit
  -c PATH, --config PATH
                        specify configuration file (default: config.json)
  -d PATH, --datadir PATH
                        path to backtest data
  -s NAME, --strategy NAME
                        specify strategy class name (default: DefaultStrategy)
  --strategy-path PATH  specify additional strategy lookup path
  --dynamic-whitelist [INT]
                        dynamically generate and update whitelist based on 24h
                        BaseVolume (default: 20)
  --db-url PATH         Override trades database URL, this is useful if
                        dry_run is enabled or in custom deployments (default:
                        sqlite:///tradesv3.sqlite)
```

### How to use a different config file?
The bot allows you to select which config file you want to use. Per 
default, the bot will load the file `./config.json`

```bash
python3 ./freqtrade/main.py -c path/far/far/away/config.json 
```

### How to use --strategy?
This parameter will allow you to load your custom strategy class.
Per default without `--strategy` or `-s` the bot will load the
`DefaultStrategy` included with the bot (`freqtrade/strategy/default_strategy.py`).

The bot will search your strategy file within `user_data/strategies` and `freqtrade/strategy`.

To load a strategy, simply pass the class name (e.g.: `CustomStrategy`) in this parameter.

**Example:**  
In `user_data/strategies` you have a file `my_awesome_strategy.py` which has
a strategy class called `AwesomeStrategy` to load it:
```bash
python3 ./freqtrade/main.py --strategy AwesomeStrategy
```

If the bot does not find your strategy file, it will display in an error 
message the reason (File not found, or errors in your code).

Learn more about strategy file in [optimize your bot](https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md).

### How to use --strategy-path?
This parameter allows you to add an additional strategy lookup path, which gets
checked before the default locations (The passed path must be a folder!):
```bash
python3 ./freqtrade/main.py --strategy AwesomeStrategy --strategy-path /some/folder
```

#### How to install a strategy?
This is very simple. Copy paste your strategy file into the folder 
`user_data/strategies` or use `--strategy-path`. And voila, the bot is ready to use it.

### How to use --dynamic-whitelist?
Per default `--dynamic-whitelist` will retrieve the 20 currencies based 
on BaseVolume. This value can be changed when you run the script.

**By Default**  
Get the 20 currencies based on BaseVolume.  
```bash
python3 ./freqtrade/main.py --dynamic-whitelist
```

**Customize the number of currencies to retrieve**  
Get the 30 currencies based on BaseVolume.  
```bash
python3 ./freqtrade/main.py --dynamic-whitelist 30
```

**Exception**  
`--dynamic-whitelist` must be greater than 0. If you enter 0 or a
negative value (e.g -2), `--dynamic-whitelist` will use the default
value (20).

### How to use --db-url?
When you run the bot in Dry-run mode, per default no transactions are 
stored in a database. If you want to store your bot actions in a DB 
using `--db-url`. This can also be used to specify a custom database
in production mode. Example command:

```bash
python3 ./freqtrade/main.py -c config.json --db-url sqlite:///tradesv3.dry_run.sqlite
```


## Backtesting commands

Backtesting also uses the config specified via `-c/--config`.

```
usage: main.py backtesting [-h] [-i TICKER_INTERVAL] [--realistic-simulation]
                           [--timerange TIMERANGE] [-l] [-r] [--export EXPORT]
                           [--export-filename EXPORTFILENAME]


optional arguments:
  -h, --help            show this help message and exit
  -i TICKER_INTERVAL, --ticker-interval TICKER_INTERVAL
                        specify ticker interval (1m, 5m, 30m, 1h, 1d)
  --enable-position-tracking
                        Disables buying the same pair multiple times to 
                        simulate real world limitations
  --timerange TIMERANGE
                        specify what timerange of data to use.
  -l, --live            using live data
  -r, --refresh-pairs-cached
                        refresh the pairs files in tests/testdata with the
                        latest data from the exchange. Use it if you want to
                        run your backtesting with up-to-date data.
  --export EXPORT       export backtest results, argument are: trades Example
                        --export=trades
  --export-filename EXPORTFILENAME
                        Save backtest results to this filename requires
                        --export to be set as well Example --export-
                        filename=backtest_today.json (default: backtest-
                        result.json
```

### How to use --refresh-pairs-cached parameter?
The first time your run Backtesting, it will take the pairs you have 
set in your config file and download data from Bittrex. 

If for any reason you want to update your data set, you use 
`--refresh-pairs-cached` to force Backtesting to update the data it has. 
**Use it only if you want to update your data set. You will not be able
to come back to the previous version.**

To test your strategy with latest data, we recommend continuing using 
the parameter `-l` or `--live`.


## Hyperopt commands

To optimize your strategy, you can use hyperopt parameter hyperoptimization
to find optimal parameter values for your stategy.

```
usage: main.py hyperopt [-h] [-i TICKER_INTERVAL] [--enable-position-tracking]
                        [--timerange TIMERANGE] [-e INT]
                        [-s {all,buy,roi,stoploss} [{all,buy,roi,stoploss} ...]]

optional arguments:
  -h, --help            show this help message and exit
  -i TICKER_INTERVAL, --ticker-interval TICKER_INTERVAL
                        specify ticker interval (1m, 5m, 30m, 1h, 1d)
  --enable-position-tracking
                        Disables buying the same pair multiple times to 
                        simulate real world limitations
  --timerange TIMERANGE specify what timerange of data to use.
  -e INT, --epochs INT  specify number of epochs (default: 100)
  -s {all,buy,roi,stoploss} [{all,buy,roi,stoploss} ...], --spaces {all,buy,roi,stoploss} [{all,buy,roi,stoploss} ...]
                        Specify which parameters to hyperopt. Space separate
                        list. Default: all
```

## A parameter missing in the configuration?
All parameters for `main.py`, `backtesting`, `hyperopt` are referenced
in [misc.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/misc.py#L84)

## Next step
The optimal strategy of the bot will change with time depending of the
market trends. The next step is to 
[optimize your bot](https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md).
