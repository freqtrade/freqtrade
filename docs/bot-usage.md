# Bot usage
This page explains the difference parameters of the bot and how to run 
it.

## Table of Contents
- [Bot commands](#bot-commands)
- [Backtesting commands](#backtesting-commands)
- [Hyperopt commands](#hyperopt-commands)

## Bot commands
```
usage: main.py [-h] [-c PATH] [-v] [--version] [--dynamic-whitelist [INT]]
               [--dry-run-db]
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
  -s PATH, --strategy PATH
                        specify strategy file (default:
                        freqtrade/strategy/default_strategy.py)
  --dry-run-db          Force dry run to use a local DB
                        "tradesv3.dry_run.sqlite" instead of memory DB. Work
                        only if dry_run is enabled.
  -dd PATH, --datadir PATH
                        path to backtest data (default freqdata/tests/testdata
  --dynamic-whitelist [INT]
                        dynamically generate and update whitelist based on 24h
                        BaseVolume (Default 20 currencies)
```

### How to use a different config file?
The bot allows you to select which config file you want to use. Per 
default, the bot will load the file `./config.json`

```bash
python3 ./freqtrade/main.py -c path/far/far/away/config.json 
```

### How to use --strategy?
This parameter will allow you to load your custom strategy file. Per 
default without `--strategy` or `-s` the bol will load the 
`default_strategy` included with the bot (`freqtrade/strategy/default_strategy.py`). 

The bot will search your strategy file into `user_data/strategies` and 
`freqtrade/strategy`.

To load a strategy, simply pass the file name (without .py) in this 
parameters.

**Example:**  
In `user_data/strategies` you have a file `my_awesome_strategy.py` to 
load it:  
```bash
python3 ./freqtrade/main.py --strategy my_awesome_strategy
```

If the bot does not find your strategy file, it will fallback to the 
`default_strategy`.

Learn more about strategy file in [optimize your bot](https://github.com/gcarq/freqtrade/blob/develop/docs/bot-optimization.md).

#### How to install a strategy?
This is very simple. Copy paste your strategy file into the folder 
`user_data/strategies`. And voila, the bot is ready to use it.

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

### How to use --dry-run-db?
When you run the bot in Dry-run mode, per default no transactions are 
stored in a database. If you want to store your bot actions in a DB 
using `--dry-run-db`. This command will use a separate database file 
`tradesv3.dry_run.sqlite`

```bash
python3 ./freqtrade/main.py -c config.json --dry-run-db
```


## Backtesting commands

Backtesting also uses the config specified via `-c/--config`.

```
usage: freqtrade backtesting [-h] [-l] [-i INT] [--realistic-simulation]
                             [-r]

optional arguments:
  -h, --help            show this help message and exit
  -l, --live            using live data
  -i INT, --ticker-interval INT
                        specify ticker interval in minutes (default: 5)
  --realistic-simulation
                        uses max_open_trades from config to simulate real
                        world limitations
  -r, --refresh-pairs-cached
                        refresh the pairs files in tests/testdata with 
                        the latest data from Bittrex. Use it if you want
                        to run your backtesting with up-to-date data.
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

It is possible to use hyperopt for trading strategy optimization.
Hyperopt uses an internal json config return by `hyperopt_optimize_conf()` 
located in `freqtrade/optimize/hyperopt_conf.py`.

```
usage: freqtrade hyperopt [-h] [-e INT] [--use-mongodb]

optional arguments:
  -h, --help            show this help message and exit
  -e INT, --epochs INT  specify number of epochs (default: 100)
  --use-mongodb         parallelize evaluations with mongodb (requires mongod
                        in PATH)

```

## A parameter missing in the configuration?
All parameters for `main.py`, `backtesting`, `hyperopt` are referenced
in [misc.py](https://github.com/gcarq/freqtrade/blob/develop/freqtrade/misc.py#L84)

## Next step
The optimal strategy of the bot will change with time depending of the
market trends. The next step is to 
[optimize your bot](https://github.com/gcarq/freqtrade/blob/develop/docs/bot-optimization.md).
