# Utility Subcommands

Besides the Live-Trade and Dry-Run run modes, the `backtesting`, `edge` and `hyperopt` optimization subcommands, and the `download-data` subcommand which prepares historical data, the bot contains a number of utility subcommands. They are described in this section.

## Create userdir

Creates the directory structure to hold your files for freqtrade.
Will also create strategy and hyperopt examples for you to get started.
Can be used multiple times - using `--reset` will reset the sample strategy and hyperopt files to their default state. 

```
usage: freqtrade create-userdir [-h] [--userdir PATH] [--reset]

optional arguments:
  -h, --help            show this help message and exit
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
  --reset               Reset sample files to their original state.
```

!!! Warning
    Using `--reset` may result in loss of data, since this will overwrite all sample files without asking again.

```
├── backtest_results
├── data
├── hyperopt_results
├── hyperopts
│   ├── sample_hyperopt_loss.py
├── notebooks
│   └── strategy_analysis_example.ipynb
├── plot
└── strategies
    └── sample_strategy.py
```

## Create new config

Creates a new configuration file, asking some questions which are important selections for a configuration.

```
usage: freqtrade new-config [-h] [-c PATH]

optional arguments:
  -h, --help            show this help message and exit
  -c PATH, --config PATH
                        Specify configuration file (default: `config.json`). Multiple --config options may be used. Can be set to `-`
                        to read config from stdin.
```

!!! Warning
    Only vital questions are asked. Freqtrade offers a lot more configuration possibilities, which are listed in the [Configuration documentation](configuration.md#configuration-parameters)

### Create config examples

```
$ freqtrade new-config --config user_data/config_binance.json

? Do you want to enable Dry-run (simulated trades)?  Yes
? Please insert your stake currency: BTC
? Please insert your stake amount: 0.05
? Please insert max_open_trades (Integer or -1 for unlimited open trades): 3
? Please insert your desired timeframe (e.g. 5m): 5m
? Please insert your display Currency (for reporting): USD
? Select exchange  binance
? Do you want to enable Telegram?  No
```

## Create new strategy

Creates a new strategy from a template similar to SampleStrategy.
The file will be named inline with your class name, and will not overwrite existing files.

Results will be located in `user_data/strategies/<strategyclassname>.py`.

``` output
usage: freqtrade new-strategy [-h] [--userdir PATH] [-s NAME]
                              [--template {full,minimal,advanced}]

optional arguments:
  -h, --help            show this help message and exit
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
  -s NAME, --strategy NAME
                        Specify strategy class name which will be used by the
                        bot.
  --template {full,minimal,advanced}
                        Use a template which is either `minimal`, `full`
                        (containing multiple sample indicators) or `advanced`.
                        Default: `full`.

```

### Sample usage of new-strategy

```bash
freqtrade new-strategy --strategy AwesomeStrategy
```

With custom user directory

```bash
freqtrade new-strategy --userdir ~/.freqtrade/ --strategy AwesomeStrategy
```

Using the advanced template (populates all optional functions and methods)

```bash
freqtrade new-strategy --strategy AwesomeStrategy --template advanced
```

## List Strategies

Use the `list-strategies` subcommand to see all strategies in one particular directory.

This subcommand is useful for finding problems in your environment with loading strategies: modules with strategies that contain errors and failed to load are printed in red (LOAD FAILED), while strategies with duplicate names are printed in yellow (DUPLICATE NAME).

```
usage: freqtrade list-strategies [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                 [-d PATH] [--userdir PATH]
                                 [--strategy-path PATH] [-1] [--no-color]
                                 [--recursive-strategy-search]

optional arguments:
  -h, --help            show this help message and exit
  --strategy-path PATH  Specify additional strategy lookup path.
  -1, --one-column      Print output in one column.
  --no-color            Disable colorization of hyperopt results. May be
                        useful if you are redirecting output to a file.
  --recursive-strategy-search
                        Recursively search for a strategy in the strategies
                        folder.

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

!!! Warning
    Using these commands will try to load all python files from a directory. This can be a security risk if untrusted files reside in this directory, since all module-level code is executed.

Example: Search default strategies directories (within the default userdir).

``` bash
freqtrade list-strategies
```

Example: Search strategies  directory within the userdir.

``` bash
freqtrade list-strategies --userdir ~/.freqtrade/
```

Example: Search dedicated strategy path.

``` bash
freqtrade list-strategies --strategy-path ~/.freqtrade/strategies/
```

## List freqAI models

Use the `list-freqaimodels` subcommand to see all freqAI models available.

This subcommand is useful for finding problems in your environment with loading freqAI models: modules with models that contain errors and failed to load are printed in red (LOAD FAILED), while models with duplicate names are printed in yellow (DUPLICATE NAME).

```
usage: freqtrade list-freqaimodels [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                   [-d PATH] [--userdir PATH]
                                   [--freqaimodel-path PATH] [-1] [--no-color]

optional arguments:
  -h, --help            show this help message and exit
  --freqaimodel-path PATH
                        Specify additional lookup path for freqaimodels.
  -1, --one-column      Print output in one column.
  --no-color            Disable colorization of hyperopt results. May be
                        useful if you are redirecting output to a file.

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

## List Exchanges

Use the `list-exchanges` subcommand to see the exchanges available for the bot.

```
usage: freqtrade list-exchanges [-h] [-1] [-a]

optional arguments:
  -h, --help        show this help message and exit
  -1, --one-column  Print output in one column.
  -a, --all         Print all exchanges known to the ccxt library.
```

Example: see exchanges available for the bot:

```
$ freqtrade list-exchanges
Exchanges available for Freqtrade:
Exchange name       Supported    Markets                 Reason
------------------  -----------  ----------------------  ------------------------------------------------------------------------
binance             Official     spot, isolated futures
bitmart             Official     spot
bybit                            spot, isolated futures
gate                Official     spot, isolated futures
htx                 Official     spot
huobi                            spot
kraken              Official     spot
okx                 Official     spot, isolated futures
```

!!! info ""
    Output reduced for clarity - supported and available exchanges may change over time.

!!! Note "missing opt exchanges"
    Values with "missing opt:" might need special configuration (e.g. using orderbook if `fetchTickers` is missing) - but should in theory work (although we cannot guarantee they will).

Example: see all exchanges supported by the ccxt library (including 'bad' ones, i.e. those that are known to not work with Freqtrade)

```
$ freqtrade list-exchanges -a
All exchanges supported by the ccxt library:
Exchange name       Valid    Supported    Markets                 Reason
------------------  -------  -----------  ----------------------  ---------------------------------------------------------------------------------
binance             True     Official     spot, isolated futures
bitflyer            False                 spot                    missing: fetchOrder. missing opt: fetchTickers.
bitmart             True     Official     spot
bybit               True                  spot, isolated futures
gate                True     Official     spot, isolated futures
htx                 True     Official     spot
kraken              True     Official     spot
okx                 True     Official     spot, isolated futures
```

!!! info ""
    Reduced output - supported and available exchanges may change over time.

## List Timeframes

Use the `list-timeframes` subcommand to see the list of timeframes available for the exchange.

```
usage: freqtrade list-timeframes [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                 [-d PATH] [--userdir PATH]
                                 [--exchange EXCHANGE] [-1]

options:
  -h, --help            show this help message and exit
  --exchange EXCHANGE   Exchange name. Only valid if no config is provided.
  -1, --one-column      Print output in one column.

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --logfile FILE, --log-file FILE
                        Log to the file specified. Special values are:
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

* Example: see the timeframes for the 'binance' exchange, set in the configuration file:

```
$ freqtrade list-timeframes -c config_binance.json
...
Timeframes available for the exchange `binance`: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
```

* Example: enumerate exchanges available for Freqtrade and print timeframes supported by each of them:
```
$ for i in `freqtrade list-exchanges -1`; do freqtrade list-timeframes --exchange $i; done
```

## List pairs/list markets

The `list-pairs` and `list-markets` subcommands allow to see the pairs/markets available on exchange.

Pairs are markets with the '/' character between the base currency part and the quote currency part in the market symbol.
For example, in the 'ETH/BTC' pair 'ETH' is the base currency, while 'BTC' is the quote currency.

For pairs traded by Freqtrade the pair quote currency is defined by the value of the `stake_currency` configuration setting.

You can print info about any pair/market with these subcommands - and you can filter output by quote-currency using `--quote BTC`, or by base-currency using `--base ETH` options correspondingly.

These subcommands have same usage and same set of available options:

```
usage: freqtrade list-markets [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                              [-d PATH] [--userdir PATH] [--exchange EXCHANGE]
                              [--print-list] [--print-json] [-1] [--print-csv]
                              [--base BASE_CURRENCY [BASE_CURRENCY ...]]
                              [--quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]]
                              [-a] [--trading-mode {spot,margin,futures}]
usage: freqtrade list-pairs [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                            [-d PATH] [--userdir PATH] [--exchange EXCHANGE]
                            [--print-list] [--print-json] [-1] [--print-csv]
                            [--base BASE_CURRENCY [BASE_CURRENCY ...]]
                            [--quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]] [-a]
                            [--trading-mode {spot,margin,futures}]
options:
  -h, --help            show this help message and exit
  --exchange EXCHANGE   Exchange name. Only valid if no config is provided.
  --print-list          Print list of pairs or market symbols. By default data
                        is printed in the tabular format.
  --print-json          Print list of pairs or market symbols in JSON format.
  -1, --one-column      Print output in one column.
  --print-csv           Print exchange pair or market data in the csv format.
  --base BASE_CURRENCY [BASE_CURRENCY ...]
                        Specify base currency(-ies). Space-separated list.
  --quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]
                        Specify quote currency(-ies). Space-separated list.
  -a, --all             Print all pairs or market symbols. By default only
                        active ones are shown.
  --trading-mode {spot,margin,futures}, --tradingmode {spot,margin,futures}
                        Select Trading mode

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --logfile FILE, --log-file FILE
                        Log to the file specified. Special values are:
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

By default, only active pairs/markets are shown. Active pairs/markets are those that can currently be traded
on the exchange. The see the list of all pairs/markets (not only the active ones), use the `-a`/`-all` option.

Pairs/markets are sorted by its symbol string in the printed output.

### Examples

* Print the list of active pairs with quote currency USD on exchange, specified in the default
configuration file (i.e. pairs on the "Binance" exchange) in JSON format:

```
$ freqtrade list-pairs --quote USD --print-json
```

* Print the list of all pairs on the exchange, specified in the `config_binance.json` configuration file
(i.e. on the "Binance" exchange) with base currencies BTC or ETH and quote currencies USDT or USD, as the
human-readable list with summary:

```
$ freqtrade list-pairs -c config_binance.json --all --base BTC ETH --quote USDT USD --print-list
```

* Print all markets on exchange "Kraken", in the tabular format:

```
$ freqtrade list-markets --exchange kraken --all
```

## Test pairlist

Use the `test-pairlist` subcommand to test the configuration of [dynamic pairlists](plugins.md#pairlists).

Requires a configuration with specified `pairlists` attribute.
Can be used to generate static pairlists to be used during backtesting / hyperopt.

```
usage: freqtrade test-pairlist [-h] [--userdir PATH] [-v] [-c PATH]
                               [--quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]]
                               [-1] [--print-json] [--exchange EXCHANGE]

options:
  -h, --help            show this help message and exit
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  -c PATH, --config PATH
                        Specify configuration file (default:
                        `userdir/config.json` or `config.json` whichever
                        exists). Multiple --config options may be used. Can be
                        set to `-` to read config from stdin.
  --quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]
                        Specify quote currency(-ies). Space-separated list.
  -1, --one-column      Print output in one column.
  --print-json          Print list of pairs or market symbols in JSON format.
  --exchange EXCHANGE   Exchange name. Only valid if no config is provided.

```

### Examples

Show whitelist when using a [dynamic pairlist](plugins.md#pairlists).

```
freqtrade test-pairlist --config config.json --quote USDT BTC
```

## Convert database

`freqtrade convert-db` can be used to convert your database from one system to another (sqlite -> postgres, postgres -> other postgres), migrating all trades, orders and Pairlocks.

Please refer to the [SQL cheatsheet](sql_cheatsheet.md#use-a-different-database-system) to learn about requirements for different database systems.

```
usage: freqtrade convert-db [-h] [--db-url PATH] [--db-url-from PATH]

optional arguments:
  -h, --help          show this help message and exit
  --db-url PATH       Override trades database URL, this is useful in custom
                      deployments (default: `sqlite:///tradesv3.sqlite` for
                      Live Run mode, `sqlite:///tradesv3.dryrun.sqlite` for
                      Dry Run).
  --db-url-from PATH  Source db url to use when migrating a database.
```

!!! Warning
    Please ensure to only use this on an empty target database. Freqtrade will perform a regular migration, but may fail if entries already existed.

## Webserver mode

!!! Warning "Experimental"
    Webserver mode is an experimental mode to increase backesting and strategy development productivity.
    There may still be bugs - so if you happen to stumble across these, please report them as github issues, thanks.

Run freqtrade in webserver mode.
Freqtrade will start the webserver and allow FreqUI to start and control backtesting processes.
This has the advantage that data will not be reloaded between backtesting runs (as long as timeframe and timerange remain identical).
FreqUI will also show the backtesting results.

```
usage: freqtrade webserver [-h] [-v] [--logfile FILE] [-V] [-c PATH] [-d PATH]
                           [--userdir PATH]

optional arguments:
  -h, --help            show this help message and exit

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

### Webserver mode - docker

You can also use webserver mode via docker.
Starting a one-off container requires the configuration of the port explicitly, as ports are not exposed by default.
You can use `docker compose run --rm -p 127.0.0.1:8080:8080 freqtrade webserver` to start a one-off container that'll be removed once you stop it. This assumes that port 8080 is still available and no other bot is running on that port.

Alternatively, you can reconfigure the docker-compose file to have the command updated:

``` yml
    command: >
      webserver
      --config /freqtrade/user_data/config.json
```

You can now use `docker compose up` to start the webserver.
This assumes that the configuration has a webserver enabled and configured for docker (listening port = `0.0.0.0`).

!!! Tip
    Don't forget to reset the command back to the trade command if you want to start a live or dry-run bot. 

## Show previous Backtest results

Allows you to show previous backtest results.
Adding `--show-pair-list` outputs a sorted pair list you can easily copy/paste into your configuration (omitting bad pairs).

??? Warning "Strategy overfitting"
    Only using winning pairs can lead to an overfitted strategy, which will not work well on future data. Make sure to extensively test your strategy in dry-run before risking real money.

```
usage: freqtrade backtesting-show [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                  [-d PATH] [--userdir PATH]
                                  [--export-filename PATH] [--show-pair-list]

optional arguments:
  -h, --help            show this help message and exit
  --export-filename PATH
                        Save backtest results to the file with this filename.
                        Requires `--export` to be set as well. Example:
                        `--export-filename=user_data/backtest_results/backtest
                        _today.json`
  --show-pair-list      Show backtesting pairlist sorted by profit.

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

## Detailed backtest analysis

Advanced backtest result analysis.

More details in the [Backtesting analysis](advanced-backtesting.md#analyze-the-buyentry-and-sellexit-tags) Section.

```
usage: freqtrade backtesting-analysis [-h] [-v] [--logfile FILE] [-V]
                                      [-c PATH] [-d PATH] [--userdir PATH]
                                      [--export-filename PATH]
                                      [--analysis-groups {0,1,2,3,4} [{0,1,2,3,4} ...]]
                                      [--enter-reason-list ENTER_REASON_LIST [ENTER_REASON_LIST ...]]
                                      [--exit-reason-list EXIT_REASON_LIST [EXIT_REASON_LIST ...]]
                                      [--indicator-list INDICATOR_LIST [INDICATOR_LIST ...]]
                                      [--timerange YYYYMMDD-[YYYYMMDD]]
                                      [--rejected]
                                      [--analysis-to-csv]
                                      [--analysis-csv-path PATH]

optional arguments:
  -h, --help            show this help message and exit
  --export-filename PATH, --backtest-filename PATH
                        Use this filename for backtest results.Requires
                        `--export` to be set as well. Example: `--export-filen
                        ame=user_data/backtest_results/backtest_today.json`
  --analysis-groups {0,1,2,3,4} [{0,1,2,3,4} ...]
                        grouping output - 0: simple wins/losses by enter tag,
                        1: by enter_tag, 2: by enter_tag and exit_tag, 3: by
                        pair and enter_tag, 4: by pair, enter_ and exit_tag
                        (this can get quite large)
  --enter-reason-list ENTER_REASON_LIST [ENTER_REASON_LIST ...]
                        Space separated list of entry signals to analyse.
                        Default: all. e.g. 'entry_tag_a entry_tag_b'
  --exit-reason-list EXIT_REASON_LIST [EXIT_REASON_LIST ...]
                        Space separated list of exit signals to analyse.
                        Default: all. e.g.
                        'exit_tag_a roi stop_loss trailing_stop_loss'
  --indicator-list INDICATOR_LIST [INDICATOR_LIST ...]
                        Space separated list of indicators to analyse. e.g.
                        'close rsi bb_lowerband profit_abs'
  --timerange YYYYMMDD-[YYYYMMDD]
                        Timerange to filter trades for analysis, 
                        start inclusive, end exclusive. e.g.
                        20220101-20220201
  --rejected
                        Print out rejected trades table
  --analysis-to-csv
                        Write out tables to individual CSVs, by default to 
                        'user_data/backtest_results' unless '--analysis-csv-path' is given.
  --analysis-csv-path [PATH]
                        Optional path where individual CSVs will be written. If not used,
                        CSVs will be written to 'user_data/backtest_results'.

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

## List Hyperopt results

You can list the hyperoptimization epochs the Hyperopt module evaluated previously with the `hyperopt-list` sub-command.

```
usage: freqtrade hyperopt-list [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                               [-d PATH] [--userdir PATH] [--best]
                               [--profitable] [--min-trades INT]
                               [--max-trades INT] [--min-avg-time FLOAT]
                               [--max-avg-time FLOAT] [--min-avg-profit FLOAT]
                               [--max-avg-profit FLOAT]
                               [--min-total-profit FLOAT]
                               [--max-total-profit FLOAT]
                               [--min-objective FLOAT] [--max-objective FLOAT]
                               [--no-color] [--print-json] [--no-details]
                               [--hyperopt-filename PATH] [--export-csv FILE]

optional arguments:
  -h, --help            show this help message and exit
  --best                Select only best epochs.
  --profitable          Select only profitable epochs.
  --min-trades INT      Select epochs with more than INT trades.
  --max-trades INT      Select epochs with less than INT trades.
  --min-avg-time FLOAT  Select epochs above average time.
  --max-avg-time FLOAT  Select epochs below average time.
  --min-avg-profit FLOAT
                        Select epochs above average profit.
  --max-avg-profit FLOAT
                        Select epochs below average profit.
  --min-total-profit FLOAT
                        Select epochs above total profit.
  --max-total-profit FLOAT
                        Select epochs below total profit.
  --min-objective FLOAT
                        Select epochs above objective.
  --max-objective FLOAT
                        Select epochs below objective.
  --no-color            Disable colorization of hyperopt results. May be
                        useful if you are redirecting output to a file.
  --print-json          Print output in JSON format.
  --no-details          Do not print best epoch details.
  --hyperopt-filename FILENAME
                        Hyperopt result filename.Example: `--hyperopt-
                        filename=hyperopt_results_2020-09-27_16-20-48.pickle`
  --export-csv FILE     Export to CSV-File. This will disable table print.
                        Example: --export-csv hyperopt.csv

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

!!! Note
    `hyperopt-list` will automatically use the latest available hyperopt results file.
    You can override this using the `--hyperopt-filename` argument, and specify another, available filename (without path!).

### Examples

List all results, print details of the best result at the end:
```
freqtrade hyperopt-list
```

List only epochs with positive profit. Do not print the details of the best epoch, so that the list can be iterated in a script:
```
freqtrade hyperopt-list --profitable --no-details
```

## Show details of Hyperopt results

You can show the details of any hyperoptimization epoch previously evaluated by the Hyperopt module with the `hyperopt-show` subcommand.

```
usage: freqtrade hyperopt-show [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                               [-d PATH] [--userdir PATH] [--best]
                               [--profitable] [-n INT] [--print-json]
                               [--hyperopt-filename FILENAME] [--no-header]
                               [--disable-param-export]
                               [--breakdown {day,week,month} [{day,week,month} ...]]

optional arguments:
  -h, --help            show this help message and exit
  --best                Select only best epochs.
  --profitable          Select only profitable epochs.
  -n INT, --index INT   Specify the index of the epoch to print details for.
  --print-json          Print output in JSON format.
  --hyperopt-filename FILENAME
                        Hyperopt result filename.Example: `--hyperopt-
                        filename=hyperopt_results_2020-09-27_16-20-48.pickle`
  --no-header           Do not print epoch details header.
  --disable-param-export
                        Disable automatic hyperopt parameter export.
  --breakdown {day,week,month} [{day,week,month} ...]
                        Show backtesting breakdown per [day, week, month].

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

!!! Note
    `hyperopt-show` will automatically use the latest available hyperopt results file.
    You can override this using the `--hyperopt-filename` argument, and specify another, available filename (without path!).

### Examples

Print details for the epoch 168 (the number of the epoch is shown by the `hyperopt-list` subcommand or by Hyperopt itself during hyperoptimization run):

```
freqtrade hyperopt-show -n 168
```

Prints JSON data with details for the last best epoch (i.e., the best of all epochs):

```
freqtrade hyperopt-show --best -n -1 --print-json --no-header
```

## Show trades

Print selected (or all) trades from database to screen.

```
usage: freqtrade show-trades [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                             [-d PATH] [--userdir PATH] [--db-url PATH]
                             [--trade-ids TRADE_IDS [TRADE_IDS ...]]
                             [--print-json]

optional arguments:
  -h, --help            show this help message and exit
  --db-url PATH         Override trades database URL, this is useful in custom
                        deployments (default: `sqlite:///tradesv3.sqlite` for
                        Live Run mode, `sqlite:///tradesv3.dryrun.sqlite` for
                        Dry Run).
  --trade-ids TRADE_IDS [TRADE_IDS ...]
                        Specify the list of trade ids.
  --print-json          Print output in JSON format.

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

### Examples

Print trades with id 2 and 3 as json

``` bash
freqtrade show-trades --db-url sqlite:///tradesv3.sqlite --trade-ids 2 3 --print-json
```

## Strategy-Updater

Updates listed strategies or all strategies within the strategies folder to be v3 compliant.
If the command runs without --strategy-list then all strategies inside the strategies folder will be converted.
Your original strategy will remain available in the `user_data/strategies_orig_updater/` directory.

!!! Warning "Conversion results"
    Strategy updater will work on a "best effort" approach. Please do your due diligence and verify the results of the conversion.
    We also recommend to run a python formatter (e.g. `black`) to format results in a sane manner.

```
usage: freqtrade strategy-updater [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                  [-d PATH] [--userdir PATH]
                                  [--strategy-list STRATEGY_LIST [STRATEGY_LIST ...]]

options:
  -h, --help            show this help message and exit
  --strategy-list STRATEGY_LIST [STRATEGY_LIST ...]
                        Provide a space-separated list of strategies to
                        be converted.

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --logfile FILE, --log-file FILE
                        Log to the file specified. Special values are:
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
