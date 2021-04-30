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
│   ├── sample_hyperopt_advanced.py
│   ├── sample_hyperopt_loss.py
│   └── sample_hyperopt.py
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
$ freqtrade new-config --config config_binance.json

? Do you want to enable Dry-run (simulated trades)?  Yes
? Please insert your stake currency: BTC
? Please insert your stake amount: 0.05
? Please insert max_open_trades (Integer or 'unlimited'): 3
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

## Create new hyperopt

Creates a new hyperopt from a template similar to SampleHyperopt.
The file will be named inline with your class name, and will not overwrite existing files.

Results will be located in `user_data/hyperopts/<classname>.py`.

``` output
usage: freqtrade new-hyperopt [-h] [--userdir PATH] [--hyperopt NAME]
                              [--template {full,minimal,advanced}]

optional arguments:
  -h, --help            show this help message and exit
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
  --hyperopt NAME       Specify hyperopt class name which will be used by the
                        bot.
  --template {full,minimal,advanced}
                        Use a template which is either `minimal`, `full`
                        (containing multiple sample indicators) or `advanced`.
                        Default: `full`.
```

### Sample usage of new-hyperopt

```bash
freqtrade new-hyperopt --hyperopt AwesomeHyperopt
```

With custom user directory

```bash
freqtrade new-hyperopt --userdir ~/.freqtrade/ --hyperopt AwesomeHyperopt
```

## List Strategies and List Hyperopts

Use the `list-strategies` subcommand to see all strategies in one particular directory and the `list-hyperopts` subcommand to list custom Hyperopts.

These subcommands are useful for finding problems in your environment with loading strategies or hyperopt classes: modules with strategies or hyperopt classes that contain errors and failed to load are printed in red (LOAD FAILED), while strategies or hyperopt classes with duplicate names are printed in yellow (DUPLICATE NAME).

```
usage: freqtrade list-strategies [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                 [-d PATH] [--userdir PATH]
                                 [--strategy-path PATH] [-1] [--no-color]

optional arguments:
  -h, --help            show this help message and exit
  --strategy-path PATH  Specify additional strategy lookup path.
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
                        Specify configuration file (default: `config.json`).
                        Multiple --config options may be used. Can be set to
                        `-` to read config from stdin.
  -d PATH, --datadir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
```
```
usage: freqtrade list-hyperopts [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                                [-d PATH] [--userdir PATH]
                                [--hyperopt-path PATH] [-1] [--no-color]

optional arguments:
  -h, --help            show this help message and exit
  --hyperopt-path PATH  Specify additional lookup path for Hyperopt and
                        Hyperopt Loss functions.
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
                        Specify configuration file (default: `config.json`).
                        Multiple --config options may be used. Can be set to
                        `-` to read config from stdin.
  -d PATH, --datadir PATH
                        Path to directory with historical backtesting data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
```

!!! Warning
    Using these commands will try to load all python files from a directory. This can be a security risk if untrusted files reside in this directory, since all module-level code is executed.

Example: Search default strategies and hyperopts directories (within the default userdir).

``` bash
freqtrade list-strategies
freqtrade list-hyperopts
```

Example: Search strategies and hyperopts directory within the userdir.

``` bash
freqtrade list-strategies --userdir ~/.freqtrade/
freqtrade list-hyperopts --userdir ~/.freqtrade/
```

Example: Search dedicated strategy path.

``` bash
freqtrade list-strategies --strategy-path ~/.freqtrade/strategies/
```

Example: Search dedicated hyperopt path.

``` bash
freqtrade list-hyperopt --hyperopt-path ~/.freqtrade/hyperopts/
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

* Example: see exchanges available for the bot:
```
$ freqtrade list-exchanges
Exchanges available for Freqtrade:
Exchange name    Valid    reason
---------------  -------  --------------------------------------------
aax              True
ascendex         True     missing opt: fetchMyTrades
bequant          True
bibox            True
bigone           True
binance          True
binanceus        True
bitbank          True     missing opt: fetchTickers
bitcoincom       True
bitfinex         True
bitforex         True     missing opt: fetchMyTrades, fetchTickers
bitget           True
bithumb          True     missing opt: fetchMyTrades
bitkk            True     missing opt: fetchMyTrades
bitmart          True
bitmax           True     missing opt: fetchMyTrades
bitpanda         True
bittrex          True
bitvavo          True
bitz             True     missing opt: fetchMyTrades
btcalpha         True     missing opt: fetchTicker, fetchTickers
btcmarkets       True     missing opt: fetchTickers
buda             True     missing opt: fetchMyTrades, fetchTickers
bw               True     missing opt: fetchMyTrades, fetchL2OrderBook
bybit            True
bytetrade        True
cdax             True
cex              True     missing opt: fetchMyTrades
coinbaseprime    True     missing opt: fetchTickers
coinbasepro      True     missing opt: fetchTickers
coinex           True
crex24           True
deribit          True
digifinex        True
equos            True     missing opt: fetchTicker, fetchTickers
eterbase         True
fcoin            True     missing opt: fetchMyTrades, fetchTickers
fcoinjp          True     missing opt: fetchMyTrades, fetchTickers
ftx              True
gateio           True
gemini           True
gopax            True
hbtc             True
hitbtc           True
huobijp          True
huobipro         True
idex             True
kraken           True
kucoin           True
lbank            True     missing opt: fetchMyTrades
mercado          True     missing opt: fetchTickers
ndax             True     missing opt: fetchTickers
novadax          True
okcoin           True
okex             True
probit           True
qtrade           True
stex             True
timex            True
upbit            True     missing opt: fetchMyTrades
vcc              True
zb               True     missing opt: fetchMyTrades

```

!!! Note "missing opt exchanges"
    Values with "missing opt:" might need special configuration (e.g. using orderbook if `fetchTickers` is missing) - but should in theory work (although we cannot guarantee they will).

* Example: see all exchanges supported by the ccxt library (including 'bad' ones, i.e. those that are known to not work with Freqtrade):
```
$ freqtrade list-exchanges -a
All exchanges supported by the ccxt library:
Exchange name       Valid    reason
------------------  -------  ---------------------------------------------------------------------------------------
aax                 True
aofex               False    missing: fetchOrder
ascendex            True     missing opt: fetchMyTrades
bequant             True
bibox               True
bigone              True
binance             True
binanceus           True
bit2c               False    missing: fetchOrder, fetchOHLCV
bitbank             True     missing opt: fetchTickers
bitbay              False    missing: fetchOrder
bitcoincom          True
bitfinex            True
bitfinex2           False    missing: fetchOrder
bitflyer            False    missing: fetchOrder, fetchOHLCV
bitforex            True     missing opt: fetchMyTrades, fetchTickers
bitget              True
bithumb             True     missing opt: fetchMyTrades
bitkk               True     missing opt: fetchMyTrades
bitmart             True
bitmax              True     missing opt: fetchMyTrades
bitmex              False    Various reasons.
bitpanda            True
bitso               False    missing: fetchOHLCV
bitstamp            False    Does not provide history. Details in https://github.com/freqtrade/freqtrade/issues/1983
bitstamp1           False    missing: fetchOrder, fetchOHLCV
bittrex             True
bitvavo             True
bitz                True     missing opt: fetchMyTrades
bl3p                False    missing: fetchOrder, fetchOHLCV
bleutrade           False    missing: fetchOrder
braziliex           False    missing: fetchOHLCV
btcalpha            True     missing opt: fetchTicker, fetchTickers
btcbox              False    missing: fetchOHLCV
btcmarkets          True     missing opt: fetchTickers
btctradeua          False    missing: fetchOrder, fetchOHLCV
btcturk             False    missing: fetchOrder
buda                True     missing opt: fetchMyTrades, fetchTickers
bw                  True     missing opt: fetchMyTrades, fetchL2OrderBook
bybit               True
bytetrade           True
cdax                True
cex                 True     missing opt: fetchMyTrades
chilebit            False    missing: fetchOrder, fetchOHLCV
coinbase            False    missing: fetchOrder, cancelOrder, createOrder, fetchOHLCV
coinbaseprime       True     missing opt: fetchTickers
coinbasepro         True     missing opt: fetchTickers
coincheck           False    missing: fetchOrder, fetchOHLCV
coinegg             False    missing: fetchOHLCV
coinex              True
coinfalcon          False    missing: fetchOHLCV
coinfloor           False    missing: fetchOrder, fetchOHLCV
coingi              False    missing: fetchOrder, fetchOHLCV
coinmarketcap       False    missing: fetchOrder, cancelOrder, createOrder, fetchBalance, fetchOHLCV
coinmate            False    missing: fetchOHLCV
coinone             False    missing: fetchOHLCV
coinspot            False    missing: fetchOrder, cancelOrder, fetchOHLCV
crex24              True
currencycom         False    missing: fetchOrder
delta               False    missing: fetchOrder
deribit             True
digifinex           True
equos               True     missing opt: fetchTicker, fetchTickers
eterbase            True
exmo                False    missing: fetchOrder
exx                 False    missing: fetchOHLCV
fcoin               True     missing opt: fetchMyTrades, fetchTickers
fcoinjp             True     missing opt: fetchMyTrades, fetchTickers
flowbtc             False    missing: fetchOrder, fetchOHLCV
foxbit              False    missing: fetchOrder, fetchOHLCV
ftx                 True
gateio              True
gemini              True
gopax               True
hbtc                True
hitbtc              True
hollaex             False    missing: fetchOrder
huobijp             True
huobipro            True
idex                True
independentreserve  False    missing: fetchOHLCV
indodax             False    missing: fetchOHLCV
itbit               False    missing: fetchOHLCV
kraken              True
kucoin              True
kuna                False    missing: fetchOHLCV
lakebtc             False    missing: fetchOrder, fetchOHLCV
latoken             False    missing: fetchOrder, fetchOHLCV
lbank               True     missing opt: fetchMyTrades
liquid              False    missing: fetchOHLCV
luno                False    missing: fetchOHLCV
lykke               False    missing: fetchOHLCV
mercado             True     missing opt: fetchTickers
mixcoins            False    missing: fetchOrder, fetchOHLCV
ndax                True     missing opt: fetchTickers
novadax             True
oceanex             False    missing: fetchOHLCV
okcoin              True
okex                True
paymium             False    missing: fetchOrder, fetchOHLCV
phemex              False    Does not provide history.
poloniex            False    missing: fetchOrder
probit              True
qtrade              True
rightbtc            False    missing: fetchOrder
ripio               False    missing: fetchOHLCV
southxchange        False    missing: fetchOrder, fetchOHLCV
stex                True
surbitcoin          False    missing: fetchOrder, fetchOHLCV
therock             False    missing: fetchOHLCV
tidebit             False    missing: fetchOrder
tidex               False    missing: fetchOHLCV
timex               True
upbit               True     missing opt: fetchMyTrades
vbtc                False    missing: fetchOrder, fetchOHLCV
vcc                 True
wavesexchange       False    missing: fetchOrder
whitebit            False    missing: fetchOrder, cancelOrder, createOrder, fetchBalance
xbtce               False    missing: fetchOrder, fetchOHLCV
xena                False    missing: fetchOrder
yobit               False    missing: fetchOHLCV
zaif                False    missing: fetchOrder, fetchOHLCV
zb                  True     missing opt: fetchMyTrades
```

## List Timeframes

Use the `list-timeframes` subcommand to see the list of timeframes available for the exchange.

```
usage: freqtrade list-timeframes [-h] [-v] [--logfile FILE] [-V] [-c PATH] [-d PATH] [--userdir PATH] [--exchange EXCHANGE] [-1]

optional arguments:
  -h, --help            show this help message and exit
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no config is provided.
  -1, --one-column      Print output in one column.

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
                              [-a]

usage: freqtrade list-pairs [-h] [-v] [--logfile FILE] [-V] [-c PATH]
                            [-d PATH] [--userdir PATH] [--exchange EXCHANGE]
                            [--print-list] [--print-json] [-1] [--print-csv]
                            [--base BASE_CURRENCY [BASE_CURRENCY ...]]
                            [--quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]] [-a]

optional arguments:
  -h, --help            show this help message and exit
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no
                        config is provided.
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

By default, only active pairs/markets are shown. Active pairs/markets are those that can currently be traded
on the exchange. The see the list of all pairs/markets (not only the active ones), use the `-a`/`-all` option.

Pairs/markets are sorted by its symbol string in the printed output.

### Examples

* Print the list of active pairs with quote currency USD on exchange, specified in the default
configuration file (i.e. pairs on the "Bittrex" exchange) in JSON format:

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
usage: freqtrade test-pairlist [-h] [-c PATH]
                               [--quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]]
                               [-1] [--print-json]

optional arguments:
  -h, --help            show this help message and exit
  -c PATH, --config PATH
                        Specify configuration file (default: `config.json`).
                        Multiple --config options may be used. Can be set to
                        `-` to read config from stdin.
  --quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]
                        Specify quote currency(-ies). Space-separated list.
  -1, --one-column      Print output in one column.
  --print-json          Print list of pairs or market symbols in JSON format.
```

### Examples

Show whitelist when using a [dynamic pairlist](plugins.md#pairlists).

```
freqtrade test-pairlist --config config.json --quote USDT BTC
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
                               [--hyperopt-filename PATH] [--no-header]

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
