# Utility Subcommands

Besides the Live-Trade and Dry-Run run modes, the `backtesting` and `hyperopt` optimization subcommands, and the `download-data` subcommand which prepares historical data, the bot contains a number of utility subcommands. They are described in this section.

## Create userdir

Creates the directory structure to hold your files for freqtrade.
Will also create strategy and hyperopt examples for you to get started.
Can be used multiple times - using `--reset` will reset the sample strategy and hyperopt files to their default state.

--8<-- "commands/create-userdir.md"

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

--8<-- "commands/new-config.md"

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

## Show config

Show configuration file (with sensitive values redacted by default).
Especially useful with [split configuration files](configuration.md#multiple-configuration-files) or [environment variables](configuration.md#environment-variables), where this command will show the merged configuration.

![Show config output](assets/show-config-output.png)

--8<-- "commands/show-config.md"

``` output
Your combined configuration is:
{
  "exit_pricing": {
    "price_side": "other",
    "use_order_book": true,
    "order_book_top": 1
  },
  "stake_currency": "USDT",
  "exchange": {
    "name": "binance",
    "key": "REDACTED",
    "secret": "REDACTED",
    "ccxt_config": {},
    "ccxt_async_config": {},
  }
  // ...
}
```

!!! Warning "Sharing information provided by this command"
    We try to remove all known sensitive information from the default output (without `--show-sensitive`). 
    Yet, please do double-check for sensitive values in your output to make sure you're not accidentally exposing some private info.

## Create new strategy

Creates a new strategy from a template similar to SampleStrategy.
The file will be named inline with your class name, and will not overwrite existing files.

Results will be located in `user_data/strategies/<strategyclassname>.py`.

--8<-- "commands/new-strategy.md"

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

--8<-- "commands/list-strategies.md"

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

## List Hyperopt-Loss functions

Use the `list-hyperoptloss` subcommand to see all hyperopt loss functions available.

It provides a quick list of all available loss functions in your environment.

This subcommand can be useful for finding problems in your environment with loading loss functions: modules with Hyperopt-Loss functions that contain errors and failed to load are printed in red (LOAD FAILED), while hyperopt-Loss functions with duplicate names are printed in yellow (DUPLICATE NAME).

--8<-- "commands/list-hyperoptloss.md"

## List freqAI models

Use the `list-freqaimodels` subcommand to see all freqAI models available.

This subcommand is useful for finding problems in your environment with loading freqAI models: modules with models that contain errors and failed to load are printed in red (LOAD FAILED), while models with duplicate names are printed in yellow (DUPLICATE NAME).

--8<-- "commands/list-freqaimodels.md"

## List Exchanges

Use the `list-exchanges` subcommand to see the exchanges available for the bot.

--8<-- "commands/list-exchanges.md"

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

--8<-- "commands/list-timeframes.md"

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

--8<-- "commands/list-pairs.md"

By default, only active pairs/markets are shown. Active pairs/markets are those that can currently be traded on the exchange.
You can use the `-a`/`-all` option to see the list of all pairs/markets, including the inactive ones.
Pairs may be listed as untradeable if the smallest tradeable price for the market is very small, i.e. less than `1e-11` (`0.00000000001`)

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

--8<-- "commands/test-pairlist.md"

### Examples

Show whitelist when using a [dynamic pairlist](plugins.md#pairlists).

```
freqtrade test-pairlist --config config.json --quote USDT BTC
```

## Convert database

`freqtrade convert-db` can be used to convert your database from one system to another (sqlite -> postgres, postgres -> other postgres), migrating all trades, orders and Pairlocks.

Please refer to the [corresponding documentation](advanced-setup.md#use-a-different-database-system) to learn about requirements for different database systems.

--8<-- "commands/convert-db.md"

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

--8<-- "commands/webserver.md"

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

--8<-- "commands/backtesting-show.md"

## Detailed backtest analysis

Advanced backtest result analysis.

More details in the [Backtesting analysis](advanced-backtesting.md#analyze-the-buyentry-and-sellexit-tags) Section.

--8<-- "commands/backtesting-analysis.md"

## List Hyperopt results

You can list the hyperoptimization epochs the Hyperopt module evaluated previously with the `hyperopt-list` sub-command.

--8<-- "commands/hyperopt-list.md"

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

--8<-- "commands/hyperopt-show.md"

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

--8<-- "commands/show-trades.md"

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

--8<-- "commands/strategy-updater.md"
