# Deprecated features

This page contains description of the command line arguments, configuration parameters
and the bot features that were declared as DEPRECATED by the bot development team
and are no longer supported. Please avoid their usage in your configuration.

## Removed features

### the `--refresh-pairs-cached` command line option

`--refresh-pairs-cached` in the context of backtesting, hyperopt and edge allows to refresh candle data for backtesting.
Since this leads to much confusion, and slows down backtesting (while not being part of backtesting) this has been singled out as a separate freqtrade sub-command `freqtrade download-data`.

This command line option was deprecated in 2019.7-dev (develop branch) and removed in 2019.9.

### The **--dynamic-whitelist** command line option

This command line option was deprecated in 2018 and removed freqtrade 2019.6-dev (develop branch) and in freqtrade 2019.7.
Please refer to [pairlists](plugins.md#pairlists-and-pairlist-handlers) instead.

### the `--live` command line option

`--live` in the context of backtesting allowed to download the latest tick data for backtesting.
Did only download the latest 500 candles, so was ineffective in getting good backtest data.
Removed in 2019-7-dev (develop branch) and in freqtrade 2019.8.

### `ticker_interval` (now `timeframe`)

Support for `ticker_interval` terminology was deprecated in 2020.6 in favor of `timeframe` - and compatibility code was removed in 2022.3.

### Allow running multiple pairlists in sequence

The former `"pairlist"` section in the configuration has been removed, and is replaced by `"pairlists"` - being a list to specify a sequence of pairlists.

The old section of configuration parameters (`"pairlist"`) has been deprecated in 2019.11 and has been removed in 2020.4.

### deprecation of bidVolume and askVolume from volume-pairlist

Since only quoteVolume can be compared between assets, the other options (bidVolume, askVolume) have been deprecated in 2020.4, and have been removed in 2020.9.

### Using order book steps for exit price

Using `order_book_min` and `order_book_max` used to allow stepping the orderbook and trying to find the next ROI slot - trying to place sell-orders early.
As this does however increase risk and provides no benefit, it's been removed for maintainability purposes in 2021.7.

### Legacy Hyperopt mode

Using separate hyperopt files was deprecated in 2021.4 and was removed in 2021.9.
Please switch to the new [Parametrized Strategies](hyperopt.md) to benefit from the new hyperopt interface.

## Strategy changes between V2 and V3

Isolated Futures / short trading was introduced in 2022.4. This required major changes to configuration settings, strategy interfaces, ...

We have put a great effort into keeping compatibility with existing strategies, so if you just want to continue using freqtrade in spot markets, there are no changes necessary.
While we may drop support for the current interface sometime in the future, we will announce this separately and have an appropriate transition period.

Please follow the [Strategy migration](strategy_migration.md) guide to migrate your strategy to the new format to start using the new functionalities.

### webhooks - changes with 2022.4

#### `buy_tag` has been renamed to `enter_tag`

This should apply only to your strategy and potentially to webhooks.
We will keep a compatibility layer for 1-2 versions (so both `buy_tag` and `enter_tag` will still work), but support for this in webhooks will disappear after that.

#### Naming changes

Webhook terminology changed from "sell" to "exit", and from "buy" to "entry", removing "webhook" in the process.

* `webhookbuy`, `webhookentry` -> `entry`
* `webhookbuyfill`, `webhookentryfill` -> `entry_fill`
* `webhookbuycancel`, `webhookentrycancel` -> `entry_cancel`
* `webhooksell`, `webhookexit` -> `exit`
* `webhooksellfill`, `webhookexitfill` -> `exit_fill`
* `webhooksellcancel`, `webhookexitcancel` -> `exit_cancel`

## Removal of `populate_any_indicators`

version 2023.3 saw the removal of `populate_any_indicators` in favor of split methods for feature engineering and targets. Please read the [migration document](strategy_migration.md#freqai-strategy) for full details.

## Removal of `protections` from configuration

Setting protections from the configuration via `"protections": [],` has been removed in 2024.10, after having raised deprecation warnings for over 3 years.

## hdf5 data storage

Using hdf5 as data storage has been deprecated in 2024.12 and was removed in 2025.1. We recommend switching to the feather data format.

Please use the [`convert-data` subcommand](data-download.md#sub-command-convert-data) to convert your existing data to one of the supported formats before updating.

## Configuring advanced logging via config

Configuring syslog and journald via `--logfile systemd` and `--logfile journald` respectively has been deprecated in 2025.3.
Please use configuration based [log setup](advanced-setup.md#advanced-logging) instead.

## Removal of the edge module

The edge module has been deprecated in 2023.9 and removed in 2025.6.
All functionalities of edge have been removed, and having edge configured will result in an error.

## Adjustment to dynamic funding rate handling

With version 2025.12, the handling of dynamic funding rates has been adjusted to also support dynamic funding rates down to 1h funding intervals.
As a consequence, the mark and funding rate timeframes have been changed to 1h for every supported futures exchange.

As the timeframe for both mark and funding_fee candles has changed (usually from 8h to 1h) - already downloaded data will have to be adjusted or partially re-downloaded.
You can either re-download everything (`freqtrade download-data [...] --erase` - :warning: can take a long time) - or download the updated data selectively.

### Strategy

Most strategies should not need adjustments to continue to work as expected - however, strategies using `@informative("8h", candle_type="funding_rate")` or similar will have to switch the timeframe to 1h.
The same is true for `dp.get_pair_dataframe(metadata["pair"], "8h", candle_type="funding_rate")` - which will need to be switched to 1h.

freqtrade will auto-adjust the timeframe and return `funding_rates` despite the wrongly given timeframe. It'll issue a warning - and may still break your strategy.

### Selective data re-download

The script below should serve as an example - you may need to adjust the timeframe and exchange to your needs!

``` bash
# Cleanup no longer needed data
rm user_data/data/<exchange>/futures/*-mark*
rm user_data/data/<exchange>/futures/*-funding_rate*

# download new data (only required once to fix the mark and funding fee data)
freqtrade download-data -t 1h --trading-mode futures --candle-types funding_rate mark [...] --timerange <full timerange you've got other data for>

```

The result of the above will be that your funding_rates and mark data will have the 1h timeframe.
you can verify this with `freqtrade list-data --exchange <yourexchange> --show`.

!!! Note "Additional arguments"
    Additional arguments to the above commands may be necessary, like configuration files or explicit user_data if they deviate from the default.

**Hyperliquid** is a special case now - which will no longer require 1h mark data - but will use regular candles instead (this data never existed and is identical to 1h futures candles). As we don't support download-data for hyperliquid (they don't provide historic data) - there won't be actions necessary for hyperliquid users.

## Catboost models in freqAI

CatBoost models have been removed with version 2025.12 and are no longer actively supported.
If you have existing bots using CatBoost models, you can still use them in your custom models by copy/pasting them from the git history (as linked below) and installing the Catboost library manually.
We do however recommend switching to other supported model libraries like LightGBM or XGBoost for better support and future compatibility.

* [CatboostRegressor](https://github.com/freqtrade/freqtrade/blob/c6f3b0081927e161a16b116cc47fb663f7831d30/freqtrade/freqai/prediction_models/CatboostRegressor.py)
* [CatboostClassifier](https://github.com/freqtrade/freqtrade/blob/c6f3b0081927e161a16b116cc47fb663f7831d30/freqtrade/freqai/prediction_models/CatboostClassifier.py)
* [CatboostClassifierMultiTarget](https://github.com/freqtrade/freqtrade/blob/c6f3b0081927e161a16b116cc47fb663f7831d30/freqtrade/freqai/prediction_models/CatboostClassifierMultiTarget.py)
