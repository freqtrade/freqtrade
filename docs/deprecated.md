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

### Allow running multiple pairlists in sequence

The former `"pairlist"` section in the configuration has been removed, and is replaced by `"pairlists"` - being a list to specify a sequence of pairlists.

The old section of configuration parameters (`"pairlist"`) has been deprecated in 2019.11 and has been removed in 2020.4.

### deprecation of bidVolume and askVolume from volume-pairlist

Since only quoteVolume can be compared between assets, the other options (bidVolume, askVolume) have been deprecated in 2020.4, and have been removed in 2020.9.

### Using order book steps for sell price

Using `order_book_min` and `order_book_max` used to allow stepping the orderbook and trying to find the next ROI slot - trying to place sell-orders early.
As this does however increase risk and provides no benefit, it's been removed for maintainability purposes in 2021.7.

### Legacy Hyperopt mode

Using separate hyperopt files was deprecated in 2021.4 and was removed in 2021.9.
Please switch to the new [Parametrized Strategies](hyperopt.md) to benefit from the new hyperopt interface.

## Margin / short changes

// TODO-lev: update version here

## Strategy changes

As strategies now have to support multiple different signal types, some things had to change.

Columns:

* `buy` -> `enter_long`
* `sell` -> `exit_long`
* `buy_tag` -> `enter_tag`

New columns are `enter_short` and `exit_short`, which will initiate short trades (requires additional configuration!)

### webhooks - `buy_tag` has been renamed to `enter_tag`

This should apply only to your strategy and potentially to webhooks.
We will keep a compatibility layer for 1-2 versions (so both `buy_tag` and `enter_tag` will still work), but support for this in webhooks will disappear after that.
