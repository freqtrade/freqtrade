# Deprecated features

This page contains description of the command line arguments, configuration parameters
and the bot features that were declared as DEPRECATED by the bot development team
and are no longer supported. Please avoid their usage in your configuration.

### the `--live` command line option

`--live` in the context of backtesting allows to download the latest tick data for backtesting.
Since this only downloads one set of data (by default 500 candles) - this is not really suitable for extendet backtesting, and has therefore been deprecated.

This command was deprecated in `2019.6-dev` and will be removed after the next release.

## Removed features

### The **--dynamic-whitelist** command line option

This command line option was deprecated in 2018 and removed freqtrade 2019.6-dev (develop branch)
and in freqtrade 2019.7 (master branch).
