# freqtrade

[![Build Status](https://travis-ci.org/freqtrade/freqtrade.svg?branch=develop)](https://travis-ci.org/freqtrade/freqtrade)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)

Simple High frequency trading bot for crypto currencies designed to support multi exchanges and be controlled via Telegram.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Disclaimer

This software is for educational purposes only. Do not risk money which 
you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS 
AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. 

Always start by running a trading bot in Dry-run and do not engage money
before you understand how it works and what profit/loss you should
expect.

We strongly recommend you to have coding and Python knowledge. Do not 
hesitate to read the source code and understand the mechanism of this bot.

## Exchange marketplaces supported

- [X] [Bittrex](https://bittrex.com/)
- [X] [Binance](https://www.binance.com/) ([*Note for binance users](#a-note-on-binance))
- [ ] [113 others to tests](https://github.com/ccxt/ccxt/). _(We cannot guarantee they will work)_

## Features

- [x] **Based on Python 3.6+**: For botting on any operating system - Windows, macOS and Linux
- [x] **Persistence**: Persistence is achieved through sqlite
- [x] **Dry-run**: Run the bot without playing money.
- [x] **Backtesting**: Run a simulation of your buy/sell strategy.
- [x] **Strategy Optimization by machine learning**: Use machine learning to optimize your buy/sell strategy parameters with real exchange data.
- [x] **Edge position sizing** Calculate your win rate, risk reward ratio, the best stoploss and adjust your position size before taking a position for each specific market. [Learn more](https://github.com/freqtrade/freqtrade/blob/develop/docs/edge.md)
- [x] **Whitelist crypto-currencies**: Select which crypto-currency you want to trade or use dynamic whitelists.
- [x] **Blacklist crypto-currencies**: Select which crypto-currency you want to avoid.
- [x] **Manageable via Telegram**: Manage the bot with Telegram
- [x] **Display profit/loss in fiat**: Display your profit/loss in 33 fiat.
- [x] **Daily summary of profit/loss**: Provide a daily summary of your profit/loss.
- [x] **Performance status report**: Provide a performance status of your current trades.


## Table of Contents

- [Quick start](#quick-start)
- [Documentations](https://github.com/freqtrade/freqtrade/blob/develop/docs/index.md)
  - [Installation](https://github.com/freqtrade/freqtrade/blob/develop/docs/installation.md)
  - [Configuration](https://github.com/freqtrade/freqtrade/blob/develop/docs/configuration.md)
  - [Strategy Optimization](https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md)
  - [Backtesting](https://github.com/freqtrade/freqtrade/blob/develop/docs/backtesting.md)
  - [Hyperopt](https://github.com/freqtrade/freqtrade/blob/develop/docs/hyperopt.md)
  - [Sandbox Testing](https://github.com/freqtrade/freqtrade/blob/develop/docs/sandbox-testing.md)
  - [Edge](https://github.com/freqtrade/freqtrade/blob/develop/docs/edge.md)
- [Basic Usage](#basic-usage)
  - [Bot commands](#bot-commands)
  - [Telegram RPC commands](#telegram-rpc-commands)
- [Support](#support)
  - [Help](#help--slack)
  - [Bugs](#bugs--issues)
  - [Feature Requests](#feature-requests)
  - [Pull Requests](#pull-requests)
- [Requirements](#requirements)
  - [Min hardware required](#min-hardware-required)
  - [Software requirements](#software-requirements)
- [Wanna help?](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)
  - [Dev - getting started](https://github.com/freqtrade/freqtrade/blob/develop/docs/developer.md) (WIP)
 

## Quick start

Freqtrade provides a Linux/macOS script to install all dependencies and help you to configure the bot.

```bash
git clone git@github.com:freqtrade/freqtrade.git
cd freqtrade
git checkout develop
./setup.sh --install
```

_Windows installation is explained in [Installation doc](https://github.com/freqtrade/freqtrade/blob/develop/docs/installation.md)_

## Documentation

We invite you to read the bot documentation to ensure you understand how the bot is working.

- [Index](https://github.com/freqtrade/freqtrade/blob/develop/docs/index.md)
- [Installation](https://github.com/freqtrade/freqtrade/blob/develop/docs/installation.md)
- [Configuration](https://github.com/freqtrade/freqtrade/blob/develop/docs/configuration.md)
- [Bot usage](https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-usage.md)
  - [How to run the bot](https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-usage.md#bot-commands)
  - [How to use Backtesting](https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-usage.md#backtesting-commands)
  - [How to use Hyperopt](https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-usage.md#hyperopt-commands)
- [Strategy Optimization](https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md)
- [Backtesting](https://github.com/freqtrade/freqtrade/blob/develop/docs/backtesting.md)
- [Hyperopt](https://github.com/freqtrade/freqtrade/blob/develop/docs/hyperopt.md)

## Basic Usage

### Bot commands

```bash
usage: main.py [-h] [-v] [--version] [-c PATH] [-d PATH] [-s NAME]
               [--strategy-path PATH] [--dynamic-whitelist [INT]]
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
  -d PATH, --datadir PATH
                        path to backtest data (default:
                        freqtrade/tests/testdata
  -s NAME, --strategy NAME
                        specify strategy class name (default: DefaultStrategy)
  --strategy-path PATH  specify additional strategy lookup path
  --dynamic-whitelist [INT]
                        dynamically generate and update whitelist based on 24h
                        BaseVolume (Default 20 currencies)
  --dry-run-db          Force dry run to use a local DB
                        "tradesv3.dry_run.sqlite" instead of memory DB. Work
                        only if dry_run is enabled.
```

### Telegram RPC commands

Telegram is not mandatory. However, this is a great way to control your bot. More details on our [documentation](https://github.com/freqtrade/freqtrade/blob/develop/docs/index.md)

- `/start`: Starts the trader
- `/stop`: Stops the trader
- `/status [table]`: Lists all open trades
- `/count`: Displays number of open trades
- `/profit`: Lists cumulative profit from all finished trades
- `/forcesell <trade_id>|all`: Instantly sells the given trade (Ignoring `minimum_roi`).
- `/performance`: Show performance of each finished trade grouped by pair
- `/balance`: Show account balance per currency
- `/daily <n>`: Shows profit or loss per day, over the last n days
- `/help`: Show help message
- `/version`: Show version


## Development branches

The project is currently setup in two main branches:

- `develop` - This branch has often new features, but might also cause breaking changes.
- `master` - This branch contains the latest stable release. The bot 'should' be stable on this branch, and is generally well tested.
- `feat/*` - These are feature branches, which are being worked on heavily. Please don't use these unless you want to test a specific feature.


## A note on Binance

For Binance, please add `"BNB/<STAKE>"` to your blacklist to avoid issues.
Accounts having BNB accounts use this to pay for fees - if your first trade happens to be on `BNB`, further trades will consume this position and make the initial BNB order unsellable as the expected amount is not there anymore.

## Support

### Help / Slack

For any questions not covered by the documentation or for further
information about the bot, we encourage you to join our slack channel.

- [Click here to join Slack channel](https://join.slack.com/t/highfrequencybot/shared_invite/enQtMjQ5NTM0OTYzMzY3LWMxYzE3M2MxNDdjMGM3ZTYwNzFjMGIwZGRjNTc3ZGU3MGE3NzdmZGMwNmU3NDM5ZTNmM2Y3NjRiNzk4NmM4OGE).

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

If you discover a bug in the bot, please 
[search our issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue) 
first. If it hasn't been reported, please 
[create a new issue](https://github.com/freqtrade/freqtrade/issues/new) and 
ensure you follow the template guide so that our team can assist you as 
quickly as possible.

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Have you a great idea to improve the bot you want to share? Please,
first search if this feature was not [already discussed](https://github.com/freqtrade/freqtrade/labels/enhancement).
If it hasn't been requested, please 
[create a new request](https://github.com/freqtrade/freqtrade/issues/new) 
and ensure you follow the template guide so that it does not get lost 
in the bug reports.

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Feel like our bot is missing a feature? We welcome your pull requests!

Please read our 
[Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)
to understand the requirements before sending your pull-requests. 

Coding is not a neccessity to contribute - maybe start with improving our documentation?
Issues labeled [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) can be good first contributions, and will help get you familiar with the codebase.

**Note** before starting any major new feature work, *please open an issue describing what you are planning to do* or talk to us on [Slack](https://join.slack.com/t/highfrequencybot/shared_invite/enQtMjQ5NTM0OTYzMzY3LWMxYzE3M2MxNDdjMGM3ZTYwNzFjMGIwZGRjNTc3ZGU3MGE3NzdmZGMwNmU3NDM5ZTNmM2Y3NjRiNzk4NmM4OGE). This will ensure that interested parties can give valuable feedback on the feature, and let others know that you are working on it.

**Important:** Always create your PR against the `develop` branch, not `master`.

## Requirements

### Uptodate clock

The clock must be accurate, syncronized to a NTP server very frequently to avoid problems with communication to the exchanges.

### Min hardware required

To run this bot we recommend you a cloud instance with a minimum of:

- Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software requirements

- [Python 3.6.x](http://docs.python-guide.org/en/latest/starting/installation/) 
- [pip](https://pip.pypa.io/en/stable/installing/)
- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [TA-Lib](https://mrjbq7.github.io/ta-lib/install.html)
- [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) (Recommended)
- [Docker](https://www.docker.com/products/docker) (Recommended)

