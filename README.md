# freqtrade

[![Build Status](https://travis-ci.org/gcarq/freqtrade.svg?branch=develop)](https://travis-ci.org/gcarq/freqtrade)
[![Coverage Status](https://coveralls.io/repos/github/gcarq/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/gcarq/freqtrade?branch=develop)


Simple High frequency trading bot for crypto currencies designed to 
support multi exchanges and be controlled via Telegram.

![freqtrade](https://raw.githubusercontent.com/gcarq/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Disclaimer
This software is for educational purposes only. Do not risk money which 
you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS 
AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. 

Always start by running a trading bot in Dry-run and do not engage money
before you understand how it works and what profit/loss you should
expect.

We strongly recommend you to have coding and Python knowledge. Do not 
hesitate to read the source code and understand the mechanism of this bot.

## Table of Contents
- [Features](#features)
- [Quick start](#quick-start)
- [Documentations](https://github.com/gcarq/freqtrade/blob/develop/docs/index.md)
   - [Installation](https://github.com/gcarq/freqtrade/blob/develop/docs/installation.md)
   - [Configuration](https://github.com/gcarq/freqtrade/blob/develop/docs/configuration.md)
   - [Strategy Optimization](https://github.com/gcarq/freqtrade/blob/develop/docs/bot-optimization.md)
   - [Backtesting](https://github.com/gcarq/freqtrade/blob/develop/docs/backtesting.md)
   - [Hyperopt](https://github.com/gcarq/freqtrade/blob/develop/docs/hyperopt.md)
- [Support](#support)
   - [Help](#help--slack)
   - [Bugs](#bugs--issues)
   - [Feature Requests](#feature-requests)
   - [Pull Requests](#pull-requests)
- [Basic Usage](#basic-usage)
  - [Bot commands](#bot-commands)
  - [Telegram RPC commands](#telegram-rpc-commands)
- [Requirements](#requirements)
    - [Min hardware required](#min-hardware-required)
    - [Software requirements](#software-requirements)

## Branches
The project is currently setup in two main branches:
- `develop` - This branch has often new features, but might also cause 
breaking changes.
- `master` - This branch contains the latest stable release. The bot 
'should' be stable on this branch, and is generally well tested. 

## Features
- [x] **Based on Python 3.6+**: For botting on any operating system - 
Windows, macOS and Linux
- [x] **Persistence**: Persistence is achieved through sqlite
- [x] **Dry-run**: Run the bot without playing money.
- [x] **Backtesting**: Run a simulation of your buy/sell strategy.
- [x] **Strategy Optimization**: Optimize your buy/sell strategy 
parameters with Hyperopts.
- [x] **Whitelist crypto-currencies**: Select which crypto-currency you
want to trade.
- [x] **Blacklist crypto-currencies**: Select which crypto-currency you
want to avoid.
- [x] **Manageable via Telegram**: Manage the bot with Telegram
- [x] **Display profit/loss in fiat**: Display your profit/loss in
33 fiat.
- [x] **Daily summary of profit/loss**: Provide a daily summary
 of your profit/loss.
- [x] **Performance status report**: Provide a performance status of 
your current trades.

### Exchange supported
- [x] Bittrex
- [ ] Binance
- [ ] Others

## Quick start
This quick start section is a very short explanation on how to test the 
bot in dry-run. We invite you to read the 
[bot documentation](https://github.com/gcarq/freqtrade/blob/develop/docs/index.md) 
to ensure you understand how the bot is working.

The following steps are made for Linux/MacOS environment

**1. Clone the repo**
```bash
git clone git@github.com:gcarq/freqtrade.git
git checkout develop
cd freqtrade
```
**2. Create the config file**  
Switch `"dry_run": true,`
```bash
cp config.json.example config.json
vi config.json
```
**3. Build your docker image and run it**
```bash
docker build -t freqtrade .
docker run --rm -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
```


### Help / Slack
For any questions not covered by the documentation or for further
information about the bot, we encourage you to join our slack channel.
- [Click here to join Slack channel](https://join.slack.com/t/highfrequencybot/shared_invite/enQtMjQ5NTM0OTYzMzY3LWMxYzE3M2MxNDdjMGM3ZTYwNzFjMGIwZGRjNTc3ZGU3MGE3NzdmZGMwNmU3NDM5ZTNmM2Y3NjRiNzk4NmM4OGE).

### [Bugs / Issues](https://github.com/gcarq/freqtrade/issues?q=is%3Aissue)
If you discover a bug in the bot, please 
[search our issue tracker](https://github.com/gcarq/freqtrade/issues?q=is%3Aissue) 
first. If it hasn't been reported, please 
[create a new issue](https://github.com/gcarq/freqtrade/issues/new) and 
ensure you follow the template guide so that our team can assist you as 
quickly as possible.

### [Feature Requests](https://github.com/gcarq/freqtrade/labels/enhancement)
Have you a great idea to improve the bot you want to share? Please,
first search if this feature was not [already discussed](https://github.com/gcarq/freqtrade/labels/enhancement).
If it hasn't been requested, please 
[create a new request](https://github.com/gcarq/freqtrade/issues/new) 
and ensure you follow the template guide so that it does not get lost 
in the bug reports.

### [Pull Requests](https://github.com/gcarq/freqtrade/pulls)
Feel like our bot is missing a feature? We welcome your pull requests! 
Please read our 
[Contributing document](https://github.com/gcarq/freqtrade/blob/develop/CONTRIBUTING.md)
to understand the requirements before sending your pull-requests. 

**Important:** Always create your PR against the `develop` branch, not 
`master`.

## Basic Usage

### Bot commands

```bash
usage: main.py [-h] [-v] [--version] [-c PATH] [--dry-run-db] [--datadir PATH]
               [--dynamic-whitelist [INT]]
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
  --dry-run-db          Force dry run to use a local DB
                        "tradesv3.dry_run.sqlite" instead of memory DB. Work
                        only if dry_run is enabled.
  --datadir PATH        path to backtest data (default freqdata/tests/testdata
  --dynamic-whitelist [INT]
                        dynamically generate and update whitelist based on 24h
                        BaseVolume (Default 20 currencies)
```
More details on:
- [How to run the bot](https://github.com/gcarq/freqtrade/blob/develop/docs/bot-usage.md#bot-commands)
- [How to use Backtesting](https://github.com/gcarq/freqtrade/blob/develop/docs/bot-usage.md#backtesting-commands)
- [How to use Hyperopt](https://github.com/gcarq/freqtrade/blob/develop/docs/bot-usage.md#hyperopt-commands)
   
### Telegram RPC commands
Telegram is not mandatory. However, this is a great way to control your
bot. More details on our 
[documentation](https://github.com/gcarq/freqtrade/blob/develop/docs/index.md)

- `/start`: Starts the trader
- `/stop`: Stops the trader
- `/status [table]`: Lists all open trades
- `/count`: Displays number of open trades
- `/profit`: Lists cumulative profit from all finished trades
- `/forcesell <trade_id>|all`: Instantly sells the given trade 
(Ignoring `minimum_roi`).
- `/performance`: Show performance of each finished trade grouped by pair
- `/balance`: Show account balance per currency
- `/daily <n>`: Shows profit or loss per day, over the last n days
- `/help`: Show help message
- `/version`: Show version

## Requirements

### Min hardware required
To run this bot we recommend you a cloud instance with a minimum of:
* Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software requirements
- [Python 3.6.x](http://docs.python-guide.org/en/latest/starting/installation/) 
- [pip](https://pip.pypa.io/en/stable/installing/)
- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [TA-Lib](https://mrjbq7.github.io/ta-lib/install.html)
- [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) (Recommended)
- [Docker](https://www.docker.com/products/docker) (Recommended)
