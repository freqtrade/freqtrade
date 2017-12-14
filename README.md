# freqtrade

[![Build Status](https://travis-ci.org/gcarq/freqtrade.svg?branch=develop)](https://travis-ci.org/gcarq/freqtrade)
[![Coverage Status](https://coveralls.io/repos/github/gcarq/freqtrade/badge.svg?branch=develop)](https://coveralls.io/github/gcarq/freqtrade?branch=develop)


Simple High frequency trading bot for crypto currencies.
Currently supports trading on Bittrex exchange.

This software is for educational purposes only.
Don't risk money which you are afraid to lose.

The command interface is accessible via Telegram (not required).
Just register a new bot on https://telegram.me/BotFather
and enter the telegram `token` and your `chat_id` in `config.json`

Persistence is achieved through sqlite.

### Telegram RPC commands:
* /start: Starts the trader
* /stop: Stops the trader
* /status [table]: Lists all open trades
* /count: Displays number of open trades
* /profit: Lists cumulative profit from all finished trades
* /forcesell <trade_id>|all: Instantly sells the given trade (Ignoring `minimum_roi`).
* /performance: Show performance of each finished trade grouped by pair
* /balance: Show account balance per currency
* /daily <n>: Shows profit or loss per day, over the last n days
* /help: Show help message
* /version: Show version

### Config
`minimal_roi` is a JSON object where the key is a duration
in minutes and the value is the minimum ROI in percent.
See the example below:
```
"minimal_roi": {
    "40": 0.0,    # Sell after 40 minutes if the profit is not negative
    "30": 0.01,   # Sell after 30 minutes if there is at least 1% profit
    "20": 0.02,   # Sell after 20 minutes if there is at least 2% profit
    "0":  0.04    # Sell immediately if there is at least 4% profit
},
```

`stoploss` is loss in percentage that should trigger a sale.
For example value `-0.10` will cause immediate sell if the
profit dips below -10% for a given trade. This parameter is optional.

`initial_state` is an optional field that defines the initial application state.
Possible values are `running` or `stopped`. (default=`running`)
If the value is `stopped` the bot has to be started with `/start` first.

`ask_last_balance` sets the bidding price. Value `0.0` will use `ask` price, `1.0` will
use the `last` price and values between those interpolate between ask and last
price. Using `ask` price will guarantee quick success in bid, but bot will also
end up paying more then would probably have been necessary.

The other values should be self-explanatory,
if not feel free to raise a github issue.

### Prerequisites
* python3.6
* sqlite
* [TA-lib](https://github.com/mrjbq7/ta-lib#dependencies) binaries

### Install

#### Arch Linux

Use your favorite AUR helper and install `python-freqtrade-git`.

#### Manually

`master` branch contains the latest stable release.

`develop` branch has often new features, but might also cause breaking changes. To use it, you are encouraged to join our [slack channel](https://join.slack.com/t/highfrequencybot/shared_invite/enQtMjQ5NTM0OTYzMzY3LWMxYzE3M2MxNDdjMGM3ZTYwNzFjMGIwZGRjNTc3ZGU3MGE3NzdmZGMwNmU3NDM5ZTNmM2Y3NjRiNzk4NmM4OGE).

```
$ cd freqtrade/
# copy example config. Dont forget to insert your api keys
$ cp config.json.example config.json
$ python -m venv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
$ pip install -e .
$ ./freqtrade/main.py
```

There is also an [article](https://www.sales4k.com/blockchain/high-frequency-trading-bot-tutorial/) about how to setup the bot (thanks [@gurghet](https://github.com/gurghet)).*

\* *Note:* that article was written for an earlier version, so it may be outdated

#### Docker

Building the image:

```
$ cd freqtrade
$ docker build -t freqtrade .
```

For security reasons, your configuration file will not be included in the
image, you will need to bind mount it. It is also advised to bind mount
a SQLite database file (see second example) to keep it between updates.

You can run a one-off container that is immediately deleted upon exiting with
the following command (config.json must be in the current working directory):

```
$ docker run --rm -v `pwd`/config.json:/freqtrade/config.json -it freqtrade
```

To run a restartable instance in the background (feel free to place your
configuration and database files wherever it feels comfortable on your
filesystem):

```
$ cd ~/.freq
$ touch tradesv3.sqlite
$ docker run -d \
  --name freqtrade \
  -v ~/.freq/config.json:/freqtrade/config.json \
  -v ~/.freq/tradesv3.sqlite:/freqtrade/tradesv3.sqlite \
  freqtrade
```
If you are using `dry_run=True` it's not necessary to mount `tradesv3.sqlite`.

You can then use the following commands to monitor and manage your container:

```
$ docker logs freqtrade
$ docker logs -f freqtrade
$ docker restart freqtrade
$ docker stop freqtrade
$ docker start freqtrade
```

You do not need to rebuild the image for configuration
changes, it will suffice to edit `config.json` and restart the container.

### Usage
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
  -c PATH, --config PATH
                        specify configuration file (default: config.json)
  -v, --verbose         be verbose
  --version             show program's version number and exit
  --dynamic-whitelist [INT]
                        dynamically generate and update whitelist based on 24h
                        BaseVolume (Default 20 currencies)
  --dry-run-db          Force dry run to use a local DB
                        "tradesv3.dry_run.sqlite" instead of memory DB. Work
                        only if dry_run is enabled.
```

#### Dynamic whitelist example
Per default `--dynamic-whitelist` will retrieve the 20 currencies based 
on BaseVolume. This value can be changed when you run the script.

**By Default**  
Get the 20 currencies based on BaseVolume.  
```bash
freqtrade --dynamic-whitelist
```

**Customize the number of currencies to retrieve**  
Get the 30 currencies based on BaseVolume.  
```bash
freqtrade --dynamic-whitelist 30
```

**Exception**  
`--dynamic-whitelist` must be greater than 0. If you enter 0 or a
negative value (e.g -2), `--dynamic-whitelist` will use the default
value (20).

### Backtesting

Backtesting also uses the config specified via `-c/--config`.

```
usage: freqtrade backtesting [-h] [-l] [-i INT] [--realistic-simulation]

optional arguments:
  -h, --help            show this help message and exit
  -l, --live            using live data
  -i INT, --ticker-interval INT
                        specify ticker interval in minutes (default: 5)
  --realistic-simulation
                        uses max_open_trades from config to simulate real
                        world limitations

```

### Hyperopt

It is possible to use hyperopt for trading strategy optimization.
Hyperopt uses an internal config named `OPTIMIZE_CONFIG` located in `freqtrade/optimize/hyperopt.py`.

```
usage: freqtrade hyperopt [-h] [-e INT] [--use-mongodb]

optional arguments:
  -h, --help            show this help message and exit
  -e INT, --epochs INT  specify number of epochs (default: 100)
  --use-mongodb         parallelize evaluations with mongodb (requires mongod
                        in PATH)

```

### Execute tests

```
$ pytest freqtrade
```

### Contributing

Feel like our bot is missing a feature? We welcome your pull requests! Few pointers for contributions:

- Create your PR against the `develop` branch, not `master`.
- New features need to contain unit tests.
- If you are unsure, discuss the feature on [slack](https://join.slack.com/t/highfrequencybot/shared_invite/enQtMjQ5NTM0OTYzMzY3LWMxYzE3M2MxNDdjMGM3ZTYwNzFjMGIwZGRjNTc3ZGU3MGE3NzdmZGMwNmU3NDM5ZTNmM2Y3NjRiNzk4NmM4OGE) or in a [issue](https://github.com/gcarq/freqtrade/issues) before a PR.
