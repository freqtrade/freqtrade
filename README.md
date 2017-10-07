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

#### Telegram RPC commands:
* /start: Starts the trader
* /stop: Stops the trader
* /status: Lists all open trades
* /profit: Lists cumulative profit from all finished trades
* /forcesell <trade_id>: Instantly sells the given trade (Ignoring `minimum_roi`).
* /performance: Show performance of each finished trade grouped by pair

#### Config
`minimal_roi` is a JSON object where the key is a duration
in minutes and the value is the minimum ROI in percent.
See the example below:
```
"minimal_roi": {
    "2880": 0.005, # Sell after 48 hours if there is at least 0.5% profit
    "1440": 0.01,  # Sell after 24 hours if there is at least 1% profit
    "720":  0.02,  # Sell after 12 hours if there is at least 2% profit
    "360":  0.02,  # Sell after 6 hours if there is at least 2% profit
    "0":    0.025  # Sell immediately if there is at least 2.5% profit
},
```

`stoploss` is loss in percentage that should trigger a sale. 
For example value `-0.10` will cause immediate sell if the
profit dips below -10% for a given trade. This parameter is optional.

`initial_state` is an optional field that defines the initial application state.
Possible values are `running` or `stopped`. (default=`running`)
If the value is `stopped` the bot has to be started with `/start` first.

`ask_last_balance` sets the bidding price. Value `0.0` will use `ask` price, `1.0` will
use the `last` price and values between those interpolate between ask and last price. Using `ask` price will guarantee quick success in bid, but bot will also end up paying more then would probably have been necessary.

The other values should be self-explanatory,
if not feel free to raise a github issue.

#### Prerequisites
* python3.6
* sqlite
* [TA-lib](https://github.com/mrjbq7/ta-lib#dependencies) binaries

#### Install
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

There is also an [article](https://www.sales4k.com/blockchain/high-frequency-trading-bot-tutorial/) about how to setup the bot (thanks [@gurghet](https://github.com/gurghet)).

#### Execute tests

```
$ pytest
```
This will by default skip the slow running backtest set. To run backtest set:

```
$ BACKTEST=true pytest
```

#### Docker
```
$ cd freqtrade
$ docker build -t freqtrade .
$ docker run --rm -it freqtrade
```

#### Contributing

Feel like our bot is missing a feature? We welcome your pull requests! Few pointers for contributions:

- Create your PR against the `develop` branch, not `master`.
- New features need to contain unit tests.
- If you are unsure, discuss the feature on [slack](https://join.slack.com/t/highfrequencybot/shared_invite/enQtMjQ5NTM0OTYzMzY3LWMxYzE3M2MxNDdjMGM3ZTYwNzFjMGIwZGRjNTc3ZGU3MGE3NzdmZGMwNmU3NDM5ZTNmM2Y3NjRiNzk4NmM4OGE) or in a [issue](https://github.com/gcarq/freqtrade/issues) before a PR.
