import logging

from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.optimize.backtesting import Backtesting


def backtest(event, context):
    """
        this method is running on the AWS server
        and back tests this application for us
        and stores the back testing results in a local database

        this event can be given as:

        :param event:
            {
                'strategy' : 'url handle where we can find the strategy'
                'stake_currency' : 'our desired stake currency'
                'asset' : '[] asset we are interested in. If empy, we fill use a default list
                'username' : user who's strategy should be evaluated
                'name' : name of the strategy we want to evaluate
                'exchange' : name of the exchange we should be using

            }
        :param context:
            standard AWS context, so pleaes ignore for now!
        :return:
            no return
    """

    name = "TestStrategy"
    user = "12345678"
    stake_currency = "USDT"
    asset = ["ETH", "BTC"]
    exchange = "binance"

    assets = list(map(lambda x: "{}/{}".format(x, stake_currency).upper(), asset))

    configuration = {
        "max_open_trades": 1,
        "stake_currency": stake_currency,
        "stake_amount": 0.001,
        "fiat_display_currency": "USD",
        "unfilledtimeout": 600,
        "bid_strategy": {
            "ask_last_balance": 0.0
        },
        "exchange": {
            "name": "bittrex",
            "enabled": True,
            "key": "key",
            "secret": "secret",
            "pair_whitelist": assets
        },
        "telegram": {
            "enabled": False,
            "token": "token",
            "chat_id": "0"
        },
        "initial_state": "running",
        "datadir": ".",
        "experimental": {
            "use_sell_signal": True,
            "sell_profit_only": True
        },
        "internals": {
            "process_throttle_secs": 5
        },
        'realistic_simulation': True,
        "loglevel": logging.DEBUG

    }

    print("generated configuration")
    print(configuration)

    print("initialized backtesting")
    backtesting = Backtesting(configuration)
    result = backtesting.start()
    print("finished test")

    print(result)
    print("persist data in dynamo")

    for index, row in result.iterrows():
        item = {
            "id": "{}.{}:{}".format(user, name, row['pair']),
            "pair": row['pair'],
            "profit": row['profit'],
            "loss": row['loss'],
            "duration": row['avg duration'],
            "avg profit": row['avg profit %'],
            "total profit": row['total profit {}'.format(stake_currency)]

        }

        print(item)
    pass


def submit(event, context):
    """

    this functions submits a new strategy to the backtesting queue

    :param event:
    :param context:
    :return:
    """
    pass


if __name__ == '__main__':
    backtest({}, {})
