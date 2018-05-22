import logging

import boto3
import os

from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.optimize.backtesting import Backtesting
import simplejson as json
from boto3.dynamodb.conditions import Key, Attr

db = boto3.resource('dynamodb')


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

    if 'body' in event:
        event['body'] = json.loads(event['body'])
        name = event['body']['name']
        user = event['body']['user']
        stake_currency = event['body']['stake_currency'].upper()
        asset = event['body']['asset']
        exchange = event['body']['exchange']

        assets = list(map(lambda x: "{}/{}".format(x, stake_currency).upper(), asset))

        table = db.Table(os.environ['strategyTable'])

        response = table.query(
            KeyConditionExpression=Key('user').eq(user) &
                                   Key('name').eq(name)

        )

        print(response)
        if "Items" in response and len(response['Items']) > 0:

            content = response['Items'][0]['content']
            configuration = {
                "max_open_trades": 1,
                "stake_currency": stake_currency,
                "stake_amount": 1,
                "fiat_display_currency": "USD",
                "unfilledtimeout": 600,
                "bid_strategy": {
                    "ask_last_balance": 0.0
                },
                "exchange": {
                    "name": exchange,
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
                "loglevel": logging.DEBUG,
                "strategy": "{}:{}".format(name, content)

            }

            print("generated configuration")
            print(configuration)

            print("initialized backtesting")
            backtesting = Backtesting(configuration)
            result = backtesting.start()
            print("finished test")

            print("persist data in dynamo")

            for index, row in result.iterrows():
                if row['loss'] > 0 or row['profit'] > 0:
                    item = {
                        "id": "{}.{}:{}".format(user, name, row['pair']),
                        "pair": row['pair'],
                        "count_profit": row['profit'],
                        "count_loss": row['loss'],
                        "avg_duration": row['avg duration'],
                        "avg profit": row['avg profit %'],
                        "total profit": row['total profit {}'.format(stake_currency)]

                    }

                print(item)
        else:
            raise Exception("sorry we did not find any matching strategy for user {} and name {}".format(user, name))
    else:
        raise Exception("no body provided")


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
