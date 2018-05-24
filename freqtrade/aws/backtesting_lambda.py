import datetime
import logging
import os
import boto3
import simplejson as json
from boto3.dynamodb.conditions import Key

from freqtrade.aws.tables import get_trade_table, get_strategy_table
from freqtrade.optimize.backtesting import Backtesting

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

        it should be invoked by SNS only to avoid abuse of the system!

        :param context:
            standard AWS context, so pleaes ignore for now!
        :return:
            no return
    """

    if 'Records' in event:
        for x in event['Records']:
            if 'Sns' in x and 'Message' in x['Sns']:

                event['body'] = json.loads(x['Sns']['Message'])
                name = event['body']['name']
                user = event['body']['user']

                trade_table = get_trade_table()
                table = get_strategy_table()

                response = table.query(
                    KeyConditionExpression=Key('user').eq(user) &
                                           Key('name').eq(name)

                )

                print(response)
                if "Items" in response and len(response['Items']) > 0:

                    today = datetime.datetime.today()
                    yesterday = today - datetime.timedelta(days=1)

                    content = response['Items'][0]['content']
                    configuration = {
                        "max_open_trades": 1,
                        "stake_currency": response['Items'][0]['stake_currency'],
                        "stake_amount": 1,
                        "fiat_display_currency": "USD",
                        "unfilledtimeout": 600,
                        "trailing_stop": response['Items'][0]['trailing_stop'],
                        "bid_strategy": {
                            "ask_last_balance": 0.0
                        },
                        "exchange": {
                            "name": response['Items'][0]['exchange'],
                            "enabled": True,
                            "key": "key",
                            "secret": "secret",
                            "pair_whitelist": list(
                                map(lambda x: "{}/{}".format(x, response['Items'][0]['stake_currency']).upper(),
                                    response['Items'][0]['assets']))
                        },
                        "telegram": {
                            "enabled": False,
                            "token": "token",
                            "chat_id": "0"
                        },
                        "initial_state": "running",
                        "datadir": ".",
                        "experimental": {
                            "use_sell_signal": response['Items'][0]['use_sell'],
                            "sell_profit_only": True
                        },
                        "internals": {
                            "process_throttle_secs": 5
                        },
                        'realistic_simulation': True,
                        "loglevel": logging.DEBUG,
                        "strategy": "{}:{}".format(name, content),
                        "timerange": "{}-{}".format(yesterday.strftime('%Y%m%d'), today.strftime('%Y%m%d')),
                        "refresh_pairs": True

                    }

                    print("generated configuration")
                    print(configuration)

                    print("initialized backtesting")
                    backtesting = Backtesting(configuration)
                    result = backtesting.start()
                    print("finished test")

                    print("persist data in dynamo")

                    print(result)
                    result_data = []
                    for index, row in result.iterrows():
                        data = {
                            "id": "{}.{}:{}".format(user, name, row['currency']),
                            "trade": "{} to {}".format(row['entry'].strftime('%Y-%m-%d %H:%M:%S'),
                                                       row['exit'].strftime('%Y-%m-%d %H:%M:%S')),
                            "pair": row['currency'],
                            "duration": row['duration'],
                            "profit_percent": row['profit_percent'],
                            "profit_stake": row['profit_BTC'],
                            "entry_date": row['entry'].strftime('%Y-%m-%d %H:%M:%S'),
                            "exit_date": row['exit'].strftime('%Y-%m-%d %H:%M:%S')
                        }

                        data = json.dumps(data, use_decimal=True)
                        data = json.loads(data, use_decimal=True)

                        # persist data
                        trade_table.put_item(Item=data)
                        result_data.append(data)

                    return {
                        "statusCode": 200,
                        "body": json.dumps(result_data)
                    }
                else:
                    raise Exception(
                        "sorry we did not find any matching strategy for user {} and name {}".format(user, name))
    else:
        raise Exception("not a valid event: {}".format(event))


def cron(event, context):
    """

    this functions submits a new strategy to the backtesting queue

    :param event:
    :param context:
    :return:
    """

    # if topic exists, we just reuse it
    client = boto3.client('sns')
    topic_arn = client.create_topic(Name=os.environ['topic'])['TopicArn']

    table = get_strategy_table()
    response = table.scan()

    def fetch(response, table):
        """
            fetches all strategies from the server
            technically code duplications
            TODO refacture
        :param response:
        :param table:
        :return:
        """

        for i in response['Items']:
            # fire a message to our queue

            print(i)
            message = {
                "user": i['user'],
                "name": i['name']
            }

            serialized = json.dumps(message, use_decimal=True)
            # submit item to queue for routing to the correct persistence

            result = client.publish(
                TopicArn=topic_arn,
                Message=json.dumps({'default': serialized}),
                Subject="schedule backtesting",
                MessageStructure='json'
            )

            print(result)

        if 'LastEvaluatedKey' in response:
            return table.scan(
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
        else:
            return {}

    # do/while simulation
    response = fetch(response, table)

    while 'LastEvaluatedKey' in response:
        response = fetch(response, table)

    pass
