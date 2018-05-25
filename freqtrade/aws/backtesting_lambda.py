import datetime
import logging
import os
import tempfile

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

                till = datetime.datetime.today()
                fromDate = till - datetime.timedelta(days=7)

                if 'from' in event['body']:
                    fromDate = datetime.datetime.strptime(event['body']['from'], '%Y%m%d')
                if 'till' in event['body']:
                    till = datetime.datetime.strptime(event['body']['till'], '%Y%m%d')

                try:
                    if "Items" in response and len(response['Items']) > 0:

                        print("backtesting from {} till {} for {} with {} vs {}".format(fromDate, till, name,
                                                                                        event['body'][
                                                                                            'stake_currency'],
                                                                                        event['body']['asset']))
                        configuration = _generate_configuration(event, fromDate, name, response, till)

                        backtesting = Backtesting(configuration)
                        result = backtesting.start()
                        result_data = []
                        for index, row in result.iterrows():
                            data = {
                                "id": "{}.{}:{}".format(user, name, row['currency'].upper()),
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

                        # fire request message to aggregate this strategy now

                        return {
                            "statusCode": 200,
                            "body": json.dumps(result_data)
                        }
                    else:
                        return {
                            "statusCode": 404,
                            "body": json.dumps({
                                "error": "sorry we did not find any matching strategy for user {} and name {}".format(
                                    user, name)})
                        }

                except ImportError as e:
                    return {
                        "statusCode": 500,
                        "body": json.dumps({"error": e})
                    }
    else:
        raise Exception("not a valid event: {}".format(event))


def _generate_configuration(event, fromDate, name, response, till):
    """
        generates the configuration for us on the fly
    :param event:
    :param fromDate:
    :param name:
    :param response:
    :param till:
    :return:
    """

    content = response['Items'][0]['content']
    configuration = {
        "max_open_trades": 1,
        "stake_currency": event['body']['stake_currency'].upper(),
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
            "pair_whitelist": [
                "{}/{}".format(event['body']['asset'].upper(),
                               event['body']['stake_currency']).upper(),

            ]
        },
        "telegram": {
            "enabled": False,
            "token": "token",
            "chat_id": "0"
        },
        "initial_state": "running",
        "datadir": tempfile.gettempdir(),
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
        "timerange": "{}-{}".format(fromDate.strftime('%Y%m%d'), till.strftime('%Y%m%d')),
        "refresh_pairs": True

    }
    return configuration


def cron(event, context):
    """

    this functions submits all strategies to the backtesting queue

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

            for x in i['assets']:
                # test each asset by it self

                message = {
                    "user": i['user'],
                    "name": i['name'],
                    "asset": x,
                    "stake_currency": i['stake_currency']
                }

                # triggered over html, let's provide
                # a date range for the backtesting
                if 'pathParameters' in event:
                    if 'from' in event['pathParameters']:
                        message['from'] = event['pathParameters']['from']
                    else:
                        message['from'] = datetime.datetime.today().strftime('%Y%m%d')
                    if 'till' in event['pathParameters']:
                        message['till'] = event['pathParameters']['till']
                    else:
                        message['till'] = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')

                serialized = json.dumps(message, use_decimal=True)
                # submit item to queue for routing to the correct persistence

                result = client.publish(
                    TopicArn=topic_arn,
                    Message=json.dumps({'default': serialized}),
                    Subject="schedule backtesting",
                    MessageStructure='json'
                )

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

    return {
        "statusCode": 200
    }
