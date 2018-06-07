import datetime
import logging
import os
import tempfile
from base64 import urlsafe_b64encode

import requests
import simplejson as json
from requests import post

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
                'asset' : '[] asset we are interested in.
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

                till = datetime.datetime.today()
                fromDate = till - datetime.timedelta(days=90)

                if 'days' in event['body']:
                    fromDate = till - datetime.timedelta(days=event['body']['days'])
                else:
                    if 'from' in event['body']:
                        fromDate = datetime.datetime.strptime(event['body']['from'], '%Y%m%d')
                    if 'till' in event['body']:
                        till = datetime.datetime.strptime(event['body']['till'], '%Y%m%d')

                timerange = (till - fromDate).days

                # by default we refresh data
                refresh = True

                if 'refresh' in event['body']:
                    refresh = event['body']['refresh']

                print("time range between dates is: {} days".format(timerange))

                try:

                    print("schedule back testing from {} till {} for {} with {} vs {}".format(fromDate, till, name,
                                                                                              event['body'][
                                                                                                  'stake_currency'],
                                                                                              event['body'][
                                                                                                  'assets']))

                    if "ticker" in event['body']:
                        ticker = event['body']['ticker']
                    else:
                        ticker = '5m'

                    if "local" in event['body'] and event['body']['local']:
                        print("running in local mode")
                        configuration = generate_configuration(fromDate, till, name, refresh, user, False)

                        run_backtest(configuration, name, user, ticker, fromDate, till)
                    else:
                        print("running in remote mode")
                        _submit_job(name, user, ticker, fromDate, till)

                    return {
                        "statusCode": 200
                    }

                except ImportError as e:
                    return {
                        "statusCode": 500,
                        "body": json.dumps({"error": e})
                    }
    else:
        raise Exception("not a valid event: {}".format(event))


def _submit_job(name, user, ticker, fromDate, till):
    """
        submits a new task to the cluster

    :param configuration:
    :param user:
    :return:
    """
    import boto3
    timerange = (till - fromDate).days

    # fire AWS fargate instance now
    # run_backtest(configuration, name, user)
    # kinda ugly right now and needs more customization
    client = boto3.client('ecs')
    response = client.run_task(
        cluster=os.environ.get('FREQ_CLUSTER_NAME', 'fargate'),  # name of the cluster
        launchType='FARGATE',
        taskDefinition=os.environ.get('FREQ_TASK_NAME', 'freqtrade-backtesting:2'),
        count=1,
        platformVersion='LATEST',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [
                    # we need at least 2, to insure network stability
                    os.environ.get('FREQ_SUBNET_1', 'subnet-c35bdcab'),
                    os.environ.get('FREQ_SUBNET_2', 'subnet-be46b9c4'),
                    os.environ.get('FREQ_SUBNET_3', 'subnet-234ab559'),
                    os.environ.get('FREQ_SUBNET_4', 'subnet-234ab559')],
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={"containerOverrides": [{
            "name": "freqtrade-backtest",
            "environment": [
                {
                    "name": "FREQ_USER",
                    "value": "{}".format(user)
                },
                {
                    "name": "FREQ_TICKER",
                    "value": "{}".format(ticker)
                },
                {
                    "name": "FREQ_FROM",
                    "value": "{}".format(fromDate.strftime('%Y%m%d'))
                },
                {
                    "name": "FREQ_TILL",
                    "value": "{}".format(till.strftime('%Y%m%d'))
                },
                {
                    "name": "FREQ_STRATEGY",
                    "value": "{}".format(name)
                },
                {
                    "name": "BASE_URL",
                    "value": "https://freq.isaac.international/dev"
                }
            ]
        }]},
    )
    return response


def run_backtest(configuration, name, user, interval, fromDate, till):
    """
        this backtests the specified evaluation

    :param configuration:
    :param name:
    :param user:
    :param interval:
    :param timerange:

    :return:
    """

    timerange = (till - fromDate).days

    configuration['ticker_interval'] = interval

    backtesting = Backtesting(configuration)
    result = backtesting.start()

    # store individual trades - not really needed
    # _store_trade_data(interval, name, result, timerange, user)

    # store aggregated values
    _store_aggregated_data(interval, name, result, timerange, user)

    return result


def _store_aggregated_data(interval, name, result, timerange, user):
    for row in result[1][2]:
        if row[1] > 0:
            data = {
                "id": "aggregate:{}:{}:{}:test".format(row[0].upper(), interval, timerange),
                "trade": "{}.{}".format(user, name),
                "pair": row[0],
                "trades": row[1],
                "losses": row[6],
                "wins": row[5],
                "duration": row[4],
                "profit_percent": row[2],
                "strategy": name,
                "user": user,
                "ticker": interval,
                "days": timerange
            }

            print(data)
            try:
                print(
                    post("{}/trade".format(os.environ.get('BASE_URL', 'https://freq.isaac.international/dev')),
                         json=data))
            except Exception as e:
                print("submission ignored: {}".format(e))


def _store_trade_data(interval, name, result, timerange, user):
    for index, row in result[0].iterrows():
        data = {
            "id": "{}.{}:{}:{}:{}:test".format(user, name, interval, timerange, row['currency'].upper()),
            "trade": "{} to {}".format(row['entry'].strftime('%Y-%m-%d %H:%M:%S'),
                                       row['exit'].strftime('%Y-%m-%d %H:%M:%S')),
            "pair": row['currency'],
            "duration": row['duration'],
            "profit_percent": row['profit_percent'],
            "profit_stake": row['profit_BTC'],
            "entry_date": row['entry'].strftime('%Y-%m-%d %H:%M:%S'),
            "exit_date": row['exit'].strftime('%Y-%m-%d %H:%M:%S'),
            "strategy": name,
            "user": user

        }

        print(data)
        try:
            print(
                post("{}/trade".format(os.environ.get('BASE_URL', 'https://freq.isaac.international/dev')),
                     json=data))
        except Exception as e:
            print("submission ignored: {}".format(e))


def generate_configuration(fromDate, till, name, refresh, user, remote=True):
    """
        generates the configuration for us on the fly for a given
        strategy. This is loaded from a remote url if specfied or
        the internal dynamodb

    :param event:
    :param fromDate:
    :param name:
    :param response:
    :param till:
    :return:
    """

    response = {}

    if remote:
        print("using remote mode to query strategy details")
        response = requests.get(
            "{}/strategies/{}/{}".format(os.environ.get('BASE_URL', "https://freq.isaac.international/dev"), user,
                                         name)).json()

        # load associated content right now this only works for public strategies obviously TODO
        content = requests.get(
            "{}/strategies/{}/{}/code".format(os.environ.get('BASE_URL', "https://freq.isaac.international/dev"), user,
                                              name)).content

        response['content'] = urlsafe_b64encode(content).decode()
        print(content)

    else:
        print("using local mode to query strategy details")
        from boto3.dynamodb.conditions import Key
        from freqtrade.aws.tables import get_strategy_table

        table = get_strategy_table()

        response = table.query(
            KeyConditionExpression=Key('user').eq(user) &
                                   Key('name').eq(name)

        )['Items'][0]

    print(response)

    content = response['content']
    configuration = {
        "max_open_trades": 1,
        "stake_currency": response['stake_currency'].upper(),
        "stake_amount": 1,
        "fiat_display_currency": "USD",
        "unfilledtimeout": 600,
        "trailing_stop": response['trailing_stop'],
        "bid_strategy": {
            "ask_last_balance": 0.0
        },
        "exchange": {
            "name": response['exchange'],
            "enabled": True,
            "key": "key",
            "secret": "secret",
            "pair_whitelist": list(
                map(lambda x: "{}/{}".format(x, response['stake_currency']).upper(),
                    response['assets']))
        },
        "telegram": {
            "enabled": False,
            "token": "token",
            "chat_id": "0"
        },
        "initial_state": "running",
        "datadir": tempfile.gettempdir(),
        "experimental": {
            "use_sell_signal": response['use_sell'],
            "sell_profit_only": True
        },
        "internals": {
            "process_throttle_secs": 5
        },
        'realistic_simulation': True,
        "loglevel": logging.INFO,
        "strategy": "{}:{}".format(name, content),
        "timerange": "{}-{}".format(fromDate.strftime('%Y%m%d'), till.strftime('%Y%m%d')),
        "refresh_pairs": refresh

    }
    return configuration


def cron(event, context):
    """

    this functions submits all strategies to the backtesting queue

    :param event:
    :param context:
    :return:
    """
    import boto3
    from freqtrade.aws.tables import get_strategy_table

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

            # we want to evaluate several time spans for the strategy
            for day in [1, 7, 30, 90]:

                # we want to evaluate several time intervals for each strategy
                for interval in ['5m', '15m', '30m', '1h']:
                    message = {
                        "user": i['user'],
                        "name": i['name'],
                        "assets": i['assets'],
                        "stake_currency": i['stake_currency'],
                        "local": False,
                        "refresh": True,
                        "ticker": interval,
                        "days": day
                    }

                    print("submitting: {}".format(message))
                    serialized = json.dumps(message, use_decimal=True)
                    # submit item to queue for routing to the correct persistence

                    result = client.publish(
                        TopicArn=topic_arn,
                        Message=json.dumps({'default': serialized}),
                        Subject="schedule",
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

    return {
        "statusCode": 200
    }


if __name__ == "__main__":
    till = datetime.datetime.today()
    fromDate = till - datetime.timedelta(days=90)
    print(_submit_job(
        "BinHV45",
        "GBPAQEFGGWCMWVFU34PMVGS4P2NJR4IDFNVI4LTCZAKJAD3JCXUMBI4J",
        "5m",
        fromDate,
        till
    ))
