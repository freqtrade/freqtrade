import datetime
import logging
import os
import tempfile
from base64 import urlsafe_b64encode

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
    from boto3.dynamodb.conditions import Key
    from freqtrade.aws.tables import get_strategy_table

    if 'Records' in event:
        for x in event['Records']:
            if 'Sns' in x and 'Message' in x['Sns']:

                event['body'] = json.loads(x['Sns']['Message'])
                name = event['body']['name']
                user = event['body']['user']

                table = get_strategy_table()

                response = table.query(
                    KeyConditionExpression=Key('user').eq(user) &
                                           Key('name').eq(name)

                )

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
                    if "Items" in response and len(response['Items']) > 0:

                        print("schedule back testing from {} till {} for {} with {} vs {}".format(fromDate, till, name,
                                                                                                  event['body'][
                                                                                                      'stake_currency'],
                                                                                                  event['body'][
                                                                                                      'assets']))
                        configuration = _generate_configuration(event, fromDate, name, response, till, refresh)

                        ticker = response['Items'][0]['ticker']

                        if "ticker" in event['body']:
                            ticker = event['body']['ticker']

                        print("using ticker of {}".format(ticker))

                        if "local" in event['body'] and event['body']['local']:
                            print("running in local mode")
                            run_backtest(configuration, name, user, ticker, timerange)
                        else:
                            print("running in remote mode")
                            json.dumps(_submit_job(configuration, user, ticker, timerange))

                        return {
                            "statusCode": 200
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


def _submit_job(configuration, user, interval, timerange):
    """
        submits a new task to the cluster

    :param configuration:
    :param user:
    :return:
    """
    import boto3
    configuration = urlsafe_b64encode(json.dumps(configuration).encode('utf-8')).decode('utf-8')
    # fire AWS fargate instance now
    # run_backtest(configuration, name, user)
    # kinda ugly right now and needs more customization
    client = boto3.client('ecs')
    response = client.run_task(
        cluster=os.environ.get('FREQ_CLUSTER_NAME', 'fargate'),  # name of the cluster
        launchType='FARGATE',
        taskDefinition=os.environ.get('FREQ_TASK_NAME', 'freqtrade-backtesting:1'),
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
                    "name": "FREQ_CONFIG",
                    "value": "{}".format(configuration)
                },
                {
                    "name": "FREQ_TICKER",
                    "value": "{}".format(interval)
                },
                {
                    "name": "FREQ_TIMERANGE",
                    "value": "{}".format(timerange)
                }

            ]
        }]},
    )
    return response


def run_backtest(configuration, name, user, interval, timerange):
    """
        this backtests the specified evaluation

    :param configuration:
    :param name:
    :param user:
    :param interval:
    :param timerange:

    :return:
    """

    configuration['ticker_interval'] = interval

    backtesting = Backtesting(configuration)
    result = backtesting.start()

    # store individual trades
    _store_trade_data(interval, name, result, timerange, user)

    # store aggregated values
    _store_aggregated_data(interval, name, result, timerange, user)

    return result


def _store_aggregated_data(interval, name, result, timerange, user):
    for row in result[1][2]:
        if row[1] > 0:
            data = {
                "id": "{}.{}:{}:{}:test".format(user, name, interval, timerange),
                "trade": "aggregate:{}".format(row[0].upper()),
                "pair": row[0],
                "trades": row[1],
                "losses": row[6],
                "wins": row[5],
                "duration": row[4],
                "profit_percent": row[2],
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
            "exit_date": row['exit'].strftime('%Y-%m-%d %H:%M:%S')
        }

        print(data)
        try:
            print(
                post("{}/trade".format(os.environ.get('BASE_URL', 'https://freq.isaac.international/dev')),
                     json=data))
        except Exception as e:
            print("submission ignored: {}".format(e))
    return data


def _generate_configuration(event, fromDate, name, response, till, refresh):
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
        "datadir": tempfile.gettempdir(),
        "experimental": {
            "use_sell_signal": response['Items'][0]['use_sell'],
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

                    serialized = json.dumps(message, use_decimal=True)
                    # submit item to queue for routing to the correct persistence

                    result = client.publish(
                        TopicArn=topic_arn,
                        Message=json.dumps({'default': serialized}),
                        Subject="schedule",
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


if __name__ == "__main__":
    import boto3

    client = boto3.client('ecs')
    response = client.run_task(
        cluster=os.environ.get('FREQ_CLUSTER_NAME', 'fargate'),  # name of the cluster
        launchType='FARGATE',
        taskDefinition=os.environ.get('FREQ_TASK_NAME', 'freqtrade-backtesting:1'),
        count=1,
        platformVersion='LATEST',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [
                    os.environ.get('FREQ_SUBNET_1', 'subnet-c35bdcab'),
                    os.environ.get('FREQ_SUBNET_2', 'subnet-be46b9c4'),
                    os.environ.get('FREQ_SUBNET_3', 'subnet-234ab559'),
                    os.environ.get('FREQ_SUBNET_4', 'subnet-234ab559')
                ],
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={"containerOverrides": [{
            "name": "freqtrade-backtest",
            "environment": [
                {
                    "name": "FREQ_USER",
                    "value": "12345"
                },
                {
                    "name": "FREQ_CONFIG",
                    "value": "eyJtYXhfb3Blbl90cmFkZXMiOiAxLCAic3Rha2VfY3VycmVuY3kiOiAiVVNEVCIsICJzdGFrZV9hbW91bnQiOiAxLCAiZmlhdF9kaXNwbGF5X2N1cnJlbmN5IjogIlVTRCIsICJ1bmZpbGxlZHRpbWVvdXQiOiA2MDAsICJ0cmFpbGluZ19zdG9wIjogZmFsc2UsICJiaWRfc3RyYXRlZ3kiOiB7ImFza19sYXN0X2JhbGFuY2UiOiAwLjB9LCAiZXhjaGFuZ2UiOiB7Im5hbWUiOiAiYmluYW5jZSIsICJlbmFibGVkIjogdHJ1ZSwgImtleSI6ICJrZXkiLCAic2VjcmV0IjogInNlY3JldCIsICJwYWlyX3doaXRlbGlzdCI6IFsiQlRDL1VTRFQiLCAiRVRIL1VTRFQiLCAiTFRDL1VTRFQiXX0sICJ0ZWxlZ3JhbSI6IHsiZW5hYmxlZCI6IGZhbHNlLCAidG9rZW4iOiAidG9rZW4iLCAiY2hhdF9pZCI6ICIwIn0sICJpbml0aWFsX3N0YXRlIjogInJ1bm5pbmciLCAiZGF0YWRpciI6ICJDOlxcVXNlcnNcXHdvaGxnXFxBcHBEYXRhXFxMb2NhbFxcVGVtcCIsICJleHBlcmltZW50YWwiOiB7InVzZV9zZWxsX3NpZ25hbCI6IHRydWUsICJzZWxsX3Byb2ZpdF9vbmx5IjogdHJ1ZX0sICJpbnRlcm5hbHMiOiB7InByb2Nlc3NfdGhyb3R0bGVfc2VjcyI6IDV9LCAicmVhbGlzdGljX3NpbXVsYXRpb24iOiB0cnVlLCAibG9nbGV2ZWwiOiAyMCwgInN0cmF0ZWd5IjogIk15RmFuY3lUZXN0U3RyYXRlZ3k6SXlBdExTMGdSRzhnYm05MElISmxiVzkyWlNCMGFHVnpaU0JzYVdKeklDMHRMUXBtY205dElHWnlaWEYwY21Ga1pTNXpkSEpoZEdWbmVTNXBiblJsY21aaFkyVWdhVzF3YjNKMElFbFRkSEpoZEdWbmVRcG1jbTl0SUhSNWNHbHVaeUJwYlhCdmNuUWdSR2xqZEN3Z1RHbHpkQW9qWm5KdmJTQm9lWEJsY205d2RDQnBiWEJ2Y25RZ2FIQWdJeUIwYUdseklIWmxjbk5wYjI0Z1pHOWxjeUJ1YjNRZ2MzVndjRzl5ZENCb2VYQmxjbTl3ZENFS1puSnZiU0JtZFc1amRHOXZiSE1nYVcxd2IzSjBJSEpsWkhWalpRcG1jbTl0SUhCaGJtUmhjeUJwYlhCdmNuUWdSR0YwWVVaeVlXMWxDaU1nTFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMHRMUzB0TFMwdExTMEtDbWx0Y0c5eWRDQjBZV3hwWWk1aFluTjBjbUZqZENCaGN5QjBZUXBwYlhCdmNuUWdabkpsY1hSeVlXUmxMblpsYm1SdmNpNXhkSEI1YkdsaUxtbHVaR2xqWVhSdmNuTWdZWE1nY1hSd2VXeHBZZ29LWTJ4aGMzTWdUWGxHWVc1amVWUmxjM1JUZEhKaGRHVm5lU2hKVTNSeVlYUmxaM2twT2dvZ0lDQWdiV2x1YVcxaGJGOXliMmtnUFNCN0NpQWdJQ0FnSUNBZ0lqQWlPaUF3TGpVS0lDQWdJSDBLSUNBZ0lITjBiM0JzYjNOeklEMGdMVEF1TWdvZ0lDQWdkR2xqYTJWeVgybHVkR1Z5ZG1Gc0lEMGdKelZ0SndvS0lDQWdJR1JsWmlCd2IzQjFiR0YwWlY5cGJtUnBZMkYwYjNKektITmxiR1lzSUdSaGRHRm1jbUZ0WlRvZ1JHRjBZVVp5WVcxbEtTQXRQaUJFWVhSaFJuSmhiV1U2Q2lBZ0lDQWdJQ0FnYldGalpDQTlJSFJoTGsxQlEwUW9aR0YwWVdaeVlXMWxLUW9nSUNBZ0lDQWdJR1JoZEdGbWNtRnRaVnNuYldGVGFHOXlkQ2RkSUQwZ2RHRXVSVTFCS0dSaGRHRm1jbUZ0WlN3Z2RHbHRaWEJsY21sdlpEMDRLUW9nSUNBZ0lDQWdJR1JoZEdGbWNtRnRaVnNuYldGTlpXUnBkVzBuWFNBOUlIUmhMa1ZOUVNoa1lYUmhabkpoYldVc0lIUnBiV1Z3WlhKcGIyUTlNakVwQ2lBZ0lDQWdJQ0FnY21WMGRYSnVJR1JoZEdGbWNtRnRaUW9LSUNBZ0lHUmxaaUJ3YjNCMWJHRjBaVjlpZFhsZmRISmxibVFvYzJWc1ppd2daR0YwWVdaeVlXMWxPaUJFWVhSaFJuSmhiV1VwSUMwLUlFUmhkR0ZHY21GdFpUb0tJQ0FnSUNBZ0lDQmtZWFJoWm5KaGJXVXViRzlqV3dvZ0lDQWdJQ0FnSUNBZ0lDQW9DaUFnSUNBZ0lDQWdJQ0FnSUNBZ0lDQnhkSEI1YkdsaUxtTnliM056WldSZllXSnZkbVVvWkdGMFlXWnlZVzFsV3lkdFlWTm9iM0owSjEwc0lHUmhkR0ZtY21GdFpWc25iV0ZOWldScGRXMG5YU2tLSUNBZ0lDQWdJQ0FnSUNBZ0tTd0tJQ0FnSUNBZ0lDQWdJQ0FnSjJKMWVTZGRJRDBnTVFvS0lDQWdJQ0FnSUNCeVpYUjFjbTRnWkdGMFlXWnlZVzFsQ2dvZ0lDQWdaR1ZtSUhCdmNIVnNZWFJsWDNObGJHeGZkSEpsYm1Rb2MyVnNaaXdnWkdGMFlXWnlZVzFsT2lCRVlYUmhSbkpoYldVcElDMC1JRVJoZEdGR2NtRnRaVG9LSUNBZ0lDQWdJQ0JrWVhSaFpuSmhiV1V1Ykc5ald3b2dJQ0FnSUNBZ0lDQWdJQ0FvQ2lBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0J4ZEhCNWJHbGlMbU55YjNOelpXUmZZV0p2ZG1Vb1pHRjBZV1p5WVcxbFd5ZHRZVTFsWkdsMWJTZGRMQ0JrWVhSaFpuSmhiV1ZiSjIxaFUyaHZjblFuWFNrS0lDQWdJQ0FnSUNBZ0lDQWdLU3dLSUNBZ0lDQWdJQ0FnSUNBZ0ozTmxiR3duWFNBOUlERUtJQ0FnSUNBZ0lDQnlaWFIxY200Z1pHRjBZV1p5WVcxbENnb0tJQ0FnSUNBZ0lDQT0iLCAidGltZXJhbmdlIjogIjIwMTgwNDAxLTIwMTgwNTAxIiwgInJlZnJlc2hfcGFpcnMiOiBmYWxzZX0="
                },
                {
                    "name": "BASE_URL",
                    "value": "https://freq.isaac.international/dev"
                }

            ]
        }]},
    )
    print(response)
