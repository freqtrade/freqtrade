import os
import time
from base64 import urlsafe_b64decode

import boto3
import simplejson as json
from jsonschema import validate

from freqtrade.aws.schemas import __SUBMIT_STRATEGY_SCHEMA__
from freqtrade.strategy.resolver import StrategyResolver

db = boto3.resource('dynamodb')


def names(event, context):
    """
            returns the names of all registered strategies, both public and private
    :param event:
    :param context:
    :return:
    """
    table = db.Table(os.environ['strategyTable'])
    response = table.scan()
    result = response['Items']

    while 'LastEvaluatedKey' in response:
        for i in response['Items']:
            result.append(i)
        response = table.scan(
            ExclusiveStartKey=response['LastEvaluatedKey']
        )

    # map results and hide informations
    data = list(map(lambda x: {'name': x['name'], 'public': x['public'], 'user': x['user']}, result))

    return {
        "statusCode": 200,
        "body": json.dumps(data)
    }


def performance(event, context):
    """
        returns the performance of the specified strategy
    :param event:
    :param context:
    :return:
    """
    pass


def code(event, context):
    """
        returns the code of the requested strategy, if it's public
    :param event:
    :param context:
    :return:
    """
    pass


def submit(event, context):
    """
        compiles the given strategy and stores it in the internal database
    :param event:
    :param context:
    :return:
    """

    # get data
    data = json.loads(event['body'])

    # print("received data")
    # print(data)

    # validate against schema
    result = validate(data, __SUBMIT_STRATEGY_SCHEMA__)

    # print("data are validated");
    # print(result)

    strategy = urlsafe_b64decode(data['content']).decode('utf-8')

    # print("loaded strategy")
    # print(strategy)
    # try to load the strategy
    StrategyResolver().compile(data['name'], strategy)

    data['time'] = int(time.time() * 1000)
    data['type'] = "strategy"

    # force serialization to deal with decimal number
    data = json.dumps(data, use_decimal=True)
    data = json.loads(data, use_decimal=True)

    table = db.Table(os.environ['strategyTable'])

    result = table.put_item(Item=data)
    return {
        "statusCode": result['ResponseMetadata']['HTTPStatusCode'],
        "body": json.dumps(result)
    }
