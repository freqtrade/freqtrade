import os
import time
from base64 import urlsafe_b64decode

import boto3
import simplejson as json
from boto3.dynamodb.conditions import Key, Attr
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


def get(event, context):
    """
        returns the code of the requested strategy, if it's public
    :param event:
    :param context:
    :return:
    """

    assert 'pathParameters' in event
    assert 'user' in event['pathParameters']
    assert 'name' in event['pathParameters']

    table = db.Table(os.environ['strategyTable'])

    response = table.query(
        KeyConditionExpression=Key('user').eq(event['pathParameters']['user']) &
                               Key('name').eq(event['pathParameters']['name'])

    )

    if "Items" in response and len(response['Items']) > 0:
        item = response['Items'][0]
        item.pop('content')

        return {
            "statusCode": response['ResponseMetadata']['HTTPStatusCode'],
            "body": json.dumps(item)
        }

    else:
        return {
            "statusCode": response['ResponseMetadata']['HTTPStatusCode'],
            "body": json.dumps(response)
        }


def code(event, context):
    """
        returns the code of the requested strategy, if it's public
    :param event:
    :param context:
    :return:
    """

    print("event")
    print(event)
    print("context")
    print(context)

    user = ""
    name = ""

    # proxy based handling
    if 'pathParameters' in event:
        assert 'user' in event['pathParameters']
        assert 'name' in event['pathParameters']
        user = event['pathParameters']['user']
        name = event['pathParameters']['name']

    # plain lambda handling
    elif 'path' in event:
        assert 'user' in event['path']
        assert 'name' in event['path']
        user = event['path']['user']
        name = event['path']['name']

    table = db.Table(os.environ['strategyTable'])

    response = table.query(
        KeyConditionExpression=Key('user').eq(user) &
                               Key('name').eq(name)

    )

    if "Items" in response and len(response['Items']) > 0:
        if response['Items'][0]["public"]:
            content = urlsafe_b64decode(response['Items'][0]['content'])

            return {
                "headers:": {"Content-Type": "text/plain"},
                "statusCode": 200,
                "body": str(content)
            }
        else:
            return {
                "statusCode": 403,
                "body": json.dumps({"success": False, "reason": "Denied"})
            }

    else:
        return {
            "statusCode": response['ResponseMetadata']['HTTPStatusCode'],
            "body": json.dumps(response)
        }


def submit(event, context):
    """
        compiles the given strategy and stores it in the internal database
    :param event:
    :param context:
    :return:
    """

    print(event)
    # get data
    data = json.loads(event['body'])

    # print("received data")

    # validate against schema
    result = validate(data, __SUBMIT_STRATEGY_SCHEMA__)

    # print("data are validated");
    # print(result)

    # validate that the user is an Isaac User
    # ToDo

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
