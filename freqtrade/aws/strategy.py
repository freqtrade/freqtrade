from freqtrade.strategy.resolver import StrategyResolver
import boto3
import os
import simplejson as json
import uuid
from jsonschema import validate
from freqtrade.aws.schemas import __SUBMIT_STRATEGY_SCHEMA__
from base64 import urlsafe_b64decode
from freqtrade.aws.service.Persistence import Persistence
import time


def names(event, context):
    """
            returns the names of all registered strategies, both public and private
    :param event:
    :param context:
    :return:
    """
    table = Persistence(os.environ['strategyTable'])

    return {
        "statusCode": 200,
        "body": json.dumps(table.list())
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

    # generate id
    data['id'] = str(uuid.uuid4())
    data['time'] = int(time.time() * 1000)

    # save to DB
    table = Persistence(os.environ['strategyTable'])

    result = table.save(data)

    return {
        "statusCode": result['ResponseMetadata']['HTTPStatusCode'],
        "body": json.dumps(result)
    }
