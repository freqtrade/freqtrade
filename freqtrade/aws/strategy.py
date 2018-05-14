from freqtrade.strategy.resolver import StrategyResolver

import simplejson as json
from jsonschema import validate
from freqtrade.aws.schemas import __SUBMIT_STRATEGY_SCHEMA__
from base64 import urlsafe_b64decode

def names(event, context):
    """
            returns the names of all registered strategies, but public and private
    :param event:
    :param context:
    :return:
    """
    pass


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

    # validate against schema
    validate(data, __SUBMIT_STRATEGY_SCHEMA__)

    strategy = urlsafe_b64decode(data['content']).decode('utf-8')

    # try to load the strategy
    StrategyResolver().compile(data['name'], strategy)

    # save to DB

    pass
