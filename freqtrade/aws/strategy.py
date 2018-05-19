import os
import ctypes

#for d, dirs, files in os.walk('lib'):
#    for f in files:
#        if f.endswith('.a') or f.endswith('.la'):
#            continue
#        print("loading: {}".format(f))
#        ctypes.cdll.LoadLibrary(os.path.join(d, f))
#



from freqtrade.strategy.resolver import StrategyResolver

import simplejson as json
from jsonschema import validate
from freqtrade.aws.schemas import __SUBMIT_STRATEGY_SCHEMA__
from base64 import urlsafe_b64decode

__HTTP_HEADERS__ = {
    'Access-Control-Allow-Origin' : '*',
    'Access-Control-Allow-Credentials' : True
}

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

    print("received data")
    print(data)

    # validate against schema
    result = validate(data, __SUBMIT_STRATEGY_SCHEMA__)

    print("data are validated");
    print(result)

    strategy = urlsafe_b64decode(data['content']).decode('utf-8')

    print("loaded strategy")
    print(strategy)
    # try to load the strategy
    StrategyResolver().compile(data['name'], strategy)

    print("compiled strategy")
    # save to DB

    return {
        "statusCode": 200,
        "headers": __HTTP_HEADERS__,
        "body": json.dumps({"success":True})
    }
