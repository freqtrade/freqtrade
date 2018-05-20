import simplejson as json
from base64 import urlsafe_b64encode
import freqtrade.aws.strategy as aws


def test_strategy(lambda_context):
    content = """# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class TestStrategy(IStrategy):
    minimal_roi = {
        "0": 0.5
    }
    stoploss = -0.2
    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['maShort'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['maMedium'] = ta.EMA(dataframe, timeperiod=21)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['maShort'], dataframe['maMedium'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['maMedium'], dataframe['maShort'])
            ),
            'sell'] = 1
        return dataframe


        """

    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG",
        "description": "simple test strategy",
        "name": "TestStrategy",
        "content": urlsafe_b64encode(content.encode('utf-8')),
        "public": False
    }

    # db should be empty
    assert (len(json.loads(aws.names({}, {})['body'])) == 0)
    # now we add an entry
    aws.submit({
        "body": json.dumps(request)
    }, {})

    # now we should have items
    assert (len(json.loads(aws.names({}, {})['body'])) == 1)

    # able to add a second strategy with the sample name, but different user

    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TH",
        "description": "simple test strategy",
        "name": "TestStrategy",
        "content": urlsafe_b64encode(content.encode('utf-8')),
        "public": True
    }

    aws.submit({
        "body": json.dumps(request)
    }, {})

    assert (len(json.loads(aws.names({}, {})['body'])) == 2)

    # able to add a duplicated strategy, which should overwrite the existing strategy

    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TH",
        "description": "simple test strategy",
        "name": "TestStrategy",
        "content": urlsafe_b64encode(content.encode('utf-8')),
        "public": True
    }

    aws.submit({
        "body": json.dumps(request)
    }, {})

    assert (len(json.loads(aws.names({}, {})['body'])) == 2)

    # we need to be able to get the code of the strategy
    code = aws.code({'pathParameters': {
        "name": "TestStrategy",
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TH"
    }}, {})

    # code should equal our initial content
    assert code == content

    # we need to be able to get a strategy ( code cannot be included )
    strategy = json.loads(aws.get({}, {}))
    assert "content" not in strategy
    assert "user" in strategy
    assert "name" in strategy
    assert "description" in strategy
    assert "public" in strategy
