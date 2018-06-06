from base64 import urlsafe_b64encode

import os
import pytest
import simplejson as json
from datetime import datetime, timedelta

from freqtrade.aws.backtesting_lambda import backtest, cron, generate_configuration
from freqtrade.aws.strategy import submit


# @pytest.mark.skip(reason="no way of currently testing this")
def test_backtest_remote(lambda_context):
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

    # now we add an entry
    submit({
        "body": json.dumps(request)
    }, {})

    # build sns request
    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG",
        "name": "MyFancyTestStrategy",
        "from": "20180401",
        "till": "20180501",
        "stake_currency": "usdt",
        "assets": ["ltc"],
        "local": False

    }

    assert backtest({
        "Records": [
            {
                "Sns": {
                    "Subject": "backtesting",
                    "Message": json.dumps(request)
                }
            }]
    }, {})['statusCode'] == 200


def test_backtest_time_frame(lambda_context):
    content = """# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MyFancyTestStrategy(IStrategy):
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
        "name": "MyFancyTestStrategy",
        "content": urlsafe_b64encode(content.encode('utf-8')),
        "public": False
    }

    # now we add an entry
    submit({
        "body": json.dumps(request)
    }, {})

    # build sns request
    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG",
        "name": "MyFancyTestStrategy",
        "from": "20180401",
        "till": "20180501",
        "stake_currency": "usdt",
        "assets": ["ltc"],
        "local": True

    }

    assert backtest({
        "Records": [
            {
                "Sns": {
                    "Subject": "backtesting",
                    "Message": json.dumps(request)
                }
            }]
    }, {})['statusCode'] == 200


def test_backtest(lambda_context):
    content = """# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List

from hyperopt import hp
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MyFancyTestStrategy(IStrategy):
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
        "name": "MyFancyTestStrategy",
        "content": urlsafe_b64encode(content.encode('utf-8')),
        "public": False
    }

    # now we add an entry
    submit({
        "body": json.dumps(request)
    }, {})

    # build sns request
    request = {
        "user": "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG",
        "name": "MyFancyTestStrategy",
        "stake_currency": "usdt",
        "assets": ["ltc"],
        "days": 2,
        "ticker": '15m',
        "local": True
    }

    assert backtest({
        "Records": [
            {
                "Sns": {
                    "Subject": "backtesting",
                    "Message": json.dumps(request)
                }
            }]
    }, {})['statusCode'] == 200


def test_cron(lambda_context):
    """ test the scheduling to the queue"""
    content = """# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MyFancyTestStrategy(IStrategy):
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
        "name": "MyFancyTestStrategy",
        "content": urlsafe_b64encode(content.encode('utf-8')),
        "public": False
    }

    # now we add an entry
    submit({
        "body": json.dumps(request)
    }, {})

    print("evaluating cron job")

    cron({}, {})

    # TODO test receiving of message some how


def test_generate_configuration(lambda_context):
    os.environ["BASE_URL"] = "https://freq.isaac.international/dev"
    till = datetime.today()
    fromDate = till - timedelta(days=90)

    config = generate_configuration(fromDate, till, "TestStrategy", True,
                           "GCU4LW2XXZW3A3FM2XZJTEJHNWHTWDKY2DIJLCZJ5ULVZ4K7LZ7D23TG", True)

    print(config)
