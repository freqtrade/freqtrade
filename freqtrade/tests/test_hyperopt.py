# pragma pylint: disable=missing-docstring
from operator import itemgetter
import logging
import os
from functools import reduce
from math import exp
import pytest
from pandas import DataFrame
from qtpylib.indicators import crossed_above

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from freqtrade.tests.test_backtesting import backtest, format_results

logging.disable(logging.DEBUG) # disable debug logs that slow backtesting a lot

# set TARGET_TRADES to suit your number concurrent trades so its realistic to 20days of data
TARGET_TRADES = 1200


@pytest.fixture
def pairs():
    return ['btc-neo', 'btc-eth', 'btc-omg', 'btc-edg', 'btc-pay',
            'btc-pivx', 'btc-qtum', 'btc-mtl', 'btc-etc', 'btc-ltc']

@pytest.fixture
def conf():
    return {
        "minimal_roi": {
            "40":  0.0,
            "30":  0.01,
            "20":  0.02,
            "0":  0.04
        },
        "stoploss": -0.05
    }

def buy_strategy_generator(params):
    print(params)
    def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS
        if params['below_sma']['enabled']:
            conditions.append(dataframe['close'] < dataframe['sma'])
        if params['over_sma']['enabled']:
            conditions.append(dataframe['close'] > dataframe['sma'])
        if params['mfi']['enabled']:
            conditions.append(dataframe['mfi'] < params['mfi']['value'])
        if params['fastd']['enabled']:
            conditions.append(dataframe['fastd'] < params['fastd']['value'])
        if params['adx']['enabled']:
            conditions.append(dataframe['adx'] > params['adx']['value'])
        if params['cci']['enabled']:
            conditions.append(dataframe['cci'] < params['cci']['value'])
        if params['rsi']['enabled']:
            conditions.append(dataframe['rsi'] < params['rsi']['value'])
        if params['over_sar']['enabled']:
            conditions.append(dataframe['close'] > dataframe['sar'])
        if params['uptrend_sma']['enabled']:
            prevsma = dataframe['sma'].shift(1)
            conditions.append(dataframe['sma'] > prevsma)

        prev_fastd = dataframe['fastd'].shift(1)
        # TRIGGERS
        triggers = {
            'lower_bb': dataframe['tema'] <= dataframe['blower'],
            'faststoch10': (dataframe['fastd'] >= 10) & (prev_fastd < 10),
            'ao_cross_zero': (crossed_above(dataframe['ao'], 0.0)),
        }
        conditions.append(triggers.get(params['trigger']['type']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1
        dataframe.loc[dataframe['buy'] == 1, 'buy_price'] = dataframe['close']

        return dataframe
    return populate_buy_trend

@pytest.mark.skipif(not os.environ.get('BACKTEST', False), reason="BACKTEST not set")
def test_hyperopt(conf, pairs, mocker):
    def optimizer(params):
        buy_strategy = buy_strategy_generator(params)
        mocker.patch('freqtrade.analyze.populate_buy_trend', side_effect=buy_strategy)
        results = backtest(conf, pairs, mocker)

        result = format_results(results)
        print(result)

        total_profit = results.profit.sum() * 1000
        trade_count = len(results.index)

        trade_loss = 1 - 0.8 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5)
        profit_loss = exp(-total_profit**3 / 10**11)

        return {
            'loss': trade_loss + profit_loss,
            'status': STATUS_OK,
            'result': result
        }

    space = {
        'mfi': hp.choice('mfi', [
            {'enabled': False},
            {'enabled': True, 'value': hp.uniform('mfi-value', 2, 40)}
        ]),
        'fastd': hp.choice('fastd', [
            {'enabled': False},
            {'enabled': True, 'value': hp.uniform('fastd-value', 2, 40)}
        ]),
        'adx': hp.choice('adx', [
            {'enabled': False},
            {'enabled': True, 'value': hp.uniform('adx-value', 2, 40)}
        ]),
        'cci': hp.choice('cci', [
            {'enabled': False},
            {'enabled': True, 'value': hp.uniform('cci-value', -200, -100)}
        ]),
        'rsi': hp.choice('rsi', [
            {'enabled': False},
            {'enabled': True, 'value': hp.uniform('rsi-value', 20, 30)}
        ]),
        'below_sma': hp.choice('below_sma', [
            {'enabled': False},
            {'enabled': True}
        ]),
        'over_sma': hp.choice('over_sma', [
            {'enabled': False},
            {'enabled': True}
        ]),
        'over_sar': hp.choice('over_sar', [
            {'enabled': False},
            {'enabled': True}
        ]),
        'uptrend_sma': hp.choice('uptrend_sma', [
            {'enabled': False},
            {'enabled': True}
        ]),
        'trigger': hp.choice('trigger', [
            {'type': 'lower_bb'},
            {'type': 'faststoch10'},
            {'type': 'ao_cross_zero'}
        ]),
    }
    trials = Trials()
    best = fmin(fn=optimizer, space=space, algo=tpe.suggest, max_evals=40, trials=trials)
    print('\n\n\n\n====================== HYPEROPT BACKTESTING REPORT ================================')
    print('Best parameters {}'.format(best))
    newlist = sorted(trials.results, key=itemgetter('loss'))
    print('Result: {}'.format(newlist[0]['result']))
