# pragma pylint: disable=missing-docstring
import logging
import os
from functools import reduce
from math import exp
from operator import itemgetter

import pytest
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from pandas import DataFrame

from freqtrade.tests.test_backtesting import backtest, format_results
from freqtrade.vendor.qtpylib.indicators import crossed_above

logging.disable(logging.DEBUG)  # disable debug logs that slow backtesting a lot

# set TARGET_TRADES to suit your number concurrent trades so its realistic to 20days of data
TARGET_TRADES = 1300
TOTAL_TRIES = 4
current_tries = 0

def buy_strategy_generator(params):
    def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS
        if params['uptrend_long_ema']['enabled']:
            conditions.append(dataframe['ema50'] > dataframe['ema100'])
        if params['uptrend_short_ema']['enabled']:
            conditions.append(dataframe['ema5'] > dataframe['ema10'])
        if params['mfi']['enabled']:
            conditions.append(dataframe['mfi'] < params['mfi']['value'])
        if params['fastd']['enabled']:
            conditions.append(dataframe['fastd'] < params['fastd']['value'])
        if params['adx']['enabled']:
            conditions.append(dataframe['adx'] > params['adx']['value'])
        if params['rsi']['enabled']:
            conditions.append(dataframe['rsi'] < params['rsi']['value'])
        if params['over_sar']['enabled']:
            conditions.append(dataframe['close'] > dataframe['sar'])
        if params['green_candle']['enabled']:
            conditions.append(dataframe['close'] > dataframe['open'])
        if params['uptrend_sma']['enabled']:
            prevsma = dataframe['sma'].shift(1)
            conditions.append(dataframe['sma'] > prevsma)

        # TRIGGERS
        triggers = {
            'lower_bb': dataframe['tema'] <= dataframe['blower'],
            'faststoch10': (crossed_above(dataframe['fastd'], 10.0)),
            'ao_cross_zero': (crossed_above(dataframe['ao'], 0.0)),
            'ema5_cross_ema10': (crossed_above(dataframe['ema5'], dataframe['ema10'])),
            'macd_cross_signal': (crossed_above(dataframe['macd'], dataframe['macdsignal'])),
            'sar_reversal': (crossed_above(dataframe['close'], dataframe['sar'])),
            'stochf_cross': (crossed_above(dataframe['fastk'], dataframe['fastd'])),
            'ht_sine': (crossed_above(dataframe['htleadsine'], dataframe['htsine'])),
        }
        conditions.append(triggers.get(params['trigger']['type']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1
        dataframe.loc[dataframe['buy'] == 1, 'buy_price'] = dataframe['close']

        return dataframe
    return populate_buy_trend


@pytest.mark.skipif(not os.environ.get('BACKTEST', False), reason="BACKTEST not set")
def test_hyperopt(backtest_conf, backdata, mocker):
    mocked_buy_trend = mocker.patch('freqtrade.analyze.populate_buy_trend')

    def optimizer(params):
        mocked_buy_trend.side_effect = buy_strategy_generator(params)

        results = backtest(backtest_conf, backdata, mocker)

        result = format_results(results)

        total_profit = results.profit.sum() * 1000
        trade_count = len(results.index)

        trade_loss = 1 - 0.4 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5.2)
        profit_loss = max(0, 1 - total_profit / 15000)  # max profit 15000

        global current_tries
        current_tries += 1
        print('{}/{}: {}'.format(current_tries, TOTAL_TRIES, result))

        return {
            'loss': trade_loss + profit_loss,
            'status': STATUS_OK,
            'result': result
        }

    space = {
        'mfi': hp.choice('mfi', [
            {'enabled': False},
            {'enabled': True, 'value': hp.quniform('mfi-value', 5, 15, 1)}
        ]),
        'fastd': hp.choice('fastd', [
            {'enabled': False},
            {'enabled': True, 'value': hp.quniform('fastd-value', 5, 40, 1)}
        ]),
        'adx': hp.choice('adx', [
            {'enabled': False},
            {'enabled': True, 'value': hp.quniform('adx-value', 10, 50, 1)}
        ]),
        'rsi': hp.choice('rsi', [
            {'enabled': False},
            {'enabled': True, 'value': hp.quniform('rsi-value', 20, 30, 1)}
        ]),
        'uptrend_long_ema': hp.choice('uptrend_long_ema', [
            {'enabled': False},
            {'enabled': True}
        ]),
        'uptrend_short_ema': hp.choice('uptrend_short_ema', [
            {'enabled': False},
            {'enabled': True}
        ]),
        'over_sar': hp.choice('over_sar', [
            {'enabled': False},
            {'enabled': True}
        ]),
        'green_candle': hp.choice('green_candle', [
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
            {'type': 'ao_cross_zero'},
            {'type': 'ema5_cross_ema10'},
            {'type': 'macd_cross_signal'},
            {'type': 'sar_reversal'},
            {'type': 'stochf_cross'},
            {'type': 'ht_sine'},
        ]),
    }
    trials = Trials()
    best = fmin(fn=optimizer, space=space, algo=tpe.suggest, max_evals=TOTAL_TRIES, trials=trials)
    print('\n\n\n\n==================== HYPEROPT BACKTESTING REPORT ==============================')
    print('Best parameters {}'.format(best))
    newlist = sorted(trials.results, key=itemgetter('loss'))
    print('Result: {}'.format(newlist[0]['result']))
