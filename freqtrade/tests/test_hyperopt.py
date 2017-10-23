# pragma pylint: disable=missing-docstring
import json
import logging
import os
from functools import reduce

import pytest
import arrow
from pandas import DataFrame

from hyperopt import fmin, tpe, hp

from freqtrade.analyze import analyze_ticker
from freqtrade.main import should_sell
from freqtrade.persistence import Trade

logging.disable(logging.DEBUG) # disable debug logs that slow backtesting a lot

def print_results(results):
    print('Made {} buys. Average profit {:.2f}%. Total profit was {:.3f}. Average duration {:.1f} mins.'.format(
        len(results.index),
        results.profit.mean() * 100.0,
        results.profit.sum(),
        results.duration.mean() * 5
    ))

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


def backtest(conf, pairs, mocker, buy_strategy):
    trades = []
    mocker.patch.dict('freqtrade.main._CONF', conf)
    for pair in pairs:
        with open('freqtrade/tests/testdata/'+pair+'.json') as data_file:
            data = json.load(data_file)

            mocker.patch('freqtrade.analyze.get_ticker_history', return_value=data)
            mocker.patch('arrow.utcnow', return_value=arrow.get('2017-08-20T14:50:00'))
            mocker.patch('freqtrade.analyze.populate_buy_trend', side_effect=buy_strategy)
            ticker = analyze_ticker(pair)
            # for each buy point
            for index, row in ticker[ticker.buy == 1].iterrows():
                trade = Trade(
                    open_rate=row['close'],
                    open_date=arrow.get(row['date']).datetime,
                    amount=1,
                )
                # calculate win/lose forwards from buy point
                for index2, row2 in ticker[index:].iterrows():
                    if should_sell(trade, row2['close'], arrow.get(row2['date']).datetime):
                        current_profit = (row2['close'] - trade.open_rate) / trade.open_rate

                        trades.append((pair, current_profit, index2 - index))
                        break

    labels = ['currency', 'profit', 'duration']
    results = DataFrame.from_records(trades, columns=labels)

    print_results(results)

    # set the value below to suit your number concurrent trades so its realistic to 20days of data
    TARGET_TRADES = 1200
    if results.profit.sum() == 0 or results.profit.mean() == 0:
        return 49999999999 # avoid division by zero, return huge value to discard result
    return abs(len(results.index) - 1200.1) / (results.profit.sum() ** 2) * results.duration.mean() # the smaller the better

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
        return backtest(conf, pairs, mocker, buy_strategy_generator(params))

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
            {'type': 'faststoch10'}
        ]),
    }

    print('Best parameters {}'.format(fmin(fn=optimizer, space=space, algo=tpe.suggest, max_evals=40)))
