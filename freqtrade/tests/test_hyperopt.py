# pragma pylint: disable=missing-docstring
import json
import logging
import os
from functools import reduce

import pytest
import arrow
from pandas import DataFrame

import hyperopt.pyll.stochastic

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
            "50":  0.0,
            "40":  0.01,
            "30":  0.02,
            "0":  0.045
        },
        "stoploss": -0.40
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
    if len(results.index) < 800:
        return 100000 # return large number to "ignore" this result
    return results.duration.mean() * results.duration.mean() / results.profit.sum() / results.profit.mean() # the smaller the better

def buy_strategy_generator(params):
    print(params)
    def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
        conditions = []
        if params['below_sma']['enabled']:
            conditions.append(dataframe['close'] < dataframe['sma'])
        if params['over_sma']['enabled']:
            conditions.append(dataframe['close'] > dataframe['sma'])
        conditions.append(dataframe['tema'] <= dataframe['blower'])
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
    }

    # print(hyperopt.pyll.stochastic.sample(space))
    print('Best parameters {}'.format(fmin(fn=optimizer, space=space, algo=tpe.suggest, max_evals=10)))
