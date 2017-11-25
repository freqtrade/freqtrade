# pragma pylint: disable=missing-docstring,W0212


import json

from functools import reduce
from math import exp
from operator import itemgetter
from pprint import pprint

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from pandas import DataFrame

from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.optimize.backtesting import backtest
from freqtrade.vendor.qtpylib.indicators import crossed_above

# set TARGET_TRADES to suit your number concurrent trades so its realistic to 20days of data
TARGET_TRADES = 1100
TOTAL_TRIES = 4
_CURRENT_TRIES = 0

# Configuration and data used by hyperopt
PROCESSED = optimize.preprocess(optimize.load_data())
OPTIMIZE_CONFIG = {
    'max_open_trades': 3,
    'stake_currency': 'BTC',
    'stake_amount': 0.01,
    'minimal_roi': {
        '40':  0.0,
        '30':  0.01,
        '20':  0.02,
        '0':  0.04,
    },
    'stoploss': -0.10,
}

SPACE = {
    'mfi': hp.choice('mfi', [
        {'enabled': False},
        {'enabled': True, 'value': hp.quniform('mfi-value', 5, 25, 1)}
    ]),
    'fastd': hp.choice('fastd', [
        {'enabled': False},
        {'enabled': True, 'value': hp.quniform('fastd-value', 10, 50, 1)}
    ]),
    'adx': hp.choice('adx', [
        {'enabled': False},
        {'enabled': True, 'value': hp.quniform('adx-value', 15, 50, 1)}
    ]),
    'rsi': hp.choice('rsi', [
        {'enabled': False},
        {'enabled': True, 'value': hp.quniform('rsi-value', 20, 40, 1)}
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


def optimizer(params):
    global _CURRENT_TRIES

    from freqtrade.optimize import backtesting
    backtesting.populate_buy_trend = buy_strategy_generator(params)

    results = backtest(OPTIMIZE_CONFIG, PROCESSED)

    result = format_results(results)

    total_profit = results.profit.sum() * 1000
    trade_count = len(results.index)

    trade_loss = 1 - 0.35 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5.2)
    profit_loss = max(0, 1 - total_profit / 10000)  # max profit 10000

    _CURRENT_TRIES += 1
    print('{:5d}/{}: {}'.format(_CURRENT_TRIES, TOTAL_TRIES, result))

    return {
        'loss': trade_loss + profit_loss,
        'status': STATUS_OK,
        'result': result
    }


def format_results(results: DataFrame):
    return ('Made {:6d} buys. Average profit {: 5.2f}%. '
            'Total profit was {: 7.3f}. Average duration {:5.1f} mins.').format(
                len(results.index),
                results.profit.mean() * 100.0,
                results.profit.sum(),
                results.duration.mean() * 5,
            )


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

        return dataframe
    return populate_buy_trend


def start(args):
    global TOTAL_TRIES
    TOTAL_TRIES = args.epochs

    # Monkey patch config
    from freqtrade import main
    main._CONF = OPTIMIZE_CONFIG

    exchange._API = Bittrex({'key': '', 'secret': ''})

    trials = Trials()
    best = fmin(fn=optimizer, space=SPACE, algo=tpe.suggest, max_evals=TOTAL_TRIES, trials=trials)
    print('\n==================== HYPEROPT BACKTESTING REPORT ==============================\n')
    print('Best parameters: {}'.format(json.dumps(best, indent=4)))
    results = sorted(trials.results, key=itemgetter('loss'))
    print('Best Result: {}\n'.format(results[0]['result']))
