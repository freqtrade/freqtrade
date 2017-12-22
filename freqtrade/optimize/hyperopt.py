# pragma pylint: disable=missing-docstring,W0212


import json
import logging
import sys
from functools import reduce
from math import exp
from operator import itemgetter

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from pandas import DataFrame
import numpy as np

from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.misc import load_config
from freqtrade.optimize.backtesting import backtest
from freqtrade.optimize.hyperopt_conf import hyperopt_optimize_conf
from freqtrade.vendor.qtpylib.indicators import crossed_above

# Remove noisy log messages
logging.getLogger('hyperopt.mongoexp').setLevel(logging.WARNING)
logging.getLogger('hyperopt.tpe').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# set TARGET_TRADES to suit your number concurrent trades so its realistic to 20days of data
TARGET_TRADES = 1100
TOTAL_TRIES = None
_CURRENT_TRIES = 0

TOTAL_PROFIT_TO_BEAT = 0
AVG_PROFIT_TO_BEAT = 0
AVG_DURATION_TO_BEAT = 100

# Configuration and data used by hyperopt
PROCESSED = optimize.preprocess(optimize.load_data())
OPTIMIZE_CONFIG = hyperopt_optimize_conf()

# Monkey patch config
from freqtrade import main  # noqa
main._CONF = OPTIMIZE_CONFIG


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


def log_results(results):
    "if results is better than _TO_BEAT show it"

    current_try = results['current_tries']
    total_tries = results['total_tries']
    result = results['result']
    profit = results['total_profit'] / 1000

    outcome = '{:5d}/{}: {}'.format(current_try, total_tries, result)

    if profit >= TOTAL_PROFIT_TO_BEAT:
        logger.info(outcome)
    else:
        print('.', end='')
        sys.stdout.flush()


def optimizer(params):
    global _CURRENT_TRIES

    from freqtrade.optimize import backtesting
    backtesting.populate_buy_trend = buy_strategy_generator(params)

    results = backtest(OPTIMIZE_CONFIG, PROCESSED)

    result = format_results(results)

    total_profit = results.profit_percent.sum() * 1000
    trade_count = len(results.index)

    trade_loss = 1 - 0.35 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5.2)
    profit_loss = max(0, 1 - total_profit / 10000)  # max profit 10000

    _CURRENT_TRIES += 1

    result_data = {
        'trade_count': trade_count,
        'total_profit': total_profit,
        'trade_loss': trade_loss,
        'profit_loss': profit_loss,
        'avg_profit': results.profit_percent.mean() * 100.0,
        'avg_duration': results.duration.mean() * 5,
        'current_tries': _CURRENT_TRIES,
        'total_tries': TOTAL_TRIES,
        'result': result,
        'results': results
    }

    # logger.info('{:5d}/{}: {}'.format(_CURRENT_TRIES, TOTAL_TRIES, result))
    log_results(result_data)

    return {
        'loss': trade_loss + profit_loss,
        'status': STATUS_OK,
        'result': result,
        'total_profit': total_profit,
        'avg_profit': result_data['avg_profit'],
    }


def format_results(results: DataFrame):
    return ('Made {:6d} buys. Average profit {: 5.2f}%. '
            'Total profit was {: 7.3f}. Average duration {:5.1f} mins.').format(
                len(results.index),
                results.profit_percent.mean() * 100.0,
                results.profit_BTC.sum(),
                results.duration.mean() * 5,
    )


def filter_nan(result, filter_key):
    return [r for r in result if not np.isnan(r[filter_key])]


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
    global TOTAL_TRIES, PROCESSED
    TOTAL_TRIES = args.epochs

    exchange._API = Bittrex({'key': '', 'secret': ''})

    # Initialize logger
    logging.basicConfig(
        level=args.loglevel,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    logger.info('Using config: %s ...', args.config)
    config = load_config(args.config)
    pairs = config['exchange']['pair_whitelist']
    PROCESSED = optimize.preprocess(optimize.load_data(
        pairs=pairs, ticker_interval=args.ticker_interval))

    if args.mongodb:
        logger.info('Using mongodb ...')
        logger.info('Start scripts/start-mongodb.sh and start-hyperopt-worker.sh manually!')

        db_name = 'freqtrade_hyperopt'
        trials = MongoTrials('mongo://127.0.0.1:1234/{}/jobs'.format(db_name), exp_key='exp1')
    else:
        trials = Trials()

    best = fmin(fn=optimizer, space=SPACE, algo=tpe.suggest, max_evals=TOTAL_TRIES, trials=trials)
    logger.info('Best parameters:\n%s', json.dumps(best, indent=4))

    filt_res = filter_nan(trials.results, 'total_profit')
    filt_res = filter_nan(filt_res, 'avg_profit')

    results = sorted(filt_res, key=itemgetter('loss'))

    logger.info('Best Result:\n%s', results[0]['result'])
