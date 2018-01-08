# pragma pylint: disable=missing-docstring,W0212,W0603


import json
import logging
import sys
from functools import reduce
from math import exp
from operator import itemgetter

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, space_eval
from hyperopt.mongoexp import MongoTrials
from pandas import DataFrame

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
CURRENT_BEST_LOSS = 100

# this is expexted avg profit * expected trade count
# for example 3.5%, 1100 trades, EXPECTED_MAX_PROFIT = 3.85
EXPECTED_MAX_PROFIT = 3.85

# Configuration and data used by hyperopt
PROCESSED = None  # optimize.preprocess(optimize.load_data())
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
    """ log results if it is better than any previous evaluation """
    global CURRENT_BEST_LOSS

    if results['loss'] < CURRENT_BEST_LOSS:
        CURRENT_BEST_LOSS = results['loss']
        logger.info('{:5d}/{}: {}'.format(
            results['current_tries'],
            results['total_tries'],
            results['result']))
    else:
        print('.', end='')
        sys.stdout.flush()


def calculate_loss(total_profit: float, trade_count: int):
    """ objective function, returns smaller number for more optimal results """
    trade_loss = 1 - 0.35 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5.2)
    profit_loss = max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
    return trade_loss + profit_loss


def optimizer(params):
    global _CURRENT_TRIES

    from freqtrade.optimize import backtesting
    backtesting.populate_buy_trend = buy_strategy_generator(params)

    results = backtest(OPTIMIZE_CONFIG['stake_amount'], PROCESSED)
    result_explanation = format_results(results)

    total_profit = results.profit_percent.sum()
    trade_count = len(results.index)

    if trade_count == 0:
        print('.', end='')
        return {
            'status': STATUS_FAIL,
            'loss': float('inf')
        }

    loss = calculate_loss(total_profit, trade_count)

    _CURRENT_TRIES += 1

    log_results({
        'loss': loss,
        'current_tries': _CURRENT_TRIES,
        'total_tries': TOTAL_TRIES,
        'result': result_explanation,
    })

    return {
        'loss': loss,
        'status': STATUS_OK,
        'result': result_explanation,
    }


def format_results(results: DataFrame):
    return ('{:6d} trades. Avg profit {: 5.2f}%. '
            'Total profit {: 11.8f} BTC. Avg duration {:5.1f} mins.').format(
                len(results.index),
                results.profit_percent.mean() * 100.0,
                results.profit_BTC.sum(),
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
    global TOTAL_TRIES, PROCESSED, SPACE
    TOTAL_TRIES = args.epochs

    exchange._API = Bittrex({'key': '', 'secret': ''})

    # Initialize logger
    logging.basicConfig(
        level=args.loglevel,
        format='\n%(message)s',
    )

    logger.info('Using config: %s ...', args.config)
    config = load_config(args.config)
    pairs = config['exchange']['pair_whitelist']
    PROCESSED = optimize.preprocess(optimize.load_data(
        args.datadir, pairs=pairs, ticker_interval=args.ticker_interval))

    if args.mongodb:
        logger.info('Using mongodb ...')
        logger.info('Start scripts/start-mongodb.sh and start-hyperopt-worker.sh manually!')

        db_name = 'freqtrade_hyperopt'
        trials = MongoTrials('mongo://127.0.0.1:1234/{}/jobs'.format(db_name), exp_key='exp1')
    else:
        trials = Trials()

    try:
        best_parameters = fmin(
            fn=optimizer,
            space=SPACE,
            algo=tpe.suggest,
            max_evals=TOTAL_TRIES,
            trials=trials
        )

        results = sorted(trials.results, key=itemgetter('loss'))
        best_result = results[0]['result']

    except ValueError:
        best_parameters = {}
        best_result = 'Sorry, Hyperopt was not able to find good parameters. Please ' \
                      'try with more epochs (param: -e).'

    # Improve best parameter logging display
    if best_parameters:
        best_parameters = space_eval(SPACE, best_parameters)

    logger.info('Best parameters:\n%s', json.dumps(best_parameters, indent=4))
    logger.info('Best Result:\n%s', best_result)
