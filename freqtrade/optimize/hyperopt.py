# pragma pylint: disable=missing-docstring,W0212,W0603


import json
import logging
import sys
import pickle
import signal
import os
from math import exp
from operator import itemgetter

from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, space_eval, tpe
from hyperopt.mongoexp import MongoTrials
from pandas import DataFrame

from freqtrade import main, misc  # noqa
from freqtrade import exchange, optimize
from freqtrade.exchange import Bittrex
from freqtrade.misc import load_config
from freqtrade.optimize.backtesting import backtest
from freqtrade.strategy.strategy import Strategy
from user_data.hyperopt_conf import hyperopt_optimize_conf

# Remove noisy log messages
logging.getLogger('hyperopt.mongoexp').setLevel(logging.WARNING)
logging.getLogger('hyperopt.tpe').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# set TARGET_TRADES to suit your number concurrent trades so its realistic to 20days of data
TARGET_TRADES = 600
TOTAL_TRIES = 0
_CURRENT_TRIES = 0
CURRENT_BEST_LOSS = 100

# max average trade duration in minutes
# if eval ends with higher value, we consider it a failed eval
MAX_ACCEPTED_TRADE_DURATION = 300

# this is expexted avg profit * expected trade count
# for example 3.5%, 1100 trades, EXPECTED_MAX_PROFIT = 3.85
# check that the reported Σ% values do not exceed this!
EXPECTED_MAX_PROFIT = 3.0

# Configuration and data used by hyperopt
PROCESSED = None  # optimize.preprocess(optimize.load_data())
OPTIMIZE_CONFIG = hyperopt_optimize_conf()

# Hyperopt Trials
TRIALS_FILE = os.path.join('freqtrade', 'optimize', 'hyperopt_trials.pickle')
TRIALS = Trials()

# Monkey patch config
from freqtrade import main  # noqa
main._CONF = OPTIMIZE_CONFIG


def save_trials(trials, trials_path=TRIALS_FILE):
    """Save hyperopt trials to file"""
    logger.info('Saving Trials to \'{}\''.format(trials_path))
    pickle.dump(trials, open(trials_path, 'wb'))


def read_trials(trials_path=TRIALS_FILE):
    """Read hyperopt trials file"""
    logger.info('Reading Trials from \'{}\''.format(trials_path))
    trials = pickle.load(open(trials_path, 'rb'))
    os.remove(trials_path)
    return trials


def log_trials_result(trials):
    vals = json.dumps(trials.best_trial['misc']['vals'], indent=4)
    results = trials.best_trial['result']['result']
    logger.info('Best result:\n%s\nwith values:\n%s', results, vals)


def log_results(results):
    """ log results if it is better than any previous evaluation """
    global CURRENT_BEST_LOSS

    if results['loss'] < CURRENT_BEST_LOSS:
        CURRENT_BEST_LOSS = results['loss']
        logger.info('{:5d}/{}: {}. Loss {:.5f}'.format(
            results['current_tries'],
            results['total_tries'],
            results['result'],
            results['loss']))
    else:
        print('.', end='')
        sys.stdout.flush()


def calculate_loss(total_profit: float, trade_count: int, trade_duration: float):
    """ objective function, returns smaller number for more optimal results """
    trade_loss = 1 - 0.25 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5.8)
    profit_loss = max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
    duration_loss = 0.7 + 0.3 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1)
    return trade_loss + profit_loss + duration_loss


def optimizer(params):
    global _CURRENT_TRIES

    from freqtrade.optimize import backtesting

    strategy = Strategy()
    backtesting.populate_buy_trend = strategy.buy_strategy_generator(params)

    results = backtest({'stake_amount': OPTIMIZE_CONFIG['stake_amount'],
                        'processed': PROCESSED,
                        'stoploss': params['stoploss']})
    result_explanation = format_results(results)

    total_profit = results.profit_percent.sum()
    trade_count = len(results.index)
    trade_duration = results.duration.mean() * 5

    if trade_count == 0 or trade_duration > MAX_ACCEPTED_TRADE_DURATION:
        print('.', end='')
        return {
            'status': STATUS_FAIL,
            'loss': float('inf')
        }

    loss = calculate_loss(total_profit, trade_count, trade_duration)

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
            'Total profit {: 11.8f} BTC ({:.4f}Σ%). Avg duration {:5.1f} mins.').format(
                len(results.index),
                results.profit_percent.mean() * 100.0,
                results.profit_BTC.sum(),
                results.profit_percent.sum(),
                results.duration.mean() * 5,
            )


def start(args):
    global TOTAL_TRIES, PROCESSED, TRIALS, _CURRENT_TRIES

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

    # init the strategy to use
    config.update({'strategy': args.strategy})
    strategy = Strategy()
    strategy.init(config)

    timerange = misc.parse_timerange(args.timerange)
    data = optimize.load_data(args.datadir, pairs=pairs,
                              ticker_interval=args.ticker_interval,
                              timerange=timerange)
    PROCESSED = optimize.tickerdata_to_dataframe(data)

    if args.mongodb:
        logger.info('Using mongodb ...')
        logger.info('Start scripts/start-mongodb.sh and start-hyperopt-worker.sh manually!')

        db_name = 'freqtrade_hyperopt'
        TRIALS = MongoTrials('mongo://127.0.0.1:1234/{}/jobs'.format(db_name), exp_key='exp1')
    else:
        logger.info('Preparing Trials..')
        signal.signal(signal.SIGINT, signal_handler)
        # read trials file if we have one
        if os.path.exists(TRIALS_FILE):
            TRIALS = read_trials()

            _CURRENT_TRIES = len(TRIALS.results)
            TOTAL_TRIES = TOTAL_TRIES + _CURRENT_TRIES
            logger.info(
                'Continuing with trials. Current: {}, Total: {}'
                .format(_CURRENT_TRIES, TOTAL_TRIES))

    try:
        best_parameters = fmin(
            fn=optimizer,
            space=strategy.hyperopt_space(),
            algo=tpe.suggest,
            max_evals=TOTAL_TRIES,
            trials=TRIALS
        )

        results = sorted(TRIALS.results, key=itemgetter('loss'))
        best_result = results[0]['result']

    except ValueError:
        best_parameters = {}
        best_result = 'Sorry, Hyperopt was not able to find good parameters. Please ' \
                      'try with more epochs (param: -e).'

    # Improve best parameter logging display
    if best_parameters:
        best_parameters = space_eval(
            strategy.hyperopt_space(),
            best_parameters
        )

    logger.info('Best parameters:\n%s', json.dumps(best_parameters, indent=4))
    logger.info('Best Result:\n%s', best_result)

    # Store trials result to file to resume next time
    save_trials(TRIALS)


def signal_handler(sig, frame):
    """Hyperopt SIGINT handler"""
    logger.info('Hyperopt received {}'.format(signal.Signals(sig).name))

    save_trials(TRIALS)
    log_trials_result(TRIALS)
    sys.exit(0)
