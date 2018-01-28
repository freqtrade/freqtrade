# pragma pylint: disable=missing-docstring,W0212,W0603


import json
import logging
import os
import pickle
import signal
import sys
from functools import reduce
from math import exp
from operator import itemgetter
from typing import Dict, Any, Callable

import numpy
import talib.abstract as ta
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.mongoexp import MongoTrials
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
# Monkey patch config
from freqtrade import main  # noqa; noqa
from freqtrade import exchange, misc, optimize
from freqtrade.exchange import Bittrex
from freqtrade.misc import load_config
from freqtrade.optimize import backtesting
from freqtrade.optimize.backtesting import backtest
from freqtrade.strategy.strategy import Strategy
from user_data.hyperopt_conf import hyperopt_optimize_conf

# Remove noisy log messages
logging.getLogger('hyperopt.mongoexp').setLevel(logging.WARNING)
logging.getLogger('hyperopt.tpe').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# set TARGET_TRADES to suit your number concurrent trades so its realistic to the number of days
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
TRIALS_FILE = os.path.join('user_data', 'hyperopt_trials.pickle')
TRIALS = Trials()

main._CONF = OPTIMIZE_CONFIG


def populate_indicators(dataframe: DataFrame) -> DataFrame:
    """
    Adds several different TA indicators to the given DataFrame
    """
    dataframe['adx'] = ta.ADX(dataframe)
    dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
    dataframe['cci'] = ta.CCI(dataframe)
    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']
    dataframe['mfi'] = ta.MFI(dataframe)
    dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
    dataframe['minus_di'] = ta.MINUS_DI(dataframe)
    dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
    dataframe['plus_di'] = ta.PLUS_DI(dataframe)
    dataframe['roc'] = ta.ROC(dataframe)
    dataframe['rsi'] = ta.RSI(dataframe)
    # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
    rsi = 0.1 * (dataframe['rsi'] - 50)
    dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
    # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
    dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)
    # Stoch
    stoch = ta.STOCH(dataframe)
    dataframe['slowd'] = stoch['slowd']
    dataframe['slowk'] = stoch['slowk']
    # Stoch fast
    stoch_fast = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch_fast['fastd']
    dataframe['fastk'] = stoch_fast['fastk']
    # Stoch RSI
    stoch_rsi = ta.STOCHRSI(dataframe)
    dataframe['fastd_rsi'] = stoch_rsi['fastd']
    dataframe['fastk_rsi'] = stoch_rsi['fastk']
    # Bollinger bands
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
    dataframe['bb_lowerband'] = bollinger['lower']
    dataframe['bb_middleband'] = bollinger['mid']
    dataframe['bb_upperband'] = bollinger['upper']
    # EMA - Exponential Moving Average
    dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
    dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
    dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
    dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
    # SAR Parabolic
    dataframe['sar'] = ta.SAR(dataframe)
    # SMA - Simple Moving Average
    dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
    # TEMA - Triple Exponential Moving Average
    dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
    # Hilbert Transform Indicator - SineWave
    hilbert = ta.HT_SINE(dataframe)
    dataframe['htsine'] = hilbert['sine']
    dataframe['htleadsine'] = hilbert['leadsine']

    # Pattern Recognition - Bullish candlestick patterns
    # ------------------------------------
    """
    # Hammer: values [0, 100]
    dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
    # Inverted Hammer: values [0, 100]
    dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
    # Dragonfly Doji: values [0, 100]
    dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
    # Piercing Line: values [0, 100]
    dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
    # Morningstar: values [0, 100]
    dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
    # Three White Soldiers: values [0, 100]
    dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]
    """

    # Pattern Recognition - Bearish candlestick patterns
    # ------------------------------------
    """
    # Hanging Man: values [0, 100]
    dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
    # Shooting Star: values [0, 100]
    dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
    # Gravestone Doji: values [0, 100]
    dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
    # Dark Cloud Cover: values [0, 100]
    dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
    # Evening Doji Star: values [0, 100]
    dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
    # Evening Star: values [0, 100]
    dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)
    """

    # Pattern Recognition - Bullish/Bearish candlestick patterns
    # ------------------------------------
    """
    # Three Line Strike: values [0, -100, 100]
    dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
    # Spinning Top: values [0, -100, 100]
    dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
    # Engulfing: values [0, -100, 100]
    dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
    # Harami: values [0, -100, 100]
    dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
    # Three Outside Up/Down: values [0, -100, 100]
    dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
    # Three Inside Up/Down: values [0, -100, 100]
    dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]
    """

    # Chart type
    # ------------------------------------
    # Heikinashi stategy
    heikinashi = qtpylib.heikinashi(dataframe)
    dataframe['ha_open'] = heikinashi['open']
    dataframe['ha_close'] = heikinashi['close']
    dataframe['ha_high'] = heikinashi['high']
    dataframe['ha_low'] = heikinashi['low']

    return dataframe


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
    duration_loss = 0.4 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1)
    return trade_loss + profit_loss + duration_loss


def generate_roi_table(params) -> Dict[str, float]:
    roi_table = {}
    roi_table["0"] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
    roi_table[str(params['roi_t3'])] = params['roi_p1'] + params['roi_p2']
    roi_table[str(params['roi_t3'] + params['roi_t2'])] = params['roi_p1']
    roi_table[str(params['roi_t3'] + params['roi_t2'] + params['roi_t1'])] = 0

    return roi_table


def roi_space() -> Dict[str, Any]:
    return {
        'roi_t1': hp.quniform('roi_t1', 10, 120, 20),
        'roi_t2': hp.quniform('roi_t2', 10, 60, 15),
        'roi_t3': hp.quniform('roi_t3', 10, 40, 10),
        'roi_p1': hp.quniform('roi_p1', 0.01, 0.04, 0.01),
        'roi_p2': hp.quniform('roi_p2', 0.01, 0.07, 0.01),
        'roi_p3': hp.quniform('roi_p3', 0.01, 0.20, 0.01),
    }


def stoploss_space() -> Dict[str, Any]:
    return {
        'stoploss': hp.quniform('stoploss', -0.5, -0.02, 0.02),
    }


def indicator_space() -> Dict[str, Any]:
    """
    Define your Hyperopt space for searching strategy parameters
    """
    return {
        'macd_below_zero': hp.choice('macd_below_zero', [
            {'enabled': False},
            {'enabled': True}
        ]),
        'mfi': hp.choice('mfi', [
            {'enabled': False},
            {'enabled': True, 'value': hp.quniform('mfi-value', 10, 25, 5)}
        ]),
        'fastd': hp.choice('fastd', [
            {'enabled': False},
            {'enabled': True, 'value': hp.quniform('fastd-value', 15, 45, 5)}
        ]),
        'adx': hp.choice('adx', [
            {'enabled': False},
            {'enabled': True, 'value': hp.quniform('adx-value', 20, 50, 5)}
        ]),
        'rsi': hp.choice('rsi', [
            {'enabled': False},
            {'enabled': True, 'value': hp.quniform('rsi-value', 20, 40, 5)}
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
            {'type': 'lower_bb_tema'},
            {'type': 'faststoch10'},
            {'type': 'ao_cross_zero'},
            {'type': 'ema3_cross_ema10'},
            {'type': 'macd_cross_signal'},
            {'type': 'sar_reversal'},
            {'type': 'ht_sine'},
            {'type': 'heiken_reversal_bull'},
            {'type': 'di_cross'},
        ]),
    }


def hyperopt_space() -> Dict[str, Any]:
    return {**indicator_space(), **roi_space(), **stoploss_space()}


def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
    """
    Define the buy strategy parameters to be used by hyperopt
    """
    def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS
        if 'uptrend_long_ema' in params and params['uptrend_long_ema']['enabled']:
            conditions.append(dataframe['ema50'] > dataframe['ema100'])
        if 'macd_below_zero' in params and params['macd_below_zero']['enabled']:
            conditions.append(dataframe['macd'] < 0)
        if 'uptrend_short_ema' in params and params['uptrend_short_ema']['enabled']:
            conditions.append(dataframe['ema5'] > dataframe['ema10'])
        if 'mfi' in params and params['mfi']['enabled']:
            conditions.append(dataframe['mfi'] < params['mfi']['value'])
        if 'fastd' in params and params['fastd']['enabled']:
            conditions.append(dataframe['fastd'] < params['fastd']['value'])
        if 'adx' in params and params['adx']['enabled']:
            conditions.append(dataframe['adx'] > params['adx']['value'])
        if 'rsi' in params and params['rsi']['enabled']:
            conditions.append(dataframe['rsi'] < params['rsi']['value'])
        if 'over_sar' in params and params['over_sar']['enabled']:
            conditions.append(dataframe['close'] > dataframe['sar'])
        if 'green_candle' in params and params['green_candle']['enabled']:
            conditions.append(dataframe['close'] > dataframe['open'])
        if 'uptrend_sma' in params and params['uptrend_sma']['enabled']:
            prevsma = dataframe['sma'].shift(1)
            conditions.append(dataframe['sma'] > prevsma)

        # TRIGGERS
        triggers = {
            'lower_bb': (
                dataframe['close'] < dataframe['bb_lowerband']
            ),
            'lower_bb_tema': (
                dataframe['tema'] < dataframe['bb_lowerband']
            ),
            'faststoch10': (qtpylib.crossed_above(
                dataframe['fastd'], 10.0
            )),
            'ao_cross_zero': (qtpylib.crossed_above(
                dataframe['ao'], 0.0
            )),
            'ema3_cross_ema10': (qtpylib.crossed_above(
                dataframe['ema3'], dataframe['ema10']
            )),
            'macd_cross_signal': (qtpylib.crossed_above(
                dataframe['macd'], dataframe['macdsignal']
            )),
            'sar_reversal': (qtpylib.crossed_above(
                dataframe['close'], dataframe['sar']
            )),
            'ht_sine': (qtpylib.crossed_above(
                dataframe['htleadsine'], dataframe['htsine']
            )),
            'heiken_reversal_bull': (
                (qtpylib.crossed_above(dataframe['ha_close'], dataframe['ha_open'])) &
                (dataframe['ha_low'] == dataframe['ha_open'])
            ),
            'di_cross': (qtpylib.crossed_above(
                dataframe['plus_di'], dataframe['minus_di']
            )),
        }
        conditions.append(triggers.get(params['trigger']['type']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    return populate_buy_trend


def optimizer(params):
    global _CURRENT_TRIES

    if 'roi_t1' in params:
        strategy = Strategy()
        strategy.minimal_roi = generate_roi_table(params)

    backtesting.populate_buy_trend = buy_strategy_generator(params)

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
    optimize.populate_indicators = populate_indicators
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
            space=hyperopt_space(),
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
            hyperopt_space(),
            best_parameters
        )

    logger.info('Best parameters:\n%s', json.dumps(best_parameters, indent=4))
    if 'roi_t1' in best_parameters:
        logger.info('ROI table:\n%s', generate_roi_table(best_parameters))
    logger.info('Best Result:\n%s', best_result)

    # Store trials result to file to resume next time
    save_trials(TRIALS)


def signal_handler(sig, frame):
    """Hyperopt SIGINT handler"""
    logger.info('Hyperopt received {}'.format(signal.Signals(sig).name))

    save_trials(TRIALS)
    log_trials_result(TRIALS)
    sys.exit(0)
