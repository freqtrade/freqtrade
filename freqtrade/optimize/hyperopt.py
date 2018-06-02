# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

"""
This module contains the hyperopt logic
"""

import json
import logging
import os
import pickle
import signal
import sys
from argparse import Namespace
from functools import reduce
from math import exp
from operator import itemgetter
from typing import Dict, Any, Callable, Optional

import numpy
import talib.abstract as ta
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.mongoexp import MongoTrials
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.optimize import load_data
from freqtrade.optimize.backtesting import Backtesting
from user_data.hyperopt_conf import hyperopt_optimize_conf

logger = logging.getLogger(__name__)


class Hyperopt(Backtesting):
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To run a backtest:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """
    def __init__(self, config: Dict[str, Any]) -> None:

        super().__init__(config)
        # set TARGET_TRADES to suit your number concurrent trades so its realistic
        # to the number of days
        self.target_trades = 600
        self.total_tries = config.get('epochs', 0)
        self.current_tries = 0
        self.current_best_loss = 100

        # max average trade duration in minutes
        # if eval ends with higher value, we consider it a failed eval
        self.max_accepted_trade_duration = 300

        # this is expexted avg profit * expected trade count
        # for example 3.5%, 1100 trades, self.expected_max_profit = 3.85
        # check that the reported Σ% values do not exceed this!
        self.expected_max_profit = 3.0

        # Configuration and data used by hyperopt
        self.processed: Optional[Dict[str, Any]] = None

        # Hyperopt Trials
        self.trials_file = os.path.join('user_data', 'hyperopt_trials.pickle')
        self.trials = Trials()

    @staticmethod
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

    def save_trials(self) -> None:
        """
        Save hyperopt trials to file
        """
        logger.info('Saving Trials to \'%s\'', self.trials_file)
        pickle.dump(self.trials, open(self.trials_file, 'wb'))

    def read_trials(self) -> Trials:
        """
        Read hyperopt trials file
        """
        logger.info('Reading Trials from \'%s\'', self.trials_file)
        trials = pickle.load(open(self.trials_file, 'rb'))
        os.remove(self.trials_file)
        return trials

    def log_trials_result(self) -> None:
        """
        Display Best hyperopt result
        """
        vals = json.dumps(self.trials.best_trial['misc']['vals'], indent=4)
        results = self.trials.best_trial['result']['result']
        logger.info('Best result:\n%s\nwith values:\n%s', results, vals)

    def log_results(self, results) -> None:
        """
        Log results if it is better than any previous evaluation
        """
        if results['loss'] < self.current_best_loss:
            self.current_best_loss = results['loss']
            log_msg = '\n{:5d}/{}: {}. Loss {:.5f}'.format(
                results['current_tries'],
                results['total_tries'],
                results['result'],
                results['loss']
            )
            print(log_msg)
        else:
            print('.', end='')
            sys.stdout.flush()

    def calculate_loss(self, total_profit: float, trade_count: int, trade_duration: float) -> float:
        """
        Objective function, returns smaller number for more optimal results
        """
        trade_loss = 1 - 0.25 * exp(-(trade_count - self.target_trades) ** 2 / 10 ** 5.8)
        profit_loss = max(0, 1 - total_profit / self.expected_max_profit)
        duration_loss = 0.4 * min(trade_duration / self.max_accepted_trade_duration, 1)
        return trade_loss + profit_loss + duration_loss

    @staticmethod
    def generate_roi_table(params: Dict) -> Dict[int, float]:
        """
        Generate the ROI table thqt will be used by Hyperopt
        """
        roi_table = {}
        roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
        roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
        roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
        roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

        return roi_table

    @staticmethod
    def roi_space() -> Dict[str, Any]:
        """
        Values to search for each ROI steps
        """
        return {
            'roi_t1': hp.quniform('roi_t1', 10, 120, 20),
            'roi_t2': hp.quniform('roi_t2', 10, 60, 15),
            'roi_t3': hp.quniform('roi_t3', 10, 40, 10),
            'roi_p1': hp.quniform('roi_p1', 0.01, 0.04, 0.01),
            'roi_p2': hp.quniform('roi_p2', 0.01, 0.07, 0.01),
            'roi_p3': hp.quniform('roi_p3', 0.01, 0.20, 0.01),
        }

    @staticmethod
    def stoploss_space() -> Dict[str, Any]:
        """
        Stoploss Value to search
        """
        return {
            'stoploss': hp.quniform('stoploss', -0.5, -0.02, 0.02),
        }

    @staticmethod
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

    def has_space(self, space: str) -> bool:
        """
        Tell if a space value is contained in the configuration
        """
        if space in self.config['spaces'] or 'all' in self.config['spaces']:
            return True
        return False

    def hyperopt_space(self) -> Dict[str, Any]:
        """
        Return the space to use during Hyperopt
        """
        spaces: Dict = {}
        if self.has_space('buy'):
            spaces = {**spaces, **Hyperopt.indicator_space()}
        if self.has_space('roi'):
            spaces = {**spaces, **Hyperopt.roi_space()}
        if self.has_space('stoploss'):
            spaces = {**spaces, **Hyperopt.stoploss_space()}
        return spaces

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by hyperopt
        """
        def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
            """
            Buy strategy Hyperopt will build and use
            """
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

    def generate_optimizer(self, params: Dict) -> Dict:
        if self.has_space('roi'):
            self.analyze.strategy.minimal_roi = self.generate_roi_table(params)

        if self.has_space('buy'):
            self.populate_buy_trend = self.buy_strategy_generator(params)

        if self.has_space('stoploss'):
            self.analyze.strategy.stoploss = params['stoploss']

        results = self.backtest(
            {
                'stake_amount': self.config['stake_amount'],
                'processed': self.processed,
                'realistic': self.config.get('realistic_simulation', False),
            }
        )
        result_explanation = self.format_results(results)

        total_profit = results.profit_percent.sum()
        trade_count = len(results.index)
        trade_duration = results.duration.mean()

        if trade_count == 0 or trade_duration > self.max_accepted_trade_duration:
            print('.', end='')
            return {
                'status': STATUS_FAIL,
                'loss': float('inf')
            }

        loss = self.calculate_loss(total_profit, trade_count, trade_duration)

        self.current_tries += 1

        self.log_results(
            {
                'loss': loss,
                'current_tries': self.current_tries,
                'total_tries': self.total_tries,
                'result': result_explanation,
            }
        )

        return {
            'loss': loss,
            'status': STATUS_OK,
            'result': result_explanation,
        }

    @staticmethod
    def format_results(results: DataFrame) -> str:
        """
        Return the format result in a string
        """
        return ('{:6d} trades. Avg profit {: 5.2f}%. '
                'Total profit {: 11.8f} BTC ({:.4f}Σ%). Avg duration {:5.1f} mins.').format(
                    len(results.index),
                    results.profit_percent.mean() * 100.0,
                    results.profit_BTC.sum(),
                    results.profit_percent.sum(),
                    results.duration.mean(),
                )

    def start(self) -> None:
        timerange = Arguments.parse_timerange(None if self.config.get(
            'timerange') is None else str(self.config.get('timerange')))
        data = load_data(  # type: ignore # timerange will be refactored
            datadir=str(self.config.get('datadir')),
            pairs=self.config['exchange']['pair_whitelist'],
            ticker_interval=self.ticker_interval,
            timerange=timerange
        )

        if self.has_space('buy'):
            self.analyze.populate_indicators = Hyperopt.populate_indicators  # type: ignore
        self.processed = self.tickerdata_to_dataframe(data)

        if self.config.get('mongodb'):
            logger.info('Using mongodb ...')
            logger.info(
                'Start scripts/start-mongodb.sh and start-hyperopt-worker.sh manually!'
            )

            db_name = 'freqtrade_hyperopt'
            self.trials = MongoTrials(
                arg='mongo://127.0.0.1:1234/{}/jobs'.format(db_name),
                exp_key='exp1'
            )
        else:
            logger.info('Preparing Trials..')
            signal.signal(signal.SIGINT, self.signal_handler)
            # read trials file if we have one
            if os.path.exists(self.trials_file) and os.path.getsize(self.trials_file) > 0:
                self.trials = self.read_trials()

                self.current_tries = len(self.trials.results)
                self.total_tries += self.current_tries
                logger.info(
                    'Continuing with trials. Current: %d, Total: %d',
                    self.current_tries,
                    self.total_tries
                )

        try:
            best_parameters = fmin(
                fn=self.generate_optimizer,
                space=self.hyperopt_space(),
                algo=tpe.suggest,
                max_evals=self.total_tries,
                trials=self.trials
            )

            results = sorted(self.trials.results, key=itemgetter('loss'))
            best_result = results[0]['result']

        except ValueError:
            best_parameters = {}
            best_result = 'Sorry, Hyperopt was not able to find good parameters. Please ' \
                          'try with more epochs (param: -e).'

        # Improve best parameter logging display
        if best_parameters:
            best_parameters = space_eval(
                self.hyperopt_space(),
                best_parameters
            )

        logger.info('Best parameters:\n%s', json.dumps(best_parameters, indent=4))
        if 'roi_t1' in best_parameters:
            logger.info('ROI table:\n%s', self.generate_roi_table(best_parameters))

        logger.info('Best Result:\n%s', best_result)

        # Store trials result to file to resume next time
        self.save_trials()

    def signal_handler(self, sig, frame) -> None:
        """
        Hyperopt SIGINT handler
        """
        logger.info(
            'Hyperopt received %s',
            signal.Signals(sig).name
        )

        self.save_trials()
        self.log_trials_result()
        sys.exit(0)


def start(args: Namespace) -> None:
    """
    Start Backtesting script
    :param args: Cli args from Arguments()
    :return: None
    """

    # Remove noisy log messages
    logging.getLogger('hyperopt.mongoexp').setLevel(logging.WARNING)
    logging.getLogger('hyperopt.tpe').setLevel(logging.WARNING)

    # Initialize configuration
    # Monkey patch the configuration with hyperopt_conf.py
    configuration = Configuration(args)
    logger.info('Starting freqtrade in Hyperopt mode')

    optimize_config = hyperopt_optimize_conf()
    config = configuration._load_common_config(optimize_config)
    config = configuration._load_backtesting_config(config)
    config = configuration._load_hyperopt_config(config)
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    # Initialize backtesting object
    hyperopt = Hyperopt(config)
    hyperopt.start()
