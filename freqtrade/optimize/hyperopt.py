# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

"""
This module contains the hyperopt logic
"""

import json
import logging
import os
import pickle
import sys
import multiprocessing

from argparse import Namespace
from functools import reduce
from math import exp
from operator import itemgetter
from typing import Dict, Any, Callable, Optional, List

import talib.abstract as ta
from pandas import DataFrame

from skopt.space import Real, Integer, Categorical, Dimension
from skopt import Optimizer
from sklearn.externals.joblib import Parallel, delayed

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.optimize import load_data
from freqtrade.optimize.backtesting import Backtesting

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

        # Previous evaluations
        self.trials_file = os.path.join('user_data', 'hyperopt_trials.pickle')
        self.trials = []

    def get_args(self, params):
        dimensions = self.hyperopt_space()
        # Ensure the number of dimensions match
        # the number of parameters in the list x.
        if len(params) != len(dimensions):
            msg = "Mismatch in number of search-space dimensions. " \
                    "len(dimensions)=={} and len(x)=={}"
            msg = msg.format(len(dimensions), len(params))
            raise ValueError(msg)

        # Create a dict where the keys are the names of the dimensions
        # and the values are taken from the list of parameters x.
        arg_dict = {dim.name: value for dim, value in zip(dimensions, params)}
        return arg_dict

    @staticmethod
    def populate_indicators(dataframe: DataFrame) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def save_trials(self) -> None:
        """
        Save hyperopt trials to file
        """
        if self.trials:
            logger.info('Saving %d evaluations to \'%s\'', len(self.trials), self.trials_file)
            pickle.dump(self.trials, open(self.trials_file, 'wb'))

    def read_trials(self) -> List:
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
        results = sorted(self.trials, key=itemgetter('loss'))
        best_result = results[0]
        logger.info(
            'Best result:\n%s\nwith values:\n%s',
            best_result['result'],
            best_result['params']
        )
        if 'roi_t1' in best_result['params']:
            logger.info('ROI table:\n%s', self.generate_roi_table(best_result['params']))

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
    def roi_space() -> List[Dimension]:
        """
        Values to search for each ROI steps
        """
        return [
            Integer(10, 120, name='roi_t1'),
            Integer(10, 60, name='roi_t2'),
            Integer(10, 40, name='roi_t3'),
            Real(0.01, 0.04, name='roi_p1'),
            Real(0.01, 0.07, name='roi_p2'),
            Real(0.01, 0.20, name='roi_p3'),
        ]

    @staticmethod
    def stoploss_space() -> List[Dimension]:
        """
        Stoploss search space
        """
        return [
            Real(-0.5, -0.02, name='stoploss'),
        ]

    @staticmethod
    def indicator_space() -> Dict[str, Any]:
        """
        Define your Hyperopt space for searching strategy parameters
        """
        return [
            Integer(10, 25, name='mfi-value'),
            Integer(15, 45, name='fastd-value'),
            Integer(20, 50, name='adx-value'),
            Integer(20, 40, name='rsi-value'),
            Categorical([True, False], name='mfi-enabled'),
            Categorical([True, False], name='fastd-enabled'),
            Categorical([True, False], name='adx-enabled'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical(['bb_lower', 'macd_cross_signal', 'sar_reversal'], name='trigger')
        ]

    def has_space(self, space: str) -> bool:
        """
        Tell if a space value is contained in the configuration
        """
        if space in self.config['spaces'] or 'all' in self.config['spaces']:
            return True
        return False

    def hyperopt_space(self) -> List[Dimension]:
        """
        Return the space to use during Hyperopt
        """
        spaces: List[Dimension] = []
        if self.has_space('buy'):
            spaces += Hyperopt.indicator_space()
        if self.has_space('roi'):
            spaces += Hyperopt.roi_space()
        if self.has_space('stoploss'):
            spaces += Hyperopt.stoploss_space()
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
            if 'mfi-enabled' in params and params['mfi-enabled']:
                conditions.append(dataframe['mfi'] < params['mfi-value'])
            if 'fastd-enabled' in params and params['fastd-enabled']:
                conditions.append(dataframe['fastd'] < params['fastd-value'])
            if 'adx-enabled' in params and params['adx-enabled']:
                conditions.append(dataframe['adx'] > params['adx-value'])
            if 'rsi-enabled' in params and params['rsi-enabled']:
                conditions.append(dataframe['rsi'] < params['rsi-value'])

            # TRIGGERS
            if params['trigger'] == 'bb_lower':
                conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
            if params['trigger'] == 'macd_cross_signal':
                conditions.append(qtpylib.crossed_above(
                    dataframe['macd'], dataframe['macdsignal']
                ))
            if params['trigger'] == 'sar_reversal':
                conditions.append(qtpylib.crossed_above(
                    dataframe['close'], dataframe['sar']
                ))

            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

            return dataframe

        return populate_buy_trend

    def generate_optimizer(self, _params) -> Dict:
        params = self.get_args(_params)

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
        trade_duration = results.trade_duration.mean()

        loss = self.calculate_loss(total_profit, trade_count, trade_duration)

        return {
            'loss': loss,
            'params': params,
            'result': result_explanation,
        }

    def format_results(self, results: DataFrame) -> str:
        """
        Return the format result in a string
        """
        return ('{:6d} trades. Avg profit {: 5.2f}%. '
                'Total profit {: 11.8f} {} ({:.4f}Σ%). Avg duration {:5.1f} mins.').format(
                    len(results.index),
                    results.profit_percent.mean() * 100.0,
                    results.profit_abs.sum(),
                    self.config['stake_currency'],
                    results.profit_percent.sum(),
                    results.trade_duration.mean(),
                )

    def start(self) -> None:
        timerange = Arguments.parse_timerange(None if self.config.get(
            'timerange') is None else str(self.config.get('timerange')))
        data = load_data(
            datadir=str(self.config.get('datadir')),
            pairs=self.config['exchange']['pair_whitelist'],
            ticker_interval=self.ticker_interval,
            timerange=timerange
        )

        if self.has_space('buy'):
            self.analyze.populate_indicators = Hyperopt.populate_indicators  # type: ignore
        self.processed = self.tickerdata_to_dataframe(data)

        self.exchange = None

        logger.info('Preparing..')
        # read trials file if we have one
        if os.path.exists(self.trials_file) and os.path.getsize(self.trials_file) > 0:
            self.trials = self.read_trials()
            logger.info(
                'Loaded %d previous evaluations from disk.',
                len(self.trials)
            )

        cpus = multiprocessing.cpu_count()
        print(f'Found {cpus}. Let\'s make them scream!')

        opt = Optimizer(
            self.hyperopt_space(),
            base_estimator="ET",
            acq_optimizer="auto",
            n_initial_points=30,
            acq_optimizer_kwargs={'n_jobs': -1}
        )

        try:
            with Parallel(n_jobs=-1) as parallel:
                for i in range(self.total_tries//cpus):
                    asked = opt.ask(n_points=cpus)
                    f_val = parallel(delayed(self.generate_optimizer)(v) for v in asked)
                    opt.tell(asked, [i['loss'] for i in f_val])

                    self.trials += f_val
                    for j in range(cpus):
                        self.log_results(
                            {
                                'loss': f_val[j]['loss'],
                                'current_tries': i * cpus + j,
                                'total_tries': self.total_tries,
                                'result': f_val[j]['result'],
                            }
                        )
        except KeyboardInterrupt:
            print('User interrupted..')

        self.save_trials()
        self.log_trials_result()


def start(args: Namespace) -> None:
    """
    Start Backtesting script
    :param args: Cli args from Arguments()
    :return: None
    """

    # Remove noisy log messages
    logging.getLogger('hyperopt.tpe').setLevel(logging.WARNING)

    # Initialize configuration
    # Monkey patch the configuration with hyperopt_conf.py
    configuration = Configuration(args)
    logger.info('Starting freqtrade in Hyperopt mode')
    config = configuration.load_config()

    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    # Initialize backtesting object
    hyperopt = Hyperopt(config)
    hyperopt.start()
