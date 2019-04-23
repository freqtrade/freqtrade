# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

"""
This module contains the hyperopt logic
"""

import logging
import multiprocessing
import os
import sys
from argparse import Namespace
from math import exp
from operator import itemgetter
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

from joblib import Parallel, delayed, dump, load, wrap_non_picklable_objects
from pandas import DataFrame
from skopt import Optimizer
from skopt.space import Dimension

from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.data.history import load_data
from freqtrade.optimize import get_timeframe
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.state import RunMode
from freqtrade.resolvers import HyperOptResolver

logger = logging.getLogger(__name__)

MAX_LOSS = 100000  # just a big enough number to be bad result in loss optimization
TICKERDATA_PICKLE = os.path.join('user_data', 'hyperopt_tickerdata.pkl')


class Hyperopt(Backtesting):
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To run a backtest:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.config = config
        self.custom_hyperopt = HyperOptResolver(self.config).hyperopt

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

        # Previous evaluations
        self.trials_file = os.path.join('user_data', 'hyperopt_results.pickle')
        self.trials: List = []

    def get_args(self, params):
        dimensions = self.hyperopt_space()
        # Ensure the number of dimensions match
        # the number of parameters in the list x.
        if len(params) != len(dimensions):
            raise ValueError('Mismatch in number of search-space dimensions. '
                             f'len(dimensions)=={len(dimensions)} and len(x)=={len(params)}')

        # Create a dict where the keys are the names of the dimensions
        # and the values are taken from the list of parameters x.
        arg_dict = {dim.name: value for dim, value in zip(dimensions, params)}
        return arg_dict

    def save_trials(self) -> None:
        """
        Save hyperopt trials to file
        """
        if self.trials:
            logger.info('Saving %d evaluations to \'%s\'', len(self.trials), self.trials_file)
            dump(self.trials, self.trials_file)

    def read_trials(self) -> List:
        """
        Read hyperopt trials file
        """
        logger.info('Reading Trials from \'%s\'', self.trials_file)
        trials = load(self.trials_file)
        os.remove(self.trials_file)
        return trials

    def log_trials_result(self) -> None:
        """
        Display Best hyperopt result
        """
        results = sorted(self.trials, key=itemgetter('loss'))
        best_result = results[0]
        logger.info(
            'Best result:\n%s\nwith values:\n',
            best_result['result']
        )
        pprint(best_result['params'], indent=4)
        if 'roi_t1' in best_result['params']:
            logger.info('ROI table:')
            pprint(self.custom_hyperopt.generate_roi_table(best_result['params']), indent=4)

    def log_results(self, results) -> None:
        """
        Log results if it is better than any previous evaluation
        """
        if self.config.get('print_all', False) or results['loss'] < self.current_best_loss:
            current = results['current_tries']
            total = results['total_tries']
            res = results['result']
            loss = results['loss']
            self.current_best_loss = results['loss']
            log_msg = f'\n{current:5d}/{total}: {res}. Loss {loss:.5f}'
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
        result = trade_loss + profit_loss + duration_loss
        return result

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
            spaces += self.custom_hyperopt.indicator_space()
        if self.has_space('sell'):
            spaces += self.custom_hyperopt.sell_indicator_space()
            # Make sure experimental is enabled
            if 'experimental' not in self.config:
                self.config['experimental'] = {}
            self.config['experimental']['use_sell_signal'] = True
        if self.has_space('roi'):
            spaces += self.custom_hyperopt.roi_space()
        if self.has_space('stoploss'):
            spaces += self.custom_hyperopt.stoploss_space()
        return spaces

    def generate_optimizer(self, _params: Dict) -> Dict:
        params = self.get_args(_params)
        if self.has_space('roi'):
            self.strategy.minimal_roi = self.custom_hyperopt.generate_roi_table(params)

        if self.has_space('buy'):
            self.advise_buy = self.custom_hyperopt.buy_strategy_generator(params)
        elif hasattr(self.custom_hyperopt, 'populate_buy_trend'):
            self.advise_buy = self.custom_hyperopt.populate_buy_trend  # type: ignore

        if self.has_space('sell'):
            self.advise_sell = self.custom_hyperopt.sell_strategy_generator(params)
        elif hasattr(self.custom_hyperopt, 'populate_sell_trend'):
            self.advise_sell = self.custom_hyperopt.populate_sell_trend  # type: ignore

        if self.has_space('stoploss'):
            self.strategy.stoploss = params['stoploss']

        processed = load(TICKERDATA_PICKLE)
        min_date, max_date = get_timeframe(processed)
        results = self.backtest(
            {
                'stake_amount': self.config['stake_amount'],
                'processed': processed,
                'position_stacking': self.config.get('position_stacking', True),
                'start_date': min_date,
                'end_date': max_date,
            }
        )
        result_explanation = self.format_results(results)

        total_profit = results.profit_percent.sum()
        trade_count = len(results.index)
        trade_duration = results.trade_duration.mean()

        if trade_count == 0:
            return {
                'loss': MAX_LOSS,
                'params': params,
                'result': result_explanation,
            }

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
        trades = len(results.index)
        avg_profit = results.profit_percent.mean() * 100.0
        total_profit = results.profit_abs.sum()
        stake_cur = self.config['stake_currency']
        profit = results.profit_percent.sum()
        duration = results.trade_duration.mean()

        return (f'{trades:6d} trades. Avg profit {avg_profit: 5.2f}%. '
                f'Total profit {total_profit: 11.8f} {stake_cur} '
                f'({profit:.4f}Σ%). Avg duration {duration:5.1f} mins.')

    def get_optimizer(self, cpu_count) -> Optimizer:
        return Optimizer(
            self.hyperopt_space(),
            base_estimator="ET",
            acq_optimizer="auto",
            n_initial_points=30,
            acq_optimizer_kwargs={'n_jobs': cpu_count}
        )

    def run_optimizer_parallel(self, parallel, asked) -> List:
        return parallel(delayed(
                        wrap_non_picklable_objects(self.generate_optimizer))(v) for v in asked)

    def load_previous_results(self):
        """ read trials file if we have one """
        if os.path.exists(self.trials_file) and os.path.getsize(self.trials_file) > 0:
            self.trials = self.read_trials()
            logger.info(
                'Loaded %d previous evaluations from disk.',
                len(self.trials)
            )

    def start(self) -> None:
        timerange = Arguments.parse_timerange(None if self.config.get(
            'timerange') is None else str(self.config.get('timerange')))
        data = load_data(
            datadir=Path(self.config['datadir']) if self.config.get('datadir') else None,
            pairs=self.config['exchange']['pair_whitelist'],
            ticker_interval=self.ticker_interval,
            timerange=timerange
        )

        if self.has_space('buy') or self.has_space('sell'):
            self.strategy.advise_indicators = \
                self.custom_hyperopt.populate_indicators  # type: ignore
        dump(self.strategy.tickerdata_to_dataframe(data), TICKERDATA_PICKLE)
        self.exchange = None  # type: ignore
        self.load_previous_results()

        cpus = multiprocessing.cpu_count()
        logger.info(f'Found {cpus} CPU cores. Let\'s make them scream!')

        opt = self.get_optimizer(cpus)
        EVALS = max(self.total_tries // cpus, 1)
        try:
            with Parallel(n_jobs=cpus) as parallel:
                for i in range(EVALS):
                    asked = opt.ask(n_points=cpus)
                    f_val = self.run_optimizer_parallel(parallel, asked)
                    opt.tell(asked, [i['loss'] for i in f_val])

                    self.trials += f_val
                    for j in range(cpus):
                        self.log_results({
                            'loss': f_val[j]['loss'],
                            'current_tries': i * cpus + j,
                            'total_tries': self.total_tries,
                            'result': f_val[j]['result'],
                        })
                        logger.debug(f"Optimizer params: {f_val[j]['params']}")
                    for j in range(cpus):
                        logger.debug(f"Opimizer state: Xi: {opt.Xi[-j-1]}, yi: {opt.yi[-j-1]}")
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
    configuration = Configuration(args, RunMode.HYPEROPT)
    logger.info('Starting freqtrade in Hyperopt mode')
    config = configuration.load_config()

    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    if config.get('strategy') and config.get('strategy') != 'DefaultStrategy':
        logger.error("Please don't use --strategy for hyperopt.")
        logger.error(
            "Read the documentation at "
            "https://github.com/freqtrade/freqtrade/blob/develop/docs/hyperopt.md "
            "to understand how to configure hyperopt.")
        raise ValueError("--strategy configured but not supported for hyperopt")
    # Initialize backtesting object
    hyperopt = Hyperopt(config)
    hyperopt.start()
