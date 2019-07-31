# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

"""
This module contains the hyperopt logic
"""

import logging
import sys

from operator import itemgetter
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

from joblib import Parallel, delayed, dump, load, wrap_non_picklable_objects, cpu_count
from pandas import DataFrame
from skopt import Optimizer
from skopt.space import Dimension

from freqtrade.configuration import Arguments
from freqtrade.data.history import load_data, get_timeframe
from freqtrade.optimize.backtesting import Backtesting
# Import IHyperOptLoss to allow users import from this file
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F4
from freqtrade.resolvers.hyperopt_resolver import HyperOptResolver, HyperOptLossResolver


logger = logging.getLogger(__name__)


INITIAL_POINTS = 30
MAX_LOSS = 100000  # just a big enough number to be bad result in loss optimization


class Hyperopt(Backtesting):
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To run a backtest:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.custom_hyperopt = HyperOptResolver(self.config).hyperopt

        self.custom_hyperoptloss = HyperOptLossResolver(self.config).hyperoptloss
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function

        self.trials_file = (self.config['user_data_dir'] /
                            'hyperopt_results' / 'hyperopt_results.pickle')
        self.tickerdata_pickle = (self.config['user_data_dir'] /
                                  'hyperopt_results' / 'hyperopt_tickerdata.pkl')
        self.total_tries = config.get('epochs', 0)
        self.current_best_loss = 100

        if not self.config.get('hyperopt_continue'):
            self.clean_hyperopt()
        else:
            logger.info("Continuing on previous hyperopt results.")

        # Previous evaluations
        self.trials: List = []

        # Populate functions here (hasattr is slow so should not be run during "regular" operations)
        if hasattr(self.custom_hyperopt, 'populate_buy_trend'):
            self.advise_buy = self.custom_hyperopt.populate_buy_trend  # type: ignore

        if hasattr(self.custom_hyperopt, 'populate_sell_trend'):
            self.advise_sell = self.custom_hyperopt.populate_sell_trend  # type: ignore

            # Use max_open_trades for hyperopt as well, except --disable-max-market-positions is set
        if self.config.get('use_max_market_positions', True):
            self.max_open_trades = self.config['max_open_trades']
        else:
            logger.debug('Ignoring max_open_trades (--disable-max-market-positions was used) ...')
            self.max_open_trades = 0
        self.position_stacking = self.config.get('position_stacking', False),

    @staticmethod
    def get_lock_filename(config) -> str:

        return str(config['user_data_dir'] / 'hyperopt.lock')

    def clean_hyperopt(self):
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        for f in [self.tickerdata_pickle, self.trials_file]:
            p = Path(f)
            if p.is_file():
                logger.info(f"Removing `{p}`.")
                p.unlink()

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
        self.trials_file.unlink()
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
        print_all = self.config.get('print_all', False)
        if print_all or results['loss'] < self.current_best_loss:
            # Output human-friendly index here (starting from 1)
            current = results['current_tries'] + 1
            total = results['total_tries']
            res = results['result']
            loss = results['loss']
            self.current_best_loss = results['loss']
            log_msg = f'{current:5d}/{total}: {res} Objective: {loss:.5f}'
            log_msg = f'*{log_msg}' if results['initial_point'] else f' {log_msg}'
            if print_all:
                print(log_msg)
            else:
                print('\n' + log_msg)
        else:
            print('.', end='')
            sys.stdout.flush()

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
        """
        Used Optimize function. Called once per epoch to optimize whatever is configured.
        Keep this function as optimized as possible!
        """
        params = self.get_args(_params)
        if self.has_space('roi'):
            self.strategy.minimal_roi = self.custom_hyperopt.generate_roi_table(params)

        if self.has_space('buy'):
            self.advise_buy = self.custom_hyperopt.buy_strategy_generator(params)

        if self.has_space('sell'):
            self.advise_sell = self.custom_hyperopt.sell_strategy_generator(params)

        if self.has_space('stoploss'):
            self.strategy.stoploss = params['stoploss']

        processed = load(self.tickerdata_pickle)

        min_date, max_date = get_timeframe(processed)

        results = self.backtest(
            {
                'stake_amount': self.config['stake_amount'],
                'processed': processed,
                'max_open_trades': self.max_open_trades,
                'position_stacking': self.position_stacking,
                'start_date': min_date,
                'end_date': max_date,
            }
        )
        result_explanation = self.format_results(results)

        trade_count = len(results.index)

        # If this evaluation contains too short amount of trades to be
        # interesting -- consider it as 'bad' (assigned max. loss value)
        # in order to cast this hyperspace point away from optimization
        # path. We do not want to optimize 'hodl' strategies.
        if trade_count < self.config['hyperopt_min_trades']:
            return {
                'loss': MAX_LOSS,
                'params': params,
                'result': result_explanation,
            }

        loss = self.calculate_loss(results=results, trade_count=trade_count,
                                   min_date=min_date.datetime, max_date=max_date.datetime)

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
        profit = results.profit_percent.sum() * 100.0
        duration = results.trade_duration.mean()

        return (f'{trades:6d} trades. Avg profit {avg_profit: 5.2f}%. '
                f'Total profit {total_profit: 11.8f} {stake_cur} '
                f'({profit: 7.2f}Σ%). Avg duration {duration:5.1f} mins.')

    def get_optimizer(self, cpu_count) -> Optimizer:
        return Optimizer(
            self.hyperopt_space(),
            base_estimator="ET",
            acq_optimizer="auto",
            n_initial_points=INITIAL_POINTS,
            acq_optimizer_kwargs={'n_jobs': cpu_count},
            random_state=self.config.get('hyperopt_random_state', None)
        )

    def run_optimizer_parallel(self, parallel, asked) -> List:
        return parallel(delayed(
                        wrap_non_picklable_objects(self.generate_optimizer))(v) for v in asked)

    def load_previous_results(self):
        """ read trials file if we have one """
        if self.trials_file.is_file() and self.trials_file.stat().st_size > 0:
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
            refresh_pairs=self.config.get('refresh_pairs', False),
            exchange=self.exchange,
            timerange=timerange
        )

        if not data:
            logger.critical("No data found. Terminating.")
            return

        min_date, max_date = get_timeframe(data)

        logger.info(
            'Hyperopting with data from %s up to %s (%s days)..',
            min_date.isoformat(),
            max_date.isoformat(),
            (max_date - min_date).days
        )

        self.strategy.advise_indicators = \
            self.custom_hyperopt.populate_indicators  # type: ignore

        preprocessed = self.strategy.tickerdata_to_dataframe(data)

        dump(preprocessed, self.tickerdata_pickle)

        # We don't need exchange instance anymore while running hyperopt
        self.exchange = None  # type: ignore

        self.load_previous_results()

        cpus = cpu_count()
        logger.info(f'Found {cpus} CPU cores. Let\'s make them scream!')
        config_jobs = self.config.get('hyperopt_jobs', -1)
        logger.info(f'Number of parallel jobs set as: {config_jobs}')

        opt = self.get_optimizer(config_jobs)
        try:
            with Parallel(n_jobs=config_jobs) as parallel:
                jobs = parallel._effective_n_jobs()
                logger.info(f'Effective number of parallel workers used: {jobs}')
                EVALS = max(self.total_tries // jobs, 1)
                for i in range(EVALS):
                    asked = opt.ask(n_points=jobs)
                    f_val = self.run_optimizer_parallel(parallel, asked)
                    opt.tell(asked, [i['loss'] for i in f_val])

                    self.trials += f_val
                    for j in range(jobs):
                        current = i * jobs + j
                        self.log_results({
                            'loss': f_val[j]['loss'],
                            'current_tries': current,
                            'initial_point': current < INITIAL_POINTS,
                            'total_tries': self.total_tries,
                            'result': f_val[j]['result'],
                        })
                        logger.debug(f"Optimizer params: {f_val[j]['params']}")
                    for j in range(jobs):
                        logger.debug(f"Optimizer state: Xi: {opt.Xi[-j-1]}, yi: {opt.yi[-j-1]}")
        except KeyboardInterrupt:
            print('User interrupted..')

        self.save_trials()
        self.log_trials_result()
