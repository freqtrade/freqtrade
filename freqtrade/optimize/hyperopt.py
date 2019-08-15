# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

"""
This module contains the hyperopt logic
"""

import logging
import os
import sys

from operator import itemgetter
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional

from colorama import init as colorama_init
from colorama import Fore, Style
from joblib import Parallel, delayed, dump, load, wrap_non_picklable_objects, cpu_count
from pandas import DataFrame
from skopt import Optimizer
from skopt.space import Dimension

from freqtrade.configuration import TimeRange
from freqtrade.data.history import load_data, get_timeframe
from freqtrade.optimize.backtesting import Backtesting
# Import IHyperOptLoss to allow users import from this file
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F4
from freqtrade.resolvers.hyperopt_resolver import HyperOptResolver, HyperOptLossResolver


logger = logging.getLogger(__name__)


INITIAL_POINTS = 30
MAX_LOSS = 100000  # just a big enough number to be bad result in loss optimization
TICKERDATA_PICKLE = os.path.join('user_data', 'hyperopt_tickerdata.pkl')
TRIALSDATA_PICKLE = os.path.join('user_data', 'hyperopt_results.pickle')
HYPEROPT_LOCKFILE = os.path.join('user_data', 'hyperopt.lock')


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

        self.total_epochs = config.get('epochs', 0)
        self.current_best_loss = 100

        if not self.config.get('hyperopt_continue'):
            self.clean_hyperopt()
        else:
            logger.info("Continuing on previous hyperopt results.")

        # Previous evaluations
        self.trials_file = TRIALSDATA_PICKLE
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

        if self.has_space('sell'):
            # Make sure experimental is enabled
            if 'experimental' not in self.config:
                self.config['experimental'] = {}
            self.config['experimental']['use_sell_signal'] = True

    def clean_hyperopt(self):
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        for f in [TICKERDATA_PICKLE, TRIALSDATA_PICKLE]:
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
        os.remove(self.trials_file)
        return trials

    def log_trials_result(self) -> None:
        """
        Display Best hyperopt result
        """
        results = sorted(self.trials, key=itemgetter('loss'))
        best_result = results[0]
        params = best_result['params']

        log_str = self.format_results_logstring(best_result)
        print(f"\nBest result:\n\n{log_str}\n")
        if self.has_space('buy'):
            print('Buy hyperspace params:')
            pprint({p.name: params.get(p.name) for p in self.hyperopt_space('buy')},
                   indent=4)
        if self.has_space('sell'):
            print('Sell hyperspace params:')
            pprint({p.name: params.get(p.name) for p in self.hyperopt_space('sell')},
                   indent=4)
        if self.has_space('roi'):
            print("ROI table:")
            pprint(self.custom_hyperopt.generate_roi_table(params), indent=4)
        if self.has_space('stoploss'):
            print(f"Stoploss: {params.get('stoploss')}")

    def log_results(self, results) -> None:
        """
        Log results if it is better than any previous evaluation
        """
        print_all = self.config.get('print_all', False)
        is_best_loss = results['loss'] < self.current_best_loss
        if print_all or is_best_loss:
            if is_best_loss:
                self.current_best_loss = results['loss']
            log_str = self.format_results_logstring(results)
            # Colorize output
            if self.config.get('print_colorized', False):
                if results['total_profit'] > 0:
                    log_str = Fore.GREEN + log_str
                if print_all and is_best_loss:
                    log_str = Style.BRIGHT + log_str
            if print_all:
                print(log_str)
            else:
                print('\n' + log_str)
        else:
            print('.', end='')
            sys.stdout.flush()

    def format_results_logstring(self, results) -> str:
        # Output human-friendly index here (starting from 1)
        current = results['current_epoch'] + 1
        total = self.total_epochs
        res = results['results_explanation']
        loss = results['loss']
        log_str = f'{current:5d}/{total}: {res} Objective: {loss:.5f}'
        log_str = f'*{log_str}' if results['is_initial_point'] else f' {log_str}'
        return log_str

    def has_space(self, space: str) -> bool:
        """
        Tell if a space value is contained in the configuration
        """
        return any(s in self.config['spaces'] for s in [space, 'all'])

    def hyperopt_space(self, space: Optional[str] = None) -> List[Dimension]:
        """
        Return the dimensions in the hyperoptimization space.
        :param space: Defines hyperspace to return dimensions for.
        If None, then the self.has_space() will be used to return dimensions
        for all hyperspaces used.
        """
        spaces: List[Dimension] = []
        if space == 'buy' or (space is None and self.has_space('buy')):
            logger.debug("Hyperopt has 'buy' space")
            spaces += self.custom_hyperopt.indicator_space()
        if space == 'sell' or (space is None and self.has_space('sell')):
            logger.debug("Hyperopt has 'sell' space")
            spaces += self.custom_hyperopt.sell_indicator_space()
        if space == 'roi' or (space is None and self.has_space('roi')):
            logger.debug("Hyperopt has 'roi' space")
            spaces += self.custom_hyperopt.roi_space()
        if space == 'stoploss' or (space is None and self.has_space('stoploss')):
            logger.debug("Hyperopt has 'stoploss' space")
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

        processed = load(TICKERDATA_PICKLE)

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
        results_explanation = self.format_results(results)

        trade_count = len(results.index)
        total_profit = results.profit_abs.sum()

        # If this evaluation contains too short amount of trades to be
        # interesting -- consider it as 'bad' (assigned max. loss value)
        # in order to cast this hyperspace point away from optimization
        # path. We do not want to optimize 'hodl' strategies.
        if trade_count < self.config['hyperopt_min_trades']:
            return {
                'loss': MAX_LOSS,
                'params': params,
                'results_explanation': results_explanation,
                'total_profit': total_profit,
            }

        loss = self.calculate_loss(results=results, trade_count=trade_count,
                                   min_date=min_date.datetime, max_date=max_date.datetime)

        return {
            'loss': loss,
            'params': params,
            'results_explanation': results_explanation,
            'total_profit': total_profit,
        }

    def format_results(self, results: DataFrame) -> str:
        """
        Return the formatted results explanation in a string
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
        if os.path.exists(self.trials_file) and os.path.getsize(self.trials_file) > 0:
            self.trials = self.read_trials()
            logger.info(
                'Loaded %d previous evaluations from disk.',
                len(self.trials)
            )

    def start(self) -> None:
        timerange = TimeRange.parse_timerange(None if self.config.get(
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

        dump(preprocessed, TICKERDATA_PICKLE)

        # We don't need exchange instance anymore while running hyperopt
        self.exchange = None  # type: ignore

        self.load_previous_results()

        cpus = cpu_count()
        logger.info(f'Found {cpus} CPU cores. Let\'s make them scream!')
        config_jobs = self.config.get('hyperopt_jobs', -1)
        logger.info(f'Number of parallel jobs set as: {config_jobs}')

        opt = self.get_optimizer(config_jobs)

        if self.config.get('print_colorized', False):
            colorama_init(autoreset=True)

        try:
            with Parallel(n_jobs=config_jobs) as parallel:
                jobs = parallel._effective_n_jobs()
                logger.info(f'Effective number of parallel workers used: {jobs}')
                EVALS = max(self.total_epochs // jobs, 1)
                for i in range(EVALS):
                    asked = opt.ask(n_points=jobs)
                    f_val = self.run_optimizer_parallel(parallel, asked)
                    opt.tell(asked, [v['loss'] for v in f_val])
                    for j in range(jobs):
                        current = i * jobs + j
                        val = f_val[j]
                        val['current_epoch'] = current
                        val['is_initial_point'] = current < INITIAL_POINTS
                        self.log_results(val)
                        self.trials.append(val)
                        logger.debug(f"Optimizer epoch evaluated: {val}")
        except KeyboardInterrupt:
            print('User interrupted..')

        self.save_trials()
        self.log_trials_result()
