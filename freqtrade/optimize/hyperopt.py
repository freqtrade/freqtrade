# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement
"""
This module contains the hyperopt logic
"""

import os
import functools
import locale
import logging
import random
import sys
import warnings
from collections import OrderedDict
from math import factorial, log
from operator import itemgetter
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional

import rapidjson
from colorama import Fore, Style
from colorama import init as colorama_init

from freqtrade.data.converter import trim_dataframe
from freqtrade.data.history import get_timerange
from freqtrade.exceptions import OperationalException
from freqtrade.misc import plural, round_dict
from freqtrade.optimize.backtesting import Backtesting
# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
from freqtrade.optimize.hyperopt_backend import CustomImmediateResultBackend
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from freqtrade.resolvers.hyperopt_resolver import (HyperOptLossResolver, HyperOptResolver)
from joblib import (Parallel, cpu_count, delayed, dump, load, wrap_non_picklable_objects)
from joblib._parallel_backends import LokyBackend
from joblib import register_parallel_backend, parallel_backend
from pandas import DataFrame

# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
    from skopt.space import Dimension

logger = logging.getLogger(__name__)

INITIAL_POINTS = 30

# Keep no more than 2*SKOPT_MODELS_MAX_NUM models
# in the skopt models list
SKOPT_MODELS_MAX_NUM = 10

MAX_LOSS = 100000  # just a big enough number to be bad result in loss optimization


class Hyperopt:
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To run a backtest:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        self.backtesting = Backtesting(self.config)

        self.custom_hyperopt = HyperOptResolver.load_hyperopt(self.config)

        self.custom_hyperoptloss = HyperOptLossResolver.load_hyperoptloss(self.config)
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function

        self.trials_file = (self.config['user_data_dir'] / 'hyperopt_results' /
                            'hyperopt_results.pickle')
        self.tickerdata_pickle = (self.config['user_data_dir'] / 'hyperopt_results' /
                                  'hyperopt_tickerdata.pkl')
        self.effort = config.get('epochs', 0) or 1
        self.total_epochs = 9999
        self.max_epoch = 9999
        self.search_space_size = 0
        self.max_epoch_reached = False

        self.min_epochs = INITIAL_POINTS
        self.current_best_loss = 100
        self.current_best_epoch = 0
        self.epochs_since_last_best = []
        self.avg_best_occurrence = 0

        if not self.config.get('hyperopt_continue'):
            self.clean_hyperopt()
        else:
            logger.info("Continuing on previous hyperopt results.")

        self.num_trials_saved = 0

        # Previous evaluations
        self.trials: List = []

        self.opt: Optimizer
        self.opt = None
        self.f_val: List = []

        # Populate functions here (hasattr is slow so should not be run during "regular" operations)
        if hasattr(self.custom_hyperopt, 'populate_indicators'):
            self.backtesting.strategy.advise_indicators = \
                self.custom_hyperopt.populate_indicators  # type: ignore
        if hasattr(self.custom_hyperopt, 'populate_buy_trend'):
            self.backtesting.strategy.advise_buy = \
                self.custom_hyperopt.populate_buy_trend  # type: ignore
        if hasattr(self.custom_hyperopt, 'populate_sell_trend'):
            self.backtesting.strategy.advise_sell = \
                self.custom_hyperopt.populate_sell_trend  # type: ignore

        # Use max_open_trades for hyperopt as well, except --disable-max-market-positions is set
        if self.config.get('use_max_market_positions', True):
            self.max_open_trades = self.config['max_open_trades']
        else:
            logger.debug('Ignoring max_open_trades (--disable-max-market-positions was used) ...')
            self.max_open_trades = 0
        self.position_stacking = self.config.get('position_stacking', False)

        if self.has_space('sell'):
            # Make sure use_sell_signal is enabled
            if 'ask_strategy' not in self.config:
                self.config['ask_strategy'] = {}
            self.config['ask_strategy']['use_sell_signal'] = True

        self.print_all = self.config.get('print_all', False)
        self.print_colorized = self.config.get('print_colorized', False)
        self.print_json = self.config.get('print_json', False)

    @staticmethod
    def get_lock_filename(config: Dict[str, Any]) -> str:

        return str(config['user_data_dir'] / 'hyperopt.lock')

    def clean_hyperopt(self) -> None:
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        for f in [self.tickerdata_pickle, self.trials_file]:
            p = Path(f)
            if p.is_file():
                logger.info(f"Removing `{p}`.")
                p.unlink()

    def _get_params_dict(self, raw_params: List[Any]) -> Dict:

        dimensions: List[Dimension] = self.dimensions

        # Ensure the number of dimensions match
        # the number of parameters in the list.
        if len(raw_params) != len(dimensions):
            raise ValueError('Mismatch in number of search-space dimensions.')

        # Return a dict where the keys are the names of the dimensions
        # and the values are taken from the list of parameters.
        return {d.name: v for d, v in zip(dimensions, raw_params)}

    def save_trials(self, final: bool = False) -> None:
        """
        Save hyperopt trials to file
        """
        num_trials = len(self.trials)
        if num_trials > self.num_trials_saved:
            logger.info(f"Saving {num_trials} {plural(num_trials, 'epoch')}.")
            dump(self.trials, self.trials_file)
            self.num_trials_saved = num_trials
        if final:
            logger.info(f"{num_trials} {plural(num_trials, 'epoch')} "
                        f"saved to '{self.trials_file}'.")

    @staticmethod
    def _read_trials(trials_file: Path) -> List:
        """
        Read hyperopt trials file
        """
        logger.info("Reading Trials from '%s'", trials_file)
        trials = load(trials_file)
        return trials

    def _get_params_details(self, params: Dict) -> Dict:
        """
        Return the params for each space
        """
        result: Dict = {}

        if self.has_space('buy'):
            result['buy'] = {p.name: params.get(p.name) for p in self.hyperopt_space('buy')}
        if self.has_space('sell'):
            result['sell'] = {p.name: params.get(p.name) for p in self.hyperopt_space('sell')}
        if self.has_space('roi'):
            result['roi'] = self.custom_hyperopt.generate_roi_table(params)
        if self.has_space('stoploss'):
            result['stoploss'] = {
                p.name: params.get(p.name)
                for p in self.hyperopt_space('stoploss')
            }
        if self.has_space('trailing'):
            result['trailing'] = self.custom_hyperopt.generate_trailing_params(params)

        return result

    @staticmethod
    def print_epoch_details(results,
                            total_epochs: int,
                            print_json: bool,
                            no_header: bool = False,
                            header_str: str = None) -> None:
        """
        Display details of the hyperopt result
        """
        params = results.get('params_details', {})

        # Default header string
        if header_str is None:
            header_str = "Best result"

        if not no_header:
            explanation_str = Hyperopt._format_explanation_string(results, total_epochs)
            print(f"\n{header_str}:\n\n{explanation_str}\n")

        if print_json:
            result_dict: Dict = {}
            for s in ['buy', 'sell', 'roi', 'stoploss', 'trailing']:
                Hyperopt._params_update_for_json(result_dict, params, s)
            print(rapidjson.dumps(result_dict, default=str, number_mode=rapidjson.NM_NATIVE))

        else:
            Hyperopt._params_pretty_print(params, 'buy', "Buy hyperspace params:")
            Hyperopt._params_pretty_print(params, 'sell', "Sell hyperspace params:")
            Hyperopt._params_pretty_print(params, 'roi', "ROI table:")
            Hyperopt._params_pretty_print(params, 'stoploss', "Stoploss:")
            Hyperopt._params_pretty_print(params, 'trailing', "Trailing stop:")

    @staticmethod
    def _params_update_for_json(result_dict, params, space: str) -> None:
        if space in params:
            space_params = Hyperopt._space_params(params, space)
            if space in ['buy', 'sell']:
                result_dict.setdefault('params', {}).update(space_params)
            elif space == 'roi':
                # Convert keys in min_roi dict to strings because
                # rapidjson cannot dump dicts with integer keys...
                # OrderedDict is used to keep the numeric order of the items
                # in the dict.
                result_dict['minimal_roi'] = OrderedDict(
                    (str(k), v) for k, v in space_params.items())
            else:  # 'stoploss', 'trailing'
                result_dict.update(space_params)

    @staticmethod
    def _params_pretty_print(params, space: str, header: str) -> None:
        if space in params:
            space_params = Hyperopt._space_params(params, space, 5)
            if space == 'stoploss':
                print(header, space_params.get('stoploss'))
            else:
                print(header)
                pprint(space_params, indent=4)

    @staticmethod
    def _space_params(params, space: str, r: int = None) -> Dict:
        d = params[space]
        # Round floats to `r` digits after the decimal point if requested
        return round_dict(d, r) if r else d

    @staticmethod
    def is_best_loss(results, current_best_loss: float) -> bool:
        return results['loss'] < current_best_loss

    def print_results(self, results) -> None:
        """
        Log results if it is better than any previous evaluation
        """
        is_best = results['is_best']
        if self.print_all or is_best:
            self.print_results_explanation(results, self.total_epochs, self.print_all,
                                           self.print_colorized)

    @staticmethod
    def print_results_explanation(results, total_epochs, highlight_best: bool,
                                  print_colorized: bool) -> None:
        """
        Log results explanation string
        """
        explanation_str = Hyperopt._format_explanation_string(results, total_epochs)
        # Colorize output
        if print_colorized:
            if results['total_profit'] > 0:
                explanation_str = Fore.GREEN + explanation_str
            if highlight_best and results['is_best']:
                explanation_str = Style.BRIGHT + explanation_str
        print(explanation_str)

    @staticmethod
    def _format_explanation_string(results, total_epochs) -> str:
        return (("*" if 'is_initial_point' in results and results['is_initial_point'] else " ") +
                f"{results['current_epoch']:5d}/{total_epochs}: " +
                f"{results['results_explanation']} " + f"Objective: {results['loss']:.5f}")

    def has_space(self, space: str) -> bool:
        """
        Tell if the space value is contained in the configuration
        """
        # The 'trailing' space is not included in the 'default' set of spaces
        if space == 'trailing':
            return any(s in self.config['spaces'] for s in [space, 'all'])
        else:
            return any(s in self.config['spaces'] for s in [space, 'all', 'default'])

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

        if space == 'trailing' or (space is None and self.has_space('trailing')):
            logger.debug("Hyperopt has 'trailing' space")
            spaces += self.custom_hyperopt.trailing_space()

        return spaces

    def generate_optimizer(self, raw_params: List[Any], iteration=None) -> Dict:
        """
        Used Optimize function. Called once per epoch to optimize whatever is configured.
        Keep this function as optimized as possible!
        """
        params_dict = self._get_params_dict(raw_params)
        params_details = self._get_params_details(params_dict)

        if self.has_space('roi'):
            self.backtesting.strategy.minimal_roi = \
                self.custom_hyperopt.generate_roi_table(params_dict)

        if self.has_space('buy'):
            self.backtesting.strategy.advise_buy = \
                self.custom_hyperopt.buy_strategy_generator(params_dict)

        if self.has_space('sell'):
            self.backtesting.strategy.advise_sell = \
                self.custom_hyperopt.sell_strategy_generator(params_dict)

        if self.has_space('stoploss'):
            self.backtesting.strategy.stoploss = params_dict['stoploss']

        if self.has_space('trailing'):
            d = self.custom_hyperopt.generate_trailing_params(params_dict)
            self.backtesting.strategy.trailing_stop = d['trailing_stop']
            self.backtesting.strategy.trailing_stop_positive = d['trailing_stop_positive']
            self.backtesting.strategy.trailing_stop_positive_offset = \
                d['trailing_stop_positive_offset']
            self.backtesting.strategy.trailing_only_offset_is_reached = \
                d['trailing_only_offset_is_reached']

        processed = load(self.tickerdata_pickle)

        min_date, max_date = get_timerange(processed)

        backtesting_results = self.backtesting.backtest(
            processed=processed,
            stake_amount=self.config['stake_amount'],
            start_date=min_date,
            end_date=max_date,
            max_open_trades=self.max_open_trades,
            position_stacking=self.position_stacking,
        )
        return self._get_results_dict(backtesting_results, min_date, max_date, params_dict,
                                      params_details)

    def _get_results_dict(self, backtesting_results, min_date, max_date, params_dict,
                          params_details):
        results_metrics = self._calculate_results_metrics(backtesting_results)
        results_explanation = self._format_results_explanation_string(results_metrics)

        trade_count = results_metrics['trade_count']
        total_profit = results_metrics['total_profit']

        # If this evaluation contains too short amount of trades to be
        # interesting -- consider it as 'bad' (assigned max. loss value)
        # in order to cast this hyperspace point away from optimization
        # path. We do not want to optimize 'hodl' strategies.
        loss: float = MAX_LOSS
        if trade_count >= self.config['hyperopt_min_trades']:
            loss = self.calculate_loss(results=backtesting_results,
                                       trade_count=trade_count,
                                       min_date=min_date.datetime,
                                       max_date=max_date.datetime)
        return {
            'loss': loss,
            'params_dict': params_dict,
            'params_details': params_details,
            'results_metrics': results_metrics,
            'results_explanation': results_explanation,
            'total_profit': total_profit,
        }

    def _calculate_results_metrics(self, backtesting_results: DataFrame) -> Dict:
        return {
            'trade_count': len(backtesting_results.index),
            'avg_profit': backtesting_results.profit_percent.mean() * 100.0,
            'total_profit': backtesting_results.profit_abs.sum(),
            'profit': backtesting_results.profit_percent.sum() * 100.0,
            'duration': backtesting_results.trade_duration.mean(),
        }

    def _format_results_explanation_string(self, results_metrics: Dict) -> str:
        """
        Return the formatted results explanation in a string
        """
        stake_cur = self.config['stake_currency']
        return (f"{results_metrics['trade_count']:6d} trades. "
                f"Avg profit {results_metrics['avg_profit']: 6.2f}%. "
                f"Total profit {results_metrics['total_profit']: 11.8f} {stake_cur} "
                f"({results_metrics['profit']: 7.2f}\N{GREEK CAPITAL LETTER SIGMA}%). "
                f"Avg duration {results_metrics['duration']:5.1f} min.").encode(
                    locale.getpreferredencoding(), 'replace').decode('utf-8')

    def get_optimizer(self, dimensions: List[Dimension], cpu_count,
                      n_initial_points=INITIAL_POINTS) -> Optimizer:
        return Optimizer(
            dimensions,
            base_estimator="ET",
            acq_optimizer="auto",
            n_initial_points=n_initial_points,
            acq_optimizer_kwargs={'n_jobs': cpu_count},
            model_queue_size=SKOPT_MODELS_MAX_NUM,
            random_state=self.random_state,
        )

    def run_optimizer_parallel(self, parallel, tries: int, first_try: int) -> List:
        result = parallel(
            delayed(wrap_non_picklable_objects(self.parallel_objective))(asked, i)
            for asked, i in zip(self.opt_generator(), range(first_try, first_try + tries)))
        return result

    def opt_generator(self):
        while True:
            if self.f_val:
                # print("opt.tell(): ",
                #       [v['params_dict'] for v in self.f_val], [v['loss'] for v in self.f_val])
                functools.partial(self.opt.tell,
                                  ([v['params_dict']
                                    for v in self.f_val], [v['loss'] for v in self.f_val]))
                self.f_val = []
            yield self.opt.ask()

    def parallel_objective(self, asked, n):
        self.log_results_immediate(n)
        return self.generate_optimizer(asked)

    def parallel_callback(self, f_val):
        self.f_val.extend(f_val)

    def log_results_immediate(self, n) -> None:
        print('.', end='')
        sys.stdout.flush()

    def log_results(self, f_val, frame_start, max_epoch) -> None:
        """
        Log results if it is better than any previous evaluation
        """
        for i, v in enumerate(f_val):
            is_best = self.is_best_loss(v, self.current_best_loss)
            current = frame_start + i + 1
            v['is_best'] = is_best
            v['current_epoch'] = current
            v['is_initial_point'] = current <= self.n_initial_points
            logger.debug(f"Optimizer epoch evaluated: {v}")
            if is_best:
                self.current_best_loss = v['loss']
                self.update_max_epoch(v, current)
            self.print_results(v)
            self.trials.append(v)
        # Save results after every batch
        print('\n')
        self.save_trials()
        # give up if no best since max epochs
        if current > self.max_epoch:
            self.max_epoch_reached = True
        # testing trapdoor
        if os.getenv('FQT_HYPEROPT_TRAP'):
            logger.debug('bypassing hyperopt loop')
            self.max_epoch = 1

    @staticmethod
    def load_previous_results(trials_file: Path) -> List:
        """
        Load data for epochs from the file if we have one
        """
        trials: List = []
        if trials_file.is_file() and trials_file.stat().st_size > 0:
            trials = Hyperopt._read_trials(trials_file)
            if trials[0].get('is_best') is None:
                raise OperationalException(
                    "The file with Hyperopt results is incompatible with this version "
                    "of Freqtrade and cannot be loaded.")
            logger.info(f"Loaded {len(trials)} previous evaluations from disk.")
        return trials

    def _set_random_state(self, random_state: Optional[int]) -> int:
        return random_state or random.randint(1, 2**16 - 1)

    @staticmethod
    def calc_epochs(dimensions: List[Dimension], config_jobs: int, effort: int):
        """ Compute a reasonable number of initial points and
        a minimum number of epochs to evaluate """
        n_dimensions = len(dimensions)
        n_parameters = 0
        # sum all the dimensions discretely, granting minimum values
        for d in dimensions:
            if type(d).__name__ == 'Integer':
                n_parameters += max(1, d.high - d.low)
            elif type(d).__name__ == 'Real':
                n_parameters += max(10, int(d.high - d.low))
            else:
                n_parameters += len(d.bounds)
        # guess the size of the search space as the count of the
        # unordered combination of the dimensions entries
        search_space_size = (factorial(n_parameters) /
                             (factorial(n_parameters - n_dimensions) * factorial(n_dimensions)))
        # logger.info(f'Search space size: {search_space_size}')
        if search_space_size < config_jobs:
            # don't waste if the space is small
            n_initial_points = config_jobs
        else:
            # extract coefficients from the search space and the jobs count
            log_sss = int(log(search_space_size, 10))
            log_jobs = int(log(config_jobs, 2))
            log_jobs = 2 if log_jobs < 0 else log_jobs
            jobs_ip = log_jobs * log_sss
            # never waste
            n_initial_points = log_sss if jobs_ip > search_space_size else jobs_ip
        # it shall run for this much, I say
        min_epochs = max(2 * n_initial_points, 3 * config_jobs) * effort
        return n_initial_points, min_epochs, search_space_size

    def update_max_epoch(self, val: Dict, current: int):
        """ calculate max epochs: store the number of non best epochs
            between each best, and get the mean of that value """
        if val['is_initial_point'] is not True:
            self.epochs_since_last_best.append(current - self.current_best_epoch)
            self.avg_best_occurrence = (sum(self.epochs_since_last_best) //
                                        len(self.epochs_since_last_best))
            self.current_best_epoch = current
            self.max_epoch = (self.current_best_epoch + self.avg_best_occurrence +
                              self.min_epochs) * self.effort
            if self.max_epoch > self.search_space_size:
                self.max_epoch = self.search_space_size
        print('\n')
        logger.info(f'Max epochs set to: {self.max_epoch}')

    def start(self) -> None:
        self.random_state = self._set_random_state(self.config.get('hyperopt_random_state', None))
        logger.info(f"Using optimizer random state: {self.random_state}")

        data, timerange = self.backtesting.load_bt_data()

        preprocessed = self.backtesting.strategy.tickerdata_to_dataframe(data)

        # Trim startup period from analyzed dataframe
        for pair, df in preprocessed.items():
            preprocessed[pair] = trim_dataframe(df, timerange)
        min_date, max_date = get_timerange(data)

        logger.info('Hyperopting with data from %s up to %s (%s days)..', min_date.isoformat(),
                    max_date.isoformat(), (max_date - min_date).days)
        dump(preprocessed, self.tickerdata_pickle)

        # We don't need exchange instance anymore while running hyperopt
        self.backtesting.exchange = None  # type: ignore

        self.trials = self.load_previous_results(self.trials_file)

        cpus = cpu_count()
        logger.info(f"Found {cpus} CPU cores. Let's make them scream!")
        config_jobs = self.config.get('hyperopt_jobs', -1)
        logger.info(f'Number of parallel jobs set as: {config_jobs}')

        self.dimensions: List[Dimension] = self.hyperopt_space()
        self.n_initial_points, self.min_epochs, self.search_space_size = self.calc_epochs(
            self.dimensions, config_jobs, self.effort)
        logger.info(f"Min epochs set to: {self.min_epochs}")
        self.max_epoch = self.min_epochs
        self.avg_best_occurrence = self.max_epoch

        logger.info(f'Initial points: {self.n_initial_points}')
        self.opt = self.get_optimizer(self.dimensions, config_jobs, self.n_initial_points)

        # last_frame_len = (self.total_epochs - 1) % self.avg_best_occurrence

        if self.print_colorized:
            colorama_init(autoreset=True)

            try:
                register_parallel_backend('custom', CustomImmediateResultBackend)
                with parallel_backend('custom'):
                    with Parallel(n_jobs=config_jobs, verbose=0) as parallel:
                        for frame in range(self.total_epochs):
                            epochs_so_far = len(self.trials)
                            # pad the frame length to the number of jobs to avoid desaturation
                            frame_len = (self.avg_best_occurrence + config_jobs -
                                         self.avg_best_occurrence % config_jobs)
                            print(
                                f"{epochs_so_far+1}-{epochs_so_far+self.avg_best_occurrence}"
                                f"/{self.total_epochs}: ",
                                end='')
                            f_val = self.run_optimizer_parallel(parallel, frame_len, epochs_so_far)
                            self.log_results(f_val, epochs_so_far, self.total_epochs)
                            if self.max_epoch_reached:
                                logger.info("Max epoch reached, terminating.")
                                break

            except KeyboardInterrupt:
                print("User interrupted..")

        self.save_trials(final=True)

        if self.trials:
            sorted_trials = sorted(self.trials, key=itemgetter('loss'))
            results = sorted_trials[0]
            self.print_epoch_details(results, self.total_epochs, self.print_json)
        else:
            # This is printed when Ctrl+C is pressed quickly, before first epochs have
            # a chance to be evaluated.
            print("No epochs evaluated yet, no best result.")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['trials']
        return state
