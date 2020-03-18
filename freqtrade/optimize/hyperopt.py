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
from collections import OrderedDict, deque
from math import factorial, log
from numpy import iinfo, int32
from operator import itemgetter
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Set

import rapidjson
from colorama import Fore, Style
from colorama import init as colorama_init
from joblib import (Parallel, cpu_count, delayed, dump, load, wrap_non_picklable_objects)
from joblib import parallel_backend
from multiprocessing import Manager
from queue import Queue
from pandas import DataFrame, json_normalize, isna
from tabulate import tabulate

from freqtrade.data.converter import trim_dataframe
from freqtrade.data.history import get_timerange
from freqtrade.exceptions import OperationalException
from freqtrade.misc import plural, round_dict
from freqtrade.optimize.backtesting import Backtesting
# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from freqtrade.resolvers.hyperopt_resolver import (HyperOptLossResolver, HyperOptResolver)

# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
    from skopt.space import Dimension
# Additional regressors already pluggable into the optimizer
# from sklearn.linear_model import ARDRegression, BayesianRidge
# possibly interesting regressors that need predict method override
# from sklearn.ensemble import HistGradientBoostingRegressor
# from xgboost import XGBoostRegressor

logger = logging.getLogger(__name__)

# supported strategies when asking for multiple points to the optimizer
LIE_STRATS = ["cl_min", "cl_mean", "cl_max"]
LIE_STRATS_N = len(LIE_STRATS)

# supported estimators
ESTIMATORS = ["GBRT", "ET", "RF"]
ESTIMATORS_N = len(ESTIMATORS)

VOID_LOSS = iinfo(int32).max  # just a big enough number to be bad result in loss optimization


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
        self.opts_file = (self.config['user_data_dir'] / 'hyperopt_results' /
                          'hyperopt_optimizers.pickle')
        self.tickerdata_pickle = (self.config['user_data_dir'] / 'hyperopt_results' /
                                  'hyperopt_tickerdata.pkl')

        self.n_jobs = self.config.get('hyperopt_jobs', -1)
        if self.n_jobs < 0:
            self.n_jobs = cpu_count() // 2 or 1
        self.effort = self.config['effort'] if 'effort' in self.config else 0
        self.total_epochs = self.config['epochs'] if 'epochs' in self.config else 0
        self.max_epoch = 0
        self.max_epoch_reached = False
        self.min_epochs = 0
        self.epochs_limit = lambda: self.total_epochs or self.max_epoch

        # a guessed number extracted by the space dimensions
        self.search_space_size = 0
        # total number of candles being backtested
        self.n_candles = 0

        self.current_best_loss = VOID_LOSS
        self.current_best_epoch = 0
        self.epochs_since_last_best: List = []
        self.avg_best_occurrence = 0

        if not self.config.get('hyperopt_continue'):
            self.clean_hyperopt()
        else:
            logger.info("Continuing on previous hyperopt results.")

        self.num_trials_saved = 0

        # evaluations
        self.trials: List = []

        # configure multi mode
        self.setup_multi()

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

    def setup_multi(self):
        # optimizers
        self.opts: List[Optimizer] = []
        self.opt: Optimizer = None

        backend.manager = Manager()
        self.mode = self.config.get('mode', 'single')
        self.shared = False
        # in multi opt one model is enough
        self.n_models = 1
        if self.mode in ('multi', 'shared'):
            self.multi = True
            if self.mode == 'shared':
                self.shared = True
                self.opt_base_estimator = lambda: 'GBRT'
            else:
                self.opt_base_estimator = self.estimators
            self.opt_acq_optimizer = 'sampling'
            backend.optimizers = backend.manager.Queue()
            backend.results_board = backend.manager.Queue(maxsize=1)
            backend.results_board.put({})
        else:
            self.multi = False
            backend.results = backend.manager.Queue()
            self.opt_acq_optimizer = 'sampling'
            self.opt_base_estimator = lambda: 'ET'
            # The GaussianProcessRegressor is heavy, which makes it not a good default
            # however longer backtests might make it a better tradeoff
            # self.opt_base_estimator = lambda: 'GP'
            # self.opt_acq_optimizer = 'lbfgs'

        # in single opt assume runs are expensive so default to 1 point per ask
        self.n_points = self.config.get('n_points', 1)
        # if 0 n_points are given, don't use any base estimator (akin to random search)
        if self.n_points < 1:
            self.n_points = 1
            self.opt_base_estimator = lambda: "DUMMY"
            self.opt_acq_optimizer = "sampling"
        if self.n_points < 2:
            # ask_points is what is used in the ask call
            # because when n_points is None, it doesn't
            # waste time generating new points
            self.ask_points = None
        else:
            self.ask_points = self.n_points
        # var used in epochs and batches calculation
        self.opt_points = self.n_jobs * (self.n_points or 1)
        # lie strategy
        lie_strat = self.config.get('lie_strat', 'default')
        if lie_strat == 'default':
            self.lie_strat = lambda: 'cl_min'
        elif lie_strat == 'random':
            self.lie_strat = self.lie_strategy
        else:
            self.lie_strat = lambda: lie_strat

    @staticmethod
    def get_lock_filename(config: Dict[str, Any]) -> str:

        return str(config['user_data_dir'] / 'hyperopt.lock')

    def clean_hyperopt(self) -> None:
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        for f in [self.tickerdata_pickle, self.trials_file, self.opts_file]:
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
        Save hyperopt trials
        """
        num_trials = len(self.trials)
        if num_trials > self.num_trials_saved:
            logger.debug(f"\nSaving {num_trials} {plural(num_trials, 'epoch')}.")
            # save_trials(self.trials, trials_path, self.num_trials_saved)
            dump(self.trials, self.trials_file)
            self.num_trials_saved = num_trials
            self.save_opts()
        if final:
            logger.info(f"\n{num_trials} {plural(num_trials, 'epoch')} "
                        f"saved to '{self.trials_file}'.")

    def save_opts(self) -> None:
        """
        Save optimizers state to disk. The minimum required state could also be constructed
        from the attributes [ models, space, rng ] with Xi, yi loaded from trials.
        All we really care about are [rng, Xi, yi] since models are never passed over queues
        and space is dependent on dimensions matching with hyperopt config
        """
        # synchronize with saved trials
        opts = []
        n_opts = 0
        if self.multi:
            while not backend.optimizers.empty():
                opts.append(backend.optimizers.get())
            n_opts = len(opts)
            for opt in opts:
                backend.optimizers.put(opt)
        else:
            if self.opt:
                n_opts = 1
                opts = [self.opt]
        logger.debug(f"Saving {n_opts} {plural(n_opts, 'optimizer')}.")
        dump(opts, self.opts_file)

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
            result['buy'] = {p.name: params.get(p.name)
                             for p in self.hyperopt_space('buy')}
        if self.has_space('sell'):
            result['sell'] = {p.name: params.get(p.name)
                              for p in self.hyperopt_space('sell')}
        if self.has_space('roi'):
            result['roi'] = self.custom_hyperopt.generate_roi_table(params)
        if self.has_space('stoploss'):
            result['stoploss'] = {p.name: params.get(p.name)
                                  for p in self.hyperopt_space('stoploss')}
        if self.has_space('trailing'):
            result['trailing'] = self.custom_hyperopt.generate_trailing_params(params)

        return result

    @staticmethod
    def print_epoch_details(results, total_epochs: int, print_json: bool,
                            no_header: bool = False, header_str: str = None) -> None:
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
                    (str(k), v) for k, v in space_params.items()
                )
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
            self.print_results_explanation(results, self.epochs_limit(), self.print_all,
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
                f"{results['results_explanation']} " +
                f"Objective: {results['loss']:.5f}")

    @staticmethod
    def print_result_table(config: dict, results: list, total_epochs: int, highlight_best: bool,
                           print_colorized: bool) -> None:
        """
        Log result table
        """
        if not results:
            return

        trials = json_normalize(results, max_level=1)
        trials['Best'] = ''
        trials = trials[['Best', 'current_epoch', 'results_metrics.trade_count',
                         'results_metrics.avg_profit', 'results_metrics.total_profit',
                         'results_metrics.profit', 'results_metrics.duration',
                         'loss', 'is_initial_point', 'is_best']]
        trials.columns = ['Best', 'Epoch', 'Trades', 'Avg profit', 'Total profit',
                          'Profit', 'Avg duration', 'Objective', 'is_initial_point', 'is_best']
        trials['is_profit'] = False
        trials.loc[trials['is_initial_point'], 'Best'] = '*'
        trials.loc[trials['is_best'], 'Best'] = 'Best'
        trials['Objective'] = trials['Objective'].astype(str)
        trials.loc[trials['Total profit'] > 0, 'is_profit'] = True
        trials['Trades'] = trials['Trades'].astype(str)

        trials['Epoch'] = trials['Epoch'].apply(
            lambda x: "{}/{}".format(x, total_epochs))
        trials['Avg profit'] = trials['Avg profit'].apply(
            lambda x: '{:,.2f}%'.format(x) if not isna(x) else x)
        trials['Profit'] = trials['Profit'].apply(
            lambda x: '{:,.2f}%'.format(x) if not isna(x) else x)
        trials['Total profit'] = trials['Total profit'].apply(
            lambda x: '{: 11.8f} '.format(x) + config['stake_currency'] if not isna(x) else x)
        trials['Avg duration'] = trials['Avg duration'].apply(
            lambda x: '{:,.1f}m'.format(x) if not isna(x) else x)
        if print_colorized:
            for i in range(len(trials)):
                if trials.loc[i]['is_profit']:
                    for z in range(len(trials.loc[i])-3):
                        trials.iat[i, z] = "{}{}{}".format(Fore.GREEN,
                                                           str(trials.loc[i][z]), Fore.RESET)
                if trials.loc[i]['is_best'] and highlight_best:
                    for z in range(len(trials.loc[i])-3):
                        trials.iat[i, z] = "{}{}{}".format(Style.BRIGHT,
                                                           str(trials.loc[i][z]), Style.RESET_ALL)

        trials = trials.drop(columns=['is_initial_point', 'is_best', 'is_profit'])

        print(tabulate(trials.to_dict(orient='list'), headers='keys', tablefmt='psql',
                       stralign="right"))

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

    def backtest_params(self, raw_params: List[Any], iteration=None) -> Dict:
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
        loss: float = VOID_LOSS
        if trade_count >= self.config['hyperopt_min_trades']:
            loss = self.calculate_loss(results=backtesting_results, trade_count=trade_count,
                                       min_date=min_date.datetime, max_date=max_date.datetime)
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
                f"Avg duration {results_metrics['duration']:5.1f} min."
                ).encode(locale.getpreferredencoding(), 'replace').decode('utf-8')

    @staticmethod
    def filter_void_losses(vals: List, opt: Optimizer) -> List:
        """ remove out of bound losses from the results """
        if opt.void_loss == VOID_LOSS and len(opt.yi) < 1:
            # only exclude results at the beginning when void loss is yet to be set
            void_filtered = list(filter(lambda v: v["loss"] != VOID_LOSS, vals))
        else:
            if opt.void_loss == VOID_LOSS:  # set void loss once
                opt.void_loss = max(opt.yi)
            void_filtered = []
            # default bad losses to set void_loss
            for k, v in enumerate(vals):
                if v["loss"] == VOID_LOSS:
                    vals[k]["loss"] = opt.void_loss
            void_filtered = vals
        return void_filtered

    def lie_strategy(self):
        """ Choose a strategy randomly among the supported ones, used in multi opt mode
        to increase the diversion of the searches of each optimizer """
        return LIE_STRATS[random.randrange(0, LIE_STRATS_N)]

    def estimators(self):
        return ESTIMATORS[random.randrange(0, ESTIMATORS_N)]

    def get_optimizer(self, dimensions: List[Dimension], n_jobs: int,
                      n_initial_points: int) -> Optimizer:
        " Construct an optimizer object "
        # https://github.com/scikit-learn/scikit-learn/issues/14265
        # lbfgs uses joblib threading backend so n_jobs has to be reduced
        # to avoid oversubscription
        if self.opt_acq_optimizer == 'lbfgs':
            n_jobs = 1
        return Optimizer(
            dimensions,
            base_estimator=self.opt_base_estimator(),
            acq_optimizer=self.opt_acq_optimizer,
            n_initial_points=n_initial_points,
            acq_optimizer_kwargs={'n_jobs': n_jobs},
            model_queue_size=self.n_models,
            random_state=self.random_state,
        )

    def run_backtest_parallel(self, parallel: Parallel, tries: int, first_try: int,
                              jobs: int) -> List:
        """ launch parallel in single opt mode, return the evaluated epochs """
        result = parallel(
            delayed(wrap_non_picklable_objects(self.parallel_objective))(asked, backend.results, i)
            for asked, i in zip(self.opt_ask_and_tell(jobs, tries),
                                range(first_try, first_try + tries)))
        return result

    def run_multi_backtest_parallel(self, parallel: Parallel, tries: int, first_try: int,
                                    jobs: int) -> List:
        """ launch parallel in multi opt mode, return the evaluated epochs"""
        results = parallel(
            delayed(wrap_non_picklable_objects(self.parallel_opt_objective))(
                i, backend.optimizers, jobs, backend.results_board)
            for i in range(first_try, first_try + tries))
        # each worker will return a list containing n_points, so compact into a single list
        return functools.reduce(lambda x, y: [*x, *y], results, [])

    def opt_ask_and_tell(self, jobs: int, tries: int):
        """
        loop to manage optimizer state in single optimizer mode, everytime a job is
        dispatched, we check the optimizer for points, to ask and to tell if any,
        but only fit a new model every n_points, because if we fit at every result previous
        points become invalid.
        """
        vals = []
        fit = False
        to_ask: deque = deque()
        evald: Set[Tuple] = set()
        opt = self.opt
        def point():
            if self.ask_points:
                if to_ask:
                    return tuple(to_ask.popleft())
                else:
                    to_ask.extend(opt.ask(n_points=self.ask_points, strategy=self.lie_strat()))
                    return tuple(to_ask.popleft())
            else:
                return tuple(opt.ask(strategy=self.lie_strat()))

        for r in range(tries):
            fit = (len(to_ask) < 1)
            while not backend.results.empty():
                vals.append(backend.results.get())
            if vals:
                # filter losses
                void_filtered = self.filter_void_losses(vals, opt)
                if void_filtered:  # again if all are filtered
                    opt.tell([list(v['params_dict'].values()) for v in void_filtered],
                             [v['loss'] for v in void_filtered],
                             fit=fit)  # only fit when out of points
                    del vals[:], void_filtered[:]

            a = point()
            while a in evald:
                logger.debug("this point was evaluated before...")
                if not fit:
                    opt.update_next()
                a = point()
            evald.add(a)
            yield a

    @staticmethod
    def opt_get_past_points(asked: dict, results_board: Queue) -> dict:
        """ fetch shared results between optimizers """
        results = results_board.get()
        results_board.put(results)
        for a in asked:
            if a in results:
                asked[a] = results[a]
        return asked

    def parallel_opt_objective(self, n: int, optimizers: Queue, jobs: int, results_board: Queue):
        """
        objective run in multi opt mode, optimizers share the results as soon as they are completed
        """
        self.log_results_immediate(n)
        is_shared = self.shared
        # get an optimizer instance
        opt = optimizers.get()
        # this is the counter used by the optimizer internally to track the initial
        # points evaluated so far..
        initial_points = opt._n_initial_points

        if is_shared:
            # get a random number before putting it back to avoid
            # replication with other workers and keep reproducibility
            rand = opt.rng.randint(0, VOID_LOSS)
            optimizers.put(opt)
            # switch the seed to get a different point
            opt.rng.seed(rand)
            opt, opt.void_loss = opt.copy(random_state=opt.rng), opt.void_loss
        # we have to get a new point if the last batch was all void
        elif opt.void:
            opt.update_next()
        # a model is only fit after initial points
        elif initial_points < 1:
            opt.tell(opt.Xi, opt.yi)

        Xi_d = []  # done
        yi_d = []
        Xi_t = []  # to do
        # ask for points according to config
        while True:
            asked = opt.ask(n_points=self.ask_points, strategy=self.lie_strat())
            if not self.ask_points:
                asked = {tuple(asked): None}
            else:
                asked = {tuple(a): None for a in asked}
            # check if some points have been evaluated by other optimizers
            p_asked = self.opt_get_past_points(asked, results_board)
            for a in p_asked:
                if p_asked[a] is not None:
                    Xi_d.append(a)
                    yi_d.append(p_asked[a])
                else:
                    Xi_t.append(a)
            if len(Xi_t) < self.n_points:
                opt.update_next()
            else:
                break
        # run the backtest for each point to do (Xi_t)
        f_val = [self.backtest_params(a) for a in Xi_t]
        # filter losses
        void_filtered = self.filter_void_losses(f_val, opt)
        # add points of the current dispatch if any
        if opt.void_loss != VOID_LOSS or len(void_filtered) > 0:
            Xi = [*Xi_d, *[list(v['params_dict'].values()) for v in void_filtered]]
            yi = [*yi_d, *[v['loss'] for v in void_filtered]]
            void = False
            if is_shared:
                # refresh the optimizer that stores all the points
                opt = optimizers.get()
            opt.tell(Xi, yi, fit=False)
        else:
            void = True
            opt.void = void
        # send back the updated optimizer only in non shared mode
        # because in shared mode if all results are void we don't
        # fetch it at all
        if not void or not is_shared:
            # don't pickle models
            del opt.models[:]
            optimizers.put(opt)
        # update the board used to skip already computed points
        # NOTE: some results at the beginning won't be published
        # because they are removed by the filter_void_losses
        if not void:
            results = results_board.get()
            for v in void_filtered:
                a = tuple(v['params_dict'].values())
                if a not in results:
                    results[a] = v['loss']
            results_board.put(results)
            # set initial point flag
            for n, v in enumerate(void_filtered):
                v['is_initial_point'] = initial_points - n > 0
        return void_filtered

    def parallel_objective(self, asked, results: Queue = None, n=0):
        """ objective run in single opt mode, run the backtest, store the results into a queue """
        self.log_results_immediate(n)
        v = self.backtest_params(asked)
        if results:
            results.put(v)
        v['is_initial_point'] = n < self.opt_n_initial_points
        return v

    def log_results_immediate(self, n) -> None:
        """ Signals that a new job has been scheduled"""
        print('.', end='')
        sys.stdout.flush()

    def log_results(self, f_val, frame_start, total_epochs: int) -> int:
        """
        Log results if it is better than any previous evaluation
        """
        current = frame_start + 1
        i = 0
        for i, v in enumerate(f_val, 1):
            is_best = self.is_best_loss(v, self.current_best_loss)
            current = frame_start + i
            v['is_best'] = is_best
            v['current_epoch'] = current
            logger.debug(f"Optimizer epoch evaluated: {v}")
            if is_best:
                self.current_best_loss = v['loss']
                self.update_max_epoch(v, current)
            self.print_results(v)
            self.trials.append(v)
        # Save results and optimizersafter every batch
        self.save_trials()
        # give up if no best since max epochs
        if current + 1 > self.epochs_limit():
            self.max_epoch_reached = True
        return i

    def setup_epochs(self) -> bool:
        """ used to resume the best epochs state from previous trials """
        len_trials = len(self.trials)
        if len_trials > 0:
            best_epochs = list(filter(lambda k: k["is_best"], self.trials))
            len_best = len(best_epochs)
            if len_best > 0:
                # sorting from lowest to highest, the first value is the current best
                best = sorted(best_epochs, key=lambda k: k["loss"])[0]
                self.current_best_epoch = best["current_epoch"]
                self.current_best_loss = best["loss"]
                self.avg_best_occurrence = len_trials // len_best
                return True
        return False

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

    @staticmethod
    def load_previous_optimizers(opts_file: Path) -> List:
        """ Load the state of previous optimizers from file """
        opts: List[Optimizer] = []
        if opts_file.is_file() and opts_file.stat().st_size > 0:
            opts = load(opts_file)
        n_opts = len(opts)
        if n_opts > 0 and type(opts[-1]) != Optimizer:
            raise OperationalException("The file storing optimizers state might be corrupted "
                                       "and cannot be loaded.")
        else:
            logger.info(f"Loaded {n_opts} previous {plural(n_opts, 'optimizer')} from disk.")
        return opts

    def _set_random_state(self, random_state: Optional[int]) -> int:
        return random_state or random.randint(1, 2**16 - 1)

    @staticmethod
    def calc_epochs(
        dimensions: List[Dimension], n_jobs: int, effort: float, total_epochs: int, n_points: int
    ):
        """ Compute a reasonable number of initial points and
        a minimum number of epochs to evaluate """
        n_dimensions = len(dimensions)
        n_parameters = 0
        opt_points = n_jobs * n_points
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
        search_space_size = int(
            (factorial(n_parameters) /
             (factorial(n_parameters - n_dimensions) * factorial(n_dimensions))))
        # logger.info(f'Search space size: {search_space_size}')
        log_opt = int(log(opt_points, 2)) if opt_points > 4 else 2
        if search_space_size < opt_points:
            # don't waste if the space is small
            n_initial_points = opt_points // 3
            min_epochs = opt_points
        elif total_epochs > 0:
            # coefficients from total epochs
            log_epp = int(log(total_epochs, 2)) * log_opt
            n_initial_points = min(log_epp, total_epochs // 3)
            min_epochs = total_epochs
        else:
            # extract coefficients from the search space
            log_sss = int(log(search_space_size, 10)) * log_opt
            # never waste
            n_initial_points = min(log_sss, search_space_size // 3)
            # it shall run for this much, I say
            min_epochs = int(max(n_initial_points, opt_points) * (1 + effort) + n_initial_points)
        return n_initial_points, min_epochs, search_space_size

    def update_max_epoch(self, val: Dict, current: int):
        """ calculate max epochs: store the number of non best epochs
            between each best, and get the mean of that value """
        if val['is_initial_point'] is not True:
            self.epochs_since_last_best.append(current - self.current_best_epoch)
            self.avg_best_occurrence = (sum(self.epochs_since_last_best) //
                                        len(self.epochs_since_last_best))
            self.current_best_epoch = current
            self.max_epoch = int(
                (self.current_best_epoch + self.avg_best_occurrence + self.min_epochs) *
                (1 + self.effort))
            if self.max_epoch > self.search_space_size:
                self.max_epoch = self.search_space_size
        logger.debug(f'\nMax epoch set to: {self.epochs_limit()}')

    def setup_optimizers(self):
        """ Setup the optimizers objects, try to load from disk, or create new ones """
        # try to load previous optimizers
        opts = self.load_previous_optimizers(self.opts_file)
        n_opts = len(opts)
        max_opts = self.n_jobs

        if self.multi:
            # when sharing results there is only one optimizer that gets copied
            if self.shared:
                max_opts = 1
            # put the restored optimizers in the queue
            if n_opts > 0:
                for n in range(n_opts):
                    backend.optimizers.put(opts[n])
            # generate as many optimizers as are still needed to fill the job count
            remaining = max_opts - backend.optimizers.qsize()
            if remaining > 0:
                opt = self.get_optimizer(self.dimensions, self.n_jobs, self.opt_n_initial_points)
                for _ in range(remaining):  # generate optimizers
                    # random state is preserved
                    opt_copy = opt.copy(random_state=opt.rng.randint(0,
                                                                     iinfo(int32).max))
                    opt_copy.void_loss = VOID_LOSS
                    opt_copy.void = False
                    backend.optimizers.put(opt_copy)
                del opt, opt_copy
        else:
            # if we have more than 1 optimizer but are using single opt,
            # pick one discard the rest...
            if n_opts > 0:
                self.opt = opts[-1]
            else:
                self.opt = self.get_optimizer(
                    self.dimensions, self.n_jobs, self.opt_n_initial_points
                )
                self.opt.void_loss = VOID_LOSS
                self.opt.void = False
        del opts[:]

    def setup_points(self):
        self.n_initial_points, self.min_epochs, self.search_space_size = self.calc_epochs(
            self.dimensions, self.n_jobs, self.effort, self.total_epochs, self.n_points
        )
        logger.info(f"Min epochs set to: {self.min_epochs}")
        # reduce random points by n_points in multi mode because asks are per job
        if self.multi:
            self.opt_n_initial_points = self.n_initial_points // self.n_points
        else:
            self.opt_n_initial_points = self.n_initial_points
        logger.info(f'Initial points: {self.n_initial_points}')
        # if total epochs are not set, max_epoch takes its place
        if self.total_epochs < 1:
            self.max_epoch = int(self.min_epochs + len(self.trials))
        # initialize average best occurrence
        self.avg_best_occurrence = self.min_epochs // self.n_jobs

    def start(self) -> None:
        """ Broom Broom """
        self.random_state = self._set_random_state(self.config.get('hyperopt_random_state', None))
        logger.info(f"Using optimizer random state: {self.random_state}")

        data, timerange = self.backtesting.load_bt_data()

        preprocessed = self.backtesting.strategy.tickerdata_to_dataframe(data)

        # Trim startup period from analyzed dataframe
        for pair, df in preprocessed.items():
            preprocessed[pair] = trim_dataframe(df, timerange)
            self.n_candles += len(preprocessed[pair])
        min_date, max_date = get_timerange(data)

        logger.info(
            'Hyperopting with data from %s up to %s (%s days)..',
            min_date.isoformat(), max_date.isoformat(), (max_date - min_date).days
        )
        dump(preprocessed, self.tickerdata_pickle)

        # We don't need exchange instance anymore while running hyperopt
        self.backtesting.exchange = None  # type: ignore

        self.trials = self.load_previous_results(self.trials_file)
        self.setup_epochs()

        logger.info(f"Found {cpu_count()} CPU cores. Let's make them scream!")
        logger.info(f'Number of parallel jobs set as: {self.n_jobs}')

        self.dimensions: List[Dimension] = self.hyperopt_space()
        self.setup_points()

        if self.print_colorized:
            colorama_init(autoreset=True)

        self.setup_optimizers()
        try:
            if self.multi:
                jobs_scheduler = self.run_multi_backtest_parallel
            else:
                jobs_scheduler = self.run_backtest_parallel
            with parallel_backend('loky', inner_max_num_threads=2):
                with Parallel(n_jobs=self.n_jobs, verbose=0, backend='loky') as parallel:
                    jobs = parallel._effective_n_jobs()
                    logger.info(f'Effective number of parallel workers used: {jobs}')
                    # update epochs count
                    n_points = self.n_points
                    prev_batch = -1
                    epochs_so_far = len(self.trials)
                    epochs_limit = self.epochs_limit
                    columns, _ = os.get_terminal_size()
                    columns -= 1
                    while epochs_so_far > prev_batch or epochs_so_far < self.min_epochs:
                        prev_batch = epochs_so_far
                        occurrence = int(self.avg_best_occurrence * (1 + self.effort))
                        # pad the batch length to the number of jobs to avoid desaturation
                        batch_len = (occurrence + jobs  -
                                     occurrence % jobs)
                        # when using multiple optimizers each worker performs
                        # n_points (epochs) in 1 dispatch but this reduces the batch len too much
                        # if self.multi: batch_len = batch_len // self.n_points
                        # don't go over the limit
                        if epochs_so_far + batch_len * n_points >= epochs_limit():
                            q, r = divmod(epochs_limit() - epochs_so_far, n_points)
                            batch_len = q + r
                        print(
                            f"{epochs_so_far+1}-{epochs_so_far+batch_len*n_points}"
                            f"/{epochs_limit()}: ",
                            end='')
                        f_val = jobs_scheduler(parallel, batch_len, epochs_so_far, jobs)
                        print(end='\r')
                        saved = self.log_results(f_val, epochs_so_far, epochs_limit())
                        print('\r', ' ' * columns, end='\r')
                        # stop if no epochs have been evaluated
                        if len(f_val) < batch_len:
                            logger.warning("Some evaluated epochs were void, "
                                           "check the loss function and the search space.")
                        if (not saved and len(f_val) > 1) or batch_len < 1:
                            break
                        # log_results add
                        epochs_so_far += saved
                        if self.max_epoch_reached:
                            logger.info("Max epoch reached, terminating.")
                            break

        except KeyboardInterrupt:
            print('User interrupted..')

        self.save_trials(final=True)

        if self.trials:
            sorted_trials = sorted(self.trials, key=itemgetter('loss'))
            results = sorted_trials[0]
            self.print_epoch_details(results, self.epochs_limit(), self.print_json)
        else:
            # This is printed when Ctrl+C is pressed quickly, before first epochs have
            # a chance to be evaluated.
            print("No epochs evaluated yet, no best result.")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['trials']
        return state
