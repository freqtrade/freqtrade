# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

"""
This module contains the hyperopt logic
"""

import logging
import random
import sys
import warnings
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rapidjson
from colorama import init as colorama_init
from joblib import Parallel, cpu_count, delayed, dump, load, wrap_non_picklable_objects
from joblib.externals import cloudpickle
from pandas import DataFrame
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from freqtrade.constants import DATETIME_PRINT_FORMAT, FTHYPT_FILEVERSION, LAST_BT_RESULT_FN, Config
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.history import get_timerange
from freqtrade.data.metrics import calculate_market_change
from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.misc import deep_merge_dicts, file_dump_json, plural
from freqtrade.optimize.backtesting import Backtesting
# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
from freqtrade.optimize.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.optimize.hyperopt_tools import (HyperoptStateContainer, HyperoptTools,
                                               hyperopt_serializer)
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver


# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
    from skopt.space import Dimension

logger = logging.getLogger(__name__)


INITIAL_POINTS = 30

# Keep no more than SKOPT_MODEL_QUEUE_SIZE models
# in the skopt model queue, to optimize memory consumption
SKOPT_MODEL_QUEUE_SIZE = 10

MAX_LOSS = 100000  # just a big enough number to be bad result in loss optimization


class Hyperopt:
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To start a hyperopt run:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """

    def __init__(self, config: Config) -> None:
        self.buy_space: List[Dimension] = []
        self.sell_space: List[Dimension] = []
        self.protection_space: List[Dimension] = []
        self.roi_space: List[Dimension] = []
        self.stoploss_space: List[Dimension] = []
        self.trailing_space: List[Dimension] = []
        self.max_open_trades_space: List[Dimension] = []
        self.dimensions: List[Dimension] = []

        self.config = config
        self.min_date: datetime
        self.max_date: datetime

        self.backtesting = Backtesting(self.config)
        self.pairlist = self.backtesting.pairlists.whitelist
        self.custom_hyperopt: HyperOptAuto
        self.analyze_per_epoch = self.config.get('analyze_per_epoch', False)
        HyperoptStateContainer.set_state(HyperoptState.STARTUP)

        if not self.config.get('hyperopt'):
            self.custom_hyperopt = HyperOptAuto(self.config)
        else:
            raise OperationalException(
                "Using separate Hyperopt files has been removed in 2021.9. Please convert "
                "your existing Hyperopt file to the new Hyperoptable strategy interface")

        self.backtesting._set_strategy(self.backtesting.strategylist[0])
        self.custom_hyperopt.strategy = self.backtesting.strategy

        self.hyperopt_pickle_magic(self.backtesting.strategy.__class__.__bases__)
        self.custom_hyperoptloss: IHyperOptLoss = HyperOptLossResolver.load_hyperoptloss(
            self.config)
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        strategy = str(self.config['strategy'])
        self.results_file: Path = (self.config['user_data_dir'] / 'hyperopt_results' /
                                   f'strategy_{strategy}_{time_now}.fthypt')
        self.data_pickle_file = (self.config['user_data_dir'] /
                                 'hyperopt_results' / 'hyperopt_tickerdata.pkl')
        self.total_epochs = config.get('epochs', 0)

        self.current_best_loss = 100

        self.clean_hyperopt()

        self.market_change = 0.0
        self.num_epochs_saved = 0
        self.current_best_epoch: Optional[Dict[str, Any]] = None

        # Use max_open_trades for hyperopt as well, except --disable-max-market-positions is set
        if not self.config.get('use_max_market_positions', True):
            logger.debug('Ignoring max_open_trades (--disable-max-market-positions was used) ...')
            self.backtesting.strategy.max_open_trades = float('inf')
            config.update({'max_open_trades': self.backtesting.strategy.max_open_trades})

        if HyperoptTools.has_space(self.config, 'sell'):
            # Make sure use_exit_signal is enabled
            self.config['use_exit_signal'] = True

        self.print_all = self.config.get('print_all', False)
        self.hyperopt_table_header = 0
        self.print_colorized = self.config.get('print_colorized', False)
        self.print_json = self.config.get('print_json', False)

    @staticmethod
    def get_lock_filename(config: Config) -> str:

        return str(config['user_data_dir'] / 'hyperopt.lock')

    def clean_hyperopt(self) -> None:
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        for f in [self.data_pickle_file, self.results_file]:
            p = Path(f)
            if p.is_file():
                logger.info(f"Removing `{p}`.")
                p.unlink()

    def hyperopt_pickle_magic(self, bases) -> None:
        """
        Hyperopt magic to allow strategy inheritance across files.
        For this to properly work, we need to register the module of the imported class
        to pickle as value.
        """
        for modules in bases:
            if modules.__name__ != 'IStrategy':
                cloudpickle.register_pickle_by_value(sys.modules[modules.__module__])
                self.hyperopt_pickle_magic(modules.__bases__)

    def _get_params_dict(self, dimensions: List[Dimension], raw_params: List[Any]) -> Dict:

        # Ensure the number of dimensions match
        # the number of parameters in the list.
        if len(raw_params) != len(dimensions):
            raise ValueError('Mismatch in number of search-space dimensions.')

        # Return a dict where the keys are the names of the dimensions
        # and the values are taken from the list of parameters.
        return {d.name: v for d, v in zip(dimensions, raw_params)}

    def _save_result(self, epoch: Dict) -> None:
        """
        Save hyperopt results to file
        Store one line per epoch.
        While not a valid json object - this allows appending easily.
        :param epoch: result dictionary for this epoch.
        """
        epoch[FTHYPT_FILEVERSION] = 2
        with self.results_file.open('a') as f:
            rapidjson.dump(epoch, f, default=hyperopt_serializer,
                           number_mode=rapidjson.NM_NATIVE | rapidjson.NM_NAN)
            f.write("\n")

        self.num_epochs_saved += 1
        logger.debug(f"{self.num_epochs_saved} {plural(self.num_epochs_saved, 'epoch')} "
                     f"saved to '{self.results_file}'.")
        # Store hyperopt filename
        latest_filename = Path.joinpath(self.results_file.parent, LAST_BT_RESULT_FN)
        file_dump_json(latest_filename, {'latest_hyperopt': str(self.results_file.name)},
                       log=False)

    def _get_params_details(self, params: Dict) -> Dict:
        """
        Return the params for each space
        """
        result: Dict = {}

        if HyperoptTools.has_space(self.config, 'buy'):
            result['buy'] = {p.name: params.get(p.name) for p in self.buy_space}
        if HyperoptTools.has_space(self.config, 'sell'):
            result['sell'] = {p.name: params.get(p.name) for p in self.sell_space}
        if HyperoptTools.has_space(self.config, 'protection'):
            result['protection'] = {p.name: params.get(p.name) for p in self.protection_space}
        if HyperoptTools.has_space(self.config, 'roi'):
            result['roi'] = {str(k): v for k, v in
                             self.custom_hyperopt.generate_roi_table(params).items()}
        if HyperoptTools.has_space(self.config, 'stoploss'):
            result['stoploss'] = {p.name: params.get(p.name) for p in self.stoploss_space}
        if HyperoptTools.has_space(self.config, 'trailing'):
            result['trailing'] = self.custom_hyperopt.generate_trailing_params(params)
        if HyperoptTools.has_space(self.config, 'trades'):
            result['max_open_trades'] = {
                'max_open_trades': self.backtesting.strategy.max_open_trades
                if self.backtesting.strategy.max_open_trades != float('inf') else -1}

        return result

    def _get_no_optimize_details(self) -> Dict[str, Any]:
        """
        Get non-optimized parameters
        """
        result: Dict[str, Any] = {}
        strategy = self.backtesting.strategy
        if not HyperoptTools.has_space(self.config, 'roi'):
            result['roi'] = {str(k): v for k, v in strategy.minimal_roi.items()}
        if not HyperoptTools.has_space(self.config, 'stoploss'):
            result['stoploss'] = {'stoploss': strategy.stoploss}
        if not HyperoptTools.has_space(self.config, 'trailing'):
            result['trailing'] = {
                'trailing_stop': strategy.trailing_stop,
                'trailing_stop_positive': strategy.trailing_stop_positive,
                'trailing_stop_positive_offset': strategy.trailing_stop_positive_offset,
                'trailing_only_offset_is_reached': strategy.trailing_only_offset_is_reached,
            }
        if not HyperoptTools.has_space(self.config, 'trades'):
            result['max_open_trades'] = {'max_open_trades': strategy.max_open_trades}
        return result

    def print_results(self, results) -> None:
        """
        Log results if it is better than any previous evaluation
        TODO: this should be moved to HyperoptTools too
        """
        is_best = results['is_best']

        if self.print_all or is_best:
            print(
                HyperoptTools.get_result_table(
                    self.config, results, self.total_epochs,
                    self.print_all, self.print_colorized,
                    self.hyperopt_table_header
                )
            )
            self.hyperopt_table_header = 2

    def init_spaces(self):
        """
        Assign the dimensions in the hyperoptimization space.
        """
        if HyperoptTools.has_space(self.config, 'protection'):
            # Protections can only be optimized when using the Parameter interface
            logger.debug("Hyperopt has 'protection' space")
            # Enable Protections if protection space is selected.
            self.config['enable_protections'] = True
            self.backtesting.enable_protections = True
            self.protection_space = self.custom_hyperopt.protection_space()

        if HyperoptTools.has_space(self.config, 'buy'):
            logger.debug("Hyperopt has 'buy' space")
            self.buy_space = self.custom_hyperopt.buy_indicator_space()

        if HyperoptTools.has_space(self.config, 'sell'):
            logger.debug("Hyperopt has 'sell' space")
            self.sell_space = self.custom_hyperopt.sell_indicator_space()

        if HyperoptTools.has_space(self.config, 'roi'):
            logger.debug("Hyperopt has 'roi' space")
            self.roi_space = self.custom_hyperopt.roi_space()

        if HyperoptTools.has_space(self.config, 'stoploss'):
            logger.debug("Hyperopt has 'stoploss' space")
            self.stoploss_space = self.custom_hyperopt.stoploss_space()

        if HyperoptTools.has_space(self.config, 'trailing'):
            logger.debug("Hyperopt has 'trailing' space")
            self.trailing_space = self.custom_hyperopt.trailing_space()

        if HyperoptTools.has_space(self.config, 'trades'):
            logger.debug("Hyperopt has 'trades' space")
            self.max_open_trades_space = self.custom_hyperopt.max_open_trades_space()

        self.dimensions = (self.buy_space + self.sell_space + self.protection_space
                           + self.roi_space + self.stoploss_space + self.trailing_space
                           + self.max_open_trades_space)

    def assign_params(self, params_dict: Dict, category: str) -> None:
        """
        Assign hyperoptable parameters
        """
        for attr_name, attr in self.backtesting.strategy.enumerate_parameters(category):
            if attr.optimize:
                # noinspection PyProtectedMember
                attr.value = params_dict[attr_name]

    def generate_optimizer(self, raw_params: List[Any]) -> Dict[str, Any]:
        """
        Used Optimize function.
        Called once per epoch to optimize whatever is configured.
        Keep this function as optimized as possible!
        """
        HyperoptStateContainer.set_state(HyperoptState.OPTIMIZE)
        backtest_start_time = datetime.now(timezone.utc)
        params_dict = self._get_params_dict(self.dimensions, raw_params)

        # Apply parameters
        if HyperoptTools.has_space(self.config, 'buy'):
            self.assign_params(params_dict, 'buy')

        if HyperoptTools.has_space(self.config, 'sell'):
            self.assign_params(params_dict, 'sell')

        if HyperoptTools.has_space(self.config, 'protection'):
            self.assign_params(params_dict, 'protection')

        if HyperoptTools.has_space(self.config, 'roi'):
            self.backtesting.strategy.minimal_roi = (
                self.custom_hyperopt.generate_roi_table(params_dict))

        if HyperoptTools.has_space(self.config, 'stoploss'):
            self.backtesting.strategy.stoploss = params_dict['stoploss']

        if HyperoptTools.has_space(self.config, 'trailing'):
            d = self.custom_hyperopt.generate_trailing_params(params_dict)
            self.backtesting.strategy.trailing_stop = d['trailing_stop']
            self.backtesting.strategy.trailing_stop_positive = d['trailing_stop_positive']
            self.backtesting.strategy.trailing_stop_positive_offset = \
                d['trailing_stop_positive_offset']
            self.backtesting.strategy.trailing_only_offset_is_reached = \
                d['trailing_only_offset_is_reached']

        if HyperoptTools.has_space(self.config, 'trades'):
            if self.config["stake_amount"] == "unlimited" and \
                    (params_dict['max_open_trades'] == -1 or params_dict['max_open_trades'] == 0):
                # Ignore unlimited max open trades if stake amount is unlimited
                params_dict.update({'max_open_trades': self.config['max_open_trades']})

            updated_max_open_trades = int(params_dict['max_open_trades']) \
                if (params_dict['max_open_trades'] != -1
                    and params_dict['max_open_trades'] != 0) else float('inf')

            self.config.update({'max_open_trades': updated_max_open_trades})

            self.backtesting.strategy.max_open_trades = updated_max_open_trades

        with self.data_pickle_file.open('rb') as f:
            processed = load(f, mmap_mode='r')
            if self.analyze_per_epoch:
                # Data is not yet analyzed, rerun populate_indicators.
                processed = self.advise_and_trim(processed)

        bt_results = self.backtesting.backtest(
            processed=processed,
            start_date=self.min_date,
            end_date=self.max_date
        )
        backtest_end_time = datetime.now(timezone.utc)
        bt_results.update({
            'backtest_start_time': int(backtest_start_time.timestamp()),
            'backtest_end_time': int(backtest_end_time.timestamp()),
        })

        return self._get_results_dict(bt_results, self.min_date, self.max_date,
                                      params_dict,
                                      processed=processed)

    def _get_results_dict(self, backtesting_results, min_date, max_date,
                          params_dict, processed: Dict[str, DataFrame]
                          ) -> Dict[str, Any]:
        params_details = self._get_params_details(params_dict)

        strat_stats = generate_strategy_stats(
            self.pairlist, self.backtesting.strategy.get_strategy_name(),
            backtesting_results, min_date, max_date, market_change=self.market_change,
            is_hyperopt=True,
        )
        results_explanation = HyperoptTools.format_results_explanation_string(
            strat_stats, self.config['stake_currency'])

        not_optimized = self.backtesting.strategy.get_no_optimize_params()
        not_optimized = deep_merge_dicts(not_optimized, self._get_no_optimize_details())

        trade_count = strat_stats['total_trades']
        total_profit = strat_stats['profit_total']

        # If this evaluation contains too short amount of trades to be
        # interesting -- consider it as 'bad' (assigned max. loss value)
        # in order to cast this hyperspace point away from optimization
        # path. We do not want to optimize 'hodl' strategies.
        loss: float = MAX_LOSS
        if trade_count >= self.config['hyperopt_min_trades']:
            loss = self.calculate_loss(results=backtesting_results['results'],
                                       trade_count=trade_count,
                                       min_date=min_date, max_date=max_date,
                                       config=self.config, processed=processed,
                                       backtest_stats=strat_stats)
        return {
            'loss': loss,
            'params_dict': params_dict,
            'params_details': params_details,
            'params_not_optimized': not_optimized,
            'results_metrics': strat_stats,
            'results_explanation': results_explanation,
            'total_profit': total_profit,
        }

    def get_optimizer(self, dimensions: List[Dimension], cpu_count) -> Optimizer:
        estimator = self.custom_hyperopt.generate_estimator(dimensions=dimensions)

        acq_optimizer = "sampling"
        if isinstance(estimator, str):
            if estimator not in ("GP", "RF", "ET", "GBRT"):
                raise OperationalException(f"Estimator {estimator} not supported.")
            else:
                acq_optimizer = "auto"

        logger.info(f"Using estimator {estimator}.")
        return Optimizer(
            dimensions,
            base_estimator=estimator,
            acq_optimizer=acq_optimizer,
            n_initial_points=INITIAL_POINTS,
            acq_optimizer_kwargs={'n_jobs': cpu_count},
            random_state=self.random_state,
            model_queue_size=SKOPT_MODEL_QUEUE_SIZE,
        )

    def run_optimizer_parallel(
            self, parallel: Parallel, asked: List[List]) -> List[Dict[str, Any]]:
        """ Start optimizer in a parallel way """
        return parallel(delayed(
                        wrap_non_picklable_objects(self.generate_optimizer))(v) for v in asked)

    def _set_random_state(self, random_state: Optional[int]) -> int:
        return random_state or random.randint(1, 2**16 - 1)

    def advise_and_trim(self, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        preprocessed = self.backtesting.strategy.advise_all_indicators(data)

        # Trim startup period from analyzed dataframe to get correct dates for output.
        # This is only used to keep track of min/max date after trimming.
        # The result is NOT returned from this method, actual trimming happens in backtesting.
        trimmed = trim_dataframes(preprocessed, self.timerange, self.backtesting.required_startup)
        self.min_date, self.max_date = get_timerange(trimmed)
        if not self.market_change:
            self.market_change = calculate_market_change(trimmed, 'close')

        # Real trimming will happen as part of backtesting.
        return preprocessed

    def prepare_hyperopt_data(self) -> None:
        HyperoptStateContainer.set_state(HyperoptState.DATALOAD)
        data, self.timerange = self.backtesting.load_bt_data()
        self.backtesting.load_bt_data_detail()
        logger.info("Dataload complete. Calculating indicators")

        if not self.analyze_per_epoch:
            HyperoptStateContainer.set_state(HyperoptState.INDICATORS)

            preprocessed = self.advise_and_trim(data)

            logger.info(f'Hyperopting with data from '
                        f'{self.min_date.strftime(DATETIME_PRINT_FORMAT)} '
                        f'up to {self.max_date.strftime(DATETIME_PRINT_FORMAT)} '
                        f'({(self.max_date - self.min_date).days} days)..')
            # Store non-trimmed data - will be trimmed after signal generation.
            dump(preprocessed, self.data_pickle_file)
        else:
            dump(data, self.data_pickle_file)

    def get_asked_points(self, n_points: int) -> Tuple[List[List[Any]], List[bool]]:
        """
        Enforce points returned from `self.opt.ask` have not been already evaluated

        Steps:
        1. Try to get points using `self.opt.ask` first
        2. Discard the points that have already been evaluated
        3. Retry using `self.opt.ask` up to 3 times
        4. If still some points are missing in respect to `n_points`, random sample some points
        5. Repeat until at least `n_points` points in the `asked_non_tried` list
        6. Return a list with length truncated at `n_points`
        """
        def unique_list(a_list):
            new_list = []
            for item in a_list:
                if item not in new_list:
                    new_list.append(item)
            return new_list
        i = 0
        asked_non_tried: List[List[Any]] = []
        is_random_non_tried: List[bool] = []
        while i < 5 and len(asked_non_tried) < n_points:
            if i < 3:
                self.opt.cache_ = {}
                asked = unique_list(self.opt.ask(n_points=n_points * 5))
                is_random = [False for _ in range(len(asked))]
            else:
                asked = unique_list(self.opt.space.rvs(n_samples=n_points * 5))
                is_random = [True for _ in range(len(asked))]
            is_random_non_tried += [rand for x, rand in zip(asked, is_random)
                                    if x not in self.opt.Xi
                                    and x not in asked_non_tried]
            asked_non_tried += [x for x in asked
                                if x not in self.opt.Xi
                                and x not in asked_non_tried]
            i += 1

        if asked_non_tried:
            return (
                asked_non_tried[:min(len(asked_non_tried), n_points)],
                is_random_non_tried[:min(len(asked_non_tried), n_points)]
            )
        else:
            return self.opt.ask(n_points=n_points), [False for _ in range(n_points)]

    def evaluate_result(self, val: Dict[str, Any], current: int, is_random: bool):
        """
        Evaluate results returned from generate_optimizer
        """
        val['current_epoch'] = current
        val['is_initial_point'] = current <= INITIAL_POINTS

        logger.debug("Optimizer epoch evaluated: %s", val)

        is_best = HyperoptTools.is_best_loss(val, self.current_best_loss)
        # This value is assigned here and not in the optimization method
        # to keep proper order in the list of results. That's because
        # evaluations can take different time. Here they are aligned in the
        # order they will be shown to the user.
        val['is_best'] = is_best
        val['is_random'] = is_random
        self.print_results(val)

        if is_best:
            self.current_best_loss = val['loss']
            self.current_best_epoch = val

        self._save_result(val)

    def start(self) -> None:
        self.random_state = self._set_random_state(self.config.get('hyperopt_random_state'))
        logger.info(f"Using optimizer random state: {self.random_state}")
        self.hyperopt_table_header = -1
        # Initialize spaces ...
        self.init_spaces()

        self.prepare_hyperopt_data()

        # We don't need exchange instance anymore while running hyperopt
        self.backtesting.exchange.close()
        self.backtesting.exchange._api = None
        self.backtesting.exchange._api_async = None
        self.backtesting.exchange.loop = None  # type: ignore
        self.backtesting.exchange._loop_lock = None  # type: ignore
        self.backtesting.exchange._cache_lock = None  # type: ignore
        # self.backtesting.exchange = None  # type: ignore
        self.backtesting.pairlists = None  # type: ignore

        cpus = cpu_count()
        logger.info(f"Found {cpus} CPU cores. Let's make them scream!")
        config_jobs = self.config.get('hyperopt_jobs', -1)
        logger.info(f'Number of parallel jobs set as: {config_jobs}')

        self.opt = self.get_optimizer(self.dimensions, config_jobs)

        if self.print_colorized:
            colorama_init(autoreset=True)

        try:
            with Parallel(n_jobs=config_jobs) as parallel:
                jobs = parallel._effective_n_jobs()
                logger.info(f'Effective number of parallel workers used: {jobs}')

                # Define progressbar
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=None),
                    MofNCompleteColumn(),
                    TaskProgressColumn(),
                    "•",
                    TimeElapsedColumn(),
                    "•",
                    TimeRemainingColumn(),
                    expand=True,
                ) as pbar:
                    task = pbar.add_task("Epochs", total=self.total_epochs)

                    start = 0

                    if self.analyze_per_epoch:
                        # First analysis not in parallel mode when using --analyze-per-epoch.
                        # This allows dataprovider to load it's informative cache.
                        asked, is_random = self.get_asked_points(n_points=1)
                        f_val0 = self.generate_optimizer(asked[0])
                        self.opt.tell(asked, [f_val0['loss']])
                        self.evaluate_result(f_val0, 1, is_random[0])
                        pbar.update(task, advance=1)
                        start += 1

                    evals = ceil((self.total_epochs - start) / jobs)
                    for i in range(evals):
                        # Correct the number of epochs to be processed for the last
                        # iteration (should not exceed self.total_epochs in total)
                        n_rest = (i + 1) * jobs - (self.total_epochs - start)
                        current_jobs = jobs - n_rest if n_rest > 0 else jobs

                        asked, is_random = self.get_asked_points(n_points=current_jobs)
                        f_val = self.run_optimizer_parallel(parallel, asked)
                        self.opt.tell(asked, [v['loss'] for v in f_val])

                        for j, val in enumerate(f_val):
                            # Use human-friendly indexes here (starting from 1)
                            current = i * jobs + j + 1 + start

                            self.evaluate_result(val, current, is_random[j])
                            pbar.update(task, advance=1)

        except KeyboardInterrupt:
            print('User interrupted..')

        logger.info(f"{self.num_epochs_saved} {plural(self.num_epochs_saved, 'epoch')} "
                    f"saved to '{self.results_file}'.")

        if self.current_best_epoch:
            HyperoptTools.try_export_params(
                self.config,
                self.backtesting.strategy.get_strategy_name(),
                self.current_best_epoch)

            HyperoptTools.show_epoch_details(self.current_best_epoch, self.total_epochs,
                                             self.print_json)
        else:
            # This is printed when Ctrl+C is pressed quickly, before first epochs have
            # a chance to be evaluated.
            print("No epochs evaluated yet, no best result.")
