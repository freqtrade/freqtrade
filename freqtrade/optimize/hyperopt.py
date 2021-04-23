# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

"""
This module contains the hyperopt logic
"""

import locale
import logging
import random
import warnings
from datetime import datetime
from math import ceil
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Optional

import progressbar
from colorama import Fore, Style
from colorama import init as colorama_init
from joblib import Parallel, cpu_count, delayed, dump, load, wrap_non_picklable_objects
from pandas import DataFrame

from freqtrade.constants import DATETIME_PRINT_FORMAT, LAST_BT_RESULT_FN
from freqtrade.data.converter import trim_dataframe
from freqtrade.data.history import get_timerange
from freqtrade.misc import file_dump_json, plural
from freqtrade.optimize.backtesting import Backtesting
# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
from freqtrade.optimize.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from freqtrade.optimize.hyperopt_tools import HyperoptTools
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver, HyperOptResolver
from freqtrade.strategy import IStrategy


# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
    from skopt.space import Dimension

progressbar.streams.wrap_stderr()
progressbar.streams.wrap_stdout()
logger = logging.getLogger(__name__)


INITIAL_POINTS = 30

# Keep no more than SKOPT_MODEL_QUEUE_SIZE models
# in the skopt model queue, to optimize memory consumption
SKOPT_MODEL_QUEUE_SIZE = 10

MAX_LOSS = 100000  # just a big enough number to be bad result in loss optimization


class Hyperopt:
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To run a backtest:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """
    custom_hyperopt: IHyperOpt

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        self.backtesting = Backtesting(self.config)

        if not self.config.get('hyperopt'):
            self.custom_hyperopt = HyperOptAuto(self.config)
        else:
            self.custom_hyperopt = HyperOptResolver.load_hyperopt(self.config)
        self.custom_hyperopt.strategy = self.backtesting.strategy

        self.custom_hyperoptloss = HyperOptLossResolver.load_hyperoptloss(self.config)
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        strategy = str(self.config['strategy'])
        self.results_file = (self.config['user_data_dir'] /
                             'hyperopt_results' /
                             f'strategy_{strategy}_hyperopt_results_{time_now}.pickle')
        self.data_pickle_file = (self.config['user_data_dir'] /
                                 'hyperopt_results' / 'hyperopt_tickerdata.pkl')
        self.total_epochs = config.get('epochs', 0)

        self.current_best_loss = 100

        self.clean_hyperopt()

        self.num_epochs_saved = 0

        # Previous evaluations
        self.epochs: List = []

        # Populate functions here (hasattr is slow so should not be run during "regular" operations)
        if hasattr(self.custom_hyperopt, 'populate_indicators'):
            self.backtesting.strategy.advise_indicators = (  # type: ignore
                self.custom_hyperopt.populate_indicators)  # type: ignore
        if hasattr(self.custom_hyperopt, 'populate_buy_trend'):
            self.backtesting.strategy.advise_buy = (  # type: ignore
                self.custom_hyperopt.populate_buy_trend)  # type: ignore
        if hasattr(self.custom_hyperopt, 'populate_sell_trend'):
            self.backtesting.strategy.advise_sell = (  # type: ignore
                self.custom_hyperopt.populate_sell_trend)  # type: ignore

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
        self.hyperopt_table_header = 0
        self.print_colorized = self.config.get('print_colorized', False)
        self.print_json = self.config.get('print_json', False)

    @staticmethod
    def get_lock_filename(config: Dict[str, Any]) -> str:

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

    def _get_params_dict(self, raw_params: List[Any]) -> Dict:

        dimensions: List[Dimension] = self.dimensions

        # Ensure the number of dimensions match
        # the number of parameters in the list.
        if len(raw_params) != len(dimensions):
            raise ValueError('Mismatch in number of search-space dimensions.')

        # Return a dict where the keys are the names of the dimensions
        # and the values are taken from the list of parameters.
        return {d.name: v for d, v in zip(dimensions, raw_params)}

    def _save_results(self) -> None:
        """
        Save hyperopt results to file
        """
        num_epochs = len(self.epochs)
        if num_epochs > self.num_epochs_saved:
            logger.debug(f"Saving {num_epochs} {plural(num_epochs, 'epoch')}.")
            dump(self.epochs, self.results_file)
            self.num_epochs_saved = num_epochs
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
            self.backtesting.strategy.minimal_roi = (  # type: ignore
                self.custom_hyperopt.generate_roi_table(params_dict))

        if self.has_space('buy'):
            self.backtesting.strategy.advise_buy = (  # type: ignore
                self.custom_hyperopt.buy_strategy_generator(params_dict))

        if self.has_space('sell'):
            self.backtesting.strategy.advise_sell = (  # type: ignore
                self.custom_hyperopt.sell_strategy_generator(params_dict))

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

        processed = load(self.data_pickle_file)

        min_date, max_date = get_timerange(processed)

        backtesting_results = self.backtesting.backtest(
            processed=processed,
            start_date=min_date.datetime,
            end_date=max_date.datetime,
            max_open_trades=self.max_open_trades,
            position_stacking=self.position_stacking,
            enable_protections=self.config.get('enable_protections', False),

        )
        return self._get_results_dict(backtesting_results, min_date, max_date,
                                      params_dict, params_details,
                                      processed=processed)

    def _get_results_dict(self, backtesting_results, min_date, max_date,
                          params_dict, params_details, processed: Dict[str, DataFrame]):
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
            loss = self.calculate_loss(results=backtesting_results, trade_count=trade_count,
                                       min_date=min_date.datetime, max_date=max_date.datetime,
                                       config=self.config, processed=processed)
        return {
            'loss': loss,
            'params_dict': params_dict,
            'params_details': params_details,
            'results_metrics': results_metrics,
            'results_explanation': results_explanation,
            'total_profit': total_profit,
        }

    def _calculate_results_metrics(self, backtesting_results: DataFrame) -> Dict:
        wins = len(backtesting_results[backtesting_results['profit_ratio'] > 0])
        draws = len(backtesting_results[backtesting_results['profit_ratio'] == 0])
        losses = len(backtesting_results[backtesting_results['profit_ratio'] < 0])
        return {
            'trade_count': len(backtesting_results.index),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'winsdrawslosses': f"{wins:>4} {draws:>4} {losses:>4}",
            'avg_profit': backtesting_results['profit_ratio'].mean() * 100.0,
            'median_profit': backtesting_results['profit_ratio'].median() * 100.0,
            'total_profit': backtesting_results['profit_abs'].sum(),
            'profit': backtesting_results['profit_ratio'].sum() * 100.0,
            'duration': backtesting_results['trade_duration'].mean(),
        }

    def _format_results_explanation_string(self, results_metrics: Dict) -> str:
        """
        Return the formatted results explanation in a string
        """
        stake_cur = self.config['stake_currency']
        return (f"{results_metrics['trade_count']:6d} trades. "
                f"{results_metrics['wins']}/{results_metrics['draws']}"
                f"/{results_metrics['losses']} Wins/Draws/Losses. "
                f"Avg profit {results_metrics['avg_profit']: 6.2f}%. "
                f"Median profit {results_metrics['median_profit']: 6.2f}%. "
                f"Total profit {results_metrics['total_profit']: 11.8f} {stake_cur} "
                f"({results_metrics['profit']: 7.2f}\N{GREEK CAPITAL LETTER SIGMA}%). "
                f"Avg duration {results_metrics['duration']:5.1f} min."
                ).encode(locale.getpreferredencoding(), 'replace').decode('utf-8')

    def get_optimizer(self, dimensions: List[Dimension], cpu_count) -> Optimizer:
        return Optimizer(
            dimensions,
            base_estimator="ET",
            acq_optimizer="auto",
            n_initial_points=INITIAL_POINTS,
            acq_optimizer_kwargs={'n_jobs': cpu_count},
            random_state=self.random_state,
            model_queue_size=SKOPT_MODEL_QUEUE_SIZE,
        )

    def run_optimizer_parallel(self, parallel, asked, i) -> List:
        return parallel(delayed(
                        wrap_non_picklable_objects(self.generate_optimizer))(v, i) for v in asked)

    def _set_random_state(self, random_state: Optional[int]) -> int:
        return random_state or random.randint(1, 2**16 - 1)

    def start(self) -> None:
        self.random_state = self._set_random_state(self.config.get('hyperopt_random_state', None))
        logger.info(f"Using optimizer random state: {self.random_state}")
        self.hyperopt_table_header = -1
        data, timerange = self.backtesting.load_bt_data()
        logger.info("Dataload complete. Calculating indicators")
        preprocessed = self.backtesting.strategy.ohlcvdata_to_dataframe(data)

        # Trim startup period from analyzed dataframe
        for pair, df in preprocessed.items():
            preprocessed[pair] = trim_dataframe(df, timerange,
                                                startup_candles=self.backtesting.required_startup)
        min_date, max_date = get_timerange(preprocessed)

        logger.info(f'Hyperopting with data from {min_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'up to {max_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'({(max_date - min_date).days} days)..')

        dump(preprocessed, self.data_pickle_file)

        # We don't need exchange instance anymore while running hyperopt
        self.backtesting.exchange.close()
        self.backtesting.exchange._api = None  # type: ignore
        self.backtesting.exchange._api_async = None  # type: ignore
        # self.backtesting.exchange = None  # type: ignore
        self.backtesting.pairlists = None  # type: ignore
        self.backtesting.strategy.dp = None  # type: ignore
        IStrategy.dp = None  # type: ignore

        cpus = cpu_count()
        logger.info(f"Found {cpus} CPU cores. Let's make them scream!")
        config_jobs = self.config.get('hyperopt_jobs', -1)
        logger.info(f'Number of parallel jobs set as: {config_jobs}')

        self.dimensions: List[Dimension] = self.hyperopt_space()
        self.opt = self.get_optimizer(self.dimensions, config_jobs)

        if self.print_colorized:
            colorama_init(autoreset=True)

        try:
            with Parallel(n_jobs=config_jobs) as parallel:
                jobs = parallel._effective_n_jobs()
                logger.info(f'Effective number of parallel workers used: {jobs}')

                # Define progressbar
                if self.print_colorized:
                    widgets = [
                        ' [Epoch ', progressbar.Counter(), ' of ', str(self.total_epochs),
                        ' (', progressbar.Percentage(), ')] ',
                        progressbar.Bar(marker=progressbar.AnimatedMarker(
                            fill='\N{FULL BLOCK}',
                            fill_wrap=Fore.GREEN + '{}' + Fore.RESET,
                            marker_wrap=Style.BRIGHT + '{}' + Style.RESET_ALL,
                        )),
                        ' [', progressbar.ETA(), ', ', progressbar.Timer(), ']',
                    ]
                else:
                    widgets = [
                        ' [Epoch ', progressbar.Counter(), ' of ', str(self.total_epochs),
                        ' (', progressbar.Percentage(), ')] ',
                        progressbar.Bar(marker=progressbar.AnimatedMarker(
                            fill='\N{FULL BLOCK}',
                        )),
                        ' [', progressbar.ETA(), ', ', progressbar.Timer(), ']',
                    ]
                with progressbar.ProgressBar(
                         max_value=self.total_epochs, redirect_stdout=False, redirect_stderr=False,
                         widgets=widgets
                     ) as pbar:
                    EVALS = ceil(self.total_epochs / jobs)
                    for i in range(EVALS):
                        # Correct the number of epochs to be processed for the last
                        # iteration (should not exceed self.total_epochs in total)
                        n_rest = (i + 1) * jobs - self.total_epochs
                        current_jobs = jobs - n_rest if n_rest > 0 else jobs

                        asked = self.opt.ask(n_points=current_jobs)
                        f_val = self.run_optimizer_parallel(parallel, asked, i)
                        self.opt.tell(asked, [v['loss'] for v in f_val])

                        # Calculate progressbar outputs
                        for j, val in enumerate(f_val):
                            # Use human-friendly indexes here (starting from 1)
                            current = i * jobs + j + 1
                            val['current_epoch'] = current
                            val['is_initial_point'] = current <= INITIAL_POINTS

                            logger.debug(f"Optimizer epoch evaluated: {val}")

                            is_best = HyperoptTools.is_best_loss(val, self.current_best_loss)
                            # This value is assigned here and not in the optimization method
                            # to keep proper order in the list of results. That's because
                            # evaluations can take different time. Here they are aligned in the
                            # order they will be shown to the user.
                            val['is_best'] = is_best
                            self.print_results(val)

                            if is_best:
                                self.current_best_loss = val['loss']
                            self.epochs.append(val)

                            # Save results after each best epoch and every 100 epochs
                            if is_best or current % 100 == 0:
                                self._save_results()

                            pbar.update(current)

        except KeyboardInterrupt:
            print('User interrupted..')

        self._save_results()
        logger.info(f"{self.num_epochs_saved} {plural(self.num_epochs_saved, 'epoch')} "
                    f"saved to '{self.results_file}'.")

        if self.epochs:
            sorted_epochs = sorted(self.epochs, key=itemgetter('loss'))
            best_epoch = sorted_epochs[0]
            HyperoptTools.print_epoch_details(best_epoch, self.total_epochs, self.print_json)
        else:
            # This is printed when Ctrl+C is pressed quickly, before first epochs have
            # a chance to be evaluated.
            print("No epochs evaluated yet, no best result.")
