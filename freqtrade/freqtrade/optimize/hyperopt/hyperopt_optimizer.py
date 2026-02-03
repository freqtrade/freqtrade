"""
This module contains the hyperopt optimizer class, which needs to be pickled
and will be sent to the hyperopt worker processes.
"""

import logging
import sys
import warnings
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from typing import Any

import optuna
from joblib import delayed, dump, load, wrap_non_picklable_objects
from joblib.externals import cloudpickle
from optuna.exceptions import ExperimentalWarning
from optuna.terminator import BestValueStagnationEvaluator, Terminator
from pandas import DataFrame

from freqtrade.constants import DATETIME_PRINT_FORMAT, Config
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.history import get_timerange
from freqtrade.data.metrics import calculate_market_change
from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.ft_types import BacktestContentType
from freqtrade.misc import deep_merge_dicts, round_dict
from freqtrade.optimize.backtesting import Backtesting

# Import IHyperOptLoss to allow unpickling classes from these modules
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt.hyperopt_logger import logging_mp_handle, logging_mp_setup
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer, HyperoptTools
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.optimize.space import (
    DimensionProtocol,
    SKDecimal,
    ft_CategoricalDistribution,
    ft_FloatDistribution,
    ft_IntDistribution,
)
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver
from freqtrade.util import dt_now
from freqtrade.util.dry_run_wallet import get_dry_run_wallet


logger = logging.getLogger(__name__)

INITIAL_POINTS = 30

MAX_LOSS = 100000  # just a big enough number to be bad result in loss optimization

optuna_samplers_dict = {
    "TPESampler": optuna.samplers.TPESampler,
    "GPSampler": optuna.samplers.GPSampler,
    "CmaEsSampler": optuna.samplers.CmaEsSampler,
    "NSGAIISampler": optuna.samplers.NSGAIISampler,
    "NSGAIIISampler": optuna.samplers.NSGAIIISampler,
    "QMCSampler": optuna.samplers.QMCSampler,
}

log_queue: Any


class HyperOptimizer:
    """
    HyperoptOptimizer class
    This class is sent to the hyperopt worker processes.
    """

    def __init__(self, config: Config, data_pickle_file: Path) -> None:
        self.spaces: dict[str, list[DimensionProtocol]] = {}
        self.dimensions: list[DimensionProtocol] = []
        self.o_dimensions: dict = {}

        self.config = config
        self.min_date: datetime
        self.max_date: datetime

        self.backtesting = Backtesting(self.config)
        self.pairlist = self.backtesting.pairlists.whitelist
        self.custom_hyperopt: HyperOptAuto
        self.analyze_per_epoch = self.config.get("analyze_per_epoch", False)

        self.custom_hyperopt = HyperOptAuto(self.config)

        self.backtesting._set_strategy(self.backtesting.strategylist[0])
        self.custom_hyperopt.strategy = self.backtesting.strategy

        self.hyperopt_pickle_magic(self.backtesting.strategy.__class__.__bases__)
        self.custom_hyperoptloss: IHyperOptLoss = HyperOptLossResolver.load_hyperoptloss(
            self.config
        )
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function

        self.data_pickle_file = data_pickle_file

        self.market_change = 0.0

        self.es_epochs = config.get("early_stop", 0)
        if self.es_epochs > 0 and self.es_epochs < 0.2 * config.get("epochs", 0):
            logger.warning(f"Early stop epochs {self.es_epochs} lower than 20% of total epochs")

        if HyperoptTools.has_space(self.config, "sell"):
            # Make sure use_exit_signal is enabled
            self.config["use_exit_signal"] = True
        self._setup_logging_mp_workaround()

    def _setup_logging_mp_workaround(self) -> None:
        """
        Workaround for logging in child processes.
        local_queue must be a global and passed to the child process via inheritance.
        """
        global log_queue
        m = Manager()
        log_queue = m.Queue()
        logger.info(f"manager queue {type(log_queue)}")

    def handle_mp_logging(self) -> None:
        """
        Handle logging from child processes.
        Must be called in the parent process to handle log messages from the child process.
        """
        logging_mp_handle(log_queue)

    def prepare_hyperopt(self) -> None:
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

    def get_strategy_name(self) -> str:
        return self.backtesting.strategy.get_strategy_name()

    def hyperopt_pickle_magic(self, bases: tuple[type, ...]) -> None:
        """
        Hyperopt magic to allow strategy inheritance across files.
        For this to properly work, we need to register the module of the imported class
        to pickle as value.
        """
        for modules in bases:
            if modules.__name__ != "IStrategy":
                if mod := sys.modules.get(modules.__module__):
                    cloudpickle.register_pickle_by_value(mod)
                self.hyperopt_pickle_magic(modules.__bases__)

    def _get_params_details(self, params: dict) -> dict:
        """
        Return the params for each space
        """
        result: dict = {}

        for space in self.spaces.keys():
            if space == "protection":
                result["protection"] = round_dict(
                    {p.name: params.get(p.name) for p in self.spaces[space]}, 13
                )
            elif space == "roi":
                result["roi"] = round_dict(
                    {str(k): v for k, v in self.custom_hyperopt.generate_roi_table(params).items()},
                    13,
                )
            elif space == "stoploss":
                result["stoploss"] = round_dict(
                    {p.name: params.get(p.name) for p in self.spaces[space]}, 13
                )
            elif space == "trailing":
                result["trailing"] = round_dict(
                    self.custom_hyperopt.generate_trailing_params(params), 13
                )
            elif space == "trades":
                result["max_open_trades"] = round_dict(
                    {
                        "max_open_trades": (
                            self.backtesting.strategy.max_open_trades
                            if self.backtesting.strategy.max_open_trades != float("inf")
                            else -1
                        )
                    },
                    13,
                )
            else:
                result[space] = round_dict(
                    {p.name: params.get(p.name) for p in self.spaces[space]}, 13
                )

        return result

    def _get_no_optimize_details(self) -> dict[str, Any]:
        """
        Get non-optimized parameters
        """
        result: dict[str, Any] = {}
        strategy = self.backtesting.strategy
        if not HyperoptTools.has_space(self.config, "roi"):
            result["roi"] = {str(k): v for k, v in strategy.minimal_roi.items()}
        if not HyperoptTools.has_space(self.config, "stoploss"):
            result["stoploss"] = {"stoploss": strategy.stoploss}
        if not HyperoptTools.has_space(self.config, "trailing"):
            result["trailing"] = {
                "trailing_stop": strategy.trailing_stop,
                "trailing_stop_positive": strategy.trailing_stop_positive,
                "trailing_stop_positive_offset": strategy.trailing_stop_positive_offset,
                "trailing_only_offset_is_reached": strategy.trailing_only_offset_is_reached,
            }
        if not HyperoptTools.has_space(self.config, "trades"):
            result["max_open_trades"] = {"max_open_trades": strategy.max_open_trades}
        return result

    def init_spaces(self):
        """
        Assign the dimensions in the hyperoptimization space.
        """
        spaces = ["buy", "sell", "protection", "roi", "stoploss", "trailing", "trades"]
        spaces += [s for s in self.custom_hyperopt.get_available_spaces() if s not in spaces]

        for space in spaces:
            if not HyperoptTools.has_space(self.config, space):
                continue
            logger.debug(f"Hyperopt has '{space}' space")
            if space == "protection":
                # Protections can only be optimized when using the Parameter interface
                # Enable Protections if protection space is selected.
                self.config["enable_protections"] = True
                self.backtesting.enable_protections = True
                self.spaces[space] = self.custom_hyperopt.get_indicator_space(space)
            elif space == "roi":
                self.spaces[space] = self.custom_hyperopt.roi_space()
            elif space == "stoploss":
                self.spaces[space] = self.custom_hyperopt.stoploss_space()
            elif space == "trailing":
                self.spaces[space] = self.custom_hyperopt.trailing_space()
            elif space == "trades":
                self.spaces[space] = self.custom_hyperopt.max_open_trades_space()
            else:
                self.spaces[space] = self.custom_hyperopt.get_indicator_space(space)

        self.dimensions = [s for space in self.spaces.values() for s in space]
        if len(self.dimensions) == 0:
            raise OperationalException(
                "No hyperopt parameters found to optimize."
                f"Available spaces: {', '.join(spaces)}. "
                "Check your strategy's parameter definitions or verify the configured spaces "
                "in your command."
            )
        self.o_dimensions = self.convert_dimensions_to_optuna_space(self.dimensions)

    @delayed
    @wrap_non_picklable_objects
    def generate_optimizer_wrapped(self, params_dict: dict[str, Any]) -> dict[str, Any]:
        logging_mp_setup(log_queue, logging.INFO if self.config["verbosity"] < 1 else logging.DEBUG)
        return self.generate_optimizer(params_dict)

    def generate_optimizer(self, params_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Used Optimize function.
        Called once per epoch to optimize whatever is configured.
        Keep this function as optimized as possible!
        """
        HyperoptStateContainer.set_state(HyperoptState.OPTIMIZE)
        backtest_start_time = dt_now()

        for attr_name, attr in self.backtesting.strategy.enumerate_parameters():
            if attr.in_space and attr.optimize:
                attr.value = params_dict[attr_name]

        if HyperoptTools.has_space(self.config, "roi"):
            self.backtesting.strategy.minimal_roi = self.custom_hyperopt.generate_roi_table(
                params_dict
            )

        if HyperoptTools.has_space(self.config, "stoploss"):
            self.backtesting.strategy.stoploss = params_dict["stoploss"]

        if HyperoptTools.has_space(self.config, "trailing"):
            d = self.custom_hyperopt.generate_trailing_params(params_dict)
            self.backtesting.strategy.trailing_stop = d["trailing_stop"]
            self.backtesting.strategy.trailing_stop_positive = d["trailing_stop_positive"]
            self.backtesting.strategy.trailing_stop_positive_offset = d[
                "trailing_stop_positive_offset"
            ]
            self.backtesting.strategy.trailing_only_offset_is_reached = d[
                "trailing_only_offset_is_reached"
            ]

        if HyperoptTools.has_space(self.config, "trades"):
            if self.config["stake_amount"] == "unlimited" and (
                params_dict["max_open_trades"] == -1 or params_dict["max_open_trades"] == 0
            ):
                # Ignore unlimited max open trades if stake amount is unlimited
                params_dict.update({"max_open_trades": self.config["max_open_trades"]})

            updated_max_open_trades = (
                int(params_dict["max_open_trades"])
                if (params_dict["max_open_trades"] != -1 and params_dict["max_open_trades"] != 0)
                else float("inf")
            )

            self.config.update({"max_open_trades": updated_max_open_trades})

            self.backtesting.strategy.max_open_trades = updated_max_open_trades

        with self.data_pickle_file.open("rb") as f:
            processed = load(f, mmap_mode="r")
        if self.analyze_per_epoch:
            # Data is not yet analyzed, rerun populate_indicators.
            processed = self.advise_and_trim(processed)

        bt_results = self.backtesting.backtest(
            processed=processed, start_date=self.min_date, end_date=self.max_date
        )
        backtest_end_time = dt_now()
        bt_results.update(
            {
                "backtest_start_time": int(backtest_start_time.timestamp()),
                "backtest_end_time": int(backtest_end_time.timestamp()),
            }
        )
        result = self._get_results_dict(
            bt_results, self.min_date, self.max_date, params_dict, processed=processed
        )
        return result

    def _get_results_dict(
        self,
        backtesting_results: BacktestContentType,
        min_date: datetime,
        max_date: datetime,
        params_dict: dict[str, Any],
        processed: dict[str, DataFrame],
    ) -> dict[str, Any]:
        params_details = self._get_params_details(params_dict)

        strat_stats = generate_strategy_stats(
            self.pairlist,
            self.backtesting.strategy.get_strategy_name(),
            backtesting_results,
            min_date,
            max_date,
            market_change=self.market_change,
            is_hyperopt=True,
        )
        results_explanation = HyperoptTools.format_results_explanation_string(
            strat_stats, self.config["stake_currency"]
        )

        not_optimized = self.backtesting.strategy.get_no_optimize_params()
        not_optimized = deep_merge_dicts(not_optimized, self._get_no_optimize_details())

        trade_count = strat_stats["total_trades"]
        total_profit = strat_stats["profit_total"]

        # If this evaluation contains too short amount of trades to be
        # interesting -- consider it as 'bad' (assigned max. loss value)
        # in order to cast this hyperspace point away from optimization
        # path. We do not want to optimize 'hodl' strategies.
        loss: float = MAX_LOSS
        if trade_count >= self.config["hyperopt_min_trades"]:
            loss = self.calculate_loss(
                results=backtesting_results["results"],
                trade_count=trade_count,
                min_date=min_date,
                max_date=max_date,
                config=self.config,
                processed=processed,
                backtest_stats=strat_stats,
                starting_balance=get_dry_run_wallet(self.config),
            )
        return {
            "loss": loss,
            "params_dict": params_dict,
            "params_details": params_details,
            "params_not_optimized": not_optimized,
            "results_metrics": strat_stats,
            "results_explanation": results_explanation,
            "total_profit": total_profit,
        }

    def convert_dimensions_to_optuna_space(self, s_dimensions: list[DimensionProtocol]) -> dict:
        o_dimensions: dict[str, optuna.distributions.BaseDistribution] = {}
        for original_dim in s_dimensions:
            if isinstance(
                original_dim,
                ft_CategoricalDistribution | ft_IntDistribution | ft_FloatDistribution | SKDecimal,
            ):
                o_dimensions[original_dim.name] = original_dim
            else:
                raise OperationalException(
                    f"Unknown search space {original_dim.name} - {original_dim} / \
                        {type(original_dim)}"
                )
        return o_dimensions

    def get_optimizer(
        self,
        random_state: int,
    ):
        o_sampler = self.custom_hyperopt.generate_estimator(
            dimensions=self.dimensions, random_state=random_state
        )

        if isinstance(o_sampler, str):
            if o_sampler not in optuna_samplers_dict.keys():
                raise OperationalException(f"Optuna Sampler {o_sampler} not supported.")
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=ExperimentalWarning)
                if o_sampler in ["NSGAIIISampler", "NSGAIISampler"]:
                    sampler = optuna_samplers_dict[o_sampler](
                        seed=random_state, population_size=INITIAL_POINTS
                    )
                elif o_sampler in ["GPSampler", "TPESampler", "CmaEsSampler"]:
                    sampler = optuna_samplers_dict[o_sampler](
                        seed=random_state, n_startup_trials=INITIAL_POINTS
                    )
                else:
                    sampler = optuna_samplers_dict[o_sampler](seed=random_state)
        else:
            sampler = o_sampler

        if self.es_epochs > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=ExperimentalWarning)
                self.es_terminator = Terminator(BestValueStagnationEvaluator(self.es_epochs))

        logger.info(f"Using optuna sampler {o_sampler}.")
        return optuna.create_study(sampler=sampler, direction="minimize")

    def advise_and_trim(self, data: dict[str, DataFrame]) -> dict[str, DataFrame]:
        preprocessed = self.backtesting.strategy.advise_all_indicators(data)

        # Trim startup period from analyzed dataframe to get correct dates for output.
        # This is only used to keep track of min/max date after trimming.
        # The result is NOT returned from this method, actual trimming happens in backtesting.
        trimmed = trim_dataframes(preprocessed, self.timerange, self.backtesting.required_startup)
        self.min_date, self.max_date = get_timerange(trimmed)
        if not self.market_change:
            self.market_change = calculate_market_change(trimmed, "close")

        # Real trimming will happen as part of backtesting.
        return preprocessed

    def prepare_hyperopt_data(self) -> None:
        HyperoptStateContainer.set_state(HyperoptState.DATALOAD)
        data, self.timerange = self.backtesting.load_bt_data()
        logger.info("Dataload complete. Calculating indicators")

        if not self.analyze_per_epoch:
            HyperoptStateContainer.set_state(HyperoptState.INDICATORS)

            preprocessed = self.advise_and_trim(data)

            logger.info(
                f"Hyperopting with data from "
                f"{self.min_date.strftime(DATETIME_PRINT_FORMAT)} "
                f"up to {self.max_date.strftime(DATETIME_PRINT_FORMAT)} "
                f"({(self.max_date - self.min_date).days} days).."
            )
            # Store non-trimmed data - will be trimmed after signal generation.
            dump(preprocessed, self.data_pickle_file)
        else:
            dump(data, self.data_pickle_file)
