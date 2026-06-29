import logging
import time
from pathlib import Path
from typing import Any

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.analysis.recursive import RecursiveAnalysis
from freqtrade.resolvers import StrategyResolver
from freqtrade.util import CustomProgress, get_progress_tracker, print_rich_table


logger = logging.getLogger(__name__)


class RecursiveAnalysisSubFunctions:
    @staticmethod
    def text_table_recursive_analysis_instances(recursive_instances: list[RecursiveAnalysis]):
        startups = recursive_instances[0]._startup_candle
        strat_scc = getattr(recursive_instances[0], "_strat_scc", 0) or 0
        headers = ["Indicators"]
        for candle in startups:
            if candle == strat_scc:
                headers.append(f"{candle} (from strategy)")
            else:
                headers.append(str(candle))

        data = []
        for inst in recursive_instances:
            if len(inst.dict_recursive) > 0:
                for indicator, values in inst.dict_recursive.items():
                    temp_data = [indicator]
                    for candle in startups:
                        if (value := values.get(candle)) is not None:
                            temp_data.append(f"{value:.3%}")
                        else:
                            temp_data.append("-")
                    data.append(temp_data)

        if len(data) > 0:
            print_rich_table(data, headers, summary="Recursive Analysis")

            return data

        return data

    @staticmethod
    def calculate_config_overrides(config: Config):
        if "timerange" not in config:
            # setting a timerange is enforced here
            raise OperationalException(
                "Please set a timerange. "
                "A timerange of 5000 candles are enough for recursive analysis."
            )

        if config.get("backtest_cache") is None:
            config["backtest_cache"] = "none"
        elif config["backtest_cache"] != "none":
            logger.info(
                f"backtest_cache = "
                f"{config['backtest_cache']} detected. "
                f"Inside recursive-analysis it is enforced to be 'none'. "
                f"Changed it to 'none'"
            )
            config["backtest_cache"] = "none"
        return config

    @staticmethod
    def initialize_single_recursive_analysis(
        config: Config, strategy_obj: dict[str, Any], progress: CustomProgress
    ):
        logger.info(f"Recursive test of {Path(strategy_obj['location']).name} started.")
        start = time.perf_counter()
        current_instance = RecursiveAnalysis(config, strategy_obj)
        current_instance.start(progress)
        elapsed = time.perf_counter() - start
        logger.info(
            f"Checking recursive and indicator-only lookahead bias of indicators "
            f"of {Path(strategy_obj['location']).name} "
            f"took {elapsed:.0f} seconds."
        )
        return current_instance

    @staticmethod
    def start(config: Config):
        config = RecursiveAnalysisSubFunctions.calculate_config_overrides(config)

        strategy_objs = StrategyResolver.search_all_objects(
            config, enum_failed=False, recursive=config.get("recursive_strategy_search", False)
        )

        RecursiveAnalysis_instances = []

        # unify --strategy and --strategy-list to one list
        if not (strategy_list := config.get("strategy_list", [])):
            if config.get("strategy") is None:
                raise OperationalException(
                    "No Strategy specified. Please specify a strategy via --strategy"
                )
            strategy_list = [config["strategy"]]

        # check if strategies can be properly loaded, only check them if they can be.
        with get_progress_tracker() as progress:
            strategy_task = (
                progress.add_task("Recursive analysis", total=len(strategy_list))
                if len(strategy_list) > 1
                else None
            )
            for strat in strategy_list:
                for strategy_obj in strategy_objs:
                    if strategy_obj["name"] == strat and strategy_obj not in strategy_list:
                        if strategy_task is not None:
                            progress.update(strategy_task, description=f"Analyzing {strat}")
                        RecursiveAnalysis_instances.append(
                            RecursiveAnalysisSubFunctions.initialize_single_recursive_analysis(
                                config, strategy_obj, progress
                            )
                        )
                        break
                if strategy_task is not None:
                    progress.update(strategy_task, advance=1)

        # report the results
        if RecursiveAnalysis_instances:
            RecursiveAnalysisSubFunctions.text_table_recursive_analysis_instances(
                RecursiveAnalysis_instances
            )
        else:
            logger.error(
                "There was no strategy specified through --strategy or timeframe was not specified."
            )
