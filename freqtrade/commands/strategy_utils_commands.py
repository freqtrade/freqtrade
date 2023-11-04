import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy.strategyupdater import StrategyUpdater


logger = logging.getLogger(__name__)


def start_strategy_update(args: Dict[str, Any]) -> None:
    """
    Start the strategy updating script
    :param args: Cli args from Arguments()
    :return: None
    """

    if sys.version_info == (3, 8):  # pragma: no cover
        sys.exit("Freqtrade strategy updater requires Python version >= 3.9")

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    strategy_objs = StrategyResolver.search_all_objects(
        config, enum_failed=False, recursive=config.get('recursive_strategy_search', False))

    filtered_strategy_objs = []
    if args['strategy_list']:
        filtered_strategy_objs = [
            strategy_obj for strategy_obj in strategy_objs
            if strategy_obj['name'] in args['strategy_list']
        ]

    else:
        # Use all available entries.
        filtered_strategy_objs = strategy_objs

    processed_locations = set()
    for strategy_obj in filtered_strategy_objs:
        if strategy_obj['location'] not in processed_locations:
            processed_locations.add(strategy_obj['location'])
            start_conversion(strategy_obj, config)


def start_conversion(strategy_obj, config):
    print(f"Conversion of {Path(strategy_obj['location']).name} started.")
    instance_strategy_updater = StrategyUpdater()
    start = time.perf_counter()
    instance_strategy_updater.start(config, strategy_obj)
    elapsed = time.perf_counter() - start
    print(f"Conversion of {Path(strategy_obj['location']).name} took {elapsed:.1f} seconds.")
