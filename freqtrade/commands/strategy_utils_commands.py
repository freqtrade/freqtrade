import logging
from typing import Any, Dict

from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.resolvers import StrategyResolver


logger = logging.getLogger(__name__)


def start_strategy_update(args: Dict[str, Any]) -> None:
    """
    Start the strategy updating script
    :param args: Cli args from Arguments()
    :return: None
    """

    # Import here to avoid loading backtesting module when it's not used
    from freqtrade.strategy.strategy_updater import strategy_updater

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    strategy_objs = StrategyResolver.search_all_objects(
        config, enum_failed=True, recursive=config.get('recursive_strategy_search', False))

    filtered_strategy_objs = []
    for args_strategy in args['strategy_list']:
        for strategy_obj in strategy_objs:
            if strategy_obj['name'] == args_strategy and strategy_obj not in filtered_strategy_objs:
                filtered_strategy_objs.append(strategy_obj)
                break

    for filtered_strategy_obj in filtered_strategy_objs:
        # Initialize backtesting object
        instance_strategy_updater = strategy_updater()
        strategy_updater.start(instance_strategy_updater, filtered_strategy_obj)
