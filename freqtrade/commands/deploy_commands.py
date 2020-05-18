import logging
import sys
from pathlib import Path
from typing import Any, Dict

from freqtrade.configuration import setup_utils_configuration
from freqtrade.configuration.directory_operations import (copy_sample_files,
                                                          create_userdata_dir)
from freqtrade.constants import USERPATH_HYPEROPTS, USERPATH_STRATEGIES
from freqtrade.exceptions import OperationalException
from freqtrade.misc import render_template, render_template_with_fallback
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def start_create_userdir(args: Dict[str, Any]) -> None:
    """
    Create "user_data" directory to contain user data strategies, hyperopt, ...)
    :param args: Cli args from Arguments()
    :return: None
    """
    if "user_data_dir" in args and args["user_data_dir"]:
        userdir = create_userdata_dir(args["user_data_dir"], create_dir=True)
        copy_sample_files(userdir, overwrite=args["reset"])
    else:
        logger.warning("`create-userdir` requires --userdir to be set.")
        sys.exit(1)


def deploy_new_strategy(strategy_name: str, strategy_path: Path, subtemplate: str) -> None:
    """
    Deploy new strategy from template to strategy_path
    """
    fallback = 'full'
    indicators = render_template_with_fallback(
        templatefile=f"subtemplates/indicators_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/indicators_{fallback}.j2",
        )
    buy_trend = render_template_with_fallback(
        templatefile=f"subtemplates/buy_trend_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/buy_trend_{fallback}.j2",
        )
    sell_trend = render_template_with_fallback(
        templatefile=f"subtemplates/sell_trend_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/sell_trend_{fallback}.j2",
        )
    plot_config = render_template_with_fallback(
        templatefile=f"subtemplates/plot_config_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/plot_config_{fallback}.j2",
    )
    additional_methods = render_template_with_fallback(
        templatefile=f"subtemplates/strategy_methods_{subtemplate}.j2",
        templatefallbackfile="subtemplates/strategy_methods_empty.j2",
    )

    strategy_text = render_template(templatefile='base_strategy.py.j2',
                                    arguments={"strategy": strategy_name,
                                               "indicators": indicators,
                                               "buy_trend": buy_trend,
                                               "sell_trend": sell_trend,
                                               "plot_config": plot_config,
                                               "additional_methods": additional_methods,
                                               })

    logger.info(f"Writing strategy to `{strategy_path}`.")
    strategy_path.write_text(strategy_text)


def start_new_strategy(args: Dict[str, Any]) -> None:

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if "strategy" in args and args["strategy"]:
        if args["strategy"] == "DefaultStrategy":
            raise OperationalException("DefaultStrategy is not allowed as name.")

        new_path = config['user_data_dir'] / USERPATH_STRATEGIES / (args["strategy"] + ".py")

        if new_path.exists():
            raise OperationalException(f"`{new_path}` already exists. "
                                       "Please choose another Strategy Name.")

        deploy_new_strategy(args['strategy'], new_path, args['template'])

    else:
        raise OperationalException("`new-strategy` requires --strategy to be set.")


def deploy_new_hyperopt(hyperopt_name: str, hyperopt_path: Path, subtemplate: str) -> None:
    """
    Deploys a new hyperopt template to hyperopt_path
    """
    fallback = 'full'
    buy_guards = render_template_with_fallback(
        templatefile=f"subtemplates/hyperopt_buy_guards_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/hyperopt_buy_guards_{fallback}.j2",
        )
    sell_guards = render_template_with_fallback(
        templatefile=f"subtemplates/hyperopt_sell_guards_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/hyperopt_sell_guards_{fallback}.j2",
        )
    buy_space = render_template_with_fallback(
        templatefile=f"subtemplates/hyperopt_buy_space_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/hyperopt_buy_space_{fallback}.j2",
        )
    sell_space = render_template_with_fallback(
        templatefile=f"subtemplates/hyperopt_sell_space_{subtemplate}.j2",
        templatefallbackfile=f"subtemplates/hyperopt_sell_space_{fallback}.j2",
        )

    strategy_text = render_template(templatefile='base_hyperopt.py.j2',
                                    arguments={"hyperopt": hyperopt_name,
                                               "buy_guards": buy_guards,
                                               "sell_guards": sell_guards,
                                               "buy_space": buy_space,
                                               "sell_space": sell_space,
                                               })

    logger.info(f"Writing hyperopt to `{hyperopt_path}`.")
    hyperopt_path.write_text(strategy_text)


def start_new_hyperopt(args: Dict[str, Any]) -> None:

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if "hyperopt" in args and args["hyperopt"]:
        if args["hyperopt"] == "DefaultHyperopt":
            raise OperationalException("DefaultHyperopt is not allowed as name.")

        new_path = config['user_data_dir'] / USERPATH_HYPEROPTS / (args["hyperopt"] + ".py")

        if new_path.exists():
            raise OperationalException(f"`{new_path}` already exists. "
                                       "Please choose another Strategy Name.")
        deploy_new_hyperopt(args['hyperopt'], new_path, args['template'])
    else:
        raise OperationalException("`new-hyperopt` requires --hyperopt to be set.")
