import logging
import sys
from pathlib import Path
from typing import Any, Dict

from freqtrade.configuration import setup_utils_configuration
from freqtrade.configuration.directory_operations import (copy_sample_files,
                                                          create_userdata_dir)
from freqtrade.constants import USERPATH_HYPEROPTS, USERPATH_STRATEGY
from freqtrade.exceptions import OperationalException
from freqtrade.misc import render_template
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


def deploy_new_strategy(strategy_name, strategy_path: Path, subtemplate: str):
    """
    Deploy new strategy from template to strategy_path
    """
    indicators = render_template(templatefile=f"subtemplates/indicators_{subtemplate}.j2",)
    buy_trend = render_template(templatefile=f"subtemplates/buy_trend_{subtemplate}.j2",)
    sell_trend = render_template(templatefile=f"subtemplates/sell_trend_{subtemplate}.j2",)
    plot_config = render_template(templatefile=f"subtemplates/plot_config_{subtemplate}.j2",)

    strategy_text = render_template(templatefile='base_strategy.py.j2',
                                    arguments={"strategy": strategy_name,
                                               "indicators": indicators,
                                               "buy_trend": buy_trend,
                                               "sell_trend": sell_trend,
                                               "plot_config": plot_config,
                                               })

    logger.info(f"Writing strategy to `{strategy_path}`.")
    strategy_path.write_text(strategy_text)


def start_new_strategy(args: Dict[str, Any]) -> None:

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if "strategy" in args and args["strategy"]:
        if args["strategy"] == "DefaultStrategy":
            raise OperationalException("DefaultStrategy is not allowed as name.")

        new_path = config['user_data_dir'] / USERPATH_STRATEGY / (args["strategy"] + ".py")

        if new_path.exists():
            raise OperationalException(f"`{new_path}` already exists. "
                                       "Please choose another Strategy Name.")

        deploy_new_strategy(args['strategy'], new_path, args['template'])

    else:
        raise OperationalException("`new-strategy` requires --strategy to be set.")


def deploy_new_hyperopt(hyperopt_name, hyperopt_path: Path, subtemplate: str):
    """
    Deploys a new hyperopt template to hyperopt_path
    """
    buy_guards = render_template(
        templatefile=f"subtemplates/hyperopt_buy_guards_{subtemplate}.j2",)
    sell_guards = render_template(
        templatefile=f"subtemplates/hyperopt_sell_guards_{subtemplate}.j2",)
    buy_space = render_template(
        templatefile=f"subtemplates/hyperopt_buy_space_{subtemplate}.j2",)
    sell_space = render_template(
        templatefile=f"subtemplates/hyperopt_sell_space_{subtemplate}.j2",)

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


def ask_user_config() -> Dict[str, Any]:
    """
    Ask user a few questions to build the configuration.
    :returns: Dict with keys to put into template
    """
    sample_selections = {
        'max_open_trades': 3,
        'stake_currency': 'USDT',
        'stake_amount': 100,
        'fiat_display_currency': 'EUR',
        'ticker_interval': '15m',
        'dry_run': True,
        'exchange_name': 'binance',
        'exchange_key': 'sampleKey',
        'exchange_secret': 'Samplesecret',
        'telegram': False,
        'telegram_token': 'asdf1244',
        'telegram_chat_id': '1144444',
    }
    return sample_selections


def deploy_new_config(config_path: Path, selections: Dict[str, Any]) -> None:
    """
    Applies selections to the template and writes the result to config_path
    :param config_path: Path object for new config file. Should not exist yet
    :param selecions: Dict containing selections taken by the user.
    """
    from jinja2.exceptions import TemplateNotFound
    try:
        selections['exchange'] = render_template(
            templatefile=f"subtemplates/exchange_{selections['exchange_name']}.j2",
            arguments=selections
            )
    except TemplateNotFound:
        selections['exchange'] = render_template(
            templatefile=f"subtemplates/exchange_generic.j2",
            arguments=selections
        )

    config_text = render_template(templatefile='base_config.json.j2',
                                  arguments=selections)

    logger.info(f"Writing config to `{config_path}`.")
    config_path.write_text(config_text)


def start_new_config(args: Dict[str, Any]) -> None:
    """
    Create a new strategy from a template
    Asking the user questions to fill out the templateaccordingly.
    """
    selections = ask_user_config()
    config_path = Path(args['config'][0])
    deploy_new_config(config_path, selections)


