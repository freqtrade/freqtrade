import logging
import sys
from pathlib import Path
from typing import Any

from freqtrade.constants import USERPATH_STRATEGIES
from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError, OperationalException


logger = logging.getLogger(__name__)


# Timeout for requests
req_timeout = 30


def start_create_userdir(args: dict[str, Any]) -> None:
    """
    Create "user_data" directory to contain user data strategies, hyperopt, ...)
    :param args: Cli args from Arguments()
    :return: None
    """
    from freqtrade.configuration.directory_operations import copy_sample_files, create_userdata_dir

    if user_data_dir := args.get("user_data_dir"):
        userdir = create_userdata_dir(user_data_dir, create_dir=True)
        copy_sample_files(userdir, overwrite=args["reset"])
    else:
        logger.warning("`create-userdir` requires --userdir to be set.")
        sys.exit(1)


def deploy_new_strategy(strategy_name: str, strategy_path: Path, subtemplate: str) -> None:
    """
    Deploy new strategy from template to strategy_path
    """
    from freqtrade.util import render_template, render_template_with_fallback

    fallback = "full"
    attributes = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/strategy_attributes_{subtemplate}.j2",
        templatefallbackfile=f"strategy_subtemplates/strategy_attributes_{fallback}.j2",
    )
    indicators = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/indicators_{subtemplate}.j2",
        templatefallbackfile=f"strategy_subtemplates/indicators_{fallback}.j2",
    )
    buy_trend = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/buy_trend_{subtemplate}.j2",
        templatefallbackfile=f"strategy_subtemplates/buy_trend_{fallback}.j2",
    )
    sell_trend = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/sell_trend_{subtemplate}.j2",
        templatefallbackfile=f"strategy_subtemplates/sell_trend_{fallback}.j2",
    )
    plot_config = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/plot_config_{subtemplate}.j2",
        templatefallbackfile=f"strategy_subtemplates/plot_config_{fallback}.j2",
    )
    additional_methods = render_template_with_fallback(
        templatefile=f"strategy_subtemplates/strategy_methods_{subtemplate}.j2",
        templatefallbackfile="strategy_subtemplates/strategy_methods_empty.j2",
    )

    strategy_text = render_template(
        templatefile="base_strategy.py.j2",
        arguments={
            "strategy": strategy_name,
            "attributes": attributes,
            "indicators": indicators,
            "buy_trend": buy_trend,
            "sell_trend": sell_trend,
            "plot_config": plot_config,
            "additional_methods": additional_methods,
        },
    )

    logger.info(f"Writing strategy to `{strategy_path}`.")
    strategy_path.write_text(strategy_text)


def start_new_strategy(args: dict[str, Any]) -> None:
    from freqtrade.configuration import setup_utils_configuration

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if strategy := args.get("strategy"):
        if strategy_path := args.get("strategy_path"):
            strategy_dir = Path(strategy_path)
        else:
            strategy_dir = config["user_data_dir"] / USERPATH_STRATEGIES
        if not strategy_dir.is_dir():
            logger.info(f"Creating strategy directory {strategy_dir}")
            strategy_dir.mkdir(parents=True)
        new_path = strategy_dir / (strategy + ".py")

        if new_path.exists():
            raise OperationalException(
                f"`{new_path}` already exists. Please choose another Strategy Name."
            )

        deploy_new_strategy(strategy, new_path, args["template"])

    else:
        raise ConfigurationError("`new-strategy` requires --strategy to be set.")


def start_install_ui(args: dict[str, Any]) -> None:
    from freqtrade.commands.deploy_ui import (
        clean_ui_subdir,
        download_and_install_ui,
        get_ui_download_url,
        read_ui_version,
    )

    dest_folder = Path(__file__).parents[1] / "rpc/api_server/ui/installed/"
    # First make sure the assets are removed.
    dl_url, latest_version = get_ui_download_url(
        args.get("ui_version"), args.get("ui_prerelease", False)
    )

    curr_version = read_ui_version(dest_folder)
    if curr_version == latest_version and not args.get("erase_ui_only"):
        logger.info(f"UI already up-to-date, FreqUI Version {curr_version}.")
        return

    clean_ui_subdir(dest_folder)
    if args.get("erase_ui_only"):
        logger.info("Erased UI directory content. Not downloading new version.")
    else:
        # Download a new version
        download_and_install_ui(dest_folder, dl_url, latest_version)
