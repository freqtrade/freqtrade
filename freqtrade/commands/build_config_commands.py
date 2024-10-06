import logging
from pathlib import Path
from typing import Any

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def start_new_config(args: dict[str, Any]) -> None:
    """
    Create a new strategy from a template
    Asking the user questions to fill out the template accordingly.
    """

    from freqtrade.configuration.deploy_config import (
        ask_user_config,
        ask_user_overwrite,
        deploy_new_config,
    )
    from freqtrade.configuration.directory_operations import chown_user_directory

    config_path = Path(args["config"][0])
    chown_user_directory(config_path.parent)
    if config_path.exists():
        overwrite = ask_user_overwrite(config_path)
        if overwrite:
            config_path.unlink()
        else:
            raise OperationalException(
                f"Configuration file `{config_path}` already exists. "
                "Please delete it or use a different configuration file name."
            )
    selections = ask_user_config()
    deploy_new_config(config_path, selections)


def start_show_config(args: dict[str, Any]) -> None:
    from freqtrade.configuration import sanitize_config
    from freqtrade.configuration.config_setup import setup_utils_configuration

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE, set_dry=False)

    print("Your combined configuration is:")
    config_sanitized = sanitize_config(
        config["original_config"], show_sensitive=args.get("show_sensitive", False)
    )

    from rich import print_json

    print_json(data=config_sanitized)
