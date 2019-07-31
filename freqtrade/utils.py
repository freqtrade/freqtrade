import logging
import sys
from argparse import Namespace
from typing import Any, Dict

from freqtrade.configuration import Configuration
from freqtrade.configuration.directory_operations import create_userdata_dir
from freqtrade.exchange import available_exchanges
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def setup_utils_configuration(args: Namespace, method: RunMode) -> Dict[str, Any]:
    """
    Prepare the configuration for utils subcommands
    :param args: Cli args from Arguments()
    :return: Configuration
    """
    configuration = Configuration(args, method)
    config = configuration.load_config()

    config['exchange']['dry_run'] = True
    # Ensure we do not use Exchange credentials
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    return config


def start_list_exchanges(args: Namespace) -> None:
    """
    Print available exchanges
    :param args: Cli args from Arguments()
    :return: None
    """

    if args.print_one_column:
        print('\n'.join(available_exchanges()))
    else:
        print(f"Exchanges supported by ccxt and available for Freqtrade: "
              f"{', '.join(available_exchanges())}")


def start_create_userdir(args: Namespace) -> None:
    """
    Create "user_data" directory to contain user data strategies, hyperopts, ...)
    :param args: Cli args from Arguments()
    :return: None
    """
    if "user_data_dir" in args and args.user_data_dir:
        create_userdata_dir(args.user_data_dir, create_dir=True)
    else:
        logger.warning("`create-userdir` requires --userdir to be set.")
        sys.exit(1)
