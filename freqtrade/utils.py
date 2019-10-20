import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import arrow

from freqtrade import OperationalException
from freqtrade.configuration import Configuration, TimeRange
from freqtrade.configuration.directory_operations import create_userdata_dir
from freqtrade.data.history import refresh_backtest_ohlcv_data
from freqtrade.exchange import available_exchanges
from freqtrade.resolvers import ExchangeResolver
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def setup_utils_configuration(args: Dict[str, Any], method: RunMode) -> Dict[str, Any]:
    """
    Prepare the configuration for utils subcommands
    :param args: Cli args from Arguments()
    :return: Configuration
    """
    configuration = Configuration(args, method)
    config = configuration.get_config()

    config['exchange']['dry_run'] = True
    # Ensure we do not use Exchange credentials
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    return config


def start_trading(args: Dict[str, Any]) -> int:
    """
    Main entry point for trading mode
    """
    from freqtrade.worker import Worker
    # Load and run worker
    worker = Worker(args)
    worker.run()
    return 0


def start_list_exchanges(args: Dict[str, Any]) -> None:
    """
    Print available exchanges
    :param args: Cli args from Arguments()
    :return: None
    """

    if args['print_one_column']:
        print('\n'.join(available_exchanges()))
    else:
        print(f"Exchanges supported by ccxt and available for Freqtrade: "
              f"{', '.join(available_exchanges())}")


def start_create_userdir(args: Dict[str, Any]) -> None:
    """
    Create "user_data" directory to contain user data strategies, hyperopts, ...)
    :param args: Cli args from Arguments()
    :return: None
    """
    if "user_data_dir" in args and args["user_data_dir"]:
        create_userdata_dir(args["user_data_dir"], create_dir=True)
    else:
        logger.warning("`create-userdir` requires --userdir to be set.")
        sys.exit(1)


def start_download_data(args: Dict[str, Any]) -> None:
    """
    Download data (former download_backtest_data.py script)
    """
    config = setup_utils_configuration(args, RunMode.OTHER)

    timerange = TimeRange()
    if 'days' in config:
        time_since = arrow.utcnow().shift(days=-config['days']).strftime("%Y%m%d")
        timerange = TimeRange.parse_timerange(f'{time_since}-')

    if 'pairs' not in config:
        raise OperationalException(
            "Downloading data requires a list of pairs. "
            "Please check the documentation on how to configure this.")

    dl_path = Path(config['datadir'])
    logger.info(f'About to download pairs: {config["pairs"]}, '
                f'intervals: {config["timeframes"]} to {dl_path}')

    pairs_not_available: List[str] = []

    try:
        # Init exchange
        exchange = ExchangeResolver(config['exchange']['name'], config).exchange

        pairs_not_available = refresh_backtest_ohlcv_data(
            exchange, pairs=config["pairs"], timeframes=config["timeframes"],
            dl_path=Path(config['datadir']), timerange=timerange, erase=config.get("erase"))

    except KeyboardInterrupt:
        sys.exit("SIGINT received, aborting ...")

    finally:
        if pairs_not_available:
            logger.info(f"Pairs [{','.join(pairs_not_available)}] not available "
                        f"on exchange {config['exchange']['name']}.")
