import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import arrow

from freqtrade.configuration import Configuration, TimeRange
from freqtrade.data.history import download_pair_history
from freqtrade.exchange import available_exchanges
from freqtrade.resolvers import ExchangeResolver
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def setup_utils_configuration(args: Namespace, method: RunMode) -> Dict[str, Any]:
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


def start_download_data(args: Namespace) -> None:
    """
    Download data based
    """
    config = setup_utils_configuration(args, RunMode.OTHER)

    timerange = TimeRange()
    if 'days' in config:
        time_since = arrow.utcnow().shift(days=-config['days']).strftime("%Y%m%d")
        timerange = TimeRange.parse_timerange(f'{time_since}-')

    dl_path = Path(config['datadir'])
    logger.info(f'About to download pairs: {config["pairs"]}, '
                f'intervals: {config["timeframes"]} to {dl_path}')

    pairs_not_available = []

    try:
        # Init exchange
        exchange = ExchangeResolver(config['exchange']['name'], config).exchange

        for pair in config["pairs"]:
            if pair not in exchange._api.markets:
                pairs_not_available.append(pair)
                logger.info(f"Skipping pair {pair}...")
                continue
            for ticker_interval in config["timeframes"]:
                pair_print = pair.replace('/', '_')
                filename = f'{pair_print}-{ticker_interval}.json'
                dl_file = dl_path.joinpath(filename)
                if args.erase and dl_file.exists():
                    logger.info(
                        f'Deleting existing data for pair {pair}, interval {ticker_interval}.')
                    dl_file.unlink()

                logger.info(f'Downloading pair {pair}, interval {ticker_interval}.')
                download_pair_history(datadir=dl_path, exchange=exchange,
                                      pair=pair, ticker_interval=str(ticker_interval),
                                      timerange=timerange)

    except KeyboardInterrupt:
        sys.exit("SIGINT received, aborting ...")

    finally:
        if pairs_not_available:
            logger.info(
                f"Pairs [{','.join(pairs_not_available)}] not available "
                f"on exchange {config['exchange']['name']}.")

    # configuration.resolve_pairs_list()
    print(config)
