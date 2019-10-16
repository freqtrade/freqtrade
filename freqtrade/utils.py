import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import arrow
import csv
import rapidjson
from tabulate import tabulate

from freqtrade import OperationalException
from freqtrade.configuration import Configuration, TimeRange
from freqtrade.configuration.directory_operations import create_userdata_dir
from freqtrade.data.history import refresh_backtest_ohlcv_data
from freqtrade.exchange import (available_exchanges, ccxt_exchanges, market_is_active,
                                market_is_pair)
from freqtrade.misc import plural
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


def start_list_exchanges(args: Dict[str, Any]) -> None:
    """
    Print available exchanges
    :param args: Cli args from Arguments()
    :return: None
    """
    exchanges = ccxt_exchanges() if args['list_exchanges_all'] else available_exchanges()
    if args['print_one_column']:
        print('\n'.join(exchanges))
    else:
        if args['list_exchanges_all']:
            print(f"All exchanges supported by the ccxt library: {', '.join(exchanges)}")
        else:
            print(f"Exchanges available for Freqtrade: {', '.join(exchanges)}")


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


def start_list_timeframes(args: Dict[str, Any]) -> None:
    """
    Print ticker intervals (timeframes) available on Exchange
    """
    config = setup_utils_configuration(args, RunMode.OTHER)
    # Do not use ticker_interval set in the config
    config['ticker_interval'] = None

    # Init exchange
    exchange = ExchangeResolver(config['exchange']['name'], config).exchange

    if args['print_one_column']:
        print('\n'.join(exchange.timeframes))
    else:
        print(f"Timeframes available for the exchange `{config['exchange']['name']}`: "
              f"{', '.join(exchange.timeframes)}.")


def start_list_pairs(args: Dict[str, Any], pairs_only: bool = False) -> None:  # noqa: C901
    """
    Print pairs on the exchange
    :param args: Cli args from Arguments()
    :param pairs_only: if True print only pairs, otherwise print all instruments (markets)
    :return: None
    """
    config = setup_utils_configuration(args, RunMode.OTHER)

    # Init exchange
    exchange = ExchangeResolver(config['exchange']['name'], config).exchange

    active_only = args.get('active_only', False)
    base_currencies = args.get('base_currencies', [])
    quote_currencies = args.get('quote_currencies', [])

    try:
        pairs = exchange.get_markets(base_currencies=base_currencies,
                                     quote_currencies=quote_currencies,
                                     pairs_only=pairs_only,
                                     active_only=active_only)
    except Exception as e:
        raise OperationalException(f"Cannot get markets. Reason: {e}") from e

    else:
        summary_str = ((f"Exchange {exchange.name} has {len(pairs)} ") +
                       ("active " if active_only else "") +
                       (plural(len(pairs), "pair" if pairs_only else "market")) +
                       (f" with {', '.join(base_currencies)} as base "
                        f"{plural(len(base_currencies), 'currency', 'currencies')}"
                        if base_currencies else "") +
                       (" and" if base_currencies and quote_currencies else "") +
                       (f" with {', '.join(quote_currencies)} as quote "
                        f"{plural(len(quote_currencies), 'currency', 'currencies')}"
                        if quote_currencies else ""))

        headers = ["Id", "Symbol", "Base", "Quote", "Active",
                   *(['Is pair'] if not pairs_only else [])]

        tabular_data = []
        for _, v in pairs.items():
            tabular_data.append({'Id': v['id'], 'Symbol': v['symbol'],
                                 'Base': v['base'], 'Quote': v['quote'],
                                 'Active': market_is_active(v),
                                 **({'Is pair': market_is_pair(v)} if not pairs_only else {})})

        if (args.get('print_one_column', False) or
                args.get('list_pairs_print_json', False) or
                args.get('print_csv', False)):
            logger.info(f"{summary_str}.")

        if args.get('print_list', False):
            # print data as a list, with human-readable summary
            print(summary_str +
                  (f": {', '.join(sorted(pairs.keys()))}" if len(pairs) else "") + ".")
        elif args.get('print_one_column', False):
            if len(pairs):
                print('\n'.join(sorted(pairs.keys())))
        elif args.get('list_pairs_print_json', False):
            if len(pairs):
                print(rapidjson.dumps(sorted(pairs.keys()), default=str))
        elif args.get('print_csv', False):
            if len(pairs):
                writer = csv.DictWriter(sys.stdout, fieldnames=headers)
                writer.writeheader()
                writer.writerows(tabular_data)
        else:
            print(summary_str +
                  (":" if len(pairs) else "."))
            if len(pairs):
                # print data as a table
                print(tabulate(tabular_data, headers='keys', tablefmt='pipe'))
