import csv
import logging
import sys
from collections import OrderedDict
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List

import arrow
import rapidjson
from colorama import init as colorama_init
from tabulate import tabulate

from freqtrade.configuration import (Configuration, TimeRange,
                                     remove_credentials,
                                     validate_config_consistency)
from freqtrade.configuration.directory_operations import (copy_sample_files,
                                                          create_userdata_dir)
from freqtrade.constants import USERPATH_HYPEROPTS, USERPATH_STRATEGY
from freqtrade.data.history import (convert_trades_to_ohlcv,
                                    refresh_backtest_ohlcv_data,
                                    refresh_backtest_trades_data)
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import (available_exchanges, ccxt_exchanges,
                                market_is_active, symbol_is_pair)
from freqtrade.misc import plural, render_template
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
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

    # Ensure we do not use Exchange credentials
    remove_credentials(config)
    validate_config_consistency(config)

    return config


def start_download_data(args: Dict[str, Any]) -> None:
    """
    Download data (former download_backtest_data.py script)
    """
    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    timerange = TimeRange()
    if 'days' in config:
        time_since = arrow.utcnow().shift(days=-config['days']).strftime("%Y%m%d")
        timerange = TimeRange.parse_timerange(f'{time_since}-')

    if 'pairs' not in config:
        raise OperationalException(
            "Downloading data requires a list of pairs. "
            "Please check the documentation on how to configure this.")

    logger.info(f'About to download pairs: {config["pairs"]}, '
                f'intervals: {config["timeframes"]} to {config["datadir"]}')

    pairs_not_available: List[str] = []

    # Init exchange
    exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config)
    try:

        if config.get('download_trades'):
            pairs_not_available = refresh_backtest_trades_data(
                exchange, pairs=config["pairs"], datadir=config['datadir'],
                timerange=timerange, erase=config.get("erase"))

            # Convert downloaded trade data to different timeframes
            convert_trades_to_ohlcv(
                pairs=config["pairs"], timeframes=config["timeframes"],
                datadir=config['datadir'], timerange=timerange, erase=config.get("erase"))
        else:
            pairs_not_available = refresh_backtest_ohlcv_data(
                exchange, pairs=config["pairs"], timeframes=config["timeframes"],
                datadir=config['datadir'], timerange=timerange, erase=config.get("erase"))

    except KeyboardInterrupt:
        sys.exit("SIGINT received, aborting ...")

    finally:
        if pairs_not_available:
            logger.info(f"Pairs [{','.join(pairs_not_available)}] not available "
                        f"on exchange {exchange.name}.")


def start_test_pairlist(args: Dict[str, Any]) -> None:
    """
    Test Pairlist configuration
    """
    from freqtrade.pairlist.pairlistmanager import PairListManager
    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config, validate=False)

    quote_currencies = args.get('quote_currencies')
    if not quote_currencies:
        quote_currencies = [config.get('stake_currency')]
    results = {}
    for curr in quote_currencies:
        config['stake_currency'] = curr
        # Do not use ticker_interval set in the config
        pairlists = PairListManager(exchange, config)
        pairlists.refresh_pairlist()
        results[curr] = pairlists.whitelist

    for curr, pairlist in results.items():
        if not args.get('print_one_column', False):
            print(f"Pairs for {curr}: ")

        if args.get('print_one_column', False):
            print('\n'.join(pairlist))
        elif args.get('list_pairs_print_json', False):
            print(rapidjson.dumps(list(pairlist), default=str))
        else:
            print(pairlist)
