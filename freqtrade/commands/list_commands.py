import csv
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from colorama import init as colorama_init
from colorama import Fore, Style
import rapidjson
from tabulate import tabulate

from freqtrade.configuration import setup_utils_configuration
from freqtrade.constants import USERPATH_HYPEROPTS, USERPATH_STRATEGIES
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import (available_exchanges, ccxt_exchanges,
                                market_is_active, symbol_is_pair)
from freqtrade.misc import plural
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


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


def _print_objs_tabular(objs: List, print_colorized: bool) -> None:
    if print_colorized:
        colorama_init(autoreset=True)
        red = Fore.RED
        yellow = Fore.YELLOW
        reset = Style.RESET_ALL
    else:
        red = ''
        yellow = ''
        reset = ''

    names = [s['name'] for s in objs]
    objss_to_print = [{
        'name': s['name'] if s['name'] else "--",
        'location': s['location'].name,
        'status': (red + "LOAD FAILED" + reset if s['class'] is None
                   else "OK" if names.count(s['name']) == 1
                   else yellow + "DUPLICATE NAME" + reset)
    } for s in objs]

    print(tabulate(objss_to_print, headers='keys', tablefmt='psql', stralign='right'))


def start_list_strategies(args: Dict[str, Any]) -> None:
    """
    Print files with Strategy custom classes available in the directory
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    directory = Path(config.get('strategy_path', config['user_data_dir'] / USERPATH_STRATEGIES))
    strategy_objs = StrategyResolver.search_all_objects(directory, not args['print_one_column'])
    # Sort alphabetically
    strategy_objs = sorted(strategy_objs, key=lambda x: x['name'])

    if args['print_one_column']:
        print('\n'.join([s['name'] for s in strategy_objs]))
    else:
        _print_objs_tabular(strategy_objs, config.get('print_colorized', False))


def start_list_hyperopts(args: Dict[str, Any]) -> None:
    """
    Print files with HyperOpt custom classes available in the directory
    """
    from freqtrade.resolvers.hyperopt_resolver import HyperOptResolver

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    directory = Path(config.get('hyperopt_path', config['user_data_dir'] / USERPATH_HYPEROPTS))
    hyperopt_objs = HyperOptResolver.search_all_objects(directory, not args['print_one_column'])
    # Sort alphabetically
    hyperopt_objs = sorted(hyperopt_objs, key=lambda x: x['name'])

    if args['print_one_column']:
        print('\n'.join([s['name'] for s in hyperopt_objs]))
    else:
        _print_objs_tabular(hyperopt_objs, config.get('print_colorized', False))


def start_list_timeframes(args: Dict[str, Any]) -> None:
    """
    Print ticker intervals (timeframes) available on Exchange
    """
    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)
    # Do not use ticker_interval set in the config
    config['ticker_interval'] = None

    # Init exchange
    exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config, validate=False)

    if args['print_one_column']:
        print('\n'.join(exchange.timeframes))
    else:
        print(f"Timeframes available for the exchange `{exchange.name}`: "
              f"{', '.join(exchange.timeframes)}")


def start_list_markets(args: Dict[str, Any], pairs_only: bool = False) -> None:
    """
    Print pairs/markets on the exchange
    :param args: Cli args from Arguments()
    :param pairs_only: if True print only pairs, otherwise print all instruments (markets)
    :return: None
    """
    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    # Init exchange
    exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config, validate=False)

    # By default only active pairs/markets are to be shown
    active_only = not args.get('list_pairs_all', False)

    base_currencies = args.get('base_currencies', [])
    quote_currencies = args.get('quote_currencies', [])

    try:
        pairs = exchange.get_markets(base_currencies=base_currencies,
                                     quote_currencies=quote_currencies,
                                     pairs_only=pairs_only,
                                     active_only=active_only)
        # Sort the pairs/markets by symbol
        pairs = OrderedDict(sorted(pairs.items()))
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
                                 **({'Is pair': symbol_is_pair(v['symbol'])}
                                    if not pairs_only else {})})

        if (args.get('print_one_column', False) or
                args.get('list_pairs_print_json', False) or
                args.get('print_csv', False)):
            # Print summary string in the log in case of machine-readable
            # regular formats.
            logger.info(f"{summary_str}.")
        else:
            # Print empty string separating leading logs and output in case of
            # human-readable formats.
            print()

        if len(pairs):
            if args.get('print_list', False):
                # print data as a list, with human-readable summary
                print(f"{summary_str}: {', '.join(pairs.keys())}.")
            elif args.get('print_one_column', False):
                print('\n'.join(pairs.keys()))
            elif args.get('list_pairs_print_json', False):
                print(rapidjson.dumps(list(pairs.keys()), default=str))
            elif args.get('print_csv', False):
                writer = csv.DictWriter(sys.stdout, fieldnames=headers)
                writer.writeheader()
                writer.writerows(tabular_data)
            else:
                # print data as a table, with the human-readable summary
                print(f"{summary_str}:")
                print(tabulate(tabular_data, headers='keys', tablefmt='psql', stralign='right'))
        elif not (args.get('print_one_column', False) or
                  args.get('list_pairs_print_json', False) or
                  args.get('print_csv', False)):
            print(f"{summary_str}.")


def start_show_trades(args: Dict[str, Any]) -> None:
    """
    Show trades
    """
    from freqtrade.persistence import init, Trade
    import json
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if 'db_url' not in config:
        raise OperationalException("--db-url is required for this command.")

    logger.info(f'Using DB: "{config["db_url"]}"')
    init(config['db_url'], clean_open_orders=False)
    tfilter = []

    if config.get('trade_ids'):
        tfilter.append(Trade.id.in_(config['trade_ids']))

    trades = Trade.get_trades(tfilter).all()
    logger.info(f"Printing {len(trades)} Trades: ")
    if config.get('print_json', False):
        print(json.dumps([trade.to_json() for trade in trades], indent=4))
    else:
        for trade in trades:
            print(trade)
