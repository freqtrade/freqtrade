import logging
from argparse import Namespace
from typing import Any, Dict

from freqtrade.configuration import Configuration
from freqtrade.exchange import available_exchanges, Exchange
from freqtrade.misc import plural
from freqtrade.state import RunMode

from tabulate import tabulate


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


def start_list_pairs(args: Namespace, pairs_only: bool = False) -> None:
    """
    Print pairs on the exchange
    :param args: Cli args from Arguments()
    :return: None
    """

    # Initialize configuration
    config = setup_utils_configuration(args, RunMode.OTHER)

    # Fetch exchange name from args, use bittrex as default exchange
    config['exchange']['name'] = args.exchange or 'bittrex'

    # Init exchange
    exchange = Exchange(config)

    logger.debug(f"Exchange markets: {exchange.markets}")

    pairs = exchange.get_markets(base_currency=args.base_currency,
                                 quote_currency=args.quote_currency,
                                 pairs_only=pairs_only,
                                 active_only=args.active_only)

    if args.print_list:
        # print data as a list
        print(f"Exchange {exchange.name} has {len(pairs)} " +
              (plural(len(pairs), "pair" if pairs_only else "market")) +
              (f" with {args.base_currency} as base currency" if args.base_currency else "") +
              (" and" if args.base_currency and args.quote_currency else "") +
              (f" with {args.quote_currency} as quote currency" if args.quote_currency else "") +
              (f": {sorted(pairs.keys())}" if len(pairs) else "") + ".")
    else:
#        # print a table of pairs (markets)
#        print('{:<15} {:<15} {:<15} {:<15} {:<15}'.format('id', 'symbol', 'base', 'quote', 'active'))
#
#        for (k, v) in pairs.items():
#            print('{:<15} {:<15} {:<15} {:<15} {:<15}'.format(v['id'], v['symbol'], v['base'], v['quote'], "Yes" if v['active'] else "No"))
        tabular_data = []
        for _, v in pairs.items():
            tabular_data.append([v['id'], v['symbol'], v['base'], v['quote'],
                                "Yes" if v['active'] else "No"])

        headers = ['Id', 'Symbol', 'Base', 'Quote', 'Active']
        print(tabulate(tabular_data, headers=headers, tablefmt='pipe'))
