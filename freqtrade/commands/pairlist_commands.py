import logging
from typing import Any

import rapidjson

from freqtrade.enums import RunMode


logger = logging.getLogger(__name__)


def start_test_pairlist(args: dict[str, Any]) -> None:
    """
    Test Pairlist configuration
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.persistence import FtNoDBContext
    from freqtrade.plugins.pairlistmanager import PairListManager
    from freqtrade.resolvers import ExchangeResolver

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    exchange = ExchangeResolver.load_exchange(config, validate=False)

    quote_currencies = args.get("quote_currencies")
    if not quote_currencies:
        quote_currencies = [config.get("stake_currency")]
    results = {}
    with FtNoDBContext():
        for curr in quote_currencies:
            config["stake_currency"] = curr
            pairlists = PairListManager(exchange, config)
            pairlists.refresh_pairlist()
            results[curr] = pairlists.whitelist

    for curr, pairlist in results.items():
        if not args.get("print_one_column", False) and not args.get("list_pairs_print_json", False):
            print(f"Pairs for {curr}: ")

        if args.get("print_one_column", False):
            print("\n".join(pairlist))
        elif args.get("list_pairs_print_json", False):
            print(rapidjson.dumps(list(pairlist), default=str))
        else:
            print(pairlist)
