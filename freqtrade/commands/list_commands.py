import csv
import logging
import sys
from typing import Any, Union

from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError, OperationalException
from freqtrade.ft_types import ValidExchangesType


logger = logging.getLogger(__name__)


def start_list_exchanges(args: dict[str, Any]) -> None:
    """
    Print available exchanges
    :param args: Cli args from Arguments()
    :return: None
    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    from freqtrade.exchange import list_available_exchanges

    available_exchanges: list[ValidExchangesType] = list_available_exchanges(
        args["list_exchanges_all"]
    )

    if args["print_one_column"]:
        print("\n".join([e["classname"] for e in available_exchanges]))
    else:
        if args["list_exchanges_all"]:
            title = (
                f"All exchanges supported by the ccxt library "
                f"({len(available_exchanges)} exchanges):"
            )
        else:
            available_exchanges = [e for e in available_exchanges if e["valid"] is not False]
            title = f"Exchanges available for Freqtrade ({len(available_exchanges)} exchanges):"

        table = Table(title=title)

        table.add_column("Exchange Name")
        table.add_column("Class Name")
        table.add_column("Markets")
        table.add_column("Reason")

        for exchange in available_exchanges:
            name = Text(exchange["name"])
            if exchange["supported"]:
                name.append(" (Supported)", style="italic")
                name.stylize("green bold")
            classname = Text(exchange["classname"])
            if exchange["is_alias"]:
                name.stylize("strike")
                classname.stylize("strike")
                classname.append(f" (use {exchange['alias_for']})", style="italic")

            trade_modes = Text(
                ", ".join(
                    (f"{a.get('margin_mode', '')} {a['trading_mode']}").lstrip()
                    for a in exchange["trade_modes"]
                ),
                style="",
            )
            if exchange["dex"]:
                trade_modes = Text("DEX: ") + trade_modes
                trade_modes.stylize("bold", 0, 3)

            table.add_row(
                name,
                classname,
                trade_modes,
                exchange["comment"],
                style=None if exchange["valid"] else "red",
            )
            # table.add_row(*[exchange[header] for header in headers])

        console = Console()
        console.print(table)


def _print_objs_tabular(objs: list, print_colorized: bool) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    names = [s["name"] for s in objs]
    objs_to_print: list[dict[str, Union[Text, str]]] = [
        {
            "name": Text(s["name"] if s["name"] else "--"),
            "location": s["location_rel"],
            "status": (
                Text("LOAD FAILED", style="bold red")
                if s["class"] is None
                else Text("OK", style="bold green")
                if names.count(s["name"]) == 1
                else Text("DUPLICATE NAME", style="bold yellow")
            ),
        }
        for s in objs
    ]
    for idx, s in enumerate(objs):
        if "hyperoptable" in s:
            objs_to_print[idx].update(
                {
                    "hyperoptable": "Yes" if s["hyperoptable"]["count"] > 0 else "No",
                    "buy-Params": str(len(s["hyperoptable"].get("buy", []))),
                    "sell-Params": str(len(s["hyperoptable"].get("sell", []))),
                }
            )
    table = Table()

    for header in objs_to_print[0].keys():
        table.add_column(header.capitalize(), justify="right")

    for row in objs_to_print:
        table.add_row(*[row[header] for header in objs_to_print[0].keys()])

    console = Console(
        color_system="auto" if print_colorized else None,
        width=200 if "pytest" in sys.modules else None,
    )
    console.print(table)


def start_list_strategies(args: dict[str, Any]) -> None:
    """
    Print files with Strategy custom classes available in the directory
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.resolvers import StrategyResolver

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    strategy_objs = StrategyResolver.search_all_objects(
        config, not args["print_one_column"], config.get("recursive_strategy_search", False)
    )
    # Sort alphabetically
    strategy_objs = sorted(strategy_objs, key=lambda x: x["name"])
    for obj in strategy_objs:
        if obj["class"]:
            obj["hyperoptable"] = obj["class"].detect_all_parameters()
        else:
            obj["hyperoptable"] = {"count": 0}

    if args["print_one_column"]:
        print("\n".join([s["name"] for s in strategy_objs]))
    else:
        _print_objs_tabular(strategy_objs, config.get("print_colorized", False))


def start_list_freqAI_models(args: dict[str, Any]) -> None:
    """
    Print files with FreqAI models custom classes available in the directory
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    model_objs = FreqaiModelResolver.search_all_objects(config, not args["print_one_column"])
    # Sort alphabetically
    model_objs = sorted(model_objs, key=lambda x: x["name"])
    if args["print_one_column"]:
        print("\n".join([s["name"] for s in model_objs]))
    else:
        _print_objs_tabular(model_objs, config.get("print_colorized", False))


def start_list_hyperopt_loss_functions(args: dict[str, Any]) -> None:
    """
    Print files with FreqAI models custom classes available in the directory
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    model_objs = HyperOptLossResolver.search_all_objects(config, not args["print_one_column"])
    # Sort alphabetically
    model_objs = sorted(model_objs, key=lambda x: x["name"])
    if args["print_one_column"]:
        print("\n".join([s["name"] for s in model_objs]))
    else:
        _print_objs_tabular(model_objs, config.get("print_colorized", False))


def start_list_timeframes(args: dict[str, Any]) -> None:
    """
    Print timeframes available on Exchange
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.resolvers import ExchangeResolver

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)
    # Do not use timeframe set in the config
    config["timeframe"] = None

    # Init exchange
    exchange = ExchangeResolver.load_exchange(config, validate=False)

    if args["print_one_column"]:
        print("\n".join(exchange.timeframes))
    else:
        print(
            f"Timeframes available for the exchange `{exchange.name}`: "
            f"{', '.join(exchange.timeframes)}"
        )


def start_list_markets(args: dict[str, Any], pairs_only: bool = False) -> None:
    """
    Print pairs/markets on the exchange
    :param args: Cli args from Arguments()
    :param pairs_only: if True print only pairs, otherwise print all instruments (markets)
    :return: None
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.exchange import market_is_active
    from freqtrade.misc import plural
    from freqtrade.resolvers import ExchangeResolver
    from freqtrade.util import print_rich_table

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    # Init exchange
    exchange = ExchangeResolver.load_exchange(config, validate=False)

    # By default only active pairs/markets are to be shown
    active_only = not args.get("list_pairs_all", False)

    base_currencies = args.get("base_currencies", [])
    quote_currencies = args.get("quote_currencies", [])

    try:
        pairs = exchange.get_markets(
            base_currencies=base_currencies,
            quote_currencies=quote_currencies,
            tradable_only=pairs_only,
            active_only=active_only,
        )
        # Sort the pairs/markets by symbol
        pairs = dict(sorted(pairs.items()))
    except Exception as e:
        raise OperationalException(f"Cannot get markets. Reason: {e}") from e

    else:
        summary_str = (
            (f"Exchange {exchange.name} has {len(pairs)} ")
            + ("active " if active_only else "")
            + (plural(len(pairs), "pair" if pairs_only else "market"))
            + (
                f" with {', '.join(base_currencies)} as base "
                f"{plural(len(base_currencies), 'currency', 'currencies')}"
                if base_currencies
                else ""
            )
            + (" and" if base_currencies and quote_currencies else "")
            + (
                f" with {', '.join(quote_currencies)} as quote "
                f"{plural(len(quote_currencies), 'currency', 'currencies')}"
                if quote_currencies
                else ""
            )
        )

        headers = [
            "Id",
            "Symbol",
            "Base",
            "Quote",
            "Active",
            "Spot",
            "Margin",
            "Future",
            "Leverage",
        ]

        tabular_data = [
            {
                "Id": v["id"],
                "Symbol": v["symbol"],
                "Base": v["base"],
                "Quote": v["quote"],
                "Active": market_is_active(v),
                "Spot": "Spot" if exchange.market_is_spot(v) else "",
                "Margin": "Margin" if exchange.market_is_margin(v) else "",
                "Future": "Future" if exchange.market_is_future(v) else "",
                "Leverage": exchange.get_max_leverage(v["symbol"], 20),
            }
            for _, v in pairs.items()
        ]

        if (
            args.get("print_one_column", False)
            or args.get("list_pairs_print_json", False)
            or args.get("print_csv", False)
        ):
            # Print summary string in the log in case of machine-readable
            # regular formats.
            logger.info(f"{summary_str}.")
        else:
            # Print empty string separating leading logs and output in case of
            # human-readable formats.
            print()

        if pairs:
            if args.get("print_list", False):
                # print data as a list, with human-readable summary
                print(f"{summary_str}: {', '.join(pairs.keys())}.")
            elif args.get("print_one_column", False):
                print("\n".join(pairs.keys()))
            elif args.get("list_pairs_print_json", False):
                import rapidjson

                print(rapidjson.dumps(list(pairs.keys()), default=str))
            elif args.get("print_csv", False):
                writer = csv.DictWriter(sys.stdout, fieldnames=headers)
                writer.writeheader()
                writer.writerows(tabular_data)
            else:
                print_rich_table(tabular_data, headers, summary_str)
        elif not (
            args.get("print_one_column", False)
            or args.get("list_pairs_print_json", False)
            or args.get("print_csv", False)
        ):
            print(f"{summary_str}.")


def start_show_trades(args: dict[str, Any]) -> None:
    """
    Show trades
    """
    import json

    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.misc import parse_db_uri_for_logging
    from freqtrade.persistence import Trade, init_db

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if "db_url" not in config:
        raise ConfigurationError("--db-url is required for this command.")

    logger.info(f'Using DB: "{parse_db_uri_for_logging(config["db_url"])}"')
    init_db(config["db_url"])
    tfilter = []

    if config.get("trade_ids"):
        tfilter.append(Trade.id.in_(config["trade_ids"]))

    trades = Trade.get_trades(tfilter).all()
    logger.info(f"Printing {len(trades)} Trades: ")
    if config.get("print_json", False):
        print(json.dumps([trade.to_json() for trade in trades], indent=4))
    else:
        for trade in trades:
            print(trade)
