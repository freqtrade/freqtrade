import logging
import sys
from collections import defaultdict
from typing import Any, Dict

from freqtrade.configuration import TimeRange, setup_utils_configuration
from freqtrade.constants import DATETIME_PRINT_FORMAT, DL_DATA_TIMEFRAMES, Config
from freqtrade.data.converter import (convert_ohlcv_format, convert_trades_format,
                                      convert_trades_to_ohlcv)
from freqtrade.data.history import download_data_main
from freqtrade.enums import CandleType, RunMode, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist
from freqtrade.resolvers import ExchangeResolver
from freqtrade.util.migrations import migrate_data


logger = logging.getLogger(__name__)


def _check_data_config_download_sanity(config: Config) -> None:
    if 'days' in config and 'timerange' in config:
        raise OperationalException("--days and --timerange are mutually exclusive. "
                                   "You can only specify one or the other.")

    if 'pairs' not in config:
        raise OperationalException(
            "Downloading data requires a list of pairs. "
            "Please check the documentation on how to configure this.")


def start_download_data(args: Dict[str, Any]) -> None:
    """
    Download data (former download_backtest_data.py script)
    """
    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    _check_data_config_download_sanity(config)

    try:
        download_data_main(config)

    except KeyboardInterrupt:
        sys.exit("SIGINT received, aborting ...")


def start_convert_trades(args: Dict[str, Any]) -> None:

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    timerange = TimeRange()

    # Remove stake-currency to skip checks which are not relevant for datadownload
    config['stake_currency'] = ''

    if 'timeframes' not in config:
        config['timeframes'] = DL_DATA_TIMEFRAMES

    # Init exchange
    exchange = ExchangeResolver.load_exchange(config, validate=False)
    # Manual validations of relevant settings

    for timeframe in config['timeframes']:
        exchange.validate_timeframes(timeframe)
    available_pairs = [
        p for p in exchange.get_markets(
            tradable_only=True, active_only=not config.get('include_inactive')
            ).keys()
    ]

    expanded_pairs = dynamic_expand_pairlist(config, available_pairs)

    # Convert downloaded trade data to different timeframes
    convert_trades_to_ohlcv(
        pairs=expanded_pairs, timeframes=config['timeframes'],
        datadir=config['datadir'], timerange=timerange, erase=bool(config.get('erase')),
        data_format_ohlcv=config['dataformat_ohlcv'],
        data_format_trades=config['dataformat_trades'],
        candle_type=config.get('candle_type_def', CandleType.SPOT)
    )


def start_convert_data(args: Dict[str, Any], ohlcv: bool = True) -> None:
    """
    Convert data from one format to another
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    if ohlcv:
        migrate_data(config)
        convert_ohlcv_format(config,
                             convert_from=args['format_from'],
                             convert_to=args['format_to'],
                             erase=args['erase'])
    else:
        convert_trades_format(config,
                              convert_from=args['format_from_trades'], convert_to=args['format_to'],
                              erase=args['erase'])


def start_list_data(args: Dict[str, Any]) -> None:
    """
    List available backtest data
    """

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from tabulate import tabulate

    from freqtrade.data.history import get_datahandler
    dhc = get_datahandler(config['datadir'], config['dataformat_ohlcv'])

    paircombs = dhc.ohlcv_get_available_data(
        config['datadir'],
        config.get('trading_mode', TradingMode.SPOT)
        )

    if args['pairs']:
        paircombs = [comb for comb in paircombs if comb[0] in args['pairs']]

    print(f"Found {len(paircombs)} pair / timeframe combinations.")
    if not config.get('show_timerange'):
        groupedpair = defaultdict(list)
        for pair, timeframe, candle_type in sorted(
            paircombs,
            key=lambda x: (x[0], timeframe_to_minutes(x[1]), x[2])
        ):
            groupedpair[(pair, candle_type)].append(timeframe)

        if groupedpair:
            print(tabulate([
                (pair, ', '.join(timeframes), candle_type)
                for (pair, candle_type), timeframes in groupedpair.items()
            ],
                headers=("Pair", "Timeframe", "Type"),
                tablefmt='psql', stralign='right'))
    else:
        paircombs1 = [(
            pair, timeframe, candle_type,
            *dhc.ohlcv_data_min_max(pair, timeframe, candle_type)
        ) for pair, timeframe, candle_type in paircombs]

        print(tabulate([
            (pair, timeframe, candle_type,
                start.strftime(DATETIME_PRINT_FORMAT),
                end.strftime(DATETIME_PRINT_FORMAT), length)
            for pair, timeframe, candle_type, start, end, length in sorted(
                paircombs1,
                key=lambda x: (x[0], timeframe_to_minutes(x[1]), x[2]))
            ],
            headers=("Pair", "Timeframe", "Type", 'From', 'To', 'Candles'),
            tablefmt='psql', stralign='right'))
