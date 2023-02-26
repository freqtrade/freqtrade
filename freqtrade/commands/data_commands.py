import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

from freqtrade.configuration import TimeRange, setup_utils_configuration
from freqtrade.constants import DATETIME_PRINT_FORMAT, Config
from freqtrade.data.converter import convert_ohlcv_format, convert_trades_format
from freqtrade.data.history import (convert_trades_to_ohlcv, refresh_backtest_ohlcv_data,
                                    refresh_backtest_trades_data)
from freqtrade.enums import CandleType, RunMode, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import market_is_active, timeframe_to_minutes
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist, expand_pairlist
from freqtrade.resolvers import ExchangeResolver
from freqtrade.util.binance_mig import migrate_binance_futures_data


logger = logging.getLogger(__name__)


def _data_download_sanity(config: Config) -> None:
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

    _data_download_sanity(config)
    timerange = TimeRange()
    if 'days' in config:
        time_since = (datetime.now() - timedelta(days=config['days'])).strftime("%Y%m%d")
        timerange = TimeRange.parse_timerange(f'{time_since}-')

    if 'timerange' in config:
        timerange = timerange.parse_timerange(config['timerange'])

    # Remove stake-currency to skip checks which are not relevant for datadownload
    config['stake_currency'] = ''

    pairs_not_available: List[str] = []

    # Init exchange
    exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config, validate=False)
    markets = [p for p, m in exchange.markets.items() if market_is_active(m)
               or config.get('include_inactive')]

    expanded_pairs = dynamic_expand_pairlist(config, markets)

    # Manual validations of relevant settings
    if not config['exchange'].get('skip_pair_validation', False):
        exchange.validate_pairs(expanded_pairs)
    logger.info(f"About to download pairs: {expanded_pairs}, "
                f"intervals: {config['timeframes']} to {config['datadir']}")

    for timeframe in config['timeframes']:
        exchange.validate_timeframes(timeframe)

    try:

        if config.get('download_trades'):
            if config.get('trading_mode') == 'futures':
                raise OperationalException("Trade download not supported for futures.")
            pairs_not_available = refresh_backtest_trades_data(
                exchange, pairs=expanded_pairs, datadir=config['datadir'],
                timerange=timerange, new_pairs_days=config['new_pairs_days'],
                erase=bool(config.get('erase')), data_format=config['dataformat_trades'])

            # Convert downloaded trade data to different timeframes
            convert_trades_to_ohlcv(
                pairs=expanded_pairs, timeframes=config['timeframes'],
                datadir=config['datadir'], timerange=timerange, erase=bool(config.get('erase')),
                data_format_ohlcv=config['dataformat_ohlcv'],
                data_format_trades=config['dataformat_trades'],
            )
        else:
            if not exchange.get_option('ohlcv_has_history', True):
                raise OperationalException(
                    f"Historic klines not available for {exchange.name}. "
                    "Please use `--dl-trades` instead for this exchange "
                    "(will unfortunately take a long time)."
                    )
            migrate_binance_futures_data(config)
            pairs_not_available = refresh_backtest_ohlcv_data(
                exchange, pairs=expanded_pairs, timeframes=config['timeframes'],
                datadir=config['datadir'], timerange=timerange,
                new_pairs_days=config['new_pairs_days'],
                erase=bool(config.get('erase')), data_format=config['dataformat_ohlcv'],
                trading_mode=config.get('trading_mode', 'spot'),
                prepend=config.get('prepend_data', False)
            )

    except KeyboardInterrupt:
        sys.exit("SIGINT received, aborting ...")

    finally:
        if pairs_not_available:
            logger.info(f"Pairs [{','.join(pairs_not_available)}] not available "
                        f"on exchange {exchange.name}.")


def start_convert_trades(args: Dict[str, Any]) -> None:

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    timerange = TimeRange()

    # Remove stake-currency to skip checks which are not relevant for datadownload
    config['stake_currency'] = ''

    if 'pairs' not in config:
        raise OperationalException(
            "Downloading data requires a list of pairs. "
            "Please check the documentation on how to configure this.")

    # Init exchange
    exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config, validate=False)
    # Manual validations of relevant settings
    if not config['exchange'].get('skip_pair_validation', False):
        exchange.validate_pairs(config['pairs'])
    expanded_pairs = expand_pairlist(config['pairs'], list(exchange.markets))

    logger.info(f"About to Convert pairs: {expanded_pairs}, "
                f"intervals: {config['timeframes']} to {config['datadir']}")

    for timeframe in config['timeframes']:
        exchange.validate_timeframes(timeframe)
    # Convert downloaded trade data to different timeframes
    convert_trades_to_ohlcv(
        pairs=expanded_pairs, timeframes=config['timeframes'],
        datadir=config['datadir'], timerange=timerange, erase=bool(config.get('erase')),
        data_format_ohlcv=config['dataformat_ohlcv'],
        data_format_trades=config['dataformat_trades'],
    )


def start_convert_data(args: Dict[str, Any], ohlcv: bool = True) -> None:
    """
    Convert data from one format to another
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    if ohlcv:
        migrate_binance_futures_data(config)
        candle_types = [CandleType.from_string(ct) for ct in config.get('candle_types', ['spot'])]
        for candle_type in candle_types:
            convert_ohlcv_format(config,
                                 convert_from=args['format_from'], convert_to=args['format_to'],
                                 erase=args['erase'], candle_type=candle_type)
    else:
        convert_trades_format(config,
                              convert_from=args['format_from'], convert_to=args['format_to'],
                              erase=args['erase'])


def start_list_data(args: Dict[str, Any]) -> None:
    """
    List available backtest data
    """

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from tabulate import tabulate

    from freqtrade.data.history.idatahandler import get_datahandler
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
                end.strftime(DATETIME_PRINT_FORMAT))
            for pair, timeframe, candle_type, start, end in paircombs1
            ],
            headers=("Pair", "Timeframe", "Type", 'From', 'To'),
            tablefmt='psql', stralign='right'))
