import logging
import sys
from typing import Any, Dict, List

import arrow

from freqtrade.configuration import TimeRange, setup_utils_configuration
from freqtrade.data.converter import (convert_ohlcv_format,
                                      convert_trades_format)
from freqtrade.data.history import (convert_trades_to_ohlcv,
                                    refresh_backtest_ohlcv_data,
                                    refresh_backtest_trades_data)
from freqtrade.exceptions import OperationalException
from freqtrade.resolvers import ExchangeResolver
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


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
    exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config, validate=False)
    # Manual validations of relevant settings
    exchange.validate_pairs(config['pairs'])
    for timeframe in config['timeframes']:
        exchange.validate_timeframes(timeframe)

    try:

        if config.get('download_trades'):
            pairs_not_available = refresh_backtest_trades_data(
                exchange, pairs=config["pairs"], datadir=config['datadir'],
                timerange=timerange, erase=bool(config.get("erase")),
                data_format=config['dataformat_trades'])

            # Convert downloaded trade data to different timeframes
            convert_trades_to_ohlcv(
                pairs=config["pairs"], timeframes=config["timeframes"],
                datadir=config['datadir'], timerange=timerange, erase=bool(config.get("erase")),
                data_format_ohlcv=config['dataformat_ohlcv'],
                data_format_trades=config['dataformat_trades'],
                )
        else:
            pairs_not_available = refresh_backtest_ohlcv_data(
                exchange, pairs=config["pairs"], timeframes=config["timeframes"],
                datadir=config['datadir'], timerange=timerange, erase=bool(config.get("erase")),
                data_format=config['dataformat_ohlcv'])

    except KeyboardInterrupt:
        sys.exit("SIGINT received, aborting ...")

    finally:
        if pairs_not_available:
            logger.info(f"Pairs [{','.join(pairs_not_available)}] not available "
                        f"on exchange {exchange.name}.")


def start_convert_data(args: Dict[str, Any], ohlcv: bool = True) -> None:
    """
    Convert data from one format to another
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    if ohlcv:
        convert_ohlcv_format(config,
                             convert_from=args['format_from'], convert_to=args['format_to'],
                             erase=args['erase'])
    else:
        convert_trades_format(config,
                              convert_from=args['format_from'], convert_to=args['format_to'],
                              erase=args['erase'])
