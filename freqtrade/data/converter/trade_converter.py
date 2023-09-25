"""
Functions to convert data from one format to another
"""
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pandas import DataFrame, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.constants import (DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS, TRADES_DTYPES,
                                 Config, TradeList)
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def trades_df_remove_duplicates(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicates from the trades DataFrame.
    Uses pandas.DataFrame.drop_duplicates to remove duplicates based on the 'timestamp' column.
    :param trades: DataFrame with the columns constants.DEFAULT_TRADES_COLUMNS
    :return: DataFrame with duplicates removed based on the 'timestamp' column
    """
    return trades.drop_duplicates(subset=['timestamp', 'id'])


def trades_dict_to_list(trades: List[Dict]) -> TradeList:
    """
    Convert fetch_trades result into a List (to be more memory efficient).
    :param trades: List of trades, as returned by ccxt.fetch_trades.
    :return: List of Lists, with constants.DEFAULT_TRADES_COLUMNS as columns
    """
    return [[t[col] for col in DEFAULT_TRADES_COLUMNS] for t in trades]


def trades_convert_types(trades: DataFrame) -> DataFrame:
    """
    Convert Trades dtypes and add 'date' column
    """
    trades = trades.astype(TRADES_DTYPES)
    trades['date'] = to_datetime(trades['timestamp'], unit='ms', utc=True)
    return trades


def trades_list_to_df(trades: TradeList, convert: bool = True):
    """
    convert trades list to dataframe
    :param trades: List of Lists with constants.DEFAULT_TRADES_COLUMNS as columns
    """
    if not trades:
        df = DataFrame(columns=DEFAULT_TRADES_COLUMNS)
    else:
        df = DataFrame(trades, columns=DEFAULT_TRADES_COLUMNS)

    if convert:
        df = trades_convert_types(df)

    return df


def trades_to_ohlcv(trades: DataFrame, timeframe: str) -> DataFrame:
    """
    Converts trades list to OHLCV list
    :param trades: List of trades, as returned by ccxt.fetch_trades.
    :param timeframe: Timeframe to resample data to
    :return: OHLCV Dataframe.
    :raises: ValueError if no trades are provided
    """
    from freqtrade.exchange import timeframe_to_minutes
    timeframe_minutes = timeframe_to_minutes(timeframe)
    if trades.empty:
        raise ValueError('Trade-list empty.')
    df = trades.set_index('date', drop=True)

    df_new = df['price'].resample(f'{timeframe_minutes}min').ohlc()
    df_new['volume'] = df['amount'].resample(f'{timeframe_minutes}min').sum()
    df_new['date'] = df_new.index
    # Drop 0 volume rows
    df_new = df_new.dropna()
    return df_new.loc[:, DEFAULT_DATAFRAME_COLUMNS]


def convert_trades_to_ohlcv(
    pairs: List[str],
    timeframes: List[str],
    datadir: Path,
    timerange: TimeRange,
    erase: bool = False,
    data_format_ohlcv: str = 'feather',
    data_format_trades: str = 'feather',
    candle_type: CandleType = CandleType.SPOT
) -> None:
    """
    Convert stored trades data to ohlcv data
    """
    from freqtrade.data.history.idatahandler import get_datahandler
    data_handler_trades = get_datahandler(datadir, data_format=data_format_trades)
    data_handler_ohlcv = get_datahandler(datadir, data_format=data_format_ohlcv)
    if not pairs:
        pairs = data_handler_trades.trades_get_pairs(datadir)

    logger.info(f"About to convert pairs: '{', '.join(pairs)}', "
                f"intervals: '{', '.join(timeframes)}' to {datadir}")

    for pair in pairs:
        trades = data_handler_trades.trades_load(pair)
        for timeframe in timeframes:
            if erase:
                if data_handler_ohlcv.ohlcv_purge(pair, timeframe, candle_type=candle_type):
                    logger.info(f'Deleting existing data for pair {pair}, interval {timeframe}.')
            try:
                ohlcv = trades_to_ohlcv(trades, timeframe)
                # Store ohlcv
                data_handler_ohlcv.ohlcv_store(pair, timeframe, data=ohlcv, candle_type=candle_type)
            except ValueError:
                logger.exception(f'Could not convert {pair} to OHLCV.')


def convert_trades_format(config: Config, convert_from: str, convert_to: str, erase: bool):
    """
    Convert trades from one format to another format.
    :param config: Config dictionary
    :param convert_from: Source format
    :param convert_to: Target format
    :param erase: Erase source data (does not apply if source and target format are identical)
    """
    if convert_from == 'kraken_csv':
        if config['exchange']['name'] != 'kraken':
            raise OperationalException(
                'Converting from csv is only supported for kraken.'
                'Please refer to the documentation for details about this special mode.'
            )
        from freqtrade.data.converter.trade_converter_kraken import import_kraken_trades_from_csv
        import_kraken_trades_from_csv(config, convert_to)
        return

    from freqtrade.data.history.idatahandler import get_datahandler
    src = get_datahandler(config['datadir'], convert_from)
    trg = get_datahandler(config['datadir'], convert_to)

    if 'pairs' not in config:
        config['pairs'] = src.trades_get_pairs(config['datadir'])
    logger.info(f"Converting trades for {config['pairs']}")

    for pair in config['pairs']:
        data = src.trades_load(pair=pair)
        logger.info(f"Converting {len(data)} trades for {pair}")
        trg.trades_store(pair, data)
        if erase and convert_from != convert_to:
            logger.info(f"Deleting source Trade data for {pair}.")
            src.trades_purge(pair=pair)
