# pylint: disable=C0103
import glob
import json
import logging
import os
from datetime import datetime, timedelta
from os.path import basename, splitext
from typing import List, Optional

import arrow
from sqlalchemy import func, text

from freqtrade import exchange
from freqtrade.exchange.interface import Exchange
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

TESTDATA_DIR = os.path.join('freqtrade', 'tests', 'testdata')
# TODO: Define a global value for analyze and backtesting
TICKER_HISTORY_INTERVAL_H: int = 24

_ROW_INDEX: int = 0
_ROW_INTERVAL: int = 0
_LEN_ROWS: int = 0
_TESTDATA: dict = {}


class Backtesting(Exchange):
    @property
    def sleep_time(self) -> float:
        return 0

    def __init__(self, config: dict, testdata_dir: Optional[str] = TESTDATA_DIR) -> None:
        global _ROW_INDEX, _ROW_INTERVAL, _LEN_ROWS, _TESTDATA

        # Disable debug logs for a quicker run
        # logging.disable(logging.DEBUG)

        # Get pairs from test data directory and inject into config replacing existing ones
        (files, pairs) = _get_testdata_pairs(testdata_dir)
        config['pair_whitelist'] = pairs

        # Load the test data for each pair
        (_TESTDATA, _LEN_ROWS) = _get_testdata(files, pairs)

        # Set first row according to shift time
        # to have some ticker history available for the first analysis
        _ROW_INDEX = _initial_row_index(_TESTDATA[pairs[0]]['result'], TICKER_HISTORY_INTERVAL_H)
        _ROW_INTERVAL = _ROW_INDEX
        if _ROW_INDEX >= _LEN_ROWS or _ROW_INDEX == 0:
            raise RuntimeError('Test data is not usable with the current row interval')

    def buy(self, pair: str, rate: float, amount: float) -> str:
        return 'backtesting'

    def sell(self, pair: str, rate: float, amount: float) -> str:
        return 'backtesting'

    def get_balance(self, currency: str) -> float:
        return 999.9

    def get_ticker(self, pair: str) -> dict:
        row = _TESTDATA[pair]['result'][_ROW_INDEX]
        return {'bid': row['C'], 'ask': row['C'], 'last': row['C']}

    def get_ticker_history(self, pair: str, minimum_date: Optional[arrow.Arrow] = None):
        minimum = _ROW_INDEX - _ROW_INTERVAL
        maximum = _ROW_INDEX
        return \
            {'success': True, 'message': '', 'result': _TESTDATA[pair]['result'][minimum:maximum]}

    def cancel_order(self, order_id: str) -> None:
        return

    def get_open_orders(self, pair: str) -> List[dict]:
        return []

    def get_pair_detail_url(self, pair: str) -> str:
        return ''

    def get_markets(self) -> List[str]:
        return []


def _get_testdata_pairs(directory: str) -> (List[str], List[str]):
    """
    Returns a file and a pair list for a given test data directory.
    :param directory: Test data directory containing JSON files
    :return: File list with paths relative to the project directory, pair list e.g. ['BTC_ETC']
    """
    files = sorted(glob.glob(os.path.join(directory, '*.json')))
    pairs = sorted([splitext(basename(p))[0].replace('-', '_').upper() for p in files])
    return files, pairs


def _get_testdata(files: List[str], pairs) -> (dict, int):
    """
    Returns test data content from a given file list.
    :param files: JSON file list with paths relative to the project directory
    :param pairs: Pair list, e.g. ['BTC_ETC']
    :return: Test data dict with currency pairs as keys, maximal common testdata length for pairs
    """
    testdata_lengths = []  # Find a common maximum for all testdata pairs
    testdata = {}
    for (file, pair) in zip(files, pairs):
        with open(file) as f:
            testdata[pair] = json.load(f)
            testdata_lengths.append(len(testdata[pair]['result']))
    len_rows = min(testdata_lengths)
    return testdata, len_rows


def _initial_row_index(ticker_history: dict, ticker_history_interval: float) -> int:
    """
    Calculates the initial row index to have a buffer for the first signal analysis.
    :param ticker_history: Ticker history of a currency pair
    :param ticker_history_interval: Time interval used during the ticker analysis
    :return: Calculated initial row index
    """
    result = 0
    t0 = arrow.get(ticker_history[0]['T'])
    t1 = t0.shift(hours=ticker_history_interval)
    for i, row in enumerate(ticker_history):
        row_time = arrow.get(row['T'])
        if row_time >= t1:
            result = i
            break
    return result


def time_step() -> bool:
    """
    Advances in time or rather increases the row counter by 1 (one).
    :return: Success status - False if the end of a data set is reached
    """
    global _ROW_INDEX

    time = _ROW_INDEX
    _ROW_INDEX += 1
    logger.debug('Row: %s/%s', time, _LEN_ROWS)
    if time >= _LEN_ROWS:  # Backtesting complete
        return False
    return True


def get_minimum_date(pair: str) -> datetime:
    """
    Subtracts the ticker history interval (e.g. 24 hours) from the current time.
    :param pair: Pair as str, format: BTC_ETH
    :return: datetime
    """
    ticker_history = _TESTDATA[pair]['result']
    minimum_row = _ROW_INDEX - _ROW_INTERVAL
    minimum_date = ticker_history[minimum_row]['T']
    return minimum_date


def current_time(pair: str) -> datetime:
    """
    Gets the time stored in the current row of a data set for a given currency pair.
    :param pair: Pair as str, format: BTC_ETC
    :return: datetime
    """
    return arrow.get(_TESTDATA[pair]['result'][_ROW_INDEX]['T']).datetime.replace(tzinfo=None)


# TODO: Common statistic methods for telegram and backtesting
def print_results() -> None:
    """
    Prints cumulative profit statistics.
    :return: None
    """
    trades = Trade.query.order_by(Trade.id).all()

    profit_amounts = []
    profits = []
    durations = []
    for trade in trades:
        if trade.close_date:
            durations.append((trade.close_date - trade.open_date).total_seconds())
        if trade.close_profit:
            profit = trade.close_profit
        else:
            # Get current rate
            current_rate = exchange.get_ticker(trade.pair)['bid']
            profit = 100 * ((current_rate - trade.open_rate) / trade.open_rate)

        profit_amounts.append((profit / 100) * trade.stake_amount)
        profits.append(profit)

    best_pair = Trade.session.query(Trade.pair, func.sum(Trade.close_profit).label('profit_sum')) \
        .filter(Trade.is_open.is_(False)) \
        .group_by(Trade.pair) \
        .order_by(text('profit_sum DESC')) \
        .first()

    if not best_pair:
        logger.info('No closed trade')
        return

    bp_pair, bp_rate = best_pair
    msg = """
ROI: {profit_btc:.2f} ({profit:.2f}%)
Trade Count: {trade_count}
First Trade opened: {first_trade_date}
Latest Trade opened: {latest_trade_date}
Avg. Duration: {avg_duration}
Best Performing: {best_pair}: {best_rate:.2f}%
    """.format(
        profit_btc=round(sum(profit_amounts), 8),
        profit=round(sum(profits), 2),
        trade_count=len(trades),
        first_trade_date=arrow.get(trades[0].open_date).humanize(),
        latest_trade_date=arrow.get(trades[-1].open_date).humanize(),
        avg_duration=str(timedelta(seconds=sum(durations) / float(len(durations)))).split('.')[0],
        best_pair=bp_pair,
        best_rate=round(bp_rate, 2),
    )
    logger.info(msg)
