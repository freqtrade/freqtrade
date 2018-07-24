# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
import operator
from argparse import Namespace
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import arrow
from pandas import DataFrame
from tabulate import tabulate

import freqtrade.optimize as optimize
from freqtrade import DependencyException, constants
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.exchange import Exchange
from freqtrade.misc import file_dump_json
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import SellType
from freqtrade.strategy.resolver import IStrategy, StrategyResolver

logger = logging.getLogger(__name__)


class BacktestResult(NamedTuple):
    """
    NamedTuple Defining BacktestResults inputs.
    """
    pair: str
    profit_percent: float
    profit_abs: float
    open_time: datetime
    close_time: datetime
    open_index: int
    close_index: int
    trade_duration: float
    open_at_end: bool
    open_rate: float
    close_rate: float
    sell_reason: SellType


class Backtesting(object):
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.strategy: IStrategy = StrategyResolver(self.config).strategy
        self.ticker_interval = self.strategy.ticker_interval
        self.tickerdata_to_dataframe = self.strategy.tickerdata_to_dataframe
        self.populate_buy_trend = self.strategy.populate_buy_trend
        self.populate_sell_trend = self.strategy.populate_sell_trend

        # Reset keys for backtesting
        self.config['exchange']['key'] = ''
        self.config['exchange']['secret'] = ''
        self.config['exchange']['password'] = ''
        self.config['exchange']['uid'] = ''
        self.config['dry_run'] = True
        self.exchange = Exchange(self.config)
        self.fee = self.exchange.get_fee()

    @staticmethod
    def get_timeframe(data: Dict[str, DataFrame]) -> Tuple[arrow.Arrow, arrow.Arrow]:
        """
        Get the maximum timeframe for the given backtest data
        :param data: dictionary with preprocessed backtesting data
        :return: tuple containing min_date, max_date
        """
        timeframe = [
            (arrow.get(min(frame.date)), arrow.get(max(frame.date)))
            for frame in data.values()
        ]
        return min(timeframe, key=operator.itemgetter(0))[0], \
            max(timeframe, key=operator.itemgetter(1))[1]

    def _generate_text_table(self, data: Dict[str, Dict], results: DataFrame) -> str:
        """
        Generates and returns a text table for the given backtest data and the results dataframe
        :return: pretty printed table with tabulate as str
        """
        stake_currency = str(self.config.get('stake_currency'))

        floatfmt = ('s', 'd', '.2f', '.2f', '.8f', 'd', '.1f', '.1f')
        tabular_data = []
        headers = ['pair', 'buy count', 'avg profit %', 'cum profit %',
                   'total profit ' + stake_currency, 'avg duration', 'profit', 'loss']
        for pair in data:
            result = results[results.pair == pair]
            tabular_data.append([
                pair,
                len(result.index),
                result.profit_percent.mean() * 100.0,
                result.profit_percent.sum() * 100.0,
                result.profit_abs.sum(),
                str(timedelta(
                    minutes=round(result.trade_duration.mean()))) if not result.empty else '0:00',
                len(result[result.profit_abs > 0]),
                len(result[result.profit_abs < 0])
            ])

        # Append Total
        tabular_data.append([
            'TOTAL',
            len(results.index),
            results.profit_percent.mean() * 100.0,
            results.profit_percent.sum() * 100.0,
            results.profit_abs.sum(),
            str(timedelta(
                minutes=round(results.trade_duration.mean()))) if not results.empty else '0:00',
            len(results[results.profit_abs > 0]),
            len(results[results.profit_abs < 0])
        ])
        return tabulate(tabular_data, headers=headers, floatfmt=floatfmt, tablefmt="pipe")

    def _generate_text_table_sell_reason(self, data: Dict[str, Dict], results: DataFrame) -> str:
        """
        Generate small table outlining Backtest results
        """
        tabular_data = []
        headers = ['Sell Reason', 'Count']
        for reason, count in results['sell_reason'].value_counts().iteritems():
            tabular_data.append([reason.value,  count])
        return tabulate(tabular_data, headers=headers, tablefmt="pipe")

    def _store_backtest_result(self, recordfilename: Optional[str], results: DataFrame) -> None:

        records = [(t.pair, t.profit_percent, t.open_time.timestamp(),
                    t.close_time.timestamp(), t.open_index - 1, t.trade_duration,
                    t.open_rate, t.close_rate, t.open_at_end, t.sell_reason.value)
                   for index, t in results.iterrows()]

        if records:
            logger.info('Dumping backtest results to %s', recordfilename)
            file_dump_json(recordfilename, records)

    def _get_sell_trade_entry(
            self, pair: str, buy_row: DataFrame,
            partial_ticker: List, trade_count_lock: Dict, args: Dict) -> Optional[BacktestResult]:

        stake_amount = args['stake_amount']
        max_open_trades = args.get('max_open_trades', 0)
        trade = Trade(
            open_rate=buy_row.open,
            open_date=buy_row.date,
            stake_amount=stake_amount,
            amount=stake_amount / buy_row.open,
            fee_open=self.fee,
            fee_close=self.fee
        )

        # calculate win/lose forwards from buy point
        for sell_row in partial_ticker:
            if max_open_trades > 0:
                # Increase trade_count_lock for every iteration
                trade_count_lock[sell_row.date] = trade_count_lock.get(sell_row.date, 0) + 1

            buy_signal = sell_row.buy
            sell = self.strategy.should_sell(trade, sell_row.open, sell_row.date, buy_signal,
                                             sell_row.sell)
            if sell.sell_flag:

                return BacktestResult(pair=pair,
                                      profit_percent=trade.calc_profit_percent(rate=sell_row.open),
                                      profit_abs=trade.calc_profit(rate=sell_row.open),
                                      open_time=buy_row.date,
                                      close_time=sell_row.date,
                                      trade_duration=int((
                                          sell_row.date - buy_row.date).total_seconds() // 60),
                                      open_index=buy_row.Index,
                                      close_index=sell_row.Index,
                                      open_at_end=False,
                                      open_rate=buy_row.open,
                                      close_rate=sell_row.open,
                                      sell_reason=sell.sell_type
                                      )
        if partial_ticker:
            # no sell condition found - trade stil open at end of backtest period
            sell_row = partial_ticker[-1]
            btr = BacktestResult(pair=pair,
                                 profit_percent=trade.calc_profit_percent(rate=sell_row.open),
                                 profit_abs=trade.calc_profit(rate=sell_row.open),
                                 open_time=buy_row.date,
                                 close_time=sell_row.date,
                                 trade_duration=int((
                                     sell_row.date - buy_row.date).total_seconds() // 60),
                                 open_index=buy_row.Index,
                                 close_index=sell_row.Index,
                                 open_at_end=True,
                                 open_rate=buy_row.open,
                                 close_rate=sell_row.open,
                                 sell_reason=SellType.FORCE_SELL
                                 )
            logger.debug('Force_selling still open trade %s with %s perc - %s', btr.pair,
                         btr.profit_percent, btr.profit_abs)
            return btr
        return None

    def backtest(self, args: Dict) -> DataFrame:
        """
        Implements backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid, logging on this method

        :param args: a dict containing:
            stake_amount: btc amount to use for each trade
            processed: a processed dictionary with format {pair, data}
            max_open_trades: maximum number of concurrent trades (default: 0, disabled)
            position_stacking: do we allow position stacking? (default: False)
        :return: DataFrame
        """
        headers = ['date', 'buy', 'open', 'close', 'sell']
        processed = args['processed']
        max_open_trades = args.get('max_open_trades', 0)
        position_stacking = args.get('position_stacking', False)
        trades = []
        trade_count_lock: Dict = {}
        for pair, pair_data in processed.items():
            pair_data['buy'], pair_data['sell'] = 0, 0  # cleanup from previous run

            ticker_data = self.populate_sell_trend(
                self.populate_buy_trend(pair_data))[headers].copy()

            # to avoid using data from future, we buy/sell with signal from previous candle
            ticker_data.loc[:, 'buy'] = ticker_data['buy'].shift(1)
            ticker_data.loc[:, 'sell'] = ticker_data['sell'].shift(1)

            ticker_data.drop(ticker_data.head(1).index, inplace=True)

            # Convert from Pandas to list for performance reasons
            # (Looping Pandas is slow.)
            ticker = [x for x in ticker_data.itertuples()]

            lock_pair_until = None
            for index, row in enumerate(ticker):
                if row.buy == 0 or row.sell == 1:
                    continue  # skip rows where no buy signal or that would immediately sell off

                if not position_stacking:
                    if lock_pair_until is not None and row.date <= lock_pair_until:
                        continue
                if max_open_trades > 0:
                    # Check if max_open_trades has already been reached for the given date
                    if not trade_count_lock.get(row.date, 0) < max_open_trades:
                        continue

                    trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1

                trade_entry = self._get_sell_trade_entry(pair, row, ticker[index + 1:],
                                                         trade_count_lock, args)

                if trade_entry:
                    lock_pair_until = trade_entry.close_time
                    trades.append(trade_entry)
                else:
                    # Set lock_pair_until to end of testing period if trade could not be closed
                    # This happens only if the buy-signal was with the last candle
                    lock_pair_until = ticker_data.iloc[-1].date

        return DataFrame.from_records(trades, columns=BacktestResult._fields)

    def start(self) -> None:
        """
        Run a backtesting end-to-end
        :return: None
        """
        data = {}
        pairs = self.config['exchange']['pair_whitelist']
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using stake_amount: %s ...', self.config['stake_amount'])

        if self.config.get('live'):
            logger.info('Downloading data for all pairs in whitelist ...')
            for pair in pairs:
                data[pair] = self.exchange.get_ticker_history(pair, self.ticker_interval)
        else:
            logger.info('Using local backtesting data (using whitelist in given config) ...')

            timerange = Arguments.parse_timerange(None if self.config.get(
                'timerange') is None else str(self.config.get('timerange')))
            data = optimize.load_data(
                self.config['datadir'],
                pairs=pairs,
                ticker_interval=self.ticker_interval,
                refresh_pairs=self.config.get('refresh_pairs', False),
                exchange=self.exchange,
                timerange=timerange
            )

        if not data:
            logger.critical("No data found. Terminating.")
            return
        # Use max_open_trades in backtesting, except --disable-max-market-positions is set
        if self.config.get('use_max_market_positions', True):
            max_open_trades = self.config['max_open_trades']
        else:
            logger.info('Ignoring max_open_trades (--disable-max-market-positions was used) ...')
            max_open_trades = 0

        preprocessed = self.tickerdata_to_dataframe(data)

        # Print timeframe
        min_date, max_date = self.get_timeframe(preprocessed)
        logger.info(
            'Measuring data from %s up to %s (%s days)..',
            min_date.isoformat(),
            max_date.isoformat(),
            (max_date - min_date).days
        )

        # Execute backtest and print results
        results = self.backtest(
            {
                'stake_amount': self.config.get('stake_amount'),
                'processed': preprocessed,
                'max_open_trades': max_open_trades,
                'position_stacking': self.config.get('position_stacking', False),
            }
        )

        if self.config.get('export', False):
            self._store_backtest_result(self.config.get('exportfilename'), results)

        logger.info(
            '\n' + '=' * 49 +
            ' BACKTESTING REPORT ' +
            '=' * 50 + '\n'
            '%s',
            self._generate_text_table(
                data,
                results
            )
        )
        # logger.info(
        #     results[['sell_reason']].groupby('sell_reason').count()
        # )

        logger.info(
            '\n' +
            ' SELL READON STATS '.center(119, '=') +
            '\n%s \n',
            self._generate_text_table_sell_reason(data, results)

        )

        logger.info(
            '\n' +
            ' LEFT OPEN TRADES REPORT '.center(119, '=') +
            '\n%s',
            self._generate_text_table(
                data,
                results.loc[results.open_at_end]
            )
        )


def setup_configuration(args: Namespace) -> Dict[str, Any]:
    """
    Prepare the configuration for the backtesting
    :param args: Cli args from Arguments()
    :return: Configuration
    """
    configuration = Configuration(args)
    config = configuration.get_config()

    # Ensure we do not use Exchange credentials
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''

    if config['stake_amount'] == constants.UNLIMITED_STAKE_AMOUNT:
        raise DependencyException('stake amount could not be "%s" for backtesting' %
                                  constants.UNLIMITED_STAKE_AMOUNT)

    return config


def start(args: Namespace) -> None:
    """
    Start Backtesting script
    :param args: Cli args from Arguments()
    :return: None
    """
    # Initialize configuration
    config = setup_configuration(args)
    logger.info('Starting freqtrade in Backtesting mode')

    # Initialize backtesting object
    backtesting = Backtesting(config)
    backtesting.start()
