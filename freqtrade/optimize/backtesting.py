# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
from argparse import Namespace
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from pandas import DataFrame
from tabulate import tabulate

import freqtrade.optimize as optimize
from freqtrade import DependencyException, constants
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
from freqtrade.exchange import Exchange
from freqtrade.misc import file_dump_json
from freqtrade.persistence import Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy.interface import SellType, IStrategy

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

        # Reset keys for backtesting
        self.config['exchange']['key'] = ''
        self.config['exchange']['secret'] = ''
        self.config['exchange']['password'] = ''
        self.config['exchange']['uid'] = ''
        self.config['dry_run'] = True
        self.strategylist: List[IStrategy] = []
        if self.config.get('strategy_list', None):
            # Force one interval
            self.ticker_interval = str(self.config.get('ticker_interval'))
            self.ticker_interval_mins = constants.TICKER_INTERVAL_MINUTES[self.ticker_interval]
            for strat in list(self.config['strategy_list']):
                stratconf = deepcopy(self.config)
                stratconf['strategy'] = strat
                self.strategylist.append(StrategyResolver(stratconf).strategy)

        else:
            # only one strategy
            self.strategylist.append(StrategyResolver(self.config).strategy)
        # Load one strategy
        self._set_strategy(self.strategylist[0])

        self.exchange = Exchange(self.config)
        self.fee = self.exchange.get_fee()

    def _set_strategy(self, strategy):
        """
        Load strategy into backtesting
        """
        self.strategy = strategy
        self.ticker_interval = self.config.get('ticker_interval')
        self.ticker_interval_mins = constants.TICKER_INTERVAL_MINUTES[self.ticker_interval]
        self.tickerdata_to_dataframe = strategy.tickerdata_to_dataframe
        self.advise_buy = strategy.advise_buy
        self.advise_sell = strategy.advise_sell

    def _generate_text_table(self, data: Dict[str, Dict], results: DataFrame,
                             skip_nan: bool = False) -> str:
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
            if skip_nan and result.profit_abs.isnull().all():
                continue

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

    def _generate_text_table_strategy(self, all_results: dict) -> str:
        """
        Generate summary table per strategy
        """
        stake_currency = str(self.config.get('stake_currency'))

        floatfmt = ('s', 'd', '.2f', '.2f', '.8f', 'd', '.1f', '.1f')
        tabular_data = []
        headers = ['Strategy', 'buy count', 'avg profit %', 'cum profit %',
                   'total profit ' + stake_currency, 'avg duration', 'profit', 'loss']
        for strategy, results in all_results.items():
            tabular_data.append([
                strategy,
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

    def _store_backtest_result(self, recordfilename: str, results: DataFrame,
                               strategyname: Optional[str] = None) -> None:

        records = [(t.pair, t.profit_percent, t.open_time.timestamp(),
                    t.close_time.timestamp(), t.open_index - 1, t.trade_duration,
                    t.open_rate, t.close_rate, t.open_at_end, t.sell_reason.value)
                   for index, t in results.iterrows()]

        if records:
            if strategyname:
                # Inject strategyname to filename
                recname = Path(recordfilename)
                recordfilename = str(Path.joinpath(
                    recname.parent, f'{recname.stem}-{strategyname}').with_suffix(recname.suffix))
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
                                             sell_row.sell, low=sell_row.low, high=sell_row.high)
            if sell.sell_flag:

                trade_dur = int((sell_row.date - buy_row.date).total_seconds() // 60)
                # Special handling if high or low hit STOP_LOSS or ROI
                if sell.sell_type in (SellType.STOP_LOSS, SellType.TRAILING_STOP_LOSS):
                    # Set close_rate to stoploss
                    closerate = trade.stop_loss
                elif sell.sell_type == (SellType.ROI):
                    # get entry in min_roi >= to trade duration
                    roi_entry = max(list(filter(lambda x: trade_dur >= x,
                                                self.strategy.minimal_roi.keys())))
                    roi = self.strategy.minimal_roi[roi_entry]

                    # - (Expected abs profit + open_rate + open_fee) / (fee_close -1)
                    closerate = - (trade.open_rate * roi + trade.open_rate *
                                   (1 + trade.fee_open)) / (trade.fee_close - 1)
                else:
                    closerate = sell_row.open

                return BacktestResult(pair=pair,
                                      profit_percent=trade.calc_profit_percent(rate=closerate),
                                      profit_abs=trade.calc_profit(rate=closerate),
                                      open_time=buy_row.date,
                                      close_time=sell_row.date,
                                      trade_duration=trade_dur,
                                      open_index=buy_row.Index,
                                      close_index=sell_row.Index,
                                      open_at_end=False,
                                      open_rate=buy_row.open,
                                      close_rate=closerate,
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
        headers = ['date', 'buy', 'open', 'close', 'sell', 'low', 'high']
        processed = args['processed']
        max_open_trades = args.get('max_open_trades', 0)
        position_stacking = args.get('position_stacking', False)
        start_date = args['start_date']
        end_date = args['end_date']
        trades = []
        trade_count_lock: Dict = {}
        ticker: Dict = {}
        pairs = []
        # Create ticker dict
        for pair, pair_data in processed.items():
            pair_data['buy'], pair_data['sell'] = 0, 0  # cleanup from previous run

            ticker_data = self.advise_sell(
                self.advise_buy(pair_data, {'pair': pair}), {'pair': pair})[headers].copy()

            # to avoid using data from future, we buy/sell with signal from previous candle
            ticker_data.loc[:, 'buy'] = ticker_data['buy'].shift(1)
            ticker_data.loc[:, 'sell'] = ticker_data['sell'].shift(1)

            ticker_data.drop(ticker_data.head(1).index, inplace=True)

            # Convert from Pandas to list for performance reasons
            # (Looping Pandas is slow.)
            ticker[pair] = [x for x in ticker_data.itertuples()]
            pairs.append(pair)

        lock_pair_until: Dict = {}
        tmp = start_date + timedelta(minutes=self.ticker_interval_mins)
        index = 0
        # Loop timerange and test per pair
        while tmp < end_date:
            # print(f"time: {tmp}")
            for i, pair in enumerate(ticker):
                try:
                    row = ticker[pair][index]
                except IndexError:
                    # missing Data for one pair ...
                    # Warnings for this are shown by `validate_backtest_data`
                    continue

                if row.buy == 0 or row.sell == 1:
                    continue  # skip rows where no buy signal or that would immediately sell off

                if not position_stacking:
                    if pair in lock_pair_until and row.date <= lock_pair_until[pair]:
                        continue
                if max_open_trades > 0:
                    # Check if max_open_trades has already been reached for the given date
                    if not trade_count_lock.get(row.date, 0) < max_open_trades:
                        continue

                    trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1

                trade_entry = self._get_sell_trade_entry(pair, row, ticker[pair][index + 1:],
                                                         trade_count_lock, args)

                if trade_entry:
                    lock_pair_until[pair] = trade_entry.close_time
                    trades.append(trade_entry)
                else:
                    # Set lock_pair_until to end of testing period if trade could not be closed
                    # This happens only if the buy-signal was with the last candle
                    lock_pair_until[pair] = end_date

            tmp += timedelta(minutes=self.ticker_interval_mins)
            index += 1
        return DataFrame.from_records(trades, columns=BacktestResult._fields)

    def start(self) -> None:
        """
        Run a backtesting end-to-end
        :return: None
        """
        data: Dict[str, Any] = {}
        pairs = self.config['exchange']['pair_whitelist']
        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using stake_amount: %s ...', self.config['stake_amount'])

        if self.config.get('live'):
            logger.info('Downloading data for all pairs in whitelist ...')
            self.exchange.refresh_tickers(pairs, self.ticker_interval)
            data = self.exchange.klines
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
        all_results = {}

        for strat in self.strategylist:
            logger.info("Running backtesting for Strategy %s", strat.get_strategy_name())
            self._set_strategy(strat)

            # need to reprocess data every time to populate signals
            preprocessed = self.strategy.tickerdata_to_dataframe(data)

            min_date, max_date = optimize.get_timeframe(preprocessed)
            # Validate dataframe for missing values
            optimize.validate_backtest_data(preprocessed, min_date, max_date,
                                            constants.TICKER_INTERVAL_MINUTES[self.ticker_interval])
            logger.info(
                'Measuring data from %s up to %s (%s days)..',
                min_date.isoformat(),
                max_date.isoformat(),
                (max_date - min_date).days
            )

            # Execute backtest and print results
            all_results[self.strategy.get_strategy_name()] = self.backtest(
                {
                    'stake_amount': self.config.get('stake_amount'),
                    'processed': preprocessed,
                    'max_open_trades': max_open_trades,
                    'position_stacking': self.config.get('position_stacking', False),
                    'start_date': min_date,
                    'end_date': max_date,
                }
            )

        for strategy, results in all_results.items():

            if self.config.get('export', False):
                self._store_backtest_result(self.config['exportfilename'], results,
                                            strategy if len(self.strategylist) > 1 else None)

            print(f"Result for strategy {strategy}")
            print(' BACKTESTING REPORT '.center(119, '='))
            print(self._generate_text_table(data, results))

            print(' SELL REASON STATS '.center(119, '='))
            print(self._generate_text_table_sell_reason(data, results))

            print(' LEFT OPEN TRADES REPORT '.center(119, '='))
            print(self._generate_text_table(data, results.loc[results.open_at_end], True))
            print()
        if len(all_results) > 1:
            # Print Strategy summary table
            print(' Strategy Summary '.center(119, '='))
            print(self._generate_text_table_strategy(all_results))
            print('\nFor more details, please look at the detail tables above')


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
