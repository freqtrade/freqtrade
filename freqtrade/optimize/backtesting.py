# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import arrow
from pandas import DataFrame

from freqtrade.configuration import (TimeRange, remove_credentials,
                                     validate_config_consistency)
from freqtrade.data import history
from freqtrade.data.converter import trim_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_seconds
from freqtrade.optimize.optimize_reports import (show_backtest_results,
                                                 store_backtest_result)
from freqtrade.pairlist.pairlistmanager import PairListManager
from freqtrade.persistence import Trade
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.state import RunMode
from freqtrade.strategy.interface import IStrategy, SellCheckTuple, SellType

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


class Backtesting:
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Reset keys for backtesting
        remove_credentials(self.config)
        self.strategylist: List[IStrategy] = []
        self.exchange = ExchangeResolver.load_exchange(self.config['exchange']['name'], self.config)

        self.pairlists = PairListManager(self.exchange, self.config)
        if 'VolumePairList' in self.pairlists.name_list:
            raise OperationalException("VolumePairList not allowed for backtesting.")

        self.pairlists.refresh_pairlist()

        if len(self.pairlists.whitelist) == 0:
            raise OperationalException("No pair in whitelist.")

        if config.get('fee'):
            self.fee = config['fee']
        else:
            self.fee = self.exchange.get_fee(symbol=self.pairlists.whitelist[0])

        if self.config.get('runmode') != RunMode.HYPEROPT:
            self.dataprovider = DataProvider(self.config, self.exchange)
            IStrategy.dp = self.dataprovider

        if self.config.get('strategy_list', None):
            for strat in list(self.config['strategy_list']):
                stratconf = deepcopy(self.config)
                stratconf['strategy'] = strat
                self.strategylist.append(StrategyResolver.load_strategy(stratconf))
                validate_config_consistency(stratconf)

        else:
            # No strategy list specified, only one strategy
            self.strategylist.append(StrategyResolver.load_strategy(self.config))
            validate_config_consistency(self.config)

        if "ticker_interval" not in self.config:
            raise OperationalException("Timeframe (ticker interval) needs to be set in either "
                                       "configuration or as cli argument `--ticker-interval 5m`")
        self.timeframe = str(self.config.get('ticker_interval'))
        self.timeframe_min = timeframe_to_minutes(self.timeframe)

        # Get maximum required startup period
        self.required_startup = max([strat.startup_candle_count for strat in self.strategylist])
        # Load one (first) strategy
        self._set_strategy(self.strategylist[0])

    def _set_strategy(self, strategy):
        """
        Load strategy into backtesting
        """
        self.strategy = strategy
        # Set stoploss_on_exchange to false for backtesting,
        # since a "perfect" stoploss-sell is assumed anyway
        # And the regular "stoploss" function would not apply to that case
        self.strategy.order_types['stoploss_on_exchange'] = False

    def load_bt_data(self) -> Tuple[Dict[str, DataFrame], TimeRange]:
        timerange = TimeRange.parse_timerange(None if self.config.get(
            'timerange') is None else str(self.config.get('timerange')))

        data = history.load_data(
            datadir=self.config['datadir'],
            pairs=self.pairlists.whitelist,
            timeframe=self.timeframe,
            timerange=timerange,
            startup_candles=self.required_startup,
            fail_without_data=True,
            data_format=self.config.get('dataformat_ohlcv', 'json'),
        )

        min_date, max_date = history.get_timerange(data)

        logger.info(
            'Loading data from %s up to %s (%s days)..',
            min_date.isoformat(), max_date.isoformat(), (max_date - min_date).days
        )
        # Adjust startts forward if not enough data is available
        timerange.adjust_start_if_necessary(timeframe_to_seconds(self.timeframe),
                                            self.required_startup, min_date)

        return data, timerange

    def _get_ohlcv_as_lists(self, processed: Dict) -> Dict[str, DataFrame]:
        """
        Helper function to convert a processed dataframes into lists for performance reasons.

        Used by backtest() - so keep this optimized for performance.
        """
        headers = ['date', 'buy', 'open', 'close', 'sell', 'low', 'high']
        data: Dict = {}
        # Create dict with data
        for pair, pair_data in processed.items():
            pair_data.loc[:, 'buy'] = 0  # cleanup from previous run
            pair_data.loc[:, 'sell'] = 0  # cleanup from previous run

            df_analyzed = self.strategy.advise_sell(
                self.strategy.advise_buy(pair_data, {'pair': pair}), {'pair': pair})[headers].copy()

            # To avoid using data from future, we use buy/sell signals shifted
            # from the previous candle
            df_analyzed.loc[:, 'buy'] = df_analyzed.loc[:, 'buy'].shift(1)
            df_analyzed.loc[:, 'sell'] = df_analyzed.loc[:, 'sell'].shift(1)

            df_analyzed.drop(df_analyzed.head(1).index, inplace=True)

            # Convert from Pandas to list for performance reasons
            # (Looping Pandas is slow.)
            data[pair] = [x for x in df_analyzed.itertuples()]
        return data

    def _get_close_rate(self, sell_row, trade: Trade, sell: SellCheckTuple,
                        trade_dur: int) -> float:
        """
        Get close rate for backtesting result
        """
        # Special handling if high or low hit STOP_LOSS or ROI
        if sell.sell_type in (SellType.STOP_LOSS, SellType.TRAILING_STOP_LOSS):
            # Set close_rate to stoploss
            return trade.stop_loss
        elif sell.sell_type == (SellType.ROI):
            roi_entry, roi = self.strategy.min_roi_reached_entry(trade_dur)
            if roi is not None:
                if roi == -1 and roi_entry % self.timeframe_min == 0:
                    # When forceselling with ROI=-1, the roi time will always be equal to trade_dur.
                    # If that entry is a multiple of the timeframe (so on candle open)
                    # - we'll use open instead of close
                    return sell_row.open

                # - (Expected abs profit + open_rate + open_fee) / (fee_close -1)
                close_rate = - (trade.open_rate * roi + trade.open_rate *
                                (1 + trade.fee_open)) / (trade.fee_close - 1)

                if (trade_dur > 0 and trade_dur == roi_entry
                        and roi_entry % self.timeframe_min == 0
                        and sell_row.open > close_rate):
                    # new ROI entry came into effect.
                    # use Open rate if open_rate > calculated sell rate
                    return sell_row.open

                # Use the maximum between close_rate and low as we
                # cannot sell outside of a candle.
                # Applies when a new ROI setting comes in place and the whole candle is above that.
                return max(close_rate, sell_row.low)

            else:
                # This should not be reached...
                return sell_row.open
        else:
            return sell_row.open

    def _get_sell_trade_entry(
            self, pair: str, buy_row: DataFrame,
            partial_ohlcv: List, trade_count_lock: Dict,
            stake_amount: float, max_open_trades: int) -> Optional[BacktestResult]:

        trade = Trade(
            pair=pair,
            open_rate=buy_row.open,
            open_date=buy_row.date,
            stake_amount=stake_amount,
            amount=stake_amount / buy_row.open,
            fee_open=self.fee,
            fee_close=self.fee,
            is_open=True,
        )
        logger.debug(f"{pair} - Backtesting emulates creation of new trade: {trade}.")
        # calculate win/lose forwards from buy point
        for sell_row in partial_ohlcv:
            if max_open_trades > 0:
                # Increase trade_count_lock for every iteration
                trade_count_lock[sell_row.date] = trade_count_lock.get(sell_row.date, 0) + 1

            sell = self.strategy.should_sell(trade, sell_row.open, sell_row.date, sell_row.buy,
                                             sell_row.sell, low=sell_row.low, high=sell_row.high)
            if sell.sell_flag:
                trade_dur = int((sell_row.date - buy_row.date).total_seconds() // 60)
                closerate = self._get_close_rate(sell_row, trade, sell, trade_dur)

                return BacktestResult(pair=pair,
                                      profit_percent=trade.calc_profit_ratio(rate=closerate),
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
        if partial_ohlcv:
            # no sell condition found - trade stil open at end of backtest period
            sell_row = partial_ohlcv[-1]
            bt_res = BacktestResult(pair=pair,
                                    profit_percent=trade.calc_profit_ratio(rate=sell_row.open),
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
            logger.debug(f"{pair} - Force selling still open trade, "
                         f"profit percent: {bt_res.profit_percent}, "
                         f"profit abs: {bt_res.profit_abs}")

            return bt_res
        return None

    def backtest(self, processed: Dict, stake_amount: float,
                 start_date: arrow.Arrow, end_date: arrow.Arrow,
                 max_open_trades: int = 0, position_stacking: bool = False) -> DataFrame:
        """
        Implement backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid extensive logging in this method and functions it calls.

        :param processed: a processed dictionary with format {pair, data}
        :param stake_amount: amount to use for each trade
        :param start_date: backtesting timerange start datetime
        :param end_date: backtesting timerange end datetime
        :param max_open_trades: maximum number of concurrent trades, <= 0 means unlimited
        :param position_stacking: do we allow position stacking?
        :return: DataFrame with trades (results of backtesting)
        """
        logger.debug(f"Run backtest, stake_amount: {stake_amount}, "
                     f"start_date: {start_date}, end_date: {end_date}, "
                     f"max_open_trades: {max_open_trades}, position_stacking: {position_stacking}"
                     )
        trades = []
        trade_count_lock: Dict = {}

        # Use dict of lists with data for performance
        # (looping lists is a lot faster than pandas DataFrames)
        data: Dict = self._get_ohlcv_as_lists(processed)

        lock_pair_until: Dict = {}
        # Indexes per pair, so some pairs are allowed to have a missing start.
        indexes: Dict = {}
        tmp = start_date + timedelta(minutes=self.timeframe_min)

        # Loop timerange and get candle for each pair at that point in time
        while tmp < end_date:

            for i, pair in enumerate(data):
                if pair not in indexes:
                    indexes[pair] = 0

                try:
                    row = data[pair][indexes[pair]]
                except IndexError:
                    # missing Data for one pair at the end.
                    # Warnings for this are shown during data loading
                    continue

                # Waits until the time-counter reaches the start of the data for this pair.
                if row.date > tmp.datetime:
                    continue

                indexes[pair] += 1

                if row.buy == 0 or row.sell == 1:
                    continue  # skip rows where no buy signal or that would immediately sell off

                if (not position_stacking and pair in lock_pair_until
                        and row.date <= lock_pair_until[pair]):
                    # without positionstacking, we can only have one open trade per pair.
                    continue

                if max_open_trades > 0:
                    # Check if max_open_trades has already been reached for the given date
                    if not trade_count_lock.get(row.date, 0) < max_open_trades:
                        continue
                    trade_count_lock[row.date] = trade_count_lock.get(row.date, 0) + 1

                # since indexes has been incremented before, we need to go one step back to
                # also check the buying candle for sell conditions.
                trade_entry = self._get_sell_trade_entry(pair, row, data[pair][indexes[pair]-1:],
                                                         trade_count_lock, stake_amount,
                                                         max_open_trades)

                if trade_entry:
                    logger.debug(f"{pair} - Locking pair till "
                                 f"close_time={trade_entry.close_time}")
                    lock_pair_until[pair] = trade_entry.close_time
                    trades.append(trade_entry)
                else:
                    # Set lock_pair_until to end of testing period if trade could not be closed
                    lock_pair_until[pair] = end_date.datetime

            # Move time one configured time_interval ahead.
            tmp += timedelta(minutes=self.timeframe_min)
        return DataFrame.from_records(trades, columns=BacktestResult._fields)

    def start(self) -> None:
        """
        Run backtesting end-to-end
        :return: None
        """
        data: Dict[str, Any] = {}

        logger.info('Using stake_currency: %s ...', self.config['stake_currency'])
        logger.info('Using stake_amount: %s ...', self.config['stake_amount'])

        # Use max_open_trades in backtesting, except --disable-max-market-positions is set
        if self.config.get('use_max_market_positions', True):
            max_open_trades = self.config['max_open_trades']
        else:
            logger.info('Ignoring max_open_trades (--disable-max-market-positions was used) ...')
            max_open_trades = 0
        position_stacking = self.config.get('position_stacking', False)

        data, timerange = self.load_bt_data()

        all_results = {}
        for strat in self.strategylist:
            logger.info("Running backtesting for Strategy %s", strat.get_strategy_name())
            self._set_strategy(strat)

            # need to reprocess data every time to populate signals
            preprocessed = self.strategy.ohlcvdata_to_dataframe(data)

            # Trim startup period from analyzed dataframe
            for pair, df in preprocessed.items():
                preprocessed[pair] = trim_dataframe(df, timerange)
            min_date, max_date = history.get_timerange(preprocessed)

            logger.info(
                'Backtesting with data from %s up to %s (%s days)..',
                min_date.isoformat(), max_date.isoformat(), (max_date - min_date).days
            )
            # Execute backtest and print results
            all_results[self.strategy.get_strategy_name()] = self.backtest(
                processed=preprocessed,
                stake_amount=self.config['stake_amount'],
                start_date=min_date,
                end_date=max_date,
                max_open_trades=max_open_trades,
                position_stacking=position_stacking,
            )

        if self.config.get('export', False):
            store_backtest_result(self.config['exportfilename'], all_results)
        # Show backtest results
        show_backtest_results(self.config, data, all_results)
