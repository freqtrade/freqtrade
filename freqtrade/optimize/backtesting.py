# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from pandas import DataFrame

from freqtrade.configuration import TimeRange, validate_config_consistency
from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.data import history
from freqtrade.data.btanalysis import trade_list_to_dataframe
from freqtrade.data.converter import trim_dataframe, trim_dataframes
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import BacktestState, SellType
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_seconds
from freqtrade.mixins import LoggingMixin
from freqtrade.optimize.bt_progress import BTProgress
from freqtrade.optimize.optimize_reports import (generate_backtest_stats, show_backtest_results,
                                                 store_backtest_stats)
from freqtrade.persistence import LocalTrade, PairLocks, Trade
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.plugins.protectionmanager import ProtectionManager
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy.interface import IStrategy, SellCheckTuple
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)

# Indexes for backtest tuples
DATE_IDX = 0
BUY_IDX = 1
OPEN_IDX = 2
CLOSE_IDX = 3
SELL_IDX = 4
LOW_IDX = 5
HIGH_IDX = 6
BUY_TAG_IDX = 7


class Backtesting:
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """

    def __init__(self, config: Dict[str, Any]) -> None:

        LoggingMixin.show_output = False
        self.config = config
        self.results: Optional[Dict[str, Any]] = None

        config['dry_run'] = True
        self.strategylist: List[IStrategy] = []
        self.all_results: Dict[str, Dict] = {}

        self.exchange = ExchangeResolver.load_exchange(self.config['exchange']['name'], self.config)
        self.dataprovider = DataProvider(self.config, None)

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

        if "timeframe" not in self.config:
            raise OperationalException("Timeframe (ticker interval) needs to be set in either "
                                       "configuration or as cli argument `--timeframe 5m`")
        self.timeframe = str(self.config.get('timeframe'))
        self.timeframe_min = timeframe_to_minutes(self.timeframe)
        self.init_backtest_detail()
        self.pairlists = PairListManager(self.exchange, self.config)
        if 'VolumePairList' in self.pairlists.name_list:
            raise OperationalException("VolumePairList not allowed for backtesting.")
        if 'PerformanceFilter' in self.pairlists.name_list:
            raise OperationalException("PerformanceFilter not allowed for backtesting.")

        if len(self.strategylist) > 1 and 'PrecisionFilter' in self.pairlists.name_list:
            raise OperationalException(
                "PrecisionFilter not allowed for backtesting multiple strategies."
            )

        self.dataprovider.add_pairlisthandler(self.pairlists)
        self.pairlists.refresh_pairlist()

        if len(self.pairlists.whitelist) == 0:
            raise OperationalException("No pair in whitelist.")

        if config.get('fee', None) is not None:
            self.fee = config['fee']
        else:
            self.fee = self.exchange.get_fee(symbol=self.pairlists.whitelist[0])

        self.timerange = TimeRange.parse_timerange(
            None if self.config.get('timerange') is None else str(self.config.get('timerange')))

        # Get maximum required startup period
        self.required_startup = max([strat.startup_candle_count for strat in self.strategylist])
        # Add maximum startup candle count to configuration for informative pairs support
        self.config['startup_candle_count'] = self.required_startup
        self.exchange.validate_required_startup_candles(self.required_startup, self.timeframe)
        self.init_backtest()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        LoggingMixin.show_output = True
        PairLocks.use_db = True
        Trade.use_db = True

    def init_backtest_detail(self):
        # Load detail timeframe if specified
        self.timeframe_detail = str(self.config.get('timeframe_detail', ''))
        if self.timeframe_detail:
            self.timeframe_detail_min = timeframe_to_minutes(self.timeframe_detail)
            if self.timeframe_min <= self.timeframe_detail_min:
                raise OperationalException(
                    "Detail timeframe must be smaller than strategy timeframe.")

        else:
            self.timeframe_detail_min = 0
        self.detail_data: Dict[str, DataFrame] = {}

    def init_backtest(self):

        self.prepare_backtest(False)

        self.wallets = Wallets(self.config, self.exchange, log=False)

        self.progress = BTProgress()
        self.abort = False

    def _set_strategy(self, strategy: IStrategy):
        """
        Load strategy into backtesting
        """
        self.strategy: IStrategy = strategy
        strategy.dp = self.dataprovider
        # Attach Wallets to Strategy baseclass
        strategy.wallets = self.wallets
        # Set stoploss_on_exchange to false for backtesting,
        # since a "perfect" stoploss-sell is assumed anyway
        # And the regular "stoploss" function would not apply to that case
        self.strategy.order_types['stoploss_on_exchange'] = False

    def _load_protections(self, strategy: IStrategy):
        if self.config.get('enable_protections', False):
            conf = self.config
            if hasattr(strategy, 'protections'):
                conf = deepcopy(conf)
                conf['protections'] = strategy.protections
            self.protections = ProtectionManager(self.config, strategy.protections)

    def load_bt_data(self) -> Tuple[Dict[str, DataFrame], TimeRange]:
        """
        Loads backtest data and returns the data combined with the timerange
        as tuple.
        """
        self.progress.init_step(BacktestState.DATALOAD, 1)

        data = history.load_data(
            datadir=self.config['datadir'],
            pairs=self.pairlists.whitelist,
            timeframe=self.timeframe,
            timerange=self.timerange,
            startup_candles=self.required_startup,
            fail_without_data=True,
            data_format=self.config.get('dataformat_ohlcv', 'json'),
        )

        min_date, max_date = history.get_timerange(data)

        logger.info(f'Loading data from {min_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'up to {max_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'({(max_date - min_date).days} days).')

        # Adjust startts forward if not enough data is available
        self.timerange.adjust_start_if_necessary(timeframe_to_seconds(self.timeframe),
                                                 self.required_startup, min_date)

        self.progress.set_new_value(1)
        return data, self.timerange

    def load_bt_data_detail(self) -> None:
        """
        Loads backtest detail data (smaller timeframe) if necessary.
        """
        if self.timeframe_detail:
            self.detail_data = history.load_data(
                datadir=self.config['datadir'],
                pairs=self.pairlists.whitelist,
                timeframe=self.timeframe_detail,
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config.get('dataformat_ohlcv', 'json'),
            )
        else:
            self.detail_data = {}

    def prepare_backtest(self, enable_protections):
        """
        Backtesting setup method - called once for every call to "backtest()".
        """
        PairLocks.use_db = False
        PairLocks.timeframe = self.config['timeframe']
        Trade.use_db = False
        PairLocks.reset_locks()
        Trade.reset_trades()
        self.rejected_trades = 0
        self.dataprovider.clear_cache()
        if enable_protections:
            self._load_protections(self.strategy)

    def check_abort(self):
        """
        Check if abort was requested, raise DependencyException if that's the case
        Only applies to Interactive backtest mode (webserver mode)
        """
        if self.abort:
            self.abort = False
            raise DependencyException("Stop requested")

    def _get_ohlcv_as_lists(self, processed: Dict[str, DataFrame]) -> Dict[str, Tuple]:
        """
        Helper function to convert a processed dataframes into lists for performance reasons.

        Used by backtest() - so keep this optimized for performance.
        """
        # Every change to this headers list must evaluate further usages of the resulting tuple
        # and eventually change the constants for indexes at the top
        headers = ['date', 'buy', 'open', 'close', 'sell', 'low', 'high', 'buy_tag']
        data: Dict = {}
        self.progress.init_step(BacktestState.CONVERT, len(processed))

        # Create dict with data
        for pair, pair_data in processed.items():
            self.check_abort()
            self.progress.increment()
            if not pair_data.empty:
                pair_data.loc[:, 'buy'] = 0  # cleanup if buy_signal is exist
                pair_data.loc[:, 'sell'] = 0  # cleanup if sell_signal is exist
                pair_data.loc[:, 'buy_tag'] = None  # cleanup if buy_tag is exist

            df_analyzed = self.strategy.advise_sell(
                self.strategy.advise_buy(pair_data, {'pair': pair}), {'pair': pair}).copy()
            # Trim startup period from analyzed dataframe
            df_analyzed = trim_dataframe(df_analyzed, self.timerange,
                                         startup_candles=self.required_startup)
            # To avoid using data from future, we use buy/sell signals shifted
            # from the previous candle
            df_analyzed.loc[:, 'buy'] = df_analyzed.loc[:, 'buy'].shift(1)
            df_analyzed.loc[:, 'sell'] = df_analyzed.loc[:, 'sell'].shift(1)
            df_analyzed.loc[:, 'buy_tag'] = df_analyzed.loc[:, 'buy_tag'].shift(1)

            # Update dataprovider cache
            self.dataprovider._set_cached_df(pair, self.timeframe, df_analyzed)

            df_analyzed = df_analyzed.drop(df_analyzed.head(1).index)

            # Convert from Pandas to list for performance reasons
            # (Looping Pandas is slow.)
            data[pair] = df_analyzed[headers].values.tolist()
        return data

    def _get_close_rate(self, sell_row: Tuple, trade: LocalTrade, sell: SellCheckTuple,
                        trade_dur: int) -> float:
        """
        Get close rate for backtesting result
        """
        # Special handling if high or low hit STOP_LOSS or ROI
        if sell.sell_type in (SellType.STOP_LOSS, SellType.TRAILING_STOP_LOSS):
            if trade.stop_loss > sell_row[HIGH_IDX]:
                # our stoploss was already higher than candle high,
                # possibly due to a cancelled trade exit.
                # sell at open price.
                return sell_row[OPEN_IDX]

            # Special case: trailing triggers within same candle as trade opened. Assume most
            # pessimistic price movement, which is moving just enough to arm stoploss and
            # immediately going down to stop price.
            if sell.sell_type == SellType.TRAILING_STOP_LOSS and trade_dur == 0:
                if (
                    not self.strategy.use_custom_stoploss and self.strategy.trailing_stop
                    and self.strategy.trailing_only_offset_is_reached
                    and self.strategy.trailing_stop_positive_offset is not None
                    and self.strategy.trailing_stop_positive
                ):
                    # Worst case: price reaches stop_positive_offset and dives down.
                    stop_rate = (sell_row[OPEN_IDX] *
                                 (1 + abs(self.strategy.trailing_stop_positive_offset) -
                                  abs(self.strategy.trailing_stop_positive)))
                else:
                    # Worst case: price ticks tiny bit above open and dives down.
                    stop_rate = sell_row[OPEN_IDX] * (1 - abs(trade.stop_loss_pct))
                    assert stop_rate < sell_row[HIGH_IDX]
                # Limit lower-end to candle low to avoid sells below the low.
                # This still remains "worst case" - but "worst realistic case".
                return max(sell_row[LOW_IDX], stop_rate)

            # Set close_rate to stoploss
            return trade.stop_loss
        elif sell.sell_type == (SellType.ROI):
            roi_entry, roi = self.strategy.min_roi_reached_entry(trade_dur)
            if roi is not None and roi_entry is not None:
                if roi == -1 and roi_entry % self.timeframe_min == 0:
                    # When forceselling with ROI=-1, the roi time will always be equal to trade_dur.
                    # If that entry is a multiple of the timeframe (so on candle open)
                    # - we'll use open instead of close
                    return sell_row[OPEN_IDX]

                # - (Expected abs profit + open_rate + open_fee) / (fee_close -1)
                close_rate = - (trade.open_rate * roi + trade.open_rate *
                                (1 + trade.fee_open)) / (trade.fee_close - 1)

                if (trade_dur > 0 and trade_dur == roi_entry
                        and roi_entry % self.timeframe_min == 0
                        and sell_row[OPEN_IDX] > close_rate):
                    # new ROI entry came into effect.
                    # use Open rate if open_rate > calculated sell rate
                    return sell_row[OPEN_IDX]

                # Use the maximum between close_rate and low as we
                # cannot sell outside of a candle.
                # Applies when a new ROI setting comes in place and the whole candle is above that.
                return min(max(close_rate, sell_row[LOW_IDX]), sell_row[HIGH_IDX])

            else:
                # This should not be reached...
                return sell_row[OPEN_IDX]
        else:
            return sell_row[OPEN_IDX]

    def _get_sell_trade_entry_for_candle(self, trade: LocalTrade,
                                         sell_row: Tuple) -> Optional[LocalTrade]:
        sell_candle_time = sell_row[DATE_IDX].to_pydatetime()
        sell = self.strategy.should_sell(trade, sell_row[OPEN_IDX],  # type: ignore
                                         sell_candle_time, sell_row[BUY_IDX],
                                         sell_row[SELL_IDX],
                                         low=sell_row[LOW_IDX], high=sell_row[HIGH_IDX])

        if sell.sell_flag:
            trade.close_date = sell_candle_time
            trade.sell_reason = sell.sell_reason
            trade_dur = int((trade.close_date_utc - trade.open_date_utc).total_seconds() // 60)
            closerate = self._get_close_rate(sell_row, trade, sell, trade_dur)

            # Confirm trade exit:
            time_in_force = self.strategy.order_time_in_force['sell']
            if not strategy_safe_wrapper(self.strategy.confirm_trade_exit, default_retval=True)(
                    pair=trade.pair, trade=trade, order_type='limit', amount=trade.amount,
                    rate=closerate,
                    time_in_force=time_in_force,
                    sell_reason=sell.sell_reason,
                    current_time=sell_candle_time):
                return None

            trade.close(closerate, show_msg=False)
            return trade

        return None

    def _get_sell_trade_entry(self, trade: LocalTrade, sell_row: Tuple) -> Optional[LocalTrade]:
        if self.timeframe_detail and trade.pair in self.detail_data:
            sell_candle_time = sell_row[DATE_IDX].to_pydatetime()
            sell_candle_end = sell_candle_time + timedelta(minutes=self.timeframe_min)

            detail_data = self.detail_data[trade.pair]
            detail_data = detail_data.loc[
                (detail_data['date'] >= sell_candle_time) &
                (detail_data['date'] < sell_candle_end)
             ].copy()
            if len(detail_data) == 0:
                # Fall back to "regular" data if no detail data was found for this candle
                return self._get_sell_trade_entry_for_candle(trade, sell_row)
            detail_data.loc[:, 'buy'] = sell_row[BUY_IDX]
            detail_data.loc[:, 'sell'] = sell_row[SELL_IDX]
            headers = ['date', 'buy', 'open', 'close', 'sell', 'low', 'high']
            for det_row in detail_data[headers].values.tolist():
                res = self._get_sell_trade_entry_for_candle(trade, det_row)
                if res:
                    return res

            return None

        else:
            return self._get_sell_trade_entry_for_candle(trade, sell_row)

    def _enter_trade(self, pair: str, row: List) -> Optional[LocalTrade]:
        try:
            stake_amount = self.wallets.get_trade_stake_amount(pair, None)
        except DependencyException:
            return None

        min_stake_amount = self.exchange.get_min_pair_stake_amount(pair, row[OPEN_IDX], -0.05) or 0
        max_stake_amount = self.wallets.get_available_stake_amount()

        stake_amount = strategy_safe_wrapper(self.strategy.custom_stake_amount,
                                             default_retval=stake_amount)(
            pair=pair, current_time=row[DATE_IDX].to_pydatetime(), current_rate=row[OPEN_IDX],
            proposed_stake=stake_amount, min_stake=min_stake_amount, max_stake=max_stake_amount)
        stake_amount = self.wallets._validate_stake_amount(pair, stake_amount, min_stake_amount)

        if not stake_amount:
            return None

        order_type = self.strategy.order_types['buy']
        time_in_force = self.strategy.order_time_in_force['sell']
        # Confirm trade entry:
        if not strategy_safe_wrapper(self.strategy.confirm_trade_entry, default_retval=True)(
                pair=pair, order_type=order_type, amount=stake_amount, rate=row[OPEN_IDX],
                time_in_force=time_in_force, current_time=row[DATE_IDX].to_pydatetime()):
            return None

        if stake_amount and (not min_stake_amount or stake_amount > min_stake_amount):
            # Enter trade
            has_buy_tag = len(row) >= BUY_TAG_IDX + 1
            trade = LocalTrade(
                pair=pair,
                open_rate=row[OPEN_IDX],
                open_date=row[DATE_IDX].to_pydatetime(),
                stake_amount=stake_amount,
                amount=round(stake_amount / row[OPEN_IDX], 8),
                fee_open=self.fee,
                fee_close=self.fee,
                is_open=True,
                buy_tag=row[BUY_TAG_IDX] if has_buy_tag else None,
                exchange='backtesting',
            )
            return trade
        return None

    def handle_left_open(self, open_trades: Dict[str, List[LocalTrade]],
                         data: Dict[str, List[Tuple]]) -> List[LocalTrade]:
        """
        Handling of left open trades at the end of backtesting
        """
        trades = []
        for pair in open_trades.keys():
            if len(open_trades[pair]) > 0:
                for trade in open_trades[pair]:
                    sell_row = data[pair][-1]

                    trade.close_date = sell_row[DATE_IDX].to_pydatetime()
                    trade.sell_reason = SellType.FORCE_SELL.value
                    trade.close(sell_row[OPEN_IDX], show_msg=False)
                    LocalTrade.close_bt_trade(trade)
                    # Deepcopy object to have wallets update correctly
                    trade1 = deepcopy(trade)
                    trade1.is_open = True
                    trades.append(trade1)
        return trades

    def trade_slot_available(self, max_open_trades: int, open_trade_count: int) -> bool:
        # Always allow trades when max_open_trades is enabled.
        if max_open_trades <= 0 or open_trade_count < max_open_trades:
            return True
        # Rejected trade
        self.rejected_trades += 1
        return False

    def backtest(self, processed: Dict,
                 start_date: datetime, end_date: datetime,
                 max_open_trades: int = 0, position_stacking: bool = False,
                 enable_protections: bool = False) -> Dict[str, Any]:
        """
        Implement backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid extensive logging in this method and functions it calls.

        :param processed: a processed dictionary with format {pair, data}
        :param start_date: backtesting timerange start datetime
        :param end_date: backtesting timerange end datetime
        :param max_open_trades: maximum number of concurrent trades, <= 0 means unlimited
        :param position_stacking: do we allow position stacking?
        :param enable_protections: Should protections be enabled?
        :return: DataFrame with trades (results of backtesting)
        """
        trades: List[LocalTrade] = []
        self.prepare_backtest(enable_protections)

        # Use dict of lists with data for performance
        # (looping lists is a lot faster than pandas DataFrames)
        data: Dict = self._get_ohlcv_as_lists(processed)

        # Indexes per pair, so some pairs are allowed to have a missing start.
        indexes: Dict = defaultdict(int)
        tmp = start_date + timedelta(minutes=self.timeframe_min)

        open_trades: Dict[str, List[LocalTrade]] = defaultdict(list)
        open_trade_count = 0

        self.progress.init_step(BacktestState.BACKTEST, int(
            (end_date - start_date) / timedelta(minutes=self.timeframe_min)))

        # Loop timerange and get candle for each pair at that point in time
        while tmp <= end_date:
            open_trade_count_start = open_trade_count
            self.check_abort()
            for i, pair in enumerate(data):
                row_index = indexes[pair]
                try:
                    # Row is treated as "current incomplete candle".
                    # Buy / sell signals are shifted by 1 to compensate for this.
                    row = data[pair][row_index]
                except IndexError:
                    # missing Data for one pair at the end.
                    # Warnings for this are shown during data loading
                    continue

                # Waits until the time-counter reaches the start of the data for this pair.
                if row[DATE_IDX] > tmp:
                    continue

                row_index += 1
                indexes[pair] = row_index
                self.dataprovider._set_dataframe_max_index(row_index)

                # without positionstacking, we can only have one open trade per pair.
                # max_open_trades must be respected
                # don't open on the last row
                if (
                    (position_stacking or len(open_trades[pair]) == 0)
                    and self.trade_slot_available(max_open_trades, open_trade_count_start)
                    and tmp != end_date
                    and row[BUY_IDX] == 1
                    and row[SELL_IDX] != 1
                    and not PairLocks.is_pair_locked(pair, row[DATE_IDX])
                ):
                    trade = self._enter_trade(pair, row)
                    if trade:
                        # TODO: hacky workaround to avoid opening > max_open_trades
                        # This emulates previous behaviour - not sure if this is correct
                        # Prevents buying if the trade-slot was freed in this candle
                        open_trade_count_start += 1
                        open_trade_count += 1
                        # logger.debug(f"{pair} - Emulate creation of new trade: {trade}.")
                        open_trades[pair].append(trade)
                        LocalTrade.add_bt_trade(trade)

                for trade in list(open_trades[pair]):
                    # also check the buying candle for sell conditions.
                    trade_entry = self._get_sell_trade_entry(trade, row)
                    # Sell occurred
                    if trade_entry:
                        # logger.debug(f"{pair} - Backtesting sell {trade}")
                        open_trade_count -= 1
                        open_trades[pair].remove(trade)

                        LocalTrade.close_bt_trade(trade)
                        trades.append(trade_entry)
                        if enable_protections:
                            self.protections.stop_per_pair(pair, row[DATE_IDX])
                            self.protections.global_stop(tmp)

            # Move time one configured time_interval ahead.
            self.progress.increment()
            tmp += timedelta(minutes=self.timeframe_min)

        trades += self.handle_left_open(open_trades, data=data)
        self.wallets.update()

        results = trade_list_to_dataframe(trades)
        return {
            'results': results,
            'config': self.strategy.config,
            'locks': PairLocks.get_all_locks(),
            'rejected_signals': self.rejected_trades,
            'final_balance': self.wallets.get_total(self.strategy.config['stake_currency']),
        }

    def backtest_one_strategy(self, strat: IStrategy, data: Dict[str, DataFrame],
                              timerange: TimeRange):
        self.progress.init_step(BacktestState.ANALYZE, 0)

        logger.info("Running backtesting for Strategy %s", strat.get_strategy_name())
        backtest_start_time = datetime.now(timezone.utc)
        self._set_strategy(strat)

        strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)()

        # Use max_open_trades in backtesting, except --disable-max-market-positions is set
        if self.config.get('use_max_market_positions', True):
            # Must come from strategy config, as the strategy may modify this setting.
            max_open_trades = self.strategy.config['max_open_trades']
        else:
            logger.info(
                'Ignoring max_open_trades (--disable-max-market-positions was used) ...')
            max_open_trades = 0

        # need to reprocess data every time to populate signals
        preprocessed = self.strategy.advise_all_indicators(data)

        # Trim startup period from analyzed dataframe
        preprocessed_tmp = trim_dataframes(preprocessed, timerange, self.required_startup)

        if not preprocessed_tmp:
            raise OperationalException(
                "No data left after adjusting for startup candles.")

        # Use preprocessed_tmp for date generation (the trimmed dataframe).
        # Backtesting will re-trim the dataframes after buy/sell signal generation.
        min_date, max_date = history.get_timerange(preprocessed_tmp)
        logger.info(f'Backtesting with data from {min_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'up to {max_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'({(max_date - min_date).days} days).')
        # Execute backtest and store results
        results = self.backtest(
            processed=preprocessed,
            start_date=min_date,
            end_date=max_date,
            max_open_trades=max_open_trades,
            position_stacking=self.config.get('position_stacking', False),
            enable_protections=self.config.get('enable_protections', False),
        )
        backtest_end_time = datetime.now(timezone.utc)
        results.update({
            'backtest_start_time': int(backtest_start_time.timestamp()),
            'backtest_end_time': int(backtest_end_time.timestamp()),
        })
        self.all_results[self.strategy.get_strategy_name()] = results

        return min_date, max_date

    def start(self) -> None:
        """
        Run backtesting end-to-end
        :return: None
        """
        data: Dict[str, Any] = {}

        data, timerange = self.load_bt_data()
        self.load_bt_data_detail()
        logger.info("Dataload complete. Calculating indicators")

        for strat in self.strategylist:
            min_date, max_date = self.backtest_one_strategy(strat, data, timerange)
        if len(self.strategylist) > 0:

            self.results = generate_backtest_stats(data, self.all_results,
                                                   min_date=min_date, max_date=max_date)

            if self.config.get('export', 'none') == 'trades':
                store_backtest_stats(self.config['exportfilename'], self.results)

            # Show backtest results
            show_backtest_results(self.config, self.results)
