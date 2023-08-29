# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the backtesting logic
"""
import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from numpy import nan
from pandas import DataFrame

from freqtrade import constants
from freqtrade.configuration import TimeRange, validate_config_consistency
from freqtrade.constants import DATETIME_PRINT_FORMAT, Config, IntOrInf, LongShort
from freqtrade.data import history
from freqtrade.data.btanalysis import find_existing_backtest_stats, trade_list_to_dataframe
from freqtrade.data.converter import trim_dataframe, trim_dataframes
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import (BacktestState, CandleType, ExitCheckTuple, ExitType, RunMode,
                             TradingMode)
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import (amount_to_contract_precision, price_to_precision,
                                timeframe_to_minutes, timeframe_to_seconds)
from freqtrade.exchange.exchange import Exchange
from freqtrade.mixins import LoggingMixin
from freqtrade.optimize.backtest_caching import get_strategy_run_id
from freqtrade.optimize.bt_progress import BTProgress
from freqtrade.optimize.optimize_reports import (generate_backtest_stats, generate_rejected_signals,
                                                 generate_trade_signal_candles,
                                                 show_backtest_results,
                                                 store_backtest_analysis_results,
                                                 store_backtest_stats)
from freqtrade.persistence import LocalTrade, Order, PairLocks, Trade
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.plugins.protectionmanager import ProtectionManager
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.types import BacktestResultType, get_BacktestResultType_default
from freqtrade.util.binance_mig import migrate_binance_futures_data
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)

# Indexes for backtest tuples
DATE_IDX = 0
OPEN_IDX = 1
HIGH_IDX = 2
LOW_IDX = 3
CLOSE_IDX = 4
LONG_IDX = 5
ELONG_IDX = 6  # Exit long
SHORT_IDX = 7
ESHORT_IDX = 8  # Exit short
ENTER_TAG_IDX = 9
EXIT_TAG_IDX = 10

# Every change to this headers list must evaluate further usages of the resulting tuple
# and eventually change the constants for indexes at the top
HEADERS = ['date', 'open', 'high', 'low', 'close', 'enter_long', 'exit_long',
           'enter_short', 'exit_short', 'enter_tag', 'exit_tag']


class Backtesting:
    """
    Backtesting class, this class contains all the logic to run a backtest

    To run a backtest:
    backtesting = Backtesting(config)
    backtesting.start()
    """

    def __init__(self, config: Config, exchange: Optional[Exchange] = None) -> None:

        LoggingMixin.show_output = False
        self.config = config
        self.results: BacktestResultType = get_BacktestResultType_default()
        self.trade_id_counter: int = 0
        self.order_id_counter: int = 0

        config['dry_run'] = True
        self.run_ids: Dict[str, str] = {}
        self.strategylist: List[IStrategy] = []
        self.all_results: Dict[str, Dict] = {}
        self.processed_dfs: Dict[str, Dict] = {}
        self.rejected_dict: Dict[str, List] = {}
        self.rejected_df: Dict[str, Dict] = {}

        self._exchange_name = self.config['exchange']['name']
        if not exchange:
            exchange = ExchangeResolver.load_exchange(self.config, load_leverage_tiers=True)
        self.exchange = exchange

        self.dataprovider = DataProvider(self.config, self.exchange)

        if self.config.get('strategy_list'):
            if self.config.get('freqai', {}).get('enabled', False):
                logger.warning("Using --strategy-list with FreqAI REQUIRES all strategies "
                               "to have identical feature_engineering_* functions.")
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
            raise OperationalException("Timeframe needs to be set in either "
                                       "configuration or as cli argument `--timeframe 5m`")
        self.timeframe = str(self.config.get('timeframe'))
        self.timeframe_min = timeframe_to_minutes(self.timeframe)
        self.init_backtest_detail()
        self.pairlists = PairListManager(self.exchange, self.config, self.dataprovider)
        self._validate_pairlists_for_backtesting()

        self.dataprovider.add_pairlisthandler(self.pairlists)
        self.pairlists.refresh_pairlist()

        if len(self.pairlists.whitelist) == 0:
            raise OperationalException("No pair in whitelist.")

        if config.get('fee', None) is not None:
            self.fee = config['fee']
        else:
            self.fee = self.exchange.get_fee(symbol=self.pairlists.whitelist[0])
        self.precision_mode = self.exchange.precisionMode

        if self.config.get('freqai_backtest_live_models', False):
            from freqtrade.freqai.utils import get_timerange_backtest_live_models
            self.config['timerange'] = get_timerange_backtest_live_models(self.config)

        self.timerange = TimeRange.parse_timerange(
            None if self.config.get('timerange') is None else str(self.config.get('timerange')))

        # Get maximum required startup period
        self.required_startup = max([strat.startup_candle_count for strat in self.strategylist])
        self.exchange.validate_required_startup_candles(self.required_startup, self.timeframe)

        if self.config.get('freqai', {}).get('enabled', False):
            # For FreqAI, increase the required_startup to includes the training data
            self.required_startup = self.dataprovider.get_required_startup(self.timeframe)

        # Add maximum startup candle count to configuration for informative pairs support
        self.config['startup_candle_count'] = self.required_startup

        self.trading_mode: TradingMode = config.get('trading_mode', TradingMode.SPOT)
        # strategies which define "can_short=True" will fail to load in Spot mode.
        self._can_short = self.trading_mode != TradingMode.SPOT
        self._position_stacking: bool = self.config.get('position_stacking', False)
        self.enable_protections: bool = self.config.get('enable_protections', False)
        migrate_binance_futures_data(config)

        self.init_backtest()

    def _validate_pairlists_for_backtesting(self):
        if 'VolumePairList' in self.pairlists.name_list:
            raise OperationalException("VolumePairList not allowed for backtesting. "
                                       "Please use StaticPairList instead.")
        if 'PerformanceFilter' in self.pairlists.name_list:
            raise OperationalException("PerformanceFilter not allowed for backtesting.")

        if len(self.strategylist) > 1 and 'PrecisionFilter' in self.pairlists.name_list:
            raise OperationalException(
                "PrecisionFilter not allowed for backtesting multiple strategies."
            )

    @staticmethod
    def cleanup():
        LoggingMixin.show_output = True
        PairLocks.use_db = True
        Trade.use_db = True

    def init_backtest_detail(self) -> None:
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
        self.futures_data: Dict[str, DataFrame] = {}

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
        # since a "perfect" stoploss-exit is assumed anyway
        # And the regular "stoploss" function would not apply to that case
        self.strategy.order_types['stoploss_on_exchange'] = False
        # Update can_short flag
        self._can_short = self.trading_mode != TradingMode.SPOT and strategy.can_short

        self.strategy.ft_bot_start()

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
            startup_candles=self.config['startup_candle_count'],
            fail_without_data=True,
            data_format=self.config['dataformat_ohlcv'],
            candle_type=self.config.get('candle_type_def', CandleType.SPOT)
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
                data_format=self.config['dataformat_ohlcv'],
                candle_type=self.config.get('candle_type_def', CandleType.SPOT)
            )
        else:
            self.detail_data = {}
        if self.trading_mode == TradingMode.FUTURES:
            # Load additional futures data.
            funding_rates_dict = history.load_data(
                datadir=self.config['datadir'],
                pairs=self.pairlists.whitelist,
                timeframe=self.exchange.get_option('mark_ohlcv_timeframe'),
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config['dataformat_ohlcv'],
                candle_type=CandleType.FUNDING_RATE
            )

            # For simplicity, assign to CandleType.Mark (might contian index candles!)
            mark_rates_dict = history.load_data(
                datadir=self.config['datadir'],
                pairs=self.pairlists.whitelist,
                timeframe=self.exchange.get_option('mark_ohlcv_timeframe'),
                timerange=self.timerange,
                startup_candles=0,
                fail_without_data=True,
                data_format=self.config['dataformat_ohlcv'],
                candle_type=CandleType.from_string(self.exchange.get_option("mark_ohlcv_price"))
            )
            # Combine data to avoid combining the data per trade.
            unavailable_pairs = []
            for pair in self.pairlists.whitelist:
                if pair not in self.exchange._leverage_tiers:
                    unavailable_pairs.append(pair)
                    continue

                self.futures_data[pair] = self.exchange.combine_funding_and_mark(
                    funding_rates=funding_rates_dict[pair],
                    mark_rates=mark_rates_dict[pair],
                    futures_funding_rate=self.config.get('futures_funding_rate', None),
                )

            if unavailable_pairs:
                raise OperationalException(
                    f"Pairs {', '.join(unavailable_pairs)} got no leverage tiers available. "
                    "It is therefore impossible to backtest with this pair at the moment.")
        else:
            self.futures_data = {}

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
        self.timedout_entry_orders = 0
        self.timedout_exit_orders = 0
        self.canceled_trade_entries = 0
        self.canceled_entry_orders = 0
        self.replaced_entry_orders = 0
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

        :param processed: a processed dictionary with format {pair, data}, which gets cleared to
        optimize memory usage!
        """

        data: Dict = {}
        self.progress.init_step(BacktestState.CONVERT, len(processed))

        # Create dict with data
        for pair in processed.keys():
            pair_data = processed[pair]
            self.check_abort()
            self.progress.increment()

            if not pair_data.empty:
                # Cleanup from prior runs
                pair_data.drop(HEADERS[5:] + ['buy', 'sell'], axis=1, errors='ignore')
            df_analyzed = self.strategy.ft_advise_signals(pair_data, {'pair': pair})
            # Update dataprovider cache
            self.dataprovider._set_cached_df(
                pair, self.timeframe, df_analyzed, self.config['candle_type_def'])

            # Trim startup period from analyzed dataframe
            df_analyzed = processed[pair] = pair_data = trim_dataframe(
                df_analyzed, self.timerange, startup_candles=self.required_startup)

            # Create a copy of the dataframe before shifting, that way the entry signal/tag
            # remains on the correct candle for callbacks.
            df_analyzed = df_analyzed.copy()

            # To avoid using data from future, we use entry/exit signals shifted
            # from the previous candle
            for col in HEADERS[5:]:
                tag_col = col in ('enter_tag', 'exit_tag')
                if col in df_analyzed.columns:
                    df_analyzed[col] = df_analyzed.loc[:, col].replace(
                        [nan], [0 if not tag_col else None]).shift(1)
                elif not df_analyzed.empty:
                    df_analyzed[col] = 0 if not tag_col else None

            df_analyzed = df_analyzed.drop(df_analyzed.head(1).index)

            # Convert from Pandas to list for performance reasons
            # (Looping Pandas is slow.)
            data[pair] = df_analyzed[HEADERS].values.tolist() if not df_analyzed.empty else []
        return data

    def _get_close_rate(self, row: Tuple, trade: LocalTrade, exit: ExitCheckTuple,
                        trade_dur: int) -> float:
        """
        Get close rate for backtesting result
        """
        # Special handling if high or low hit STOP_LOSS or ROI
        if exit.exit_type in (
                ExitType.STOP_LOSS, ExitType.TRAILING_STOP_LOSS, ExitType.LIQUIDATION):
            return self._get_close_rate_for_stoploss(row, trade, exit, trade_dur)
        elif exit.exit_type == (ExitType.ROI):
            return self._get_close_rate_for_roi(row, trade, exit, trade_dur)
        else:
            return row[OPEN_IDX]

    def _get_close_rate_for_stoploss(self, row: Tuple, trade: LocalTrade, exit: ExitCheckTuple,
                                     trade_dur: int) -> float:
        # our stoploss was already lower than candle high,
        # possibly due to a cancelled trade exit.
        # exit at open price.
        is_short = trade.is_short or False
        leverage = trade.leverage or 1.0
        side_1 = -1 if is_short else 1
        if exit.exit_type == ExitType.LIQUIDATION and trade.liquidation_price:
            stoploss_value = trade.liquidation_price
        else:
            stoploss_value = trade.stop_loss

        if is_short:
            if stoploss_value < row[LOW_IDX]:
                return row[OPEN_IDX]
        else:
            if stoploss_value > row[HIGH_IDX]:
                return row[OPEN_IDX]

        # Special case: trailing triggers within same candle as trade opened. Assume most
        # pessimistic price movement, which is moving just enough to arm stoploss and
        # immediately going down to stop price.
        if exit.exit_type == ExitType.TRAILING_STOP_LOSS and trade_dur == 0:
            if (
                not self.strategy.use_custom_stoploss and self.strategy.trailing_stop
                and self.strategy.trailing_only_offset_is_reached
                and self.strategy.trailing_stop_positive_offset is not None
                and self.strategy.trailing_stop_positive
            ):
                # Worst case: price reaches stop_positive_offset and dives down.
                stop_rate = (row[OPEN_IDX] *
                             (1 + side_1 * abs(self.strategy.trailing_stop_positive_offset) -
                              side_1 * abs(self.strategy.trailing_stop_positive / leverage)))
            else:
                # Worst case: price ticks tiny bit above open and dives down.
                stop_rate = row[OPEN_IDX] * (1 - side_1 * abs(
                    (trade.stop_loss_pct or 0.0) / leverage))

            # Limit lower-end to candle low to avoid exits below the low.
            # This still remains "worst case" - but "worst realistic case".
            if is_short:
                return min(row[HIGH_IDX], stop_rate)
            else:
                return max(row[LOW_IDX], stop_rate)

        # Set close_rate to stoploss
        return stoploss_value

    def _get_close_rate_for_roi(self, row: Tuple, trade: LocalTrade, exit: ExitCheckTuple,
                                trade_dur: int) -> float:
        is_short = trade.is_short or False
        leverage = trade.leverage or 1.0
        side_1 = -1 if is_short else 1
        roi_entry, roi = self.strategy.min_roi_reached_entry(trade_dur)
        if roi is not None and roi_entry is not None:
            if roi == -1 and roi_entry % self.timeframe_min == 0:
                # When force_exiting with ROI=-1, the roi time will always be equal to trade_dur.
                # If that entry is a multiple of the timeframe (so on candle open)
                # - we'll use open instead of close
                return row[OPEN_IDX]

            # - (Expected abs profit - open_rate - open_fee) / (fee_close -1)
            roi_rate = trade.open_rate * roi / leverage
            open_fee_rate = side_1 * trade.open_rate * (1 + side_1 * trade.fee_open)
            close_rate = -(roi_rate + open_fee_rate) / ((trade.fee_close or 0.0) - side_1 * 1)
            if is_short:
                is_new_roi = row[OPEN_IDX] < close_rate
            else:
                is_new_roi = row[OPEN_IDX] > close_rate
            if (trade_dur > 0 and trade_dur == roi_entry
                    and roi_entry % self.timeframe_min == 0
                    and is_new_roi):
                # new ROI entry came into effect.
                # use Open rate if open_rate > calculated exit rate
                return row[OPEN_IDX]

            if (trade_dur == 0 and (
                (
                    is_short
                    # Red candle (for longs)
                    and row[OPEN_IDX] < row[CLOSE_IDX]  # Red candle
                    and trade.open_rate > row[OPEN_IDX]  # trade-open above open_rate
                    and close_rate < row[CLOSE_IDX]  # closes below close
                )
                or
                (
                    not is_short
                    # green candle (for shorts)
                    and row[OPEN_IDX] > row[CLOSE_IDX]  # green candle
                    and trade.open_rate < row[OPEN_IDX]  # trade-open below open_rate
                    and close_rate > row[CLOSE_IDX]  # closes above close
                )
            )):
                # ROI on opening candles with custom pricing can only
                # trigger if the entry was at Open or lower wick.
                # details: https: // github.com/freqtrade/freqtrade/issues/6261
                # If open_rate is < open, only allow exits below the close on red candles.
                raise ValueError("Opening candle ROI on red candles.")

            # Use the maximum between close_rate and low as we
            # cannot exit outside of a candle.
            # Applies when a new ROI setting comes in place and the whole candle is above that.
            return min(max(close_rate, row[LOW_IDX]), row[HIGH_IDX])

        else:
            # This should not be reached...
            return row[OPEN_IDX]

    def _get_adjust_trade_entry_for_candle(self, trade: LocalTrade, row: Tuple
                                           ) -> LocalTrade:
        current_rate = row[OPEN_IDX]
        current_date = row[DATE_IDX].to_pydatetime()
        current_profit = trade.calc_profit_ratio(current_rate)
        min_stake = self.exchange.get_min_pair_stake_amount(trade.pair, current_rate, -0.1)
        max_stake = self.exchange.get_max_pair_stake_amount(trade.pair, current_rate)
        stake_available = self.wallets.get_available_stake_amount()
        stake_amount = strategy_safe_wrapper(self.strategy.adjust_trade_position,
                                             default_retval=None, supress_error=True)(
            trade=trade,  # type: ignore[arg-type]
            current_time=current_date, current_rate=current_rate,
            current_profit=current_profit, min_stake=min_stake,
            max_stake=min(max_stake, stake_available),
            current_entry_rate=current_rate, current_exit_rate=current_rate,
            current_entry_profit=current_profit, current_exit_profit=current_profit)

        # Check if we should increase our position
        if stake_amount is not None and stake_amount > 0.0:
            check_adjust_entry = True
            if self.strategy.max_entry_position_adjustment > -1:
                entry_count = trade.nr_of_successful_entries
                check_adjust_entry = (entry_count <= self.strategy.max_entry_position_adjustment)
            if check_adjust_entry:
                pos_trade = self._enter_trade(
                    trade.pair, row, 'short' if trade.is_short else 'long', stake_amount, trade)
                if pos_trade is not None:
                    self.wallets.update()
                    return pos_trade

        if stake_amount is not None and stake_amount < 0.0:
            amount = amount_to_contract_precision(
                abs(stake_amount * trade.leverage) / current_rate, trade.amount_precision,
                self.precision_mode, trade.contract_size)
            if amount == 0.0:
                return trade
            if amount > trade.amount:
                # This is currently ineffective as remaining would become < min tradable
                amount = trade.amount
            remaining = (trade.amount - amount) * current_rate
            if remaining < min_stake:
                # Remaining stake is too low to be sold.
                return trade
            exit_ = ExitCheckTuple(ExitType.PARTIAL_EXIT)
            pos_trade = self._get_exit_for_signal(trade, row, exit_, amount)
            if pos_trade is not None:
                order = pos_trade.orders[-1]
                if self._try_close_open_order(order, trade, current_date, row):
                    trade.recalc_trade_from_orders()
                self.wallets.update()
                return pos_trade

        return trade

    def _get_order_filled(self, rate: float, row: Tuple) -> bool:
        """ Rate is within candle, therefore filled"""
        return row[LOW_IDX] <= rate <= row[HIGH_IDX]

    def _call_adjust_stop(self, current_date: datetime, trade: LocalTrade, current_rate: float):
        profit = trade.calc_profit_ratio(current_rate)
        self.strategy.ft_stoploss_adjust(current_rate, trade,  # type: ignore
                                         current_date, profit, 0, after_fill=True)

    def _try_close_open_order(
            self, order: Optional[Order], trade: LocalTrade, current_date: datetime,
            row: Tuple) -> bool:
        """
        Check if an order is open and if it should've filled.
        :return:  True if the order filled.
        """
        if order and self._get_order_filled(order.ft_price, row):
            order.close_bt_order(current_date, trade)
            trade.open_order_id = None
            if not (order.ft_order_side == trade.exit_side and order.safe_amount == trade.amount):
                self._call_adjust_stop(current_date, trade, order.ft_price)
                # pass
            return True
        return False

    def _get_exit_for_signal(
            self, trade: LocalTrade, row: Tuple, exit_: ExitCheckTuple,
            amount: Optional[float] = None) -> Optional[LocalTrade]:

        exit_candle_time: datetime = row[DATE_IDX].to_pydatetime()
        if exit_.exit_flag:
            trade.close_date = exit_candle_time
            exit_reason = exit_.exit_reason
            amount_ = amount if amount is not None else trade.amount
            trade_dur = int((trade.close_date_utc - trade.open_date_utc).total_seconds() // 60)
            try:
                close_rate = self._get_close_rate(row, trade, exit_, trade_dur)
            except ValueError:
                return None
            # call the custom exit price,with default value as previous close_rate
            current_profit = trade.calc_profit_ratio(close_rate)
            order_type = self.strategy.order_types['exit']
            if exit_.exit_type in (ExitType.EXIT_SIGNAL, ExitType.CUSTOM_EXIT,
                                   ExitType.PARTIAL_EXIT):
                # Checks and adds an exit tag, after checking that the length of the
                # row has the length for an exit tag column
                if (
                    len(row) > EXIT_TAG_IDX
                    and row[EXIT_TAG_IDX] is not None
                    and len(row[EXIT_TAG_IDX]) > 0
                    and exit_.exit_type in (ExitType.EXIT_SIGNAL,)
                ):
                    exit_reason = row[EXIT_TAG_IDX]
                # Custom exit pricing only for exit-signals
                if order_type == 'limit':
                    rate = strategy_safe_wrapper(self.strategy.custom_exit_price,
                                                 default_retval=close_rate)(
                        pair=trade.pair,
                        trade=trade,  # type: ignore[arg-type]
                        current_time=exit_candle_time,
                        proposed_rate=close_rate, current_profit=current_profit,
                        exit_tag=exit_reason)
                    if rate != close_rate:
                        close_rate = price_to_precision(rate, trade.price_precision,
                                                        self.precision_mode)
                    # We can't place orders lower than current low.
                    # freqtrade does not support this in live, and the order would fill immediately
                    if trade.is_short:
                        close_rate = min(close_rate, row[HIGH_IDX])
                    else:
                        close_rate = max(close_rate, row[LOW_IDX])
            # Confirm trade exit:
            time_in_force = self.strategy.order_time_in_force['exit']

            if (exit_.exit_type not in (ExitType.LIQUIDATION, ExitType.PARTIAL_EXIT)
                    and not strategy_safe_wrapper(
                    self.strategy.confirm_trade_exit, default_retval=True)(
                        pair=trade.pair,
                        trade=trade,  # type: ignore[arg-type]
                        order_type=order_type,
                        amount=amount_,
                        rate=close_rate,
                        time_in_force=time_in_force,
                        sell_reason=exit_reason,  # deprecated
                        exit_reason=exit_reason,
                        current_time=exit_candle_time)):
                return None

            trade.exit_reason = exit_reason

            return self._exit_trade(trade, row, close_rate, amount_)
        return None

    def _exit_trade(self, trade: LocalTrade, sell_row: Tuple,
                    close_rate: float, amount: Optional[float] = None) -> Optional[LocalTrade]:
        self.order_id_counter += 1
        exit_candle_time = sell_row[DATE_IDX].to_pydatetime()
        order_type = self.strategy.order_types['exit']
        # amount = amount or trade.amount
        amount = amount_to_contract_precision(amount or trade.amount, trade.amount_precision,
                                              self.precision_mode, trade.contract_size)
        order = Order(
            id=self.order_id_counter,
            ft_trade_id=trade.id,
            order_date=exit_candle_time,
            order_update_date=exit_candle_time,
            ft_is_open=True,
            ft_pair=trade.pair,
            order_id=str(self.order_id_counter),
            symbol=trade.pair,
            ft_order_side=trade.exit_side,
            side=trade.exit_side,
            order_type=order_type,
            status="open",
            ft_price=close_rate,
            price=close_rate,
            average=close_rate,
            amount=amount,
            filled=0,
            remaining=amount,
            cost=amount * close_rate,
        )
        order._trade_bt = trade
        trade.orders.append(order)
        return trade

    def _check_trade_exit(self, trade: LocalTrade, row: Tuple) -> Optional[LocalTrade]:
        exit_candle_time: datetime = row[DATE_IDX].to_pydatetime()

        if self.trading_mode == TradingMode.FUTURES:
            trade.funding_fees = self.exchange.calculate_funding_fees(
                self.futures_data[trade.pair],
                amount=trade.amount,
                is_short=trade.is_short,
                open_date=trade.date_last_filled_utc,
                close_date=exit_candle_time,
            )

        # Check if we need to adjust our current positions
        if self.strategy.position_adjustment_enable:
            trade = self._get_adjust_trade_entry_for_candle(trade, row)

        enter = row[SHORT_IDX] if trade.is_short else row[LONG_IDX]
        exit_sig = row[ESHORT_IDX] if trade.is_short else row[ELONG_IDX]
        exits = self.strategy.should_exit(
            trade, row[OPEN_IDX], row[DATE_IDX].to_pydatetime(),  # type: ignore
            enter=enter, exit_=exit_sig,
            low=row[LOW_IDX], high=row[HIGH_IDX]
        )
        for exit_ in exits:
            t = self._get_exit_for_signal(trade, row, exit_)
            if t:
                return t
        return None

    def get_valid_price_and_stake(
        self, pair: str, row: Tuple, propose_rate: float, stake_amount: float,
        direction: LongShort, current_time: datetime, entry_tag: Optional[str],
        trade: Optional[LocalTrade], order_type: str, price_precision: Optional[float]
    ) -> Tuple[float, float, float, float]:

        if order_type == 'limit':
            new_rate = strategy_safe_wrapper(self.strategy.custom_entry_price,
                                             default_retval=propose_rate)(
                pair=pair, current_time=current_time,
                proposed_rate=propose_rate, entry_tag=entry_tag,
                side=direction,
            )  # default value is the open rate
            # We can't place orders higher than current high (otherwise it'd be a stop limit entry)
            # which freqtrade does not support in live.
            if new_rate != propose_rate:
                propose_rate = price_to_precision(new_rate, price_precision,
                                                  self.precision_mode)
            if direction == "short":
                propose_rate = max(propose_rate, row[LOW_IDX])
            else:
                propose_rate = min(propose_rate, row[HIGH_IDX])

        pos_adjust = trade is not None
        leverage = trade.leverage if trade else 1.0
        if not pos_adjust:
            try:
                stake_amount = self.wallets.get_trade_stake_amount(pair, None, update=False)
            except DependencyException:
                return 0, 0, 0, 0

            max_leverage = self.exchange.get_max_leverage(pair, stake_amount)
            leverage = strategy_safe_wrapper(self.strategy.leverage, default_retval=1.0)(
                pair=pair,
                current_time=current_time,
                current_rate=row[OPEN_IDX],
                proposed_leverage=1.0,
                max_leverage=max_leverage,
                side=direction, entry_tag=entry_tag,
            ) if self.trading_mode != TradingMode.SPOT else 1.0
            # Cap leverage between 1.0 and max_leverage.
            leverage = min(max(leverage, 1.0), max_leverage)

        min_stake_amount = self.exchange.get_min_pair_stake_amount(
            pair, propose_rate, -0.05 if not pos_adjust else 0.0, leverage=leverage) or 0
        max_stake_amount = self.exchange.get_max_pair_stake_amount(
            pair, propose_rate, leverage=leverage)
        stake_available = self.wallets.get_available_stake_amount()

        if not pos_adjust:
            stake_amount = strategy_safe_wrapper(self.strategy.custom_stake_amount,
                                                 default_retval=stake_amount)(
                pair=pair, current_time=current_time, current_rate=propose_rate,
                proposed_stake=stake_amount, min_stake=min_stake_amount,
                max_stake=min(stake_available, max_stake_amount),
                leverage=leverage, entry_tag=entry_tag, side=direction)

        stake_amount_val = self.wallets.validate_stake_amount(
            pair=pair,
            stake_amount=stake_amount,
            min_stake_amount=min_stake_amount,
            max_stake_amount=max_stake_amount,
            trade_amount=trade.stake_amount if trade else None
        )

        return propose_rate, stake_amount_val, leverage, min_stake_amount

    def _enter_trade(self, pair: str, row: Tuple, direction: LongShort,
                     stake_amount: Optional[float] = None,
                     trade: Optional[LocalTrade] = None,
                     requested_rate: Optional[float] = None,
                     requested_stake: Optional[float] = None) -> Optional[LocalTrade]:
        """
        :param trade: Trade to adjust - initial entry if None
        :param requested_rate: Adjusted entry rate
        :param requested_stake: Stake amount for adjusted orders (`adjust_entry_price`).
        """

        current_time = row[DATE_IDX].to_pydatetime()
        entry_tag = row[ENTER_TAG_IDX] if len(row) >= ENTER_TAG_IDX + 1 else None
        # let's call the custom entry price, using the open price as default price
        order_type = self.strategy.order_types['entry']
        pos_adjust = trade is not None and requested_rate is None

        stake_amount_ = stake_amount or (trade.stake_amount if trade else 0.0)
        precision_price = self.exchange.get_precision_price(pair)

        propose_rate, stake_amount, leverage, min_stake_amount = self.get_valid_price_and_stake(
            pair, row, row[OPEN_IDX], stake_amount_, direction, current_time, entry_tag, trade,
            order_type, precision_price,
        )

        # replace proposed rate if another rate was requested
        propose_rate = requested_rate if requested_rate else propose_rate
        stake_amount = requested_stake if requested_stake else stake_amount

        if not stake_amount:
            # In case of pos adjust, still return the original trade
            # If not pos adjust, trade is None
            return trade
        time_in_force = self.strategy.order_time_in_force['entry']

        if stake_amount and (not min_stake_amount or stake_amount >= min_stake_amount):
            self.order_id_counter += 1
            base_currency = self.exchange.get_pair_base_currency(pair)
            amount_p = (stake_amount / propose_rate) * leverage

            contract_size = self.exchange.get_contract_size(pair)
            precision_amount = self.exchange.get_precision_amount(pair)
            amount = amount_to_contract_precision(amount_p, precision_amount, self.precision_mode,
                                                  contract_size)
            # Backcalculate actual stake amount.
            stake_amount = amount * propose_rate / leverage

            if not pos_adjust:
                # Confirm trade entry:
                if not strategy_safe_wrapper(
                        self.strategy.confirm_trade_entry, default_retval=True)(
                            pair=pair, order_type=order_type, amount=amount, rate=propose_rate,
                            time_in_force=time_in_force, current_time=current_time,
                            entry_tag=entry_tag, side=direction):
                    return trade

            is_short = (direction == 'short')
            # Necessary for Margin trading. Disabled until support is enabled.
            # interest_rate = self.exchange.get_interest_rate()

            if trade is None:
                # Enter trade
                self.trade_id_counter += 1
                trade = LocalTrade(
                    id=self.trade_id_counter,
                    open_order_id=self.order_id_counter,
                    pair=pair,
                    base_currency=base_currency,
                    stake_currency=self.config['stake_currency'],
                    open_rate=propose_rate,
                    open_rate_requested=propose_rate,
                    open_date=current_time,
                    stake_amount=stake_amount,
                    amount=amount,
                    amount_requested=amount,
                    fee_open=self.fee,
                    fee_close=self.fee,
                    is_open=True,
                    enter_tag=entry_tag,
                    exchange=self._exchange_name,
                    is_short=is_short,
                    trading_mode=self.trading_mode,
                    leverage=leverage,
                    # interest_rate=interest_rate,
                    amount_precision=precision_amount,
                    price_precision=precision_price,
                    precision_mode=self.precision_mode,
                    contract_size=contract_size,
                    orders=[],
                )

            trade.adjust_stop_loss(trade.open_rate, self.strategy.stoploss, initial=True)

            trade.set_liquidation_price(self.exchange.get_liquidation_price(
                pair=pair,
                open_rate=propose_rate,
                amount=amount,
                stake_amount=trade.stake_amount,
                leverage=trade.leverage,
                wallet_balance=trade.stake_amount,
                is_short=is_short,
            ))

            order = Order(
                id=self.order_id_counter,
                ft_trade_id=trade.id,
                ft_is_open=True,
                ft_pair=trade.pair,
                order_id=str(self.order_id_counter),
                symbol=trade.pair,
                ft_order_side=trade.entry_side,
                side=trade.entry_side,
                order_type=order_type,
                status="open",
                order_date=current_time,
                order_filled_date=current_time,
                order_update_date=current_time,
                ft_price=propose_rate,
                price=propose_rate,
                average=propose_rate,
                amount=amount,
                filled=0,
                remaining=amount,
                cost=amount * propose_rate + trade.fee_open,
            )
            order._trade_bt = trade
            trade.orders.append(order)
            if not self._try_close_open_order(order, trade, current_time, row):
                trade.open_order_id = str(self.order_id_counter)
            trade.recalc_trade_from_orders()

        return trade

    def handle_left_open(self, open_trades: Dict[str, List[LocalTrade]],
                         data: Dict[str, List[Tuple]]) -> None:
        """
        Handling of left open trades at the end of backtesting
        """
        for pair in open_trades.keys():
            for trade in list(open_trades[pair]):
                if trade.open_order_id and trade.nr_of_successful_entries == 0:
                    # Ignore trade if entry-order did not fill yet
                    continue
                exit_row = data[pair][-1]
                self._exit_trade(trade, exit_row, exit_row[OPEN_IDX], trade.amount)
                trade.orders[-1].close_bt_order(exit_row[DATE_IDX].to_pydatetime(), trade)

                trade.close_date = exit_row[DATE_IDX].to_pydatetime()
                trade.exit_reason = ExitType.FORCE_EXIT.value
                trade.close(exit_row[OPEN_IDX], show_msg=False)
                LocalTrade.close_bt_trade(trade)

    def trade_slot_available(self, open_trade_count: int) -> bool:
        # Always allow trades when max_open_trades is enabled.
        max_open_trades: IntOrInf = self.config['max_open_trades']
        if max_open_trades <= 0 or open_trade_count < max_open_trades:
            return True
        # Rejected trade
        self.rejected_trades += 1
        return False

    def check_for_trade_entry(self, row) -> Optional[LongShort]:
        enter_long = row[LONG_IDX] == 1
        exit_long = row[ELONG_IDX] == 1
        enter_short = self._can_short and row[SHORT_IDX] == 1
        exit_short = self._can_short and row[ESHORT_IDX] == 1

        if enter_long == 1 and not any([exit_long, enter_short]):
            # Long
            return 'long'
        if enter_short == 1 and not any([exit_short, enter_long]):
            # Short
            return 'short'
        return None

    def run_protections(self, pair: str, current_time: datetime, side: LongShort):
        if self.enable_protections:
            self.protections.stop_per_pair(pair, current_time, side)
            self.protections.global_stop(current_time, side)

    def manage_open_orders(self, trade: LocalTrade, current_time: datetime, row: Tuple) -> bool:
        """
        Check if any open order needs to be cancelled or replaced.
        Returns True if the trade should be deleted.
        """
        for order in [o for o in trade.orders if o.ft_is_open]:
            oc = self.check_order_cancel(trade, order, current_time)
            if oc:
                # delete trade due to order timeout
                return True
            elif oc is None and self.check_order_replace(trade, order, current_time, row):
                # delete trade due to user request
                self.canceled_trade_entries += 1
                return True
        # default maintain trade
        return False

    def check_order_cancel(
            self, trade: LocalTrade, order: Order, current_time: datetime) -> Optional[bool]:
        """
        Check if current analyzed order has to be canceled.
        Returns True if the trade should be Deleted (initial order was canceled),
                False if it's Canceled
                None if the order is still active.
        """
        timedout = self.strategy.ft_check_timed_out(
            trade,  # type: ignore[arg-type]
            order, current_time)
        if timedout:
            if order.side == trade.entry_side:
                self.timedout_entry_orders += 1
                if trade.nr_of_successful_entries == 0:
                    # Remove trade due to entry timeout expiration.
                    return True
                else:
                    # Close additional entry order
                    del trade.orders[trade.orders.index(order)]
                    trade.open_order_id = None
                    return False
            if order.side == trade.exit_side:
                self.timedout_exit_orders += 1
                # Close exit order and retry exiting on next signal.
                del trade.orders[trade.orders.index(order)]
                trade.open_order_id = None
                return False
        return None

    def check_order_replace(self, trade: LocalTrade, order: Order, current_time,
                            row: Tuple) -> bool:
        """
        Check if current analyzed entry order has to be replaced and do so.
        If user requested cancellation and there are no filled orders in the trade will
        instruct caller to delete the trade.
        Returns True if the trade should be deleted.
        """
        # only check on new candles for open entry orders
        if order.side == trade.entry_side and current_time > order.order_date_utc:
            requested_rate = strategy_safe_wrapper(self.strategy.adjust_entry_price,
                                                   default_retval=order.ft_price)(
                trade=trade,  # type: ignore[arg-type]
                order=order, pair=trade.pair, current_time=current_time,
                proposed_rate=row[OPEN_IDX], current_order_rate=order.ft_price,
                entry_tag=trade.enter_tag, side=trade.trade_direction
            )  # default value is current order price

            # cancel existing order whenever a new rate is requested (or None)
            if requested_rate == order.ft_price:
                # assumption: there can't be multiple open entry orders at any given time
                return False
            else:
                del trade.orders[trade.orders.index(order)]
                trade.open_order_id = None
                self.canceled_entry_orders += 1

            # place new order if result was not None
            if requested_rate:
                self._enter_trade(pair=trade.pair, row=row, trade=trade,
                                  requested_rate=requested_rate,
                                  requested_stake=(
                                    order.safe_remaining * order.ft_price / trade.leverage),
                                  direction='short' if trade.is_short else 'long')
                # Delete trade if no successful entries happened (if placing the new order failed)
                if trade.open_order_id is None and trade.nr_of_successful_entries == 0:
                    return True
                self.replaced_entry_orders += 1
            else:
                # assumption: there can't be multiple open entry orders at any given time
                return (trade.nr_of_successful_entries == 0)
        return False

    def validate_row(
            self, data: Dict, pair: str, row_index: int, current_time: datetime) -> Optional[Tuple]:
        try:
            # Row is treated as "current incomplete candle".
            # entry / exit signals are shifted by 1 to compensate for this.
            row = data[pair][row_index]
        except IndexError:
            # missing Data for one pair at the end.
            # Warnings for this are shown during data loading
            return None

        # Waits until the time-counter reaches the start of the data for this pair.
        if row[DATE_IDX] > current_time:
            return None
        return row

    def _collate_rejected(self, pair, row):
        """
        Temporarily store rejected signal information for downstream use in backtesting_analysis
        """
        # It could be fun to enable hyperopt mode to write
        # a loss function to reduce rejected signals
        if (self.config.get('export', 'none') == 'signals' and
                self.dataprovider.runmode == RunMode.BACKTEST):
            if pair not in self.rejected_dict:
                self.rejected_dict[pair] = []
            self.rejected_dict[pair].append([row[DATE_IDX], row[ENTER_TAG_IDX]])

    def backtest_loop(
            self, row: Tuple, pair: str, current_time: datetime, end_date: datetime,
            open_trade_count_start: int, trade_dir: Optional[LongShort],
            is_first: bool = True) -> int:
        """
        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.

        Backtesting processing for one candle/pair.
        """
        for t in list(LocalTrade.bt_trades_open_pp[pair]):
            # 1. Manage currently open orders of active trades
            if self.manage_open_orders(t, current_time, row):
                # Close trade
                open_trade_count_start -= 1
                LocalTrade.remove_bt_trade(t)
                self.wallets.update()

        # 2. Process entries.
        # without positionstacking, we can only have one open trade per pair.
        # max_open_trades must be respected
        # don't open on the last row
        # We only open trades on the main candle, not on detail candles
        if (
            (self._position_stacking or len(LocalTrade.bt_trades_open_pp[pair]) == 0)
            and is_first
            and current_time != end_date
            and trade_dir is not None
            and not PairLocks.is_pair_locked(pair, row[DATE_IDX], trade_dir)
        ):
            if (self.trade_slot_available(open_trade_count_start)):
                trade = self._enter_trade(pair, row, trade_dir)
                if trade:
                    # TODO: hacky workaround to avoid opening > max_open_trades
                    # This emulates previous behavior - not sure if this is correct
                    # Prevents entering if the trade-slot was freed in this candle
                    open_trade_count_start += 1
                    # logger.debug(f"{pair} - Emulate creation of new trade: {trade}.")
                    LocalTrade.add_bt_trade(trade)
                    self.wallets.update()
            else:
                self._collate_rejected(pair, row)

        for trade in list(LocalTrade.bt_trades_open_pp[pair]):
            # 3. Process entry orders.
            order = trade.select_order(trade.entry_side, is_open=True)
            if self._try_close_open_order(order, trade, current_time, row):
                self.wallets.update()

            # 4. Create exit orders (if any)
            if not trade.open_order_id:
                self._check_trade_exit(trade, row)  # Place exit order if necessary

            # 5. Process exit orders.
            order = trade.select_order(trade.exit_side, is_open=True)
            if order and self._try_close_open_order(order, trade, current_time, row):
                sub_trade = order.safe_amount_after_fee != trade.amount
                if sub_trade:
                    trade.recalc_trade_from_orders()
                else:
                    trade.close_date = current_time
                    trade.close(order.ft_price, show_msg=False)

                    # logger.debug(f"{pair} - Backtesting exit {trade}")
                    LocalTrade.close_bt_trade(trade)
                self.wallets.update()
                self.run_protections(pair, current_time, trade.trade_direction)
        return open_trade_count_start

    def backtest(self, processed: Dict,
                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Implement backtesting functionality

        NOTE: This method is used by Hyperopt at each iteration. Please keep it optimized.
        Of course try to not have ugly code. By some accessor are sometime slower than functions.
        Avoid extensive logging in this method and functions it calls.

        :param processed: a processed dictionary with format {pair, data}, which gets cleared to
        optimize memory usage!
        :param start_date: backtesting timerange start datetime
        :param end_date: backtesting timerange end datetime
        :return: DataFrame with trades (results of backtesting)
        """
        self.prepare_backtest(self.enable_protections)
        # Ensure wallets are uptodate (important for --strategy-list)
        self.wallets.update()
        # Use dict of lists with data for performance
        # (looping lists is a lot faster than pandas DataFrames)
        data: Dict = self._get_ohlcv_as_lists(processed)

        # Indexes per pair, so some pairs are allowed to have a missing start.
        indexes: Dict = defaultdict(int)
        current_time = start_date + timedelta(minutes=self.timeframe_min)

        self.progress.init_step(BacktestState.BACKTEST, int(
            (end_date - start_date) / timedelta(minutes=self.timeframe_min)))
        # Loop timerange and get candle for each pair at that point in time
        while current_time <= end_date:
            open_trade_count_start = LocalTrade.bt_open_open_trade_count
            self.check_abort()
            strategy_safe_wrapper(self.strategy.bot_loop_start, supress_error=True)(
                current_time=current_time)
            for i, pair in enumerate(data):
                row_index = indexes[pair]
                row = self.validate_row(data, pair, row_index, current_time)
                if not row:
                    continue

                row_index += 1
                indexes[pair] = row_index
                self.dataprovider._set_dataframe_max_index(self.required_startup + row_index)
                self.dataprovider._set_dataframe_max_date(current_time)
                current_detail_time: datetime = row[DATE_IDX].to_pydatetime()
                trade_dir: Optional[LongShort] = self.check_for_trade_entry(row)

                if (
                    (trade_dir is not None or len(LocalTrade.bt_trades_open_pp[pair]) > 0)
                    and self.timeframe_detail and pair in self.detail_data
                ):
                    # Spread out into detail timeframe.
                    # Should only happen when we are either in a trade for this pair
                    # or when we got the signal for a new trade.
                    exit_candle_end = current_detail_time + timedelta(minutes=self.timeframe_min)

                    detail_data = self.detail_data[pair]
                    detail_data = detail_data.loc[
                        (detail_data['date'] >= current_detail_time) &
                        (detail_data['date'] < exit_candle_end)
                    ].copy()
                    if len(detail_data) == 0:
                        # Fall back to "regular" data if no detail data was found for this candle
                        open_trade_count_start = self.backtest_loop(
                            row, pair, current_time, end_date,
                            open_trade_count_start, trade_dir)
                        continue
                    detail_data.loc[:, 'enter_long'] = row[LONG_IDX]
                    detail_data.loc[:, 'exit_long'] = row[ELONG_IDX]
                    detail_data.loc[:, 'enter_short'] = row[SHORT_IDX]
                    detail_data.loc[:, 'exit_short'] = row[ESHORT_IDX]
                    detail_data.loc[:, 'enter_tag'] = row[ENTER_TAG_IDX]
                    detail_data.loc[:, 'exit_tag'] = row[EXIT_TAG_IDX]
                    is_first = True
                    current_time_det = current_time
                    for det_row in detail_data[HEADERS].values.tolist():
                        self.dataprovider._set_dataframe_max_date(current_time_det)
                        open_trade_count_start = self.backtest_loop(
                            det_row, pair, current_time_det, end_date,
                            open_trade_count_start, trade_dir, is_first)
                        current_time_det += timedelta(minutes=self.timeframe_detail_min)
                        is_first = False
                else:
                    self.dataprovider._set_dataframe_max_date(current_time)
                    open_trade_count_start = self.backtest_loop(
                        row, pair, current_time, end_date,
                        open_trade_count_start, trade_dir)

            # Move time one configured time_interval ahead.
            self.progress.increment()
            current_time += timedelta(minutes=self.timeframe_min)

        self.handle_left_open(LocalTrade.bt_trades_open_pp, data=data)
        self.wallets.update()

        results = trade_list_to_dataframe(LocalTrade.trades)
        return {
            'results': results,
            'config': self.strategy.config,
            'locks': PairLocks.get_all_locks(),
            'rejected_signals': self.rejected_trades,
            'timedout_entry_orders': self.timedout_entry_orders,
            'timedout_exit_orders': self.timedout_exit_orders,
            'canceled_trade_entries': self.canceled_trade_entries,
            'canceled_entry_orders': self.canceled_entry_orders,
            'replaced_entry_orders': self.replaced_entry_orders,
            'final_balance': self.wallets.get_total(self.strategy.config['stake_currency']),
        }

    def backtest_one_strategy(self, strat: IStrategy, data: Dict[str, DataFrame],
                              timerange: TimeRange):
        self.progress.init_step(BacktestState.ANALYZE, 0)
        strategy_name = strat.get_strategy_name()
        logger.info(f"Running backtesting for Strategy {strategy_name}")
        backtest_start_time = datetime.now(timezone.utc)
        self._set_strategy(strat)

        # Use max_open_trades in backtesting, except --disable-max-market-positions is set
        if not self.config.get('use_max_market_positions', True):
            logger.info(
                'Ignoring max_open_trades (--disable-max-market-positions was used) ...')
            self.strategy.max_open_trades = float('inf')
            self.config.update({'max_open_trades': self.strategy.max_open_trades})

        # need to reprocess data every time to populate signals
        preprocessed = self.strategy.advise_all_indicators(data)

        # Trim startup period from analyzed dataframe
        # This only used to determine if trimming would result in an empty dataframe
        preprocessed_tmp = trim_dataframes(preprocessed, timerange, self.required_startup)

        if not preprocessed_tmp:
            raise OperationalException(
                "No data left after adjusting for startup candles.")

        # Use preprocessed_tmp for date generation (the trimmed dataframe).
        # Backtesting will re-trim the dataframes after entry/exit signal generation.
        min_date, max_date = history.get_timerange(preprocessed_tmp)
        logger.info(f'Backtesting with data from {min_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'up to {max_date.strftime(DATETIME_PRINT_FORMAT)} '
                    f'({(max_date - min_date).days} days).')
        # Execute backtest and store results
        results = self.backtest(
            processed=preprocessed,
            start_date=min_date,
            end_date=max_date,
        )
        backtest_end_time = datetime.now(timezone.utc)
        results.update({
            'run_id': self.run_ids.get(strategy_name, ''),
            'backtest_start_time': int(backtest_start_time.timestamp()),
            'backtest_end_time': int(backtest_end_time.timestamp()),
        })
        self.all_results[strategy_name] = results

        if (self.config.get('export', 'none') == 'signals' and
                self.dataprovider.runmode == RunMode.BACKTEST):
            self.processed_dfs[strategy_name] = generate_trade_signal_candles(
                preprocessed_tmp, results)
            self.rejected_df[strategy_name] = generate_rejected_signals(
                preprocessed_tmp, self.rejected_dict)

        return min_date, max_date

    def _get_min_cached_backtest_date(self):
        min_backtest_date = None
        backtest_cache_age = self.config.get('backtest_cache', constants.BACKTEST_CACHE_DEFAULT)
        if self.timerange.stopts == 0 or self.timerange.stopdt > datetime.now(tz=timezone.utc):
            logger.warning('Backtest result caching disabled due to use of open-ended timerange.')
        elif backtest_cache_age == 'day':
            min_backtest_date = datetime.now(tz=timezone.utc) - timedelta(days=1)
        elif backtest_cache_age == 'week':
            min_backtest_date = datetime.now(tz=timezone.utc) - timedelta(weeks=1)
        elif backtest_cache_age == 'month':
            min_backtest_date = datetime.now(tz=timezone.utc) - timedelta(weeks=4)
        return min_backtest_date

    def load_prior_backtest(self):
        self.run_ids = {
            strategy.get_strategy_name(): get_strategy_run_id(strategy)
            for strategy in self.strategylist
        }

        # Load previous result that will be updated incrementally.
        # This can be circumvented in certain instances in combination with downloading more data
        min_backtest_date = self._get_min_cached_backtest_date()
        if min_backtest_date is not None:
            self.results = find_existing_backtest_stats(
                self.config['user_data_dir'] / 'backtest_results', self.run_ids, min_backtest_date)

    def start(self) -> None:
        """
        Run backtesting end-to-end
        :return: None
        """
        data: Dict[str, Any] = {}

        data, timerange = self.load_bt_data()
        self.load_bt_data_detail()
        logger.info("Dataload complete. Calculating indicators")

        self.load_prior_backtest()

        for strat in self.strategylist:
            if self.results and strat.get_strategy_name() in self.results['strategy']:
                # When previous result hash matches - reuse that result and skip backtesting.
                logger.info(f'Reusing result of previous backtest for {strat.get_strategy_name()}')
                continue
            min_date, max_date = self.backtest_one_strategy(strat, data, timerange)

        # Update old results with new ones.
        if len(self.all_results) > 0:
            results = generate_backtest_stats(
                data, self.all_results, min_date=min_date, max_date=max_date)
            if self.results:
                self.results['metadata'].update(results['metadata'])
                self.results['strategy'].update(results['strategy'])
                self.results['strategy_comparison'].extend(results['strategy_comparison'])
            else:
                self.results = results
            dt_appendix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if self.config.get('export', 'none') in ('trades', 'signals'):
                store_backtest_stats(self.config['exportfilename'], self.results, dt_appendix)

            if (self.config.get('export', 'none') == 'signals' and
                    self.dataprovider.runmode == RunMode.BACKTEST):
                store_backtest_analysis_results(
                    self.config['exportfilename'], self.processed_dfs, self.rejected_df,
                    dt_appendix)

        # Results may be mixed up now. Sort them so they follow --strategy-list order.
        if 'strategy_list' in self.config and len(self.results) > 0:
            self.results['strategy_comparison'] = sorted(
                self.results['strategy_comparison'],
                key=lambda c: self.config['strategy_list'].index(c['key']))
            self.results['strategy'] = dict(
                sorted(self.results['strategy'].items(),
                       key=lambda kv: self.config['strategy_list'].index(kv[0])))

        if len(self.strategylist) > 0:
            # Show backtest results
            show_backtest_results(self.config, self.results)
