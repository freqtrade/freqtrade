"""
IStrategy interface
This module defines the interface to apply for strategies
"""
import logging
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

import arrow
from pandas import DataFrame

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import SellType, SignalTagType, SignalType
from freqtrade.exceptions import OperationalException, StrategyError
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_seconds
from freqtrade.exchange.exchange import timeframe_to_next_date
from freqtrade.persistence import PairLocks, Trade
from freqtrade.strategy.hyper import HyperStrategyMixin
from freqtrade.strategy.informative_decorator import (InformativeData, PopulateIndicators,
                                                      _create_and_merge_informative_pair,
                                                      _format_pair_name)
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)
CUSTOM_SELL_MAX_LENGTH = 64


class SellCheckTuple(object):
    """
    NamedTuple for Sell type + reason
    """
    sell_type: SellType
    sell_reason: str = ''

    def __init__(self, sell_type: SellType, sell_reason: str = ''):
        self.sell_type = sell_type
        self.sell_reason = sell_reason or sell_type.value

    @property
    def sell_flag(self):
        return self.sell_type != SellType.NONE


class IStrategy(ABC, HyperStrategyMixin):
    """
    Interface for freqtrade strategies
    Defines the mandatory structure must follow any custom strategies

    Attributes you can use:
        minimal_roi -> Dict: Minimal ROI designed for the strategy
        stoploss -> float: optimal stoploss designed for the strategy
        timeframe -> str: value of the timeframe (ticker interval) to use with the strategy
    """
    # Strategy interface version
    # Default to version 2
    # Version 1 is the initial interface without metadata dict
    # Version 2 populate_* include metadata dict
    INTERFACE_VERSION: int = 2

    _populate_fun_len: int = 0
    _buy_fun_len: int = 0
    _sell_fun_len: int = 0
    _ft_params_from_file: Dict
    # associated minimal roi
    minimal_roi: Dict = {}

    # associated stoploss
    stoploss: float

    # trailing stoploss
    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached = False
    use_custom_stoploss: bool = False

    # associated timeframe
    ticker_interval: str  # DEPRECATED
    timeframe: str

    # Optional order types
    order_types: Dict = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
    }

    # Optional time in force
    order_time_in_force: Dict = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    # run "populate_indicators" only for new candle
    process_only_new_candles: bool = False

    use_sell_signal: bool
    sell_profit_only: bool
    sell_profit_offset: float
    ignore_roi_if_buy_signal: bool

    # Number of seconds after which the candle will no longer result in a buy on expired candles
    ignore_buying_expired_candle_after: int = 0

    # Disable checking the dataframe (converts the error into a warning message)
    disable_dataframe_checks: bool = False

    # Count of candles the strategy requires before producing valid signals
    startup_candle_count: int = 0

    # Protections
    protections: List = []

    # Class level variables (intentional) containing
    # the dataprovider (dp) (access to other candles, historic data, ...)
    # and wallets - access to the current balance.
    dp: Optional[DataProvider]
    wallets: Optional[Wallets] = None
    # Filled from configuration
    stake_currency: str
    # container variable for strategy source code
    __source__: str = ''

    # Definition of plot_config. See plotting documentation for more details.
    plot_config: Dict = {}

    def __init__(self, config: dict) -> None:
        self.config = config
        # Dict to determine if analysis is necessary
        self._last_candle_seen_per_pair: Dict[str, datetime] = {}
        super().__init__(config)

        # Gather informative pairs from @informative-decorated methods.
        self._ft_informative: List[Tuple[InformativeData, PopulateIndicators]] = []
        for attr_name in dir(self.__class__):
            cls_method = getattr(self.__class__, attr_name)
            if not callable(cls_method):
                continue
            informative_data_list = getattr(cls_method, '_ft_informative', None)
            if not isinstance(informative_data_list, list):
                # Type check is required because mocker would return a mock object that evaluates to
                # True, confusing this code.
                continue
            strategy_timeframe_minutes = timeframe_to_minutes(self.timeframe)
            for informative_data in informative_data_list:
                if timeframe_to_minutes(informative_data.timeframe) < strategy_timeframe_minutes:
                    raise OperationalException('Informative timeframe must be equal or higher than '
                                               'strategy timeframe!')
                self._ft_informative.append((informative_data, cls_method))

    @abstractmethod
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        :param dataframe: DataFrame with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        return dataframe

    @abstractmethod
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        return dataframe

    @abstractmethod
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        return dataframe

    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        """
        Check buy timeout function callback.
        This method can be used to override the buy-timeout.
        It is called whenever a limit buy order has been created,
        and is not yet fully filled.
        Configuration options in `unfilledtimeout` will be verified before this,
        so ensure to set these timeouts high enough.

        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: trade object.
        :param order: Order dictionary as returned from CCXT.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is cancelled.
        """
        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        """
        Check sell timeout function callback.
        This method can be used to override the sell-timeout.
        It is called whenever a limit sell order has been created,
        and is not yet fully filled.
        Configuration options in `unfilledtimeout` will be verified before this,
        so ensure to set these timeouts high enough.

        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: trade object.
        :param order: Order dictionary as returned from CCXT.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the sell-order is cancelled.
        """
        return False

    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pass

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a buy order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (quote) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular sell order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be sold.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in quote currency.
        :param rate: Rate that's going to be used when using limit orders
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param sell_reason: Sell reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'sell_signal', 'force_sell', 'emergency_sell']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the sell-order is placed on the exchange.
            False aborts the process
        """
        return True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the current_rate
        """
        return self.stoploss

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           **kwargs) -> float:
        """
        Custom entry price logic, returning the new entry price.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None, orderbook is used to set entry price

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New entry price value if provided
        """
        return proposed_rate

    def custom_exit_price(self, pair: str, trade: Trade,
                          current_time: datetime, proposed_rate: float,
                          current_profit: float, **kwargs) -> float:
        """
        Custom exit price logic, returning the new exit price.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None, orderbook is used to set exit price

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New exit price value if provided
        """
        return proposed_rate

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """
        Custom sell signal logic indicating that specified position should be sold. Returning a
        string or True from this method is equal to setting sell signal on a candle at specified
        time. This method is not called when sell signal is set.

        This method should be overridden to create sell signals that depend on trade parameters. For
        example you could implement a sell relative to the candle when the trade was opened,
        or a custom 1:2 risk-reward ROI.

        Custom sell reason max length is 64. Exceeding characters will be removed.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return: To execute sell, return a string with custom sell reason or True. Otherwise return
        None or False.
        """
        return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        """
        Customize stake size for each new trade. This method is not called when edge module is
        enabled.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param proposed_stake: A stake amount proposed by the bot.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :return: A stake size, which is between min_stake and max_stake.
        """
        return proposed_stake

    def informative_pairs(self) -> ListPairsWithTimeframes:
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

###
# END - Intended to be overridden by strategy
###

    def gather_informative_pairs(self) -> ListPairsWithTimeframes:
        """
        Internal method which gathers all informative pairs (user or automatically defined).
        """
        informative_pairs = self.informative_pairs()
        for inf_data, _ in self._ft_informative:
            if inf_data.asset:
                pair_tf = (_format_pair_name(self.config, inf_data.asset), inf_data.timeframe)
                informative_pairs.append(pair_tf)
            else:
                if not self.dp:
                    raise OperationalException('@informative decorator with unspecified asset '
                                               'requires DataProvider instance.')
                for pair in self.dp.current_whitelist():
                    informative_pairs.append((pair, inf_data.timeframe))
        return list(set(informative_pairs))

    def get_strategy_name(self) -> str:
        """
        Returns strategy class name
        """
        return self.__class__.__name__

    def lock_pair(self, pair: str, until: datetime, reason: str = None) -> None:
        """
        Locks pair until a given timestamp happens.
        Locked pairs are not analyzed, and are prevented from opening new trades.
        Locks can only count up (allowing users to lock pairs for a longer period of time).
        To remove a lock from a pair, use `unlock_pair()`
        :param pair: Pair to lock
        :param until: datetime in UTC until the pair should be blocked from opening new trades.
                Needs to be timezone aware `datetime.now(timezone.utc)`
        :param reason: Optional string explaining why the pair was locked.
        """
        PairLocks.lock_pair(pair, until, reason)

    def unlock_pair(self, pair: str) -> None:
        """
        Unlocks a pair previously locked using lock_pair.
        Not used by freqtrade itself, but intended to be used if users lock pairs
        manually from within the strategy, to allow an easy way to unlock pairs.
        :param pair: Unlock pair to allow trading again
        """
        PairLocks.unlock_pair(pair, datetime.now(timezone.utc))

    def unlock_reason(self, reason: str) -> None:
        """
        Unlocks all pairs previously locked using lock_pair with specified reason.
        Not used by freqtrade itself, but intended to be used if users lock pairs
        manually from within the strategy, to allow an easy way to unlock pairs.
        :param reason: Unlock pairs to allow trading again
        """
        PairLocks.unlock_reason(reason, datetime.now(timezone.utc))

    def is_pair_locked(self, pair: str, candle_date: datetime = None) -> bool:
        """
        Checks if a pair is currently locked
        The 2nd, optional parameter ensures that locks are applied until the new candle arrives,
        and not stop at 14:00:00 - while the next candle arrives at 14:00:02 leaving a gap
        of 2 seconds for a buy to happen on an old signal.
        :param pair: "Pair to check"
        :param candle_date: Date of the last candle. Optional, defaults to current date
        :returns: locking state of the pair in question.
        """

        if not candle_date:
            # Simple call ...
            return PairLocks.is_pair_locked(pair)
        else:
            lock_time = timeframe_to_next_date(self.timeframe, candle_date)
            return PairLocks.is_pair_locked(pair, lock_time)

    def analyze_ticker(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Parses the given candle (OHLCV) data and returns a populated DataFrame
        add several TA indicators and buy signal to it
        :param dataframe: Dataframe containing data from exchange
        :param metadata: Metadata dictionary with additional data (e.g. 'pair')
        :return: DataFrame of candle (OHLCV) data with indicator data and signals added
        """
        logger.debug("TA Analysis Launched")
        dataframe = self.advise_indicators(dataframe, metadata)
        dataframe = self.advise_buy(dataframe, metadata)
        dataframe = self.advise_sell(dataframe, metadata)
        return dataframe

    def _analyze_ticker_internal(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Parses the given candle (OHLCV) data and returns a populated DataFrame
        add several TA indicators and buy signal to it
        WARNING: Used internally only, may skip analysis if `process_only_new_candles` is set.
        :param dataframe: Dataframe containing data from exchange
        :param metadata: Metadata dictionary with additional data (e.g. 'pair')
        :return: DataFrame of candle (OHLCV) data with indicator data and signals added
        """
        pair = str(metadata.get('pair'))

        # Test if seen this pair and last candle before.
        # always run if process_only_new_candles is set to false
        if (not self.process_only_new_candles or
                self._last_candle_seen_per_pair.get(pair, None) != dataframe.iloc[-1]['date']):
            # Defs that only make change on new candle data.
            dataframe = self.analyze_ticker(dataframe, metadata)
            self._last_candle_seen_per_pair[pair] = dataframe.iloc[-1]['date']
            if self.dp:
                self.dp._set_cached_df(pair, self.timeframe, dataframe)
        else:
            logger.debug("Skipping TA Analysis for already analyzed candle")
            dataframe['buy'] = 0
            dataframe['sell'] = 0
            dataframe['buy_tag'] = None

        # Other Defs in strategy that want to be called every loop here
        # twitter_sell = self.watch_twitter_feed(dataframe, metadata)
        logger.debug("Loop Analysis Launched")

        return dataframe

    def analyze_pair(self, pair: str) -> None:
        """
        Fetch data for this pair from dataprovider and analyze.
        Stores the dataframe into the dataprovider.
        The analyzed dataframe is then accessible via `dp.get_analyzed_dataframe()`.
        :param pair: Pair to analyze.
        """
        if not self.dp:
            raise OperationalException("DataProvider not found.")
        dataframe = self.dp.ohlcv(pair, self.timeframe)
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning('Empty candle (OHLCV) data for pair %s', pair)
            return

        try:
            df_len, df_close, df_date = self.preserve_df(dataframe)

            dataframe = strategy_safe_wrapper(
                self._analyze_ticker_internal, message=""
            )(dataframe, {'pair': pair})

            self.assert_df(dataframe, df_len, df_close, df_date)
        except StrategyError as error:
            logger.warning(f"Unable to analyze candle (OHLCV) data for pair {pair}: {error}")
            return

        if dataframe.empty:
            logger.warning('Empty dataframe for pair %s', pair)
            return

    def analyze(self, pairs: List[str]) -> None:
        """
        Analyze all pairs using analyze_pair().
        :param pairs: List of pairs to analyze
        """
        for pair in pairs:
            self.analyze_pair(pair)

    @staticmethod
    def preserve_df(dataframe: DataFrame) -> Tuple[int, float, datetime]:
        """ keep some data for dataframes """
        return len(dataframe), dataframe["close"].iloc[-1], dataframe["date"].iloc[-1]

    def assert_df(self, dataframe: DataFrame, df_len: int, df_close: float, df_date: datetime):
        """
        Ensure dataframe (length, last candle) was not modified, and has all elements we need.
        """
        message_template = "Dataframe returned from strategy has mismatching {}."
        message = ""
        if dataframe is None:
            message = "No dataframe returned (return statement missing?)."
        elif 'buy' not in dataframe:
            message = "Buy column not set."
        elif df_len != len(dataframe):
            message = message_template.format("length")
        elif df_close != dataframe["close"].iloc[-1]:
            message = message_template.format("last close price")
        elif df_date != dataframe["date"].iloc[-1]:
            message = message_template.format("last date")
        if message:
            if self.disable_dataframe_checks:
                logger.warning(message)
            else:
                raise StrategyError(message)

    def get_signal(
        self,
        pair: str,
        timeframe: str,
        dataframe: DataFrame
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        Calculates current signal based based on the buy / sell columns of the dataframe.
        Used by Bot to get the signal to buy or sell
        :param pair: pair in format ANT/BTC
        :param timeframe: timeframe to use
        :param dataframe: Analyzed dataframe to get signal from.
        :return: (Buy, Sell) A bool-tuple indicating buy/sell signal
        """
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning(f'Empty candle (OHLCV) data for pair {pair}')
            return False, False, None

        latest_date = dataframe['date'].max()
        latest = dataframe.loc[dataframe['date'] == latest_date].iloc[-1]
        # Explicitly convert to arrow object to ensure the below comparison does not fail
        latest_date = arrow.get(latest_date)

        # Check if dataframe is out of date
        timeframe_minutes = timeframe_to_minutes(timeframe)
        offset = self.config.get('exchange', {}).get('outdated_offset', 5)
        if latest_date < (arrow.utcnow().shift(minutes=-(timeframe_minutes * 2 + offset))):
            logger.warning(
                'Outdated history for pair %s. Last tick is %s minutes old',
                pair, int((arrow.utcnow() - latest_date).total_seconds() // 60)
            )
            return False, False, None

        buy = latest[SignalType.BUY.value] == 1

        sell = False
        if SignalType.SELL.value in latest:
            sell = latest[SignalType.SELL.value] == 1

        buy_tag = latest.get(SignalTagType.BUY_TAG.value, None)

        logger.debug('trigger: %s (pair=%s) buy=%s sell=%s',
                     latest['date'], pair, str(buy), str(sell))
        timeframe_seconds = timeframe_to_seconds(timeframe)
        if self.ignore_expired_candle(latest_date=latest_date,
                                      current_time=datetime.now(timezone.utc),
                                      timeframe_seconds=timeframe_seconds,
                                      buy=buy):
            return False, sell, buy_tag
        return buy, sell, buy_tag

    def ignore_expired_candle(self, latest_date: datetime, current_time: datetime,
                              timeframe_seconds: int, buy: bool):
        if self.ignore_buying_expired_candle_after and buy:
            time_delta = current_time - (latest_date + timedelta(seconds=timeframe_seconds))
            return time_delta.total_seconds() > self.ignore_buying_expired_candle_after
        else:
            return False

    def should_sell(self, trade: Trade, rate: float, date: datetime, buy: bool,
                    sell: bool, low: float = None, high: float = None,
                    force_stoploss: float = 0) -> SellCheckTuple:
        """
        This function evaluates if one of the conditions required to trigger a sell
        has been reached, which can either be a stop-loss, ROI or sell-signal.
        :param low: Only used during backtesting to simulate stoploss
        :param high: Only used during backtesting, to simulate ROI
        :param force_stoploss: Externally provided stoploss
        :return: True if trade should be sold, False otherwise
        """
        current_rate = rate
        current_profit = trade.calc_profit_ratio(current_rate)

        trade.adjust_min_max_rates(high or current_rate, low or current_rate)

        stoplossflag = self.stop_loss_reached(current_rate=current_rate, trade=trade,
                                              current_time=date, current_profit=current_profit,
                                              force_stoploss=force_stoploss, low=low, high=high)

        # Set current rate to high for backtesting sell
        current_rate = high or rate
        current_profit = trade.calc_profit_ratio(current_rate)

        # if buy signal and ignore_roi is set, we don't need to evaluate min_roi.
        roi_reached = (not (buy and self.ignore_roi_if_buy_signal)
                       and self.min_roi_reached(trade=trade, current_profit=current_profit,
                                                current_time=date))

        sell_signal = SellType.NONE
        custom_reason = ''
        # use provided rate in backtesting, not high/low.
        current_rate = rate
        current_profit = trade.calc_profit_ratio(current_rate)

        if (self.sell_profit_only and current_profit <= self.sell_profit_offset):
            # sell_profit_only and profit doesn't reach the offset - ignore sell signal
            pass
        elif self.use_sell_signal and not buy:
            if sell:
                sell_signal = SellType.SELL_SIGNAL
            else:
                custom_reason = strategy_safe_wrapper(self.custom_sell, default_retval=False)(
                    pair=trade.pair, trade=trade, current_time=date, current_rate=current_rate,
                    current_profit=current_profit)
                if custom_reason:
                    sell_signal = SellType.CUSTOM_SELL
                    if isinstance(custom_reason, str):
                        if len(custom_reason) > CUSTOM_SELL_MAX_LENGTH:
                            logger.warning(f'Custom sell reason returned from custom_sell is too '
                                           f'long and was trimmed to {CUSTOM_SELL_MAX_LENGTH} '
                                           f'characters.')
                            custom_reason = custom_reason[:CUSTOM_SELL_MAX_LENGTH]
                    else:
                        custom_reason = None
            # TODO: return here if sell-signal should be favored over ROI

        # Start evaluations
        # Sequence:
        # ROI (if not stoploss)
        # Sell-signal
        # Stoploss
        if roi_reached and stoplossflag.sell_type != SellType.STOP_LOSS:
            logger.debug(f"{trade.pair} - Required profit reached. sell_type=SellType.ROI")
            return SellCheckTuple(sell_type=SellType.ROI)

        if sell_signal != SellType.NONE:
            logger.debug(f"{trade.pair} - Sell signal received. "
                         f"sell_type=SellType.{sell_signal.name}" +
                         (f", custom_reason={custom_reason}" if custom_reason else ""))
            return SellCheckTuple(sell_type=sell_signal, sell_reason=custom_reason)

        if stoplossflag.sell_flag:

            logger.debug(f"{trade.pair} - Stoploss hit. sell_type={stoplossflag.sell_type}")
            return stoplossflag

        # This one is noisy, commented out...
        # logger.debug(f"{trade.pair} - No sell signal.")
        return SellCheckTuple(sell_type=SellType.NONE)

    def stop_loss_reached(self, current_rate: float, trade: Trade,
                          current_time: datetime, current_profit: float,
                          force_stoploss: float, low: float = None,
                          high: float = None) -> SellCheckTuple:
        """
        Based on current profit of the trade and configured (trailing) stoploss,
        decides to sell or not
        :param current_profit: current profit as ratio
        :param low: Low value of this candle, only set in backtesting
        :param high: High value of this candle, only set in backtesting
        """
        stop_loss_value = force_stoploss if force_stoploss else self.stoploss

        # Initiate stoploss with open_rate. Does nothing if stoploss is already set.
        trade.adjust_stop_loss(trade.open_rate, stop_loss_value, initial=True)

        if self.use_custom_stoploss and trade.stop_loss < (low or current_rate):
            stop_loss_value = strategy_safe_wrapper(self.custom_stoploss, default_retval=None
                                                    )(pair=trade.pair, trade=trade,
                                                      current_time=current_time,
                                                      current_rate=current_rate,
                                                      current_profit=current_profit)
            # Sanity check - error cases will return None
            if stop_loss_value:
                # logger.info(f"{trade.pair} {stop_loss_value=} {current_profit=}")
                trade.adjust_stop_loss(current_rate, stop_loss_value)
            else:
                logger.warning("CustomStoploss function did not return valid stoploss")

        if self.trailing_stop and trade.stop_loss < (low or current_rate):
            # trailing stoploss handling
            sl_offset = self.trailing_stop_positive_offset

            # Make sure current_profit is calculated using high for backtesting.
            high_profit = current_profit if not high else trade.calc_profit_ratio(high)

            # Don't update stoploss if trailing_only_offset_is_reached is true.
            if not (self.trailing_only_offset_is_reached and high_profit < sl_offset):
                # Specific handling for trailing_stop_positive
                if self.trailing_stop_positive is not None and high_profit > sl_offset:
                    stop_loss_value = self.trailing_stop_positive
                    logger.debug(f"{trade.pair} - Using positive stoploss: {stop_loss_value} "
                                 f"offset: {sl_offset:.4g} profit: {current_profit:.4f}%")

                trade.adjust_stop_loss(high or current_rate, stop_loss_value)

        # evaluate if the stoploss was hit if stoploss is not on exchange
        # in Dry-Run, this handles stoploss logic as well, as the logic will not be different to
        # regular stoploss handling.
        if ((trade.stop_loss >= (low or current_rate)) and
                (not self.order_types.get('stoploss_on_exchange') or self.config['dry_run'])):

            sell_type = SellType.STOP_LOSS

            # If initial stoploss is not the same as current one then it is trailing.
            if trade.initial_stop_loss != trade.stop_loss:
                sell_type = SellType.TRAILING_STOP_LOSS
                logger.debug(
                    f"{trade.pair} - HIT STOP: current price at {(low or current_rate):.6f}, "
                    f"stoploss is {trade.stop_loss:.6f}, "
                    f"initial stoploss was at {trade.initial_stop_loss:.6f}, "
                    f"trade opened at {trade.open_rate:.6f}")
                logger.debug(f"{trade.pair} - Trailing stop saved "
                             f"{trade.stop_loss - trade.initial_stop_loss:.6f}")

            return SellCheckTuple(sell_type=sell_type)

        return SellCheckTuple(sell_type=SellType.NONE)

    def min_roi_reached_entry(self, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:
        """
        Based on trade duration defines the ROI entry that may have been reached.
        :param trade_dur: trade duration in minutes
        :return: minimal ROI entry value or None if none proper ROI entry was found.
        """
        # Get highest entry in ROI dict where key <= trade-duration
        roi_list = list(filter(lambda x: x <= trade_dur, self.minimal_roi.keys()))
        if not roi_list:
            return None, None
        roi_entry = max(roi_list)
        return roi_entry, self.minimal_roi[roi_entry]

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        """
        Based on trade duration, current profit of the trade and ROI configuration,
        decides whether bot should sell.
        :param current_profit: current profit as ratio
        :return: True if bot should sell at current rate
        """
        # Check if time matches and current rate is above threshold
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        _, roi = self.min_roi_reached_entry(trade_dur)
        if roi is None:
            return False
        else:
            return current_profit > roi

    def advise_all_indicators(self, data: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Populates indicators for given candle (OHLCV) data (for multiple pairs)
        Does not run advise_buy or advise_sell!
        Used by optimize operations only, not during dry / live runs.
        Using .copy() to get a fresh copy of the dataframe for every strategy run.
        Also copy on output to avoid PerformanceWarnings pandas 1.3.0 started to show.
        Has positive effects on memory usage for whatever reason - also when
        using only one strategy.
        """
        return {pair: self.advise_indicators(pair_data.copy(), {'pair': pair}).copy()
                for pair, pair_data in data.items()}

    def advise_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        This method should not be overridden.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        logger.debug(f"Populating indicators for pair {metadata.get('pair')}.")

        # call populate_indicators_Nm() which were tagged with @informative decorator.
        for inf_data, populate_fn in self._ft_informative:
            dataframe = _create_and_merge_informative_pair(
                self, dataframe, metadata, inf_data, populate_fn)

        if self._populate_fun_len == 2:
            warnings.warn("deprecated - check out the Sample strategy to see "
                          "the current function headers!", DeprecationWarning)
            return self.populate_indicators(dataframe)  # type: ignore
        else:
            return self.populate_indicators(dataframe, metadata)

    def advise_buy(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        This method should not be overridden.
        :param dataframe: DataFrame
        :param metadata: Additional information dictionary, with details like the
            currently traded pair
        :return: DataFrame with buy column
        """
        logger.debug(f"Populating buy signals for pair {metadata.get('pair')}.")

        if self._buy_fun_len == 2:
            warnings.warn("deprecated - check out the Sample strategy to see "
                          "the current function headers!", DeprecationWarning)
            return self.populate_buy_trend(dataframe)  # type: ignore
        else:
            return self.populate_buy_trend(dataframe, metadata)

    def advise_sell(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        This method should not be overridden.
        :param dataframe: DataFrame
        :param metadata: Additional information dictionary, with details like the
            currently traded pair
        :return: DataFrame with sell column
        """
        logger.debug(f"Populating sell signals for pair {metadata.get('pair')}.")
        if self._sell_fun_len == 2:
            warnings.warn("deprecated - check out the Sample strategy to see "
                          "the current function headers!", DeprecationWarning)
            return self.populate_sell_trend(dataframe)  # type: ignore
        else:
            return self.populate_sell_trend(dataframe, metadata)
