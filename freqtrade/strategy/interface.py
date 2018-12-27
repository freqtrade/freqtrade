"""
IStrategy interface
This module defines the interface to apply for strategies
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, NamedTuple, Tuple
import warnings

import arrow
from pandas import DataFrame

from freqtrade import constants
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """
    Enum to distinguish between buy and sell signals
    """
    BUY = "buy"
    SELL = "sell"


class SellType(Enum):
    """
    Enum to distinguish between sell reasons
    """
    ROI = "roi"
    STOP_LOSS = "stop_loss"
    STOPLOSS_ON_EXCHANGE = "stoploss_on_exchange"
    TRAILING_STOP_LOSS = "trailing_stop_loss"
    SELL_SIGNAL = "sell_signal"
    FORCE_SELL = "force_sell"
    NONE = ""


class SellCheckTuple(NamedTuple):
    """
    NamedTuple for Sell type + reason
    """
    sell_flag: bool
    sell_type: SellType


class IStrategy(ABC):
    """
    Interface for freqtrade strategies
    Defines the mandatory structure must follow any custom strategies

    Attributes you can use:
        minimal_roi -> Dict: Minimal ROI designed for the strategy
        stoploss -> float: optimal stoploss designed for the strategy
        ticker_interval -> str: value of the ticker interval to use for the strategy
    """

    _populate_fun_len: int = 0
    _buy_fun_len: int = 0
    _sell_fun_len: int = 0
    # associated minimal roi
    minimal_roi: Dict

    # associated stoploss
    stoploss: float

    # associated ticker interval
    ticker_interval: str

    # Optional order types
    order_types: Dict = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    # Optional time in force
    order_time_in_force: Dict = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    # run "populate_indicators" only for new candle
    process_only_new_candles: bool = False

    # Dict to determine if analysis is necessary
    _last_candle_seen_per_pair: Dict[str, datetime] = {}

    def __init__(self, config: dict) -> None:
        self.config = config
        self._last_candle_seen_per_pair = {}

    @abstractmethod
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

    @abstractmethod
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

    @abstractmethod
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """

    def get_strategy_name(self) -> str:
        """
        Returns strategy class name
        """
        return self.__class__.__name__

    def analyze_ticker(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Parses the given ticker history and returns a populated DataFrame
        add several TA indicators and buy signal to it
        :return DataFrame with ticker data and indicator data
        """

        pair = str(metadata.get('pair'))

        # Test if seen this pair and last candle before.
        # always run if process_only_new_candles is set to false
        if (not self.process_only_new_candles or
                self._last_candle_seen_per_pair.get(pair, None) != dataframe.iloc[-1]['date']):
            # Defs that only make change on new candle data.
            logging.debug("TA Analysis Launched")
            dataframe = self.advise_indicators(dataframe, metadata)
            dataframe = self.advise_buy(dataframe, metadata)
            dataframe = self.advise_sell(dataframe, metadata)
            self._last_candle_seen_per_pair[pair] = dataframe.iloc[-1]['date']
        else:
            logging.debug("Skippinig TA Analysis for already analyzed candle")
            dataframe['buy'] = 0
            dataframe['sell'] = 0

        # Other Defs in strategy that want to be called every loop here
        # twitter_sell = self.watch_twitter_feed(dataframe, metadata)
        logging.debug("Loop Analysis Launched")

        return dataframe

    def get_signal(self, pair: str, interval: str,
                   dataframe: DataFrame) -> Tuple[bool, bool]:
        """
        Calculates current signal based several technical analysis indicators
        :param pair: pair in format ANT/BTC
        :param interval: Interval to use (in min)
        :param dataframe: Dataframe to analyze
        :return: (Buy, Sell) A bool-tuple indicating buy/sell signal
        """
        if not isinstance(dataframe, DataFrame) or dataframe.empty:
            logger.warning('Empty ticker history for pair %s', pair)
            return False, False

        try:
            dataframe = self.analyze_ticker(dataframe, {'pair': pair})
        except ValueError as error:
            logger.warning(
                'Unable to analyze ticker for pair %s: %s',
                pair,
                str(error)
            )
            return False, False
        except Exception as error:
            logger.exception(
                'Unexpected error when analyzing ticker for pair %s: %s',
                pair,
                str(error)
            )
            return False, False

        if dataframe.empty:
            logger.warning('Empty dataframe for pair %s', pair)
            return False, False

        latest = dataframe.iloc[-1]

        # Check if dataframe is out of date
        signal_date = arrow.get(latest['date'])
        interval_minutes = constants.TICKER_INTERVAL_MINUTES[interval]
        offset = self.config.get('exchange', {}).get('outdated_offset', 5)
        if signal_date < (arrow.utcnow().shift(minutes=-(interval_minutes * 2 + offset))):
            logger.warning(
                'Outdated history for pair %s. Last tick is %s minutes old',
                pair,
                (arrow.utcnow() - signal_date).seconds // 60
            )
            return False, False

        (buy, sell) = latest[SignalType.BUY.value] == 1, latest[SignalType.SELL.value] == 1
        logger.debug(
            'trigger: %s (pair=%s) buy=%s sell=%s',
            latest['date'],
            pair,
            str(buy),
            str(sell)
        )
        return buy, sell

    def should_sell(self, trade: Trade, rate: float, date: datetime, buy: bool,
                    sell: bool, low: float = None, high: float = None,
                    force_stoploss: float = 0) -> SellCheckTuple:
        """
        This function evaluate if on the condition required to trigger a sell has been reached
        if the threshold is reached and updates the trade record.
        :return: True if trade should be sold, False otherwise
        """

        # Set current rate to low for backtesting sell
        current_rate = low or rate
        current_profit = trade.calc_profit_percent(current_rate)

        if self.order_types.get('stoploss_on_exchange'):
            stoplossflag = SellCheckTuple(sell_flag=False, sell_type=SellType.NONE)
        else:
            stoplossflag = self.stop_loss_reached(current_rate=current_rate, trade=trade,
                                                  current_time=date, current_profit=current_profit,
                                                  force_stoploss=force_stoploss)

        if stoplossflag.sell_flag:
            return stoplossflag

        # Set current rate to low for backtesting sell
        current_rate = high or rate
        current_profit = trade.calc_profit_percent(current_rate)
        experimental = self.config.get('experimental', {})

        if buy and experimental.get('ignore_roi_if_buy_signal', False):
            logger.debug('Buy signal still active - not selling.')
            return SellCheckTuple(sell_flag=False, sell_type=SellType.NONE)

        # Check if minimal roi has been reached and no longer in buy conditions (avoiding a fee)
        if self.min_roi_reached(trade=trade, current_profit=current_profit, current_time=date):
            logger.debug('Required profit reached. Selling..')
            return SellCheckTuple(sell_flag=True, sell_type=SellType.ROI)

        if experimental.get('sell_profit_only', False):
            logger.debug('Checking if trade is profitable..')
            if trade.calc_profit(rate=rate) <= 0:
                return SellCheckTuple(sell_flag=False, sell_type=SellType.NONE)
        if sell and not buy and experimental.get('use_sell_signal', False):
            logger.debug('Sell signal received. Selling..')
            return SellCheckTuple(sell_flag=True, sell_type=SellType.SELL_SIGNAL)

        return SellCheckTuple(sell_flag=False, sell_type=SellType.NONE)

    def stop_loss_reached(self, current_rate: float, trade: Trade, current_time: datetime,
                          current_profit: float, force_stoploss: float) -> SellCheckTuple:
        """
        Based on current profit of the trade and configured (trailing) stoploss,
        decides to sell or not
        :param current_profit: current profit in percent
        """

        trailing_stop = self.config.get('trailing_stop', False)

        trade.adjust_stop_loss(trade.open_rate, force_stoploss if force_stoploss
                               else self.stoploss, initial=True)

        # evaluate if the stoploss was hit
        if self.stoploss is not None and trade.stop_loss >= current_rate:
            selltype = SellType.STOP_LOSS
            if trailing_stop:
                selltype = SellType.TRAILING_STOP_LOSS
                logger.debug(
                    f"HIT STOP: current price at {current_rate:.6f}, "
                    f"stop loss is {trade.stop_loss:.6f}, "
                    f"initial stop loss was at {trade.initial_stop_loss:.6f}, "
                    f"trade opened at {trade.open_rate:.6f}")
                logger.debug(f"trailing stop saved {trade.stop_loss - trade.initial_stop_loss:.6f}")

            logger.debug('Stop loss hit.')
            return SellCheckTuple(sell_flag=True, sell_type=selltype)

        # update the stop loss afterwards, after all by definition it's supposed to be hanging
        if trailing_stop:

            # check if we have a special stop loss for positive condition
            # and if profit is positive
            stop_loss_value = self.stoploss
            sl_offset = self.config.get('trailing_stop_positive_offset', 0.0)

            if 'trailing_stop_positive' in self.config and current_profit > sl_offset:

                # Ignore mypy error check in configuration that this is a float
                stop_loss_value = self.config.get('trailing_stop_positive')  # type: ignore
                logger.debug(f"using positive stop loss mode: {stop_loss_value} "
                             f"with offset {sl_offset:.4g} "
                             f"since we have profit {current_profit:.4f}%")

            trade.adjust_stop_loss(current_rate, stop_loss_value)

        return SellCheckTuple(sell_flag=False, sell_type=SellType.NONE)

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        """
        Based an earlier trade and current price and ROI configuration, decides whether bot should
        sell
        :return True if bot should sell at current rate
        """

        # Check if time matches and current rate is above threshold
        time_diff = (current_time.timestamp() - trade.open_date.timestamp()) / 60
        for duration, threshold in self.minimal_roi.items():
            if time_diff <= duration:
                return False
            if current_profit > threshold:
                return True

        return False

    def tickerdata_to_dataframe(self, tickerdata: Dict[str, List]) -> Dict[str, DataFrame]:
        """
        Creates a dataframe and populates indicators for given ticker data
        """
        return {pair: self.advise_indicators(pair_data, {'pair': pair})
                for pair, pair_data in tickerdata.items()}

    def advise_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        This method should not be overridden.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
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
        :param pair: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
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
        :param pair: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        if self._sell_fun_len == 2:
            warnings.warn("deprecated - check out the Sample strategy to see "
                          "the current function headers!", DeprecationWarning)
            return self.populate_sell_trend(dataframe)  # type: ignore
        else:
            return self.populate_sell_trend(dataframe, metadata)
