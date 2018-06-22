"""
Functions to analyze ticker data with indicators and produce buy and sell signals
"""
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple

import arrow
from pandas import DataFrame, to_datetime

from freqtrade import constants
from freqtrade.exchange import Exchange
from freqtrade.persistence import Trade
from freqtrade.strategy.resolver import StrategyResolver, IStrategy


logger = logging.getLogger(__name__)


class SignalType(Enum):
    """
    Enum to distinguish between buy and sell signals
    """
    BUY = "buy"
    SELL = "sell"


class Analyze(object):
    """
    Analyze class contains everything the bot need to determine if the situation is good for
    buying or selling.
    """
    def __init__(self, config: dict) -> None:
        """
        Init Analyze
        :param config: Bot configuration (use the one from Configuration())
        """
        self.config = config
        self.strategy: IStrategy = StrategyResolver(self.config).strategy

    @staticmethod
    def parse_ticker_dataframe(ticker: list) -> DataFrame:
        """
        Analyses the trend for the given ticker history
        :param ticker: See exchange.get_ticker_history
        :return: DataFrame
        """
        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        frame = DataFrame(ticker, columns=cols)

        frame['date'] = to_datetime(frame['date'],
                                    unit='ms',
                                    utc=True,
                                    infer_datetime_format=True)

        # group by index and aggregate results to eliminate duplicate ticks
        frame = frame.groupby(by='date', as_index=False, sort=True).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'max',
        })
        frame.drop(frame.tail(1).index, inplace=True)     # eliminate partial candle
        return frame

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        return self.strategy.populate_indicators(dataframe=dataframe)

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        return self.strategy.populate_buy_trend(dataframe=dataframe)

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        return self.strategy.populate_sell_trend(dataframe=dataframe)

    def get_ticker_interval(self) -> str:
        """
        Return ticker interval to use
        :return: Ticker interval value to use
        """
        return self.strategy.ticker_interval

    def analyze_ticker(self, ticker_history: List[Dict]) -> DataFrame:
        """
        Parses the given ticker history and returns a populated DataFrame
        add several TA indicators and buy signal to it
        :return DataFrame with ticker data and indicator data
        """
        dataframe = self.parse_ticker_dataframe(ticker_history)
        dataframe = self.populate_indicators(dataframe)
        dataframe = self.populate_buy_trend(dataframe)
        dataframe = self.populate_sell_trend(dataframe)
        return dataframe

    def get_signal(self, exchange: Exchange, pair: str, interval: str) -> Tuple[bool, bool]:
        """
        Calculates current signal based several technical analysis indicators
        :param pair: pair in format ANT/BTC
        :param interval: Interval to use (in min)
        :return: (Buy, Sell) A bool-tuple indicating buy/sell signal
        """
        ticker_hist = exchange.get_ticker_history(pair, interval)
        if not ticker_hist:
            logger.warning('Empty ticker history for pair %s', pair)
            return False, False

        try:
            dataframe = self.analyze_ticker(ticker_hist)
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
        if signal_date < (arrow.utcnow() - timedelta(minutes=(interval_minutes + 5))):
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

    def should_sell(self, trade: Trade, rate: float, date: datetime, buy: bool, sell: bool) -> bool:
        """
        This function evaluate if on the condition required to trigger a sell has been reached
        if the threshold is reached and updates the trade record.
        :return: True if trade should be sold, False otherwise
        """
        # Check if minimal roi has been reached and no longer in buy conditions (avoiding a fee)
        if self.min_roi_reached(trade=trade, current_rate=rate, current_time=date):
            logger.debug('Required profit reached. Selling..')
            return True

        # Experimental: Check if the trade is profitable before selling it (avoid selling at loss)
        if self.config.get('experimental', {}).get('sell_profit_only', False):
            logger.debug('Checking if trade is profitable..')
            if trade.calc_profit(rate=rate) <= 0:
                return False

        if sell and not buy and self.config.get('experimental', {}).get('use_sell_signal', False):
            logger.debug('Sell signal received. Selling..')
            return True

        return False

    def min_roi_reached(self, trade: Trade, current_rate: float, current_time: datetime) -> bool:
        """
        Based an earlier trade and current price and ROI configuration, decides whether bot should
        sell
        :return True if bot should sell at current rate
        """
        current_profit = trade.calc_profit_percent(current_rate)
        if self.strategy.stoploss is not None and current_profit < self.strategy.stoploss:
            logger.debug('Stop loss hit.')
            return True

        # Check if time matches and current rate is above threshold
        time_diff = (current_time.timestamp() - trade.open_date.timestamp()) / 60
        for duration, threshold in self.strategy.minimal_roi.items():
            if time_diff <= duration:
                return False
            if current_profit > threshold:
                return True

        return False

    def tickerdata_to_dataframe(self, tickerdata: Dict[str, List]) -> Dict[str, DataFrame]:
        """
        Creates a dataframe and populates indicators for given ticker data
        """
        return {pair: self.populate_indicators(self.parse_ticker_dataframe(pair_data))
                for pair, pair_data in tickerdata.items()}
