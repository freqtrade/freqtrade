"""
Functions to analyze ticker data with indicators and produce buy and sell signals
"""
import arrow
from datetime import datetime, timedelta
from enum import Enum
from pandas import DataFrame, to_datetime
from typing import Dict, List
from freqtrade.exchange import get_ticker_history
from freqtrade.logger import Logger
from freqtrade.strategy.strategy import Strategy
from freqtrade.persistence import Trade


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
        self.logger = Logger(name=__name__).get_logger()

        self.config = config
        self.strategy = Strategy()
        self.strategy.init(self.config)

    @staticmethod
    def parse_ticker_dataframe(ticker: list) -> DataFrame:
        """
        Analyses the trend for the given ticker history
        :param ticker: See exchange.get_ticker_history
        :return: DataFrame
        """
        columns = {'C': 'close', 'V': 'volume', 'O': 'open', 'H': 'high', 'L': 'low', 'T': 'date'}
        frame = DataFrame(ticker) \
            .rename(columns=columns)
        if 'BV' in frame:
            frame.drop('BV', 1, inplace=True)
        frame['date'] = to_datetime(frame['date'], utc=True, infer_datetime_format=True)
        frame.sort_values('date', inplace=True)
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

    # FIX: Maybe return False, if an error has occured,
    #      Otherwise we might mask an error as an non-signal-scenario
    def get_signal(self, pair: str, interval: int) -> (bool, bool):
        """
        Calculates current signal based several technical analysis indicators
        :param pair: pair in format BTC_ANT or BTC-ANT
        :return: (Buy, Sell) A bool-tuple indicating buy/sell signal
        """
        ticker_hist = get_ticker_history(pair, interval)
        if not ticker_hist:
            self.logger.warning('Empty ticker history for pair %s', pair)
            return (False, False)  # return False ?

        try:
            dataframe = self.analyze_ticker(ticker_hist)
        except ValueError as error:
            self.logger.warning(
                'Unable to analyze ticker for pair %s: %s',
                pair,
                str(error)
            )
            return (False, False)  # return False ?
        except Exception as error:
            self.logger.exception(
                'Unexpected error when analyzing ticker for pair %s: %s',
                pair,
                str(error)
            )
            return (False, False)  # return False ?

        if dataframe.empty:
            self.logger.warning('Empty dataframe for pair %s', pair)
            return (False, False)  # return False ?

        latest = dataframe.iloc[-1]

        # Check if dataframe is out of date
        signal_date = arrow.get(latest['date'])
        if signal_date < arrow.now() - timedelta(minutes=(interval + 5)):
            self.logger.warning('Too old dataframe for pair %s', pair)
            return (False, False)  # return False ?

        (buy, sell) = latest[SignalType.BUY.value] == 1, latest[SignalType.SELL.value] == 1
        self.logger.debug(
            'trigger: %s (pair=%s) buy=%s sell=%s',
            latest['date'],
            pair,
            str(buy),
            str(sell)
        )
        return (buy, sell)

    def should_sell(self, trade: Trade, rate: float, date: datetime, buy: bool, sell: bool) -> bool:
        """
        This function evaluate if on the condition required to trigger a sell has been reached
        if the threshold is reached and updates the trade record.
        :return: True if trade should be sold, False otherwise
        """
        # Check if minimal roi has been reached and no longer in buy conditions (avoiding a fee)
        if self.min_roi_reached(trade=trade, current_rate=rate, current_time=date):
            self.logger.debug('Executing sell due to ROI ...')
            return True

        # Experimental: Check if the trade is profitable before selling it (avoid selling at loss)
        if self.config.get('experimental', {}).get('sell_profit_only', False):
            self.logger.debug('Checking if trade is profitable ...')
            if trade.calc_profit(rate=rate) <= 0:
                return False

        if sell and not buy and self.config.get('experimental', {}).get('use_sell_signal', False):
            self.logger.debug('Executing sell due to sell signal ...')
            return True

        return False

    def min_roi_reached(self, trade: Trade, current_rate: float, current_time: datetime) -> bool:
        """
        Based an earlier trade and current price and ROI configuration, decides whether bot should
        sell
        :return True if bot should sell at current rate
        """
        current_profit = trade.calc_profit_percent(current_rate)
        if self.strategy.stoploss is not None and current_profit < float(self.strategy.stoploss):
            self.logger.debug('Stop loss hit.')
            return True

        # Check if time matches and current rate is above threshold
        time_diff = (current_time - trade.open_date).total_seconds() / 60
        for duration, threshold in sorted(self.strategy.minimal_roi.items()):
            if time_diff > float(duration) and current_profit > threshold:
                return True

        self.logger.debug(
            'Threshold not reached. (cur_profit: %1.2f%%)',
            float(current_profit) * 100.0
        )
        return False
