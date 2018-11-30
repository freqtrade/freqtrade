"""
Dataprovider
Responsible to provide data to the bot
including Klines, tickers, historic data
Common Interface for bot and strategy to access data.
"""
import logging

from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class DataProvider(object):

    def __init__(self, exchange: Exchange) -> None:
        pass

    def refresh() -> None:
        """
        Refresh data, called with each cycle
        """
        pass

    def kline(pair: str):
        """
        get ohlcv data for the given pair
        """
        pass

    def historic_kline(pair: str):
        """
        get historic ohlcv data stored for backtesting
        """
        pass

    def ticker(pair: str):
        pass

    def orderbook(pair: str, max: int):
        pass

    def balance(pair):
        # TODO: maybe use wallet directly??
        pass
