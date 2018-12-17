"""
Dataprovider
Responsible to provide data to the bot
including Klines, tickers, historic data
Common Interface for bot and strategy to access data.
"""
import logging
from typing import List, Dict

from pandas import DataFrame

from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class DataProvider(object):

    def __init__(self, config: dict, exchange: Exchange) -> None:
        self._config = config
        self._exchange = exchange

    def refresh(self, pairlist: List[str]) -> None:
        """
        Refresh data, called with each cycle
        """
        self._exchange.refresh_tickers(pairlist, self._config['ticker_interval'])

    def ohlcv(self, pair: str) -> List[str]:
        """
        get ohlcv data for the given pair as DataFrame
        """
        # TODO: Should not be stored in exchange but in this class
        # TODO: should return dataframe, not list
        return self._exchange.klines(pair)

    def historic_ohlcv(self, pair: str) -> DataFrame:
        """
        get historic ohlcv data stored for backtesting
        """
        pass

    def ticker(self, pair: str):
        """
        Return last ticker data
        """
        pass

    def orderbook(self, pair: str, max: int):
        """
        return latest orderbook data
        """
        pass

    def balance(self, pair):
        # TODO: maybe use wallet directly??
        pass

    @property
    def runmode(self) -> str:
        """
        Get runmode of the bot
        can be "live", "dry-run", "backtest", "edgecli", "hyperopt".
        """
        return self._config.get['runmode']
