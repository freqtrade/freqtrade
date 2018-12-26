"""
Dataprovider
Responsible to provide data to the bot
including Klines, tickers, historic data
Common Interface for bot and strategy to access data.
"""
import logging
from pathlib import Path
from typing import List

from pandas import DataFrame

from freqtrade.data.history import load_pair_history
from freqtrade.exchange import Exchange
from freqtrade.state import RunMode

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

    @property
    def available_pairs(self) -> List[str]:
        """
        Return a list of pairs for which data is currently cached.
        Should be whitelist + open trades.
        """
        return list(self._exchange._klines.keys())

    def ohlcv(self, pair: str, copy: bool = True) -> List[str]:
        """
        get ohlcv data for the given pair as DataFrame
        :param pair: pair to get the data for
        :param copy: copy dataframe before returning.
                     Use false only for RO operations (where the dataframe is not modified)
        """
        # TODO: Should not be stored in exchange but in this class
        return self._exchange.klines(pair, copy)

    def historic_ohlcv(self, pair: str, ticker_interval: str) -> DataFrame:
        """
        get historic ohlcv data stored for backtesting
        """
        return load_pair_history(pair=pair,
                                 ticker_interval=ticker_interval,
                                 refresh_pairs=False,
                                 datadir=Path(self._config['datadir']) if self._config.get(
                                     'datadir') else None
                                 )

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
    def runmode(self) -> RunMode:
        """
        Get runmode of the bot
        can be "live", "dry-run", "backtest", "edgecli", "hyperopt".
        """
        return RunMode(self._config.get('runmode', RunMode.OTHER))
