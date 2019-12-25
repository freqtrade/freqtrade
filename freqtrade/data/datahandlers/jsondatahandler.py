import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pandas import DataFrame, read_json, to_datetime

from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.data.converter import clean_ohlcv_dataframe
from freqtrade.data.history import trim_dataframe

from .idatahandler import IDataHandler


class JsonDataHandler(IDataHandler):

    _use_zip = False
    _columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str) -> List[str]:
        """
        Returns a list of all pairs available in this datadir
        """

        _tmp = [re.search(r'^(\S+)(?=\-' + timeframe + '.json)', p.name)
                for p in datadir.glob(f"*{timeframe}.{cls._get_file_extension()}")]
        # Check if regex found something and only return these results
        return [match[0].replace('_', '/') for match in _tmp if match]

    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame) -> None:
        """
        Store data in json format "values".
            format looks as follows:
            [[<date>,<open>,<high>,<low>,<close>]]
        :param pair: Pair - used to generate filename
        :timeframe: Timeframe - used to generate filename
        :data: Dataframe containing OHLCV data
        :return: None
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        _data = data.copy()
        # Convert date to int
        _data['date'] = _data['date'].astype(np.int64) // 1000 // 1000

        # Reset index, select only appropriate columns and save as json
        _data.reset_index(drop=True).loc[:, self._columns].to_json(
            filename, orient="values",
            compression='gzip' if self._use_zip else None)

    def _ohlcv_load(self, pair: str, timeframe: str,
                    timerange: Optional[TimeRange] = None,
                    fill_missing: bool = True,
                    drop_incomplete: bool = True,
                    ) -> DataFrame:
        """
        Internal method used to load data for one pair from disk.
        Implements the loading and conversation to a Pandas dataframe.
        :param pair: Pair to load data for
        :param timeframe: Ticker timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange
        :param fill_missing: Fill missing values with "No action"-candles
        :param drop_incomplete: Drop last candle assuming it may be incomplete.
        :param startup_candles: Additional candles to load at the start of the period
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        if not filename.is_file():
            return DataFrame(columns=self._columns)
        pairdata = read_json(filename, orient='values')
        pairdata.columns = self._columns
        pairdata['date'] = to_datetime(pairdata['date'],
                                       unit='ms',
                                       utc=True,
                                       infer_datetime_format=True)

        if timerange:
            pairdata = trim_dataframe(pairdata, timerange)

        return clean_ohlcv_dataframe(pairdata, timeframe,
                                     pair=pair,
                                     fill_missing=fill_missing,
                                     drop_incomplete=drop_incomplete)

    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.

        """
        raise NotImplementedError()

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        """
        Returns a list of all pairs available in this datadir
        """
        _tmp = [re.search(r'^(\S+)(?=\-trades.json)', p.name)
                for p in datadir.glob(f"*trades.{cls._get_file_extension()}")]
        # Check if regex found something and only return these results to avoid exceptions.
        return [match[0].replace('_', '/') for match in _tmp if match]

    def trades_store(self, pair: str, data: List[Dict]):
        """
        Store data
        """
        filename = self._pair_trades_filename(self._datadir, pair)
        misc.file_dump_json(filename, data, is_zip=self._use_zip)

    def trades_append(self, pair: str, data: DataFrame):
        """
        Append data to existing files
        """
        raise NotImplementedError()

    def trades_load(self, pair: str, timerange: Optional[TimeRange] = None) -> List[Dict]:
        """
        Load a pair from file, either .json.gz or .json
        # TODO: validate timerange ...
        :return: List of trades
        """
        filename = self._pair_trades_filename(self._datadir, pair)
        tradesdata = misc.file_load_json(filename)
        if not tradesdata:
            return []

        return tradesdata

    @classmethod
    def _pair_data_filename(cls, datadir: Path, pair: str, timeframe: str) -> Path:
        pair_s = pair.replace("/", "_")
        filename = datadir.joinpath(f'{pair_s}-{timeframe}.{cls._get_file_extension()}')
        return filename

    @classmethod
    def _get_file_extension(cls):
        return "json.gz" if cls._use_zip else "json"

    @classmethod
    def _pair_trades_filename(cls, datadir: Path, pair: str) -> Path:
        pair_s = pair.replace("/", "_")
        filename = datadir.joinpath(f'{pair_s}-trades.{cls._get_file_extension()}')
        return filename


class JsonGzDataHandler(JsonDataHandler):

    _use_zip = True
