import re
from pathlib import Path
from typing import Dict, List, Optional

from pandas import DataFrame

from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.data.converter import parse_ticker_dataframe

from .idatahandler import IDataHandler


class JsonDataHandler(IDataHandler):

    _use_zip = False

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
        Store data
        """
        filename = JsonDataHandler._pair_data_filename(self._datadir, pair, timeframe)
        misc.file_dump_json(filename, data, is_zip=self._use_zip)

    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame) -> None:
        """
        Append data to existing files
        """
        raise NotImplementedError()

    def _ohlcv_load(self, pair: str, timeframe: str,
                    timerange: Optional[TimeRange] = None,
                    fill_up_missing: bool = True,
                    drop_incomplete: bool = True,
                    ) -> DataFrame:
        """
        Load data for one pair from disk.
        Implements the loading and conversation to a Pandas dataframe.
        :return: Dataframe
        """
        filename = JsonDataHandler._pair_data_filename(self._datadir, pair, timeframe)
        pairdata = misc.file_load_json(filename)
        if not pairdata:
            return DataFrame()

        if timerange:
            pairdata = IDataHandler.trim_tickerlist(pairdata, timerange)
        return parse_ticker_dataframe(pairdata, timeframe,
                                      pair=self._pair,
                                      fill_missing=fill_up_missing,
                                      drop_incomplete=drop_incomplete)
        return pairdata

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
