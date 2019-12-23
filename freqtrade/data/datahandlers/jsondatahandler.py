import re
from pathlib import Path
from typing import Dict, List, Optional

from pandas import DataFrame

from freqtrade import misc
from freqtrade.configuration import TimeRange

from .idatahandler import IDataHandler


class JsonDataHandler(IDataHandler):

    _use_zip = False

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str) -> List[str]:
        """
        Returns a list of all pairs available in this datadir
        """
        return [re.search(r'^(\S+)(?=\-' + timeframe + '.json)', p.name)[0].replace('_', ' /')
                for p in datadir.glob(f"*{timeframe}.{cls._get_file_extension()}")]

    def ohlcv_store(self, timeframe: str, data: DataFrame):
        """
        Store data
        """
        raise NotImplementedError()

    def ohlcv_append(self, timeframe: str, data: DataFrame):
        """
        Append data to existing files
        """
        raise NotImplementedError()

    def ohlcv_load(self, timeframe: str, timerange: Optional[TimeRange] = None) -> DataFrame:
        """
        Load data for one pair
        :return: Dataframe
        """
        filename = JsonDataHandler._pair_data_filename(self.datadir, self._pair,
                                                       self._pair, timeframe)
        pairdata = misc.file_load_json(filename)
        if not pairdata:
            return []

        if timerange:
            pairdata = IDataHandler.trim_tickerlist(pairdata, timerange)
        return pairdata

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        """
        Returns a list of all pairs available in this datadir
        """
        return [re.search(r'^(\S+)(?=\-trades.json)', p.name)[0].replace('_', '/')
                for p in datadir.glob(f"*trades.{cls._get_file_extension()}")]

    def trades_store(self, data: List[Dict]):
        """
        Store data
        """
        filename = self._pair_trades_filename(self._datadir, self._pair)
        misc.file_dump_json(filename, data, is_zip=self._use_zip)

    def trades_append(self, data: DataFrame):
        """
        Append data to existing files
        """
        raise NotImplementedError()

    def trades_load(self, timerange: Optional[TimeRange] = None) -> List[Dict]:
        """
        Load a pair from file, either .json.gz or .json
        # TODO: validate timerange ...
        :return: List of trades
        """
        filename = self._pair_trades_filename(self._datadir, self._pair)
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
