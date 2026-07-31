from pathlib import Path

from pandas import DataFrame, read_feather

from .arrowdatahandler import ArrowDataHandler


class FeatherDataHandler(ArrowDataHandler):
    @classmethod
    def _get_file_extension(cls):
        return "feather"

    def _store_dataframe(self, data: DataFrame, filename: Path) -> None:
        data.to_feather(filename, compression_level=9, compression="lz4")

    def _load_dataframe(self, filename: Path) -> DataFrame:
        return read_feather(filename)
