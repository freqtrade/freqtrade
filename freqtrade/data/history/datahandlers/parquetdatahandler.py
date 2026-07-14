import logging
from pathlib import Path

from pandas import DataFrame, read_parquet

from .arrowdatahandler import ArrowDataHandler


logger = logging.getLogger(__name__)


class ParquetDataHandler(ArrowDataHandler):
    @classmethod
    def _get_file_extension(cls):
        return "parquet"

    def _store_dataframe(self, data: DataFrame, filename: Path) -> None:
        data.to_parquet(filename)

    def _load_dataframe(self, filename: Path) -> DataFrame:
        return read_parquet(filename)
