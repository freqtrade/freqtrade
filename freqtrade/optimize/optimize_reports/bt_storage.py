import logging
from pathlib import Path
from typing import Optional

from pandas import DataFrame

from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.ft_types import BacktestResultType
from freqtrade.misc import file_dump_joblib, file_dump_json
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename


logger = logging.getLogger(__name__)


def _generate_filename(recordfilename: Path, appendix: str, suffix: str) -> Path:
    """
    Generates a filename based on the provided parameters.
    :param recordfilename: Path object, which can either be a filename or a directory.
    :param appendix: use for the filename. e.g. backtest-result-<datetime>
    :param suffix: Suffix to use for the file, e.g. .json, .pkl
    :return: Generated filename as a Path object
    """
    if recordfilename.is_dir():
        filename = (recordfilename / f"backtest-result-{appendix}").with_suffix(suffix)
    else:
        filename = Path.joinpath(
            recordfilename.parent, f"{recordfilename.stem}-{appendix}"
        ).with_suffix(suffix)
    return filename


def store_backtest_stats(
    recordfilename: Path,
    stats: BacktestResultType,
    dtappendix: str,
    *,
    market_change_data: Optional[DataFrame] = None,
) -> Path:
    """
    Stores backtest results
    :param recordfilename: Path object, which can either be a filename or a directory.
        Filenames will be appended with a timestamp right before the suffix
        while for directories, <directory>/backtest-result-<datetime>.json will be used as filename
    :param stats: Dataframe containing the backtesting statistics
    :param dtappendix: Datetime to use for the filename
    """
    filename = _generate_filename(recordfilename, dtappendix, ".json")

    # Store metadata separately.
    file_dump_json(get_backtest_metadata_filename(filename), stats["metadata"])
    # Don't mutate the original stats dict.
    stats_copy = {
        "strategy": stats["strategy"],
        "strategy_comparison": stats["strategy_comparison"],
    }

    file_dump_json(filename, stats_copy)

    latest_filename = Path.joinpath(filename.parent, LAST_BT_RESULT_FN)
    file_dump_json(latest_filename, {"latest_backtest": str(filename.name)})

    if market_change_data is not None:
        filename_mc = _generate_filename(recordfilename, f"{dtappendix}_market_change", ".feather")
        market_change_data.reset_index().to_feather(
            filename_mc, compression_level=9, compression="lz4"
        )

    return filename


def _store_backtest_analysis_data(
    recordfilename: Path, data: dict[str, dict], dtappendix: str, name: str
) -> Path:
    """
    Stores backtest trade candles for analysis
    :param recordfilename: Path object, which can either be a filename or a directory.
        Filenames will be appended with a timestamp right before the suffix
        while for directories, <directory>/backtest-result-<datetime>_<name>.pkl will be used
        as filename
    :param candles: Dict containing the backtesting data for analysis
    :param dtappendix: Datetime to use for the filename
    :param name: Name to use for the file, e.g. signals, rejected
    """
    filename = _generate_filename(recordfilename, f"{dtappendix}_{name}", ".pkl")

    file_dump_joblib(filename, data)

    return filename


def store_backtest_analysis_results(
    recordfilename: Path,
    candles: dict[str, dict],
    trades: dict[str, dict],
    exited: dict[str, dict],
    dtappendix: str,
) -> None:
    _store_backtest_analysis_data(recordfilename, candles, dtappendix, "signals")
    _store_backtest_analysis_data(recordfilename, trades, dtappendix, "rejected")
    _store_backtest_analysis_data(recordfilename, exited, dtappendix, "exited")
