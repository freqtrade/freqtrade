"""
Helpers when analyzing backtest data
"""

import logging
import zipfile
from copy import copy
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.exceptions import ConfigurationError, OperationalException
from freqtrade.ft_types import BacktestHistoryEntryType, BacktestResultType
from freqtrade.misc import file_dump_json, json_load
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename
from freqtrade.persistence import LocalTrade, Trade, init_db


logger = logging.getLogger(__name__)

# Newest format
BT_DATA_COLUMNS = [
    "pair",
    "stake_amount",
    "max_stake_amount",
    "amount",
    "open_date",
    "close_date",
    "open_rate",
    "close_rate",
    "fee_open",
    "fee_close",
    "trade_duration",
    "profit_ratio",
    "profit_abs",
    "exit_reason",
    "initial_stop_loss_abs",
    "initial_stop_loss_ratio",
    "stop_loss_abs",
    "stop_loss_ratio",
    "min_rate",
    "max_rate",
    "is_open",
    "enter_tag",
    "leverage",
    "is_short",
    "open_timestamp",
    "close_timestamp",
    "orders",
    "funding_fees",
]


def get_latest_optimize_filename(directory: Path | str, variant: str) -> str:
    """
    Get latest backtest export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :param variant: 'backtest' or 'hyperopt' - the method to return
    :return: string containing the filename of the latest backtest result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Directory '{directory}' does not exist.")
    filename = directory / LAST_BT_RESULT_FN

    if not filename.is_file():
        raise ValueError(
            f"Directory '{directory}' does not seem to contain backtest statistics yet."
        )

    with filename.open() as file:
        data = json_load(file)

    if f"latest_{variant}" not in data:
        raise ValueError(f"Invalid '{LAST_BT_RESULT_FN}' format.")

    return data[f"latest_{variant}"]


def get_latest_backtest_filename(directory: Path | str) -> str:
    """
    Get latest backtest export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :return: string containing the filename of the latest backtest result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    return get_latest_optimize_filename(directory, "backtest")


def get_latest_hyperopt_filename(directory: Path | str) -> str:
    """
    Get latest hyperopt export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :return: string containing the filename of the latest hyperopt result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    try:
        return get_latest_optimize_filename(directory, "hyperopt")
    except ValueError:
        # Return default (legacy) pickle filename
        return "hyperopt_results.pickle"


def get_latest_hyperopt_file(directory: Path | str, predef_filename: str | None = None) -> Path:
    """
    Get latest hyperopt export based on '.last_result.json'.
    :param directory: Directory to search for last result
    :return: string containing the filename of the latest hyperopt result
    :raises: ValueError in the following cases:
        * Directory does not exist
        * `directory/.last_result.json` does not exist
        * `directory/.last_result.json` has the wrong content
    """
    if isinstance(directory, str):
        directory = Path(directory)
    if predef_filename:
        if Path(predef_filename).is_absolute():
            raise ConfigurationError(
                "--hyperopt-filename expects only the filename, not an absolute path."
            )
        return directory / predef_filename
    return directory / get_latest_hyperopt_filename(directory)


def load_backtest_metadata(filename: Path | str) -> dict[str, Any]:
    """
    Read metadata dictionary from backtest results file without reading and deserializing entire
    file.
    :param filename: path to backtest results file.
    :return: metadata dict or None if metadata is not present.
    """
    filename = get_backtest_metadata_filename(filename)
    try:
        with filename.open() as fp:
            return json_load(fp)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise OperationalException("Unexpected error while loading backtest metadata.") from e


def _normalize_filename(file_or_directory: Path | str, filename: Path | str | None) -> Path:
    """
    Normalize the filename by ensuring it is a Path object.
    :param file_or_directory: The directory or file to normalize.
    :param filename: The filename to normalize.
    :return: A Path object representing the normalized filename.
    """
    if isinstance(file_or_directory, str):
        file_or_directory = Path(file_or_directory)
    if file_or_directory.is_dir():
        if not filename:
            filename = get_latest_backtest_filename(file_or_directory)
        if Path(filename).is_file():
            fn = Path(filename)
        else:
            fn = file_or_directory / filename
    else:
        fn = file_or_directory
    return fn


def load_backtest_stats(
    file_or_directory: Path | str, filename: Path | str | None = None
) -> BacktestResultType:
    """
    Load backtest statistics file.
    :param file_or_directory: pathlib.Path object, or string pointing to the directory,
        or absolute/relative path to the backtest results file.
    :param filename: Optional filename to load from (if different from the main filename).
        Only valid when loading from a directory.
    :return: a dictionary containing the resulting file.
    """
    fn = _normalize_filename(file_or_directory, filename)

    if not fn.is_file():
        raise ValueError(f"File or directory {fn} does not exist.")
    logger.info(f"Loading backtest result from {fn}")

    if fn.suffix == ".zip":
        data = json_load(
            StringIO(load_file_from_zip(fn, fn.with_suffix(".json").name).decode("utf-8"))
        )
    else:
        with fn.open() as file:
            data = json_load(file)

    # Legacy list format does not contain metadata.
    if isinstance(data, dict):
        data["metadata"] = load_backtest_metadata(fn)
    return data


def load_and_merge_backtest_result(strategy_name: str, filename: Path, results: dict[str, Any]):
    """
    Load one strategy from multi-strategy result and merge it with results
    :param strategy_name: Name of the strategy contained in the result
    :param filename: Backtest-result-filename to load
    :param results: dict to merge the result to.
    """
    bt_data = load_backtest_stats(filename)
    k: Literal["metadata", "strategy"]
    for k in ("metadata", "strategy"):
        results[k][strategy_name] = bt_data[k][strategy_name]
    results["metadata"][strategy_name]["filename"] = filename.stem
    comparison = bt_data["strategy_comparison"]
    for i in range(len(comparison)):
        if comparison[i]["key"] == strategy_name:
            results["strategy_comparison"].append(comparison[i])
            break


def _get_backtest_files(dirname: Path) -> list[Path]:
    # Get both json and zip files separately and combine the results
    json_files = dirname.glob("backtest-result-*-[0-9][0-9]*.json")
    zip_files = dirname.glob("backtest-result-*-[0-9][0-9]*.zip")
    return list(reversed(sorted(list(json_files) + list(zip_files))))


def _extract_backtest_result(filename: Path) -> list[BacktestHistoryEntryType]:
    metadata = load_backtest_metadata(filename)
    return [
        {
            "filename": filename.stem,
            "strategy": s,
            "run_id": v["run_id"],
            "notes": v.get("notes", ""),
            # Backtest "run" time
            "backtest_start_time": v["backtest_start_time"],
            # Backtest timerange
            "backtest_start_ts": v.get("backtest_start_ts", None),
            "backtest_end_ts": v.get("backtest_end_ts", None),
            "timeframe": v.get("timeframe", None),
            "timeframe_detail": v.get("timeframe_detail", None),
        }
        for s, v in metadata.items()
    ]


def get_backtest_result(filename: Path) -> list[BacktestHistoryEntryType]:
    """
    Get backtest result read from metadata file
    """
    return _extract_backtest_result(filename)


def get_backtest_resultlist(dirname: Path) -> list[BacktestHistoryEntryType]:
    """
    Get list of backtest results read from metadata files
    """
    return [
        result
        for filename in _get_backtest_files(dirname)
        for result in _extract_backtest_result(filename)
    ]


def delete_backtest_result(file_abs: Path):
    """
    Delete backtest result file and corresponding metadata file.
    """
    # *.meta.json
    logger.info(f"Deleting backtest result file: {file_abs.name}")

    for file in file_abs.parent.glob(f"{file_abs.stem}*"):
        logger.info(f"Deleting file: {file}")
        file.unlink()


def update_backtest_metadata(filename: Path, strategy: str, content: dict[str, Any]):
    """
    Updates backtest metadata file with new content.
    :raises: ValueError if metadata file does not exist, or strategy is not in this file.
    """
    metadata = load_backtest_metadata(filename)
    if not metadata:
        raise ValueError("File does not exist.")
    if strategy not in metadata:
        raise ValueError("Strategy not in metadata.")
    metadata[strategy].update(content)
    # Write data again.
    file_dump_json(get_backtest_metadata_filename(filename), metadata)


def get_backtest_market_change(filename: Path, include_ts: bool = True) -> pd.DataFrame:
    """
    Read backtest market change file.
    """
    if filename.suffix == ".zip":
        data = load_file_from_zip(filename, f"{filename.stem}_market_change.feather")
        df = pd.read_feather(BytesIO(data))
    else:
        df = pd.read_feather(filename)
    if include_ts:
        df.loc[:, "__date_ts"] = df.loc[:, "date"].astype(np.int64) // 1000 // 1000
    return df


def find_existing_backtest_stats(
    dirname: Path | str, run_ids: dict[str, str], min_backtest_date: datetime | None = None
) -> dict[str, Any]:
    """
    Find existing backtest stats that match specified run IDs and load them.
    :param dirname: pathlib.Path object, or string pointing to the file.
    :param run_ids: {strategy_name: id_string} dictionary.
    :param min_backtest_date: do not load a backtest older than specified date.
    :return: results dict.
    """
    # Copy so we can modify this dict without affecting parent scope.
    run_ids = copy(run_ids)
    dirname = Path(dirname)
    results: dict[str, Any] = {
        "metadata": {},
        "strategy": {},
        "strategy_comparison": [],
    }

    for filename in _get_backtest_files(dirname):
        metadata = load_backtest_metadata(filename)
        if not metadata:
            # Files are sorted from newest to oldest. When file without metadata is encountered it
            # is safe to assume older files will also not have any metadata.
            break

        for strategy_name, run_id in list(run_ids.items()):
            strategy_metadata = metadata.get(strategy_name, None)
            if not strategy_metadata:
                # This strategy is not present in analyzed backtest.
                continue

            if min_backtest_date is not None:
                backtest_date = strategy_metadata["backtest_start_time"]
                backtest_date = datetime.fromtimestamp(backtest_date, tz=UTC)
                if backtest_date < min_backtest_date:
                    # Do not use a cached result for this strategy as first result is too old.
                    del run_ids[strategy_name]
                    continue

            if strategy_metadata["run_id"] == run_id:
                del run_ids[strategy_name]
                load_and_merge_backtest_result(strategy_name, filename, results)

        if len(run_ids) == 0:
            break
    return results


def _load_backtest_data_df_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compatibility support for older backtest data.
    """
    df["open_date"] = pd.to_datetime(df["open_date"], utc=True)
    df["close_date"] = pd.to_datetime(df["close_date"], utc=True)
    # Compatibility support for pre short Columns
    if "is_short" not in df.columns:
        df["is_short"] = False
    if "leverage" not in df.columns:
        df["leverage"] = 1.0
    if "enter_tag" not in df.columns:
        df["enter_tag"] = df["buy_tag"]
        df = df.drop(["buy_tag"], axis=1)
    if "max_stake_amount" not in df.columns:
        df["max_stake_amount"] = df["stake_amount"]
    if "orders" not in df.columns:
        df["orders"] = None
    if "funding_fees" not in df.columns:
        df["funding_fees"] = 0.0
    return df


def load_backtest_data(
    file_or_directory: Path | str, strategy: str | None = None, filename: Path | str | None = None
) -> pd.DataFrame:
    """
    Load backtest data file, returns a dataframe with the individual trades.
    :param file_or_directory: pathlib.Path object, or string pointing to the directory,
        or absolute/relative path to the backtest results file.
    :param strategy: Strategy to load - mainly relevant for multi-strategy backtests
                     Can also serve as protection to load the correct result.
    :param filename: Optional filename to load from (if different from the main filename).
        Only valid when loading from a directory.
    :return: a dataframe with the analysis results
    :raise: ValueError if loading goes wrong.
    """
    data = load_backtest_stats(file_or_directory, filename)
    if not isinstance(data, list):
        # new, nested format
        if "strategy" not in data:
            raise ValueError("Unknown dataformat.")

        if not strategy:
            if len(data["strategy"]) == 1:
                strategy = next(iter(data["strategy"].keys()))
            else:
                raise ValueError(
                    "Detected backtest result with more than one strategy. "
                    "Please specify a strategy."
                )

        if strategy not in data["strategy"]:
            raise ValueError(
                f"Strategy {strategy} not available in the backtest result. "
                f"Available strategies are '{','.join(data['strategy'].keys())}'"
            )

        data = data["strategy"][strategy]["trades"]
        df = pd.DataFrame(data)
        if not df.empty:
            df = _load_backtest_data_df_compatibility(df)

    else:
        # old format - only with lists.
        raise OperationalException(
            "Backtest-results with only trades data are no longer supported."
        )
    if not df.empty:
        df = df.sort_values("open_date").reset_index(drop=True)
    return df


def load_file_from_zip(zip_path: Path, filename: str) -> bytes:
    """
    Load a file from a zip file
    :param zip_path: Path to the zip file
    :param filename: Name of the file to load
    :return: Bytes of the file
    :raises: ValueError if loading goes wrong.
    """
    try:
        with zipfile.ZipFile(zip_path) as zipf:
            try:
                with zipf.open(filename) as file:
                    return file.read()
            except KeyError:
                logger.error(f"File {filename} not found in zip: {zip_path}")
                raise ValueError(f"File {filename} not found in zip: {zip_path}") from None
    except FileNotFoundError:
        raise ValueError(f"Zip file {zip_path} not found.")
    except zipfile.BadZipFile:
        logger.error(f"Bad zip file: {zip_path}.")
        raise ValueError(f"Bad zip file: {zip_path}.") from None


def load_backtest_analysis_data(
    file_or_directory: Path,
    name: Literal["signals", "rejected", "exited"],
    filename: Path | str | None = None,
):
    """
    Load backtest analysis data either from a pickle file or from within a zip file
    :param file_or_directory: pathlib.Path object, or string pointing to the directory,
        or absolute/relative path to the backtest results file.
    :param name: Name of the analysis data to load (signals, rejected, exited)
    :param filename: Optional filename to load from (if different from the main filename).
        Only valid when loading from a directory.
    :return: Analysis data
    """
    import joblib

    zip_path = _normalize_filename(file_or_directory, filename)

    if zip_path.suffix == ".zip":
        # Load from zip file
        analysis_name = f"{zip_path.stem}_{name}.pkl"
        data = load_file_from_zip(zip_path, analysis_name)
        if not data:
            return None
        loaded_data = joblib.load(BytesIO(data))

        logger.info(f"Loaded {name} candles from zip: {str(zip_path)}:{analysis_name}")
        return loaded_data

    else:
        # Load from separate pickle file
        if file_or_directory.is_dir():
            scpf = Path(file_or_directory, f"{zip_path.stem}_{name}.pkl")
        else:
            scpf = Path(file_or_directory.parent / f"{file_or_directory.stem}_{name}.pkl")

        try:
            with scpf.open("rb") as scp:
                loaded_data = joblib.load(scp)
                logger.info(f"Loaded {name} candles: {str(scpf)}")
                return loaded_data
        except Exception:
            logger.exception(f"Cannot load {name} data from pickled results.")
            return None


def trade_list_to_dataframe(trades: list[Trade] | list[LocalTrade]) -> pd.DataFrame:
    """
    Convert list of Trade objects to pandas Dataframe
    :param trades: List of trade objects
    :return: Dataframe with BT_DATA_COLUMNS
    """
    df = pd.DataFrame.from_records([t.to_json(True) for t in trades], columns=BT_DATA_COLUMNS)
    if len(df) > 0:
        df["close_date"] = pd.to_datetime(df["close_timestamp"], unit="ms", utc=True)
        df["open_date"] = pd.to_datetime(df["open_timestamp"], unit="ms", utc=True)
        df["close_rate"] = df["close_rate"].astype("float64")
    return df


def load_trades_from_db(db_url: str, strategy: str | None = None) -> pd.DataFrame:
    """
    Load trades from a DB (using dburl)
    :param db_url: Sqlite url (default format sqlite:///tradesv3.dry-run.sqlite)
    :param strategy: Strategy to load - mainly relevant for multi-strategy backtests
                     Can also serve as protection to load the correct result.
    :return: Dataframe containing Trades
    """
    init_db(db_url)

    filters = []
    if strategy:
        filters.append(Trade.strategy == strategy)
    trades = trade_list_to_dataframe(list(Trade.get_trades(filters).all()))

    return trades


def load_trades(
    source: str,
    db_url: str,
    exportfilename: Path,
    no_trades: bool = False,
    strategy: str | None = None,
) -> pd.DataFrame:
    """
    Based on configuration option 'trade_source':
    * loads data from DB (using `db_url`)
    * loads data from backtestfile (using `exportfilename`)
    :param source: "DB" or "file" - specify source to load from
    :param db_url: sqlalchemy formatted url to a database
    :param exportfilename: Json file generated by backtesting
    :param no_trades: Skip using trades, only return backtesting data columns
    :return: DataFrame containing trades
    """
    if no_trades:
        df = pd.DataFrame(columns=BT_DATA_COLUMNS)
        return df

    if source == "DB":
        return load_trades_from_db(db_url)
    elif source == "file":
        return load_backtest_data(exportfilename, strategy)


def extract_trades_of_period(
    dataframe: pd.DataFrame, trades: pd.DataFrame, date_index=False
) -> pd.DataFrame:
    """
    Compare trades and backtested pair DataFrames to get trades performed on backtested period
    :return: the DataFrame of a trades of period
    """
    if date_index:
        trades_start = dataframe.index[0]
        trades_stop = dataframe.index[-1]
    else:
        trades_start = dataframe.iloc[0]["date"]
        trades_stop = dataframe.iloc[-1]["date"]
    trades = trades.loc[
        (trades["open_date"] >= trades_start) & (trades["close_date"] <= trades_stop)
    ]
    return trades
