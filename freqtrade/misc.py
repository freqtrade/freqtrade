"""
Various tool function for Freqtrade and scripts
"""
import gzip
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, TextIO, Union
from urllib.parse import urlparse

import pandas as pd
import rapidjson

from freqtrade.enums import SignalTagType, SignalType


logger = logging.getLogger(__name__)


def file_dump_json(filename: Path, data: Any, is_zip: bool = False, log: bool = True) -> None:
    """
    Dump JSON data into a file
    :param filename: file to create
    :param is_zip: if file should be zip
    :param data: JSON Data to save
    :return:
    """

    if is_zip:
        if filename.suffix != '.gz':
            filename = filename.with_suffix('.gz')
        if log:
            logger.info(f'dumping json to "{filename}"')

        with gzip.open(filename, 'w') as fpz:
            rapidjson.dump(data, fpz, default=str, number_mode=rapidjson.NM_NATIVE)
    else:
        if log:
            logger.info(f'dumping json to "{filename}"')
        with filename.open('w') as fp:
            rapidjson.dump(data, fp, default=str, number_mode=rapidjson.NM_NATIVE)

    logger.debug(f'done json to "{filename}"')


def file_dump_joblib(filename: Path, data: Any, log: bool = True) -> None:
    """
    Dump object data into a file
    :param filename: file to create
    :param data: Object data to save
    :return:
    """
    import joblib

    if log:
        logger.info(f'dumping joblib to "{filename}"')
    with filename.open('wb') as fp:
        joblib.dump(data, fp)
    logger.debug(f'done joblib dump to "{filename}"')


def json_load(datafile: Union[gzip.GzipFile, TextIO]) -> Any:
    """
    load data with rapidjson
    Use this to have a consistent experience,
    set number_mode to "NM_NATIVE" for greatest speed
    """
    return rapidjson.load(datafile, number_mode=rapidjson.NM_NATIVE)


def file_load_json(file: Path):

    if file.suffix != ".gz":
        gzipfile = file.with_suffix(file.suffix + '.gz')
    else:
        gzipfile = file
    # Try gzip file first, otherwise regular json file.
    if gzipfile.is_file():
        logger.debug(f"Loading historical data from file {gzipfile}")
        with gzip.open(gzipfile) as datafile:
            pairdata = json_load(datafile)
    elif file.is_file():
        logger.debug(f"Loading historical data from file {file}")
        with file.open() as datafile:
            pairdata = json_load(datafile)
    else:
        return None
    return pairdata


def is_file_in_dir(file: Path, directory: Path) -> bool:
    """
    Helper function to check if file is in directory.
    """
    return file.is_file() and file.parent.samefile(directory)


def pair_to_filename(pair: str) -> str:
    for ch in ['/', ' ', '.', '@', '$', '+', ':']:
        pair = pair.replace(ch, '_')
    return pair


def deep_merge_dicts(source, destination, allow_null_overrides: bool = True):
    """
    Values from Source override destination, destination is returned (and modified!!)
    Sample:
    >>> a = { 'first' : { 'rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            deep_merge_dicts(value, node, allow_null_overrides)
        elif value is not None or allow_null_overrides:
            destination[key] = value

    return destination


def round_dict(d, n):
    """
    Rounds float values in the dict to n digits after the decimal point.
    """
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


def safe_value_fallback(obj: dict, key1: str, key2: Optional[str] = None, default_value=None):
    """
    Search a value in obj, return this if it's not None.
    Then search key2 in obj - return that if it's not none - then use default_value.
    Else falls back to None.
    """
    if key1 in obj and obj[key1] is not None:
        return obj[key1]
    else:
        if key2 and key2 in obj and obj[key2] is not None:
            return obj[key2]
    return default_value


dictMap = Union[Dict[str, Any], Mapping[str, Any]]


def safe_value_fallback2(dict1: dictMap, dict2: dictMap, key1: str, key2: str, default_value=None):
    """
    Search a value in dict1, return this if it's not None.
    Fall back to dict2 - return key2 from dict2 if it's not None.
    Else falls back to None.

    """
    if key1 in dict1 and dict1[key1] is not None:
        return dict1[key1]
    else:
        if key2 in dict2 and dict2[key2] is not None:
            return dict2[key2]
    return default_value


def plural(num: float, singular: str, plural: Optional[str] = None) -> str:
    return singular if (num == 1 or num == -1) else plural or singular + 's'


def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """
    Split lst into chunks of the size n.
    :param lst: list to split into chunks
    :param n: number of max elements per chunk
    :return: None
    """
    for chunk in range(0, len(lst), n):
        yield (lst[chunk:chunk + n])


def parse_db_uri_for_logging(uri: str):
    """
    Helper method to parse the DB URI and return the same DB URI with the password censored
    if it contains it. Otherwise, return the DB URI unchanged
    :param uri: DB URI to parse for logging
    """
    parsed_db_uri = urlparse(uri)
    if not parsed_db_uri.netloc:  # No need for censoring as no password was provided
        return uri
    pwd = parsed_db_uri.netloc.split(':')[1].split('@')[0]
    return parsed_db_uri.geturl().replace(f':{pwd}@', ':*****@')


def dataframe_to_json(dataframe: pd.DataFrame) -> str:
    """
    Serialize a DataFrame for transmission over the wire using JSON
    :param dataframe: A pandas DataFrame
    :returns: A JSON string of the pandas DataFrame
    """
    return dataframe.to_json(orient='split')


def json_to_dataframe(data: str) -> pd.DataFrame:
    """
    Deserialize JSON into a DataFrame
    :param data: A JSON string
    :returns: A pandas DataFrame from the JSON string
    """
    dataframe = pd.read_json(StringIO(data), orient='split')
    if 'date' in dataframe.columns:
        dataframe['date'] = pd.to_datetime(dataframe['date'], unit='ms', utc=True)

    return dataframe


def remove_entry_exit_signals(dataframe: pd.DataFrame):
    """
    Remove Entry and Exit signals from a DataFrame

    :param dataframe: The DataFrame to remove signals from
    """
    dataframe[SignalType.ENTER_LONG.value] = 0
    dataframe[SignalType.EXIT_LONG.value] = 0
    dataframe[SignalType.ENTER_SHORT.value] = 0
    dataframe[SignalType.EXIT_SHORT.value] = 0
    dataframe[SignalTagType.ENTER_TAG.value] = None
    dataframe[SignalTagType.EXIT_TAG.value] = None

    return dataframe


def append_candles_to_dataframe(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    Append the `right` dataframe to the `left` dataframe

    :param left: The full dataframe you want appended to
    :param right: The new dataframe containing the data you want appended
    :returns: The dataframe with the right data in it
    """
    if left.iloc[-1]['date'] != right.iloc[-1]['date']:
        left = pd.concat([left, right])

    # Only keep the last 1500 candles in memory
    left = left[-1500:] if len(left) > 1500 else left
    left.reset_index(drop=True, inplace=True)

    return left
