"""
Various tool function for Freqtrade and scripts
"""

import gzip
import logging
import re
from datetime import datetime
from typing import Dict

import numpy as np
from pandas import DataFrame
import rapidjson

logger = logging.getLogger(__name__)


def shorten_date(_date: str) -> str:
    """
    Trim the date so it fits on small screens
    """
    new_date = re.sub('seconds?', 'sec', _date)
    new_date = re.sub('minutes?', 'min', new_date)
    new_date = re.sub('hours?', 'h', new_date)
    new_date = re.sub('days?', 'd', new_date)
    new_date = re.sub('^an?', '1', new_date)
    return new_date


############################################
# Used by scripts                          #
# Matplotlib doesn't support ::datetime64, #
# so we need to convert it into ::datetime #
############################################
def datesarray_to_datetimearray(dates: np.ndarray) -> np.ndarray:
    """
    Convert an pandas-array of timestamps into
    An numpy-array of datetimes
    :return: numpy-array of datetime
    """
    times = []
    dates = dates.astype(datetime)
    for index in range(0, dates.size):
        date = dates[index].to_pydatetime()
        times.append(date)
    return np.array(times)


def common_datearray(dfs: Dict[str, DataFrame]) -> np.ndarray:
    """
    Return dates from Dataframe
    :param dfs: Dict with format pair: pair_data
    :return: List of dates
    """
    alldates = {}
    for pair, pair_data in dfs.items():
        dates = datesarray_to_datetimearray(pair_data['date'])
        for date in dates:
            alldates[date] = 1
    lst = []
    for date, _ in alldates.items():
        lst.append(date)
    arr = np.array(lst)
    return np.sort(arr, axis=0)


def file_dump_json(filename, data, is_zip=False) -> None:
    """
    Dump JSON data into a file
    :param filename: file to create
    :param data: JSON Data to save
    :return:
    """
    print(f'dumping json to "{filename}"')

    if is_zip:
        if not filename.endswith('.gz'):
            filename = filename + '.gz'
        with gzip.open(filename, 'w') as fp:
            rapidjson.dump(data, fp, default=str, number_mode=rapidjson.NM_NATIVE)
    else:
        with open(filename, 'w') as fp:
            rapidjson.dump(data, fp, default=str, number_mode=rapidjson.NM_NATIVE)


def json_load(data):
    """
    load data with rapidjson
    Use this to have a consistent experience,
    sete number_mode to "NM_NATIVE" for greatest speed
    """
    return rapidjson.load(data, number_mode=rapidjson.NM_NATIVE)


def format_ms_time(date: int) -> str:
    """
    convert MS date to readable format.
    : epoch-string in ms
    """
    return datetime.fromtimestamp(date/1000.0).strftime('%Y-%m-%dT%H:%M:%S')
