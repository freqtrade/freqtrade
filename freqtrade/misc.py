"""
Various tool function for Freqtrade and scripts
"""

import re
import json
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


def shorten_date(_date):
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
def datesarray_to_datetimearray(dates):
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


def common_datearray(dfs):
    """
    Return dates from Dataframe
    :param dfs: Dataframe
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


def file_dump_json(filename, data) -> None:
    """
    Dump JSON data into a file
    :param filename: file to create
    :param data: JSON Data to save
    :return:
    """
    with open(filename, 'w') as fp:
        json.dump(data, fp, default=str)
