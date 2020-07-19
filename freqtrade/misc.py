"""
Various tool function for Freqtrade and scripts
"""
import gzip
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from typing.io import IO

import numpy as np
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
    return dates.dt.to_pydatetime()


def file_dump_json(filename: Path, data: Any, is_zip: bool = False) -> None:
    """
    Dump JSON data into a file
    :param filename: file to create
    :param data: JSON Data to save
    :return:
    """

    if is_zip:
        if filename.suffix != '.gz':
            filename = filename.with_suffix('.gz')
        logger.info(f'dumping json to "{filename}"')

        with gzip.open(filename, 'w') as fp:
            rapidjson.dump(data, fp, default=str, number_mode=rapidjson.NM_NATIVE)
    else:
        logger.info(f'dumping json to "{filename}"')
        with open(filename, 'w') as fp:
            rapidjson.dump(data, fp, default=str, number_mode=rapidjson.NM_NATIVE)

    logger.debug(f'done json to "{filename}"')


def json_load(datafile: IO) -> Any:
    """
    load data with rapidjson
    Use this to have a consistent experience,
    sete number_mode to "NM_NATIVE" for greatest speed
    """
    return rapidjson.load(datafile, number_mode=rapidjson.NM_NATIVE)


def file_load_json(file):

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
        with open(file) as datafile:
            pairdata = json_load(datafile)
    else:
        return None
    return pairdata


def pair_to_filename(pair: str) -> str:
    for ch in ['/', '-', ' ', '.', '@', '$', '+', ':']:
        pair = pair.replace(ch, '_')
    return pair


def format_ms_time(date: int) -> str:
    """
    convert MS date to readable format.
    : epoch-string in ms
    """
    return datetime.fromtimestamp(date/1000.0).strftime('%Y-%m-%dT%H:%M:%S')


def deep_merge_dicts(source, destination):
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
            deep_merge_dicts(value, node)
        else:
            destination[key] = value

    return destination


def round_dict(d, n):
    """
    Rounds float values in the dict to n digits after the decimal point.
    """
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


def safe_value_fallback(dict1: dict, dict2: dict, key1: str, key2: str, default_value=None):
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


def plural(num: float, singular: str, plural: str = None) -> str:
    return singular if (num == 1 or num == -1) else plural or singular + 's'


def render_template(templatefile: str, arguments: dict = {}) -> str:

    from jinja2 import Environment, PackageLoader, select_autoescape

    env = Environment(
        loader=PackageLoader('freqtrade', 'templates'),
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template(templatefile)
    return template.render(**arguments)


def render_template_with_fallback(templatefile: str, templatefallbackfile: str,
                                  arguments: dict = {}) -> str:
    """
    Use templatefile if possible, otherwise fall back to templatefallbackfile
    """
    from jinja2.exceptions import TemplateNotFound
    try:
        return render_template(templatefile, arguments)
    except TemplateNotFound:
        return render_template(templatefallbackfile, arguments)
