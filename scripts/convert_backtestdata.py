#!/usr/bin/env python3
"""
Script to display when the bot will buy a specific pair

Mandatory Cli parameters:
-p / --pair: pair to examine

Optional Cli parameters
-d / --datadir: path to pair backtest data
--timerange: specify what timerange of data to use.
-l / --live: Live, to download the latest ticker for the pair
"""

import sys
from argparse import Namespace
from os import path
import glob
import json
import re
from typing import List, Dict
import gzip

from freqtrade.arguments import Arguments
from freqtrade import misc
from freqtrade.logger import Logger
from pandas import DataFrame
from freqtrade.constants import Constants

import dateutil.parser

logger = Logger(name="freqtrade").get_logger()


def load_old_file(filename) -> (List[Dict], bool):
    if not path.isfile(filename):
        logger.warning("filename %s does not exist", filename)
        return (None, False)
    logger.debug('Loading ticker data from file %s', filename)

    pairdata = None

    if filename.endswith('.gz'):
        logger.debug('Loading ticker data from file %s', filename)
        is_zip = True
        with gzip.open(filename) as tickerdata:
            pairdata = json.load(tickerdata)
    else:
        is_zip = False
        with open(filename) as tickerdata:
            pairdata = json.load(tickerdata)
    return (pairdata, is_zip)


def parse_old_backtest_data(ticker) -> DataFrame:
    """
    Reads old backtest data
    Format: "O": 8.794e-05,
            "H": 8.948e-05,
            "L": 8.794e-05,
            "C": 8.88e-05,
            "V": 991.09056638,
            "T": "2017-11-26T08:50:00",
            "BV": 0.0877869
    """

    columns = {'C': 'close', 'V': 'volume', 'O': 'open',
               'H': 'high', 'L': 'low', 'T': 'date'}

    frame = DataFrame(ticker) \
        .rename(columns=columns)
    if 'BV' in frame:
        frame.drop('BV', 1, inplace=True)
    if not 'date' in frame:
        logger.warning("Date not in frame - probably not a Ticker file")
        return None
    frame.sort_values('date', inplace=True)
    return frame


def convert_dataframe(frame: DataFrame):
    """Convert dataframe to new format"""
    # reorder columns:
    cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    frame = frame[cols]

    # Make sure parsing/printing data is assumed to be UTC
    frame['date'] = frame['date'].apply(
        lambda d: int(dateutil.parser.parse(d+'+00:00').timestamp()) * 1000)
    frame['date'] = frame['date'].astype('int64')
    # Convert columns one by one to preserve type.
    by_column = [frame[x].values.tolist() for x in frame.columns]
    return list(list(x) for x in zip(*by_column))


def convert_file(filename: str, filename_new: str) -> None:
    """Converts a file from old format to ccxt format"""
    (pairdata, is_zip) = load_old_file(filename)
    if pairdata and type(pairdata) is list:
        if type(pairdata[0]) is list:
            logger.error("pairdata for %s already in new format", filename)
            return

    frame = parse_old_backtest_data(pairdata)
    # Convert frame to new format
    if frame is not None:
        frame1 = convert_dataframe(frame)
        misc.file_dump_json(filename_new, frame1, is_zip)


def convert_main(args: Namespace) -> None:
    """
    converts a folder given in --datadir from old to new format to support ccxt
    """

    workdir = path.join(args.datadir, "")
    logger.info("Workdir: %s", workdir)

    for filename in glob.glob(workdir + "*.json"):
        # swap currency names
        ret = re.search(r'[A-Z_]{7,}', path.basename(filename))
        if args.norename:
            filename_new = filename
        else:
            if not ret:
                logger.warning("file %s could not be converted, could not extract currencies",
                               filename)
                continue
            pair = ret.group(0)
            currencies = pair.split("_")
            if len(currencies) != 2:
                logger.warning("file %s could not be converted, could not extract currencies",
                               filename)
                continue

            ret_integer = re.search(r'\d+(?=\.json)', path.basename(filename))
            ret_string = re.search(r'(\d+[mhdw])(?=\.json)', path.basename(filename))

            if ret_integer:
                minutes = int(ret_integer.group(0))
                interval = str(minutes) + 'm'  # default to adding 'm' to end of minutes for new interval name
                # but check if there is a mapping between int and string also
                for str_interval, minutes_interval in Constants.TICKER_INTERVAL_MINUTES.items():
                    if minutes_interval == minutes:
                        interval = str_interval
                        break
                # change order on pairs if old ticker interval found
                filename_new = path.join(path.dirname(filename),
                                         "{}_{}-{}.json".format(currencies[1],
                                                                currencies[0], interval))

            elif ret_string:
                interval = ret_string.group(0)
                filename_new = path.join(path.dirname(filename),
                                         "{}_{}-{}.json".format(currencies[0],
                                                                currencies[1], interval))

            else:
                logger.warning("file %s could not be converted, interval not found", filename)
                continue

        logger.debug("Converting and renaming %s to %s", filename, filename_new)
        convert_file(filename, filename_new)


def convert_parse_args(args: List[str]) -> Namespace:
    """
    Parse args passed to the script
    :param args: Cli arguments
    :return: args: Array with all arguments
    """
    arguments = Arguments(args, 'Convert datafiles')
    arguments.parser.add_argument(
        '-d', '--datadir',
        help='path to backtest data (default: %(default)s',
        dest='datadir',
        default=path.join('freqtrade', 'tests', 'testdata'),
        type=str,
        metavar='PATH',
    )
    arguments.parser.add_argument(
        '-n', '--norename',
        help='don''t rename files from BTC_<PAIR> to <PAIR>_BTC - '
             'Note that not renaming will overwrite source files',
        dest='norename',
        default=False,
        action='store_true'
    )

    return arguments.parse_args()


def main(sysargv: List[str]) -> None:
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """
    logger.info('Starting Dataframe conversation')
    convert_main(convert_parse_args(sysargv))


if __name__ == '__main__':
    main(sys.argv[1:])
