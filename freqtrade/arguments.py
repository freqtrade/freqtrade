"""
This module contains the argument manager class
"""

import argparse
import logging
import os
import re
import arrow
from typing import List, Tuple, Optional

from freqtrade import __version__
from freqtrade.constants import Constants


class Arguments(object):
    """
    Arguments Class. Manage the arguments received by the cli
    """

    def __init__(self, args: List[str], description: str):
        self.args = args
        self.parsed_arg = None
        self.parser = argparse.ArgumentParser(description=description)

    def _load_args(self) -> None:
        self.common_args_parser()
        self._build_subcommands()

    def get_parsed_arg(self) -> argparse.Namespace:
        """
        Return the list of arguments
        :return: List[str] List of arguments
        """
        if self.parsed_arg is None:
            self._load_args()
            self.parsed_arg = self.parse_args()

        return self.parsed_arg

    def parse_args(self) -> argparse.Namespace:
        """
        Parses given arguments and returns an argparse Namespace instance.
        """
        parsed_arg = self.parser.parse_args(self.args)

        return parsed_arg

    def common_args_parser(self) -> None:
        """
        Parses given common arguments and returns them as a parsed object.
        """
        self.parser.add_argument(
            '-v', '--verbose',
            help='be verbose',
            action='store_const',
            dest='loglevel',
            const=logging.DEBUG,
            default=logging.INFO,
        )
        self.parser.add_argument(
            '--version',
            action='version',
            version='%(prog)s {}'.format(__version__),
        )
        self.parser.add_argument(
            '-c', '--config',
            help='specify configuration file (default: %(default)s)',
            dest='config',
            default='config.json',
            type=str,
            metavar='PATH',
        )
        self.parser.add_argument(
            '-d', '--datadir',
            help='path to backtest data (default: %(default)s',
            dest='datadir',
            default=os.path.join('freqtrade', 'tests', 'testdata'),
            type=str,
            metavar='PATH',
        )
        self.parser.add_argument(
            '-s', '--strategy',
            help='specify strategy file (default: %(default)s)',
            dest='strategy',
            default='default_strategy',
            type=str,
            metavar='PATH',
        )
        self.parser.add_argument(
            '--dynamic-whitelist',
            help='dynamically generate and update whitelist \
                                  based on 24h BaseVolume (Default 20 currencies)',  # noqa
            dest='dynamic_whitelist',
            const=Constants.DYNAMIC_WHITELIST,
            type=int,
            metavar='INT',
            nargs='?',
        )
        self.parser.add_argument(
            '--dry-run-db',
            help='Force dry run to use a local DB "tradesv3.dry_run.sqlite" \
                                  instead of memory DB. Work only if dry_run is enabled.',
            action='store_true',
            dest='dry_run_db',
        )

    @staticmethod
    def backtesting_options(parser: argparse.ArgumentParser) -> None:
        """
        Parses given arguments for Backtesting scripts.
        """
        parser.add_argument(
            '-l', '--live',
            help='using live data',
            action='store_true',
            dest='live',
        )
        parser.add_argument(
            '-r', '--refresh-pairs-cached',
            help='refresh the pairs files in tests/testdata with the latest data from Bittrex. \
                  Use it if you want to run your backtesting with up-to-date data.',
            action='store_true',
            dest='refresh_pairs',
        )
        parser.add_argument(
            '--export',
            help='export backtest results, argument are: trades\
                  Example --export=trades',
            type=str,
            default=None,
            dest='export',
        )

    @staticmethod
    def optimizer_shared_options(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            '-i', '--ticker-interval',
            help='specify ticker interval (1m, 5m, 30m, 1h, 1d)',
            dest='ticker_interval',
            type=str,
        )
        parser.add_argument(
            '--realistic-simulation',
            help='uses max_open_trades from config to simulate real world limitations',
            action='store_true',
            dest='realistic_simulation',
        )
        parser.add_argument(
            '--timerange',
            help='specify what timerange of data to use.',
            default=None,
            type=str,
            dest='timerange',
        )

    @staticmethod
    def hyperopt_options(parser: argparse.ArgumentParser) -> None:
        """
        Parses given arguments for Hyperopt scripts.
        """
        parser.add_argument(
            '-e', '--epochs',
            help='specify number of epochs (default: %(default)d)',
            dest='epochs',
            default=Constants.HYPEROPT_EPOCH,
            type=int,
            metavar='INT',
        )
        parser.add_argument(
            '--use-mongodb',
            help='parallelize evaluations with mongodb (requires mongod in PATH)',
            dest='mongodb',
            action='store_true',
        )
        parser.add_argument(
            '-s', '--spaces',
            help='Specify which parameters to hyperopt. Space separate list. \
                  Default: %(default)s',
            choices=['all', 'buy', 'roi', 'stoploss'],
            default='all',
            nargs='+',
            dest='spaces',
        )

    def _build_subcommands(self) -> None:
        """
        Builds and attaches all subcommands
        :return: None
        """
        from freqtrade.optimize import backtesting, hyperopt

        subparsers = self.parser.add_subparsers(dest='subparser')

        # Add backtesting subcommand
        backtesting_cmd = subparsers.add_parser('backtesting', help='backtesting module')
        backtesting_cmd.set_defaults(func=backtesting.start)
        self.optimizer_shared_options(backtesting_cmd)
        self.backtesting_options(backtesting_cmd)

        # Add hyperopt subcommand
        hyperopt_cmd = subparsers.add_parser('hyperopt', help='hyperopt module')
        hyperopt_cmd.set_defaults(func=hyperopt.start)
        self.optimizer_shared_options(hyperopt_cmd)
        self.hyperopt_options(hyperopt_cmd)

    @staticmethod
    def parse_timerange(text: str) -> Optional[Tuple[List, int, int]]:
        """
        Parse the value of the argument --timerange to determine what is the range desired
        :param text: value from --timerange
        :return: Start and End range period
        """
        if text is None:
            return None
        syntax = [(r'^-(\d{8})$', (None, 'date')),
                  (r'^(\d{8})-$', ('date', None)),
                  (r'^(\d{8})-(\d{8})$', ('date', 'date')),
                  (r'^(-\d+)$', (None, 'line')),
                  (r'^(\d+)-$', ('line', None)),
                  (r'^(\d+)-(\d+)$', ('index', 'index'))]
        for rex, stype in syntax:
            # Apply the regular expression to text
            match = re.match(rex, text)
            if match:  # Regex has matched
                rvals = match.groups()
                index = 0
                start = None
                stop = None
                if stype[0]:
                    start = rvals[index]
                    if stype[0] == 'date':
                        start = arrow.get(start, 'YYYYMMDD').timestamp
                    else:
                        start = int(start)
                    index += 1
                if stype[1]:
                    stop = rvals[index]
                    if stype[1] == 'date':
                        stop = arrow.get(stop, 'YYYYMMDD').timestamp
                    else:
                        stop = int(stop)
                return stype, start, stop
        raise Exception('Incorrect syntax for timerange "%s"' % text)

    def scripts_options(self) -> None:
        """
        Parses given arguments for scripts.
        """
        self.parser.add_argument(
            '-p', '--pair',
            help='Show profits for only this pairs. Pairs are comma-separated.',
            dest='pair',
            default=None
        )

    def testdata_dl_options(self) -> None:
        """
        Parses given arguments for testdata download
        """
        self.parser.add_argument(
            '--pairs-file',
            help='File containing a list of pairs to download',
            dest='pairs_file',
            default=None
        )

        self.parser.add_argument(
            '--export',
            help='Export files to given dir',
            dest='export',
            default=None)

        self.parser.add_argument(
            '--days',
            help='Download data for number of days',
            dest='days',
            type=int,
            default=None)

        self.parser.add_argument(
            '--exchange',
            help='Exchange name',
            dest='exchange',
            type=str,
            default='bittrex')
