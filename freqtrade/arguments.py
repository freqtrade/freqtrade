"""
This module contains the argument manager class
"""

import argparse
import os
import re
import logging
from typing import List

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
        self._common_args_parser()
        self._build_subcommands()

    def get_parsed_arg(self) -> List[str]:
        """
        Return the list of arguments
        :return: List[str] List of arguments
        """
        self.parsed_arg = self._parse_args()

        return self.parsed_arg

    def _parse_args(self) -> List[str]:
        """
        Parses given arguments and returns an argparse Namespace instance.
        """
        parsed_arg = self.parser.parse_args(self.args)

        return parsed_arg

    def _common_args_parser(self) -> None:
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
            help='specify configuration file (default: config.json)',
            dest='config',
            default='config.json',
            type=str,
            metavar='PATH',
        )
        self.parser.add_argument(
            '--datadir',
            help='path to backtest data (default freqdata/tests/testdata)',
            dest='datadir',
            default=os.path.join('freqtrade', 'tests', 'testdata'),
            type=str,
            metavar='PATH',
        )
        self.parser.add_argument(
            '-s', '--strategy',
            help='specify strategy file (default: freqtrade/strategy/default_strategy.py)',
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
    def _backtesting_options(parser: argparse.ArgumentParser) -> None:
        """
        Parses given arguments for Backtesting scripts.
        """
        parser.add_argument(
            '-l', '--live',
            action='store_true',
            dest='live',
            help='using live data',
        )
        parser.add_argument(
            '-i', '--ticker-interval',
            help='specify ticker interval in minutes (1, 5, 30, 60, 1440)',
            dest='ticker_interval',
            type=int,
            metavar='INT',
        )
        parser.add_argument(
            '--realistic-simulation',
            help='uses max_open_trades from config to simulate real world limitations',
            action='store_true',
            dest='realistic_simulation',
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
            help='Export backtest results, argument are: trades\
                  Example --export=trades',
            type=str,
            default=None,
            dest='export',
        )
        parser.add_argument(
            '--timerange',
            help='Specify what timerange of data to use.',
            default=None,
            type=str,
            dest='timerange',
        )

    @staticmethod
    def _hyperopt_options(parser: argparse.ArgumentParser) -> None:
        """
        Parses given arguments for Hyperopt scripts.
        """
        parser.add_argument(
            '-e', '--epochs',
            help='specify number of epochs (default: 100)',
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
            '-i', '--ticker-interval',
            help='specify ticker interval in minutes (1, 5, 30, 60, 1440)',
            dest='ticker_interval',
            type=int,
            metavar='INT',
        )
        parser.add_argument(
            '--timerange',
            help='Specify what timerange of data to use.',
            default=None,
            type=str,
            dest='timerange',
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
        self._backtesting_options(backtesting_cmd)

        # Add hyperopt subcommand
        hyperopt_cmd = subparsers.add_parser('hyperopt', help='hyperopt module')
        hyperopt_cmd.set_defaults(func=hyperopt.start)
        self._hyperopt_options(hyperopt_cmd)

    @staticmethod
    def parse_timerange(text: str) -> (List, int, int):
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
                    if stype[0] != 'date':
                        start = int(start)
                    index += 1
                if stype[1]:
                    stop = rvals[index]
                    if stype[1] != 'date':
                        stop = int(stop)
                return (stype, start, stop)
        raise Exception('Incorrect syntax for timerange "%s"' % text)

    def scripts_options(self):
        """
        Parses given arguments for plot scripts.
        """
        self.parser.add_argument(
            '-p', '--pair',
            help='Show profits for only this pairs. Pairs are comma-separated.',
            dest='pair',
            default=None
        )
