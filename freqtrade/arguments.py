"""
This module contains the argument manager class
"""

import argparse
import os
import re
from typing import List, NamedTuple, Optional
import arrow
from freqtrade import __version__, constants


class TimeRange(NamedTuple):
    """
    NamedTuple Defining timerange inputs.
    [start/stop]type defines if [start/stop]ts shall be used.
    if *type is none, don't use corresponding startvalue.
    """
    starttype: Optional[str] = None
    stoptype: Optional[str] = None
    startts: int = 0
    stopts: int = 0


class Arguments(object):
    """
    Arguments Class. Manage the arguments received by the cli
    """

    def __init__(self, args: Optional[List[str]], description: str) -> None:
        self.args = args
        self.parsed_arg: Optional[argparse.Namespace] = None
        self.parser = argparse.ArgumentParser(description=description)

    def _load_args(self) -> None:
        self.common_options()
        self.main_options()
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

    def parse_args(self, no_default_config: bool = False) -> argparse.Namespace:
        """
        Parses given arguments and returns an argparse Namespace instance.
        """
        parsed_arg = self.parser.parse_args(self.args)

        # Workaround issue in argparse with action='append' and default value
        # (see https://bugs.python.org/issue16399)
        if not no_default_config and parsed_arg.config is None:
            parsed_arg.config = [constants.DEFAULT_CONFIG]

        return parsed_arg

    def common_options(self) -> None:
        """
        Parses arguments that are common for the main Freqtrade, all subcommands and scripts.
        """
        parser = self.parser

        parser.add_argument(
            '-v', '--verbose',
            help='Verbose mode (-vv for more, -vvv to get all messages).',
            action='count',
            dest='loglevel',
            default=0,
        )
        parser.add_argument(
            '--logfile',
            help='Log to the file specified.',
            dest='logfile',
            metavar='FILE',
        )
        parser.add_argument(
            '--version',
            action='version',
            version=f'%(prog)s {__version__}'
        )
        parser.add_argument(
            '-c', '--config',
            help=f'Specify configuration file (default: `{constants.DEFAULT_CONFIG}`). '
                 f'Multiple --config options may be used. '
                 f'Can be set to `-` to read config from stdin.',
            dest='config',
            action='append',
            metavar='PATH',
        )
        parser.add_argument(
            '-d', '--datadir',
            help='Path to backtest data.',
            dest='datadir',
            metavar='PATH',
        )

    def main_options(self) -> None:
        """
        Parses arguments for the main Freqtrade.
        """
        parser = self.parser

        parser.add_argument(
            '-s', '--strategy',
            help='Specify strategy class name (default: `%(default)s`).',
            dest='strategy',
            default='DefaultStrategy',
            metavar='NAME',
        )
        parser.add_argument(
            '--strategy-path',
            help='Specify additional strategy lookup path.',
            dest='strategy_path',
            metavar='PATH',
        )
        parser.add_argument(
            '--dynamic-whitelist',
            help='Dynamically generate and update whitelist '
                 'based on 24h BaseVolume (default: %(const)s). '
                 'DEPRECATED.',
            dest='dynamic_whitelist',
            const=constants.DYNAMIC_WHITELIST,
            type=int,
            metavar='INT',
            nargs='?',
        )
        parser.add_argument(
            '--db-url',
            help=f'Override trades database URL, this is useful in custom deployments '
                 f'(default: `{constants.DEFAULT_DB_PROD_URL}` for Live Run mode, '
                 f'`{constants.DEFAULT_DB_DRYRUN_URL}` for Dry Run).',
            dest='db_url',
            metavar='PATH',
        )
        parser.add_argument(
            '--sd-notify',
            help='Notify systemd service manager.',
            action='store_true',
            dest='sd_notify',
        )

    def common_optimize_options(self, subparser: argparse.ArgumentParser = None) -> None:
        """
        Parses arguments common for Backtesting, Edge and Hyperopt modules.
        :param parser:
        """
        parser = subparser or self.parser

        parser.add_argument(
            '-i', '--ticker-interval',
            help='Specify ticker interval (`1m`, `5m`, `30m`, `1h`, `1d`).',
            dest='ticker_interval',
        )
        parser.add_argument(
            '--timerange',
            help='Specify what timerange of data to use.',
            dest='timerange',
        )
        parser.add_argument(
            '--max_open_trades',
            help='Specify max_open_trades to use.',
            type=int,
            dest='max_open_trades',
        )
        parser.add_argument(
            '--stake_amount',
            help='Specify stake_amount.',
            type=float,
            dest='stake_amount',
        )
        parser.add_argument(
            '-r', '--refresh-pairs-cached',
            help='Refresh the pairs files in tests/testdata with the latest data from the '
                 'exchange. Use it if you want to run your optimization commands with '
                 'up-to-date data.',
            action='store_true',
            dest='refresh_pairs',
        )

    def backtesting_options(self, subparser: argparse.ArgumentParser = None) -> None:
        """
        Parses given arguments for Backtesting module.
        """
        parser = subparser or self.parser

        parser.add_argument(
            '--eps', '--enable-position-stacking',
            help='Allow buying the same pair multiple times (position stacking).',
            action='store_true',
            dest='position_stacking',
            default=False
        )
        parser.add_argument(
            '--dmmp', '--disable-max-market-positions',
            help='Disable applying `max_open_trades` during backtest '
                 '(same as setting `max_open_trades` to a very high number).',
            action='store_false',
            dest='use_max_market_positions',
            default=True
        )
        parser.add_argument(
            '-l', '--live',
            help='Use live data.',
            action='store_true',
            dest='live',
        )
        parser.add_argument(
            '--strategy-list',
            help='Provide a comma-separated list of strategies to backtest. '
                 'Please note that ticker-interval needs to be set either in config '
                 'or via command line. When using this together with `--export trades`, '
                 'the strategy-name is injected into the filename '
                 '(so `backtest-data.json` becomes `backtest-data-DefaultStrategy.json`',
            nargs='+',
            dest='strategy_list',
        )
        parser.add_argument(
            '--export',
            help='Export backtest results, argument are: trades. '
                 'Example: `--export=trades`',
            dest='export',
        )
        parser.add_argument(
            '--export-filename',
            help='Save backtest results to the file with this filename (default: `%(default)s`). '
                 'Requires `--export` to be set as well. '
                 'Example: `--export-filename=user_data/backtest_data/backtest_today.json`',
            default=os.path.join('user_data', 'backtest_data', 'backtest-result.json'),
            dest='exportfilename',
            metavar='PATH',
        )

    def edge_options(self, subparser: argparse.ArgumentParser = None) -> None:
        """
        Parses given arguments for Edge module.
        """
        parser = subparser or self.parser

        parser.add_argument(
            '--stoplosses',
            help='Defines a range of stoploss values against which edge will assess the strategy. '
                 'The format is "min,max,step" (without any space). '
                 'Example: `--stoplosses=-0.01,-0.1,-0.001`',
            dest='stoploss_range',
        )

    def hyperopt_options(self, subparser: argparse.ArgumentParser = None) -> None:
        """
        Parses given arguments for Hyperopt module.
        """
        parser = subparser or self.parser

        parser.add_argument(
            '--customhyperopt',
            help='Specify hyperopt class name (default: `%(default)s`).',
            dest='hyperopt',
            default=constants.DEFAULT_HYPEROPT,
            metavar='NAME',
        )
        parser.add_argument(
            '--eps', '--enable-position-stacking',
            help='Allow buying the same pair multiple times (position stacking).',
            action='store_true',
            dest='position_stacking',
            default=False
        )
        parser.add_argument(
            '--dmmp', '--disable-max-market-positions',
            help='Disable applying `max_open_trades` during backtest '
                 '(same as setting `max_open_trades` to a very high number).',
            action='store_false',
            dest='use_max_market_positions',
            default=True
        )
        parser.add_argument(
            '-e', '--epochs',
            help='Specify number of epochs (default: %(default)d).',
            dest='epochs',
            default=constants.HYPEROPT_EPOCH,
            type=int,
            metavar='INT',
        )
        parser.add_argument(
            '-s', '--spaces',
            help='Specify which parameters to hyperopt. Space-separated list. '
                 'Default: `%(default)s`.',
            choices=['all', 'buy', 'sell', 'roi', 'stoploss'],
            default='all',
            nargs='+',
            dest='spaces',
        )
        parser.add_argument(
            '--print-all',
            help='Print all results, not only the best ones.',
            action='store_true',
            dest='print_all',
            default=False
        )
        parser.add_argument(
            '-j', '--job-workers',
            help='The number of concurrently running jobs for hyperoptimization '
                 '(hyperopt worker processes). '
                 'If -1 (default), all CPUs are used, for -2, all CPUs but one are used, etc. '
                 'If 1 is given, no parallel computing code is used at all.',
            dest='hyperopt_jobs',
            default=-1,
            type=int,
            metavar='JOBS',
        )
        parser.add_argument(
            '--random-state',
            help='Set random state to some positive integer for reproducible hyperopt results.',
            dest='hyperopt_random_state',
            type=Arguments.check_int_positive,
            metavar='INT',
        )
        parser.add_argument(
            '--min-trades',
            help="Set minimal desired number of trades for evaluations in the hyperopt "
                 "optimization path (default: 1).",
            dest='hyperopt_min_trades',
            default=1,
            type=Arguments.check_int_positive,
            metavar='INT',
        )

    def list_exchanges_options(self, subparser: argparse.ArgumentParser = None) -> None:
        """
        Parses given arguments for the list-exchanges command.
        """
        parser = subparser or self.parser

        parser.add_argument(
            '-1', '--one-column',
            help='Print exchanges in one column.',
            action='store_true',
            dest='print_one_column',
        )

    def _build_subcommands(self) -> None:
        """
        Builds and attaches all subcommands.
        :return: None
        """
        from freqtrade.optimize import start_backtesting, start_hyperopt, start_edge
        from freqtrade.utils import start_list_exchanges

        subparsers = self.parser.add_subparsers(dest='subparser')

        # Add backtesting subcommand
        backtesting_cmd = subparsers.add_parser('backtesting', help='Backtesting module.')
        backtesting_cmd.set_defaults(func=start_backtesting)
        self.common_optimize_options(backtesting_cmd)
        self.backtesting_options(backtesting_cmd)

        # Add edge subcommand
        edge_cmd = subparsers.add_parser('edge', help='Edge module.')
        edge_cmd.set_defaults(func=start_edge)
        self.common_optimize_options(edge_cmd)
        self.edge_options(edge_cmd)

        # Add hyperopt subcommand
        hyperopt_cmd = subparsers.add_parser('hyperopt', help='Hyperopt module.')
        hyperopt_cmd.set_defaults(func=start_hyperopt)
        self.common_optimize_options(hyperopt_cmd)
        self.hyperopt_options(hyperopt_cmd)

        # Add list-exchanges subcommand
        list_exchanges_cmd = subparsers.add_parser(
            'list-exchanges',
            help='Print available exchanges.'
        )
        list_exchanges_cmd.set_defaults(func=start_list_exchanges)
        self.list_exchanges_options(list_exchanges_cmd)

    @staticmethod
    def parse_timerange(text: Optional[str]) -> TimeRange:
        """
        Parse the value of the argument --timerange to determine what is the range desired
        :param text: value from --timerange
        :return: Start and End range period
        """
        if text is None:
            return TimeRange(None, None, 0, 0)
        syntax = [(r'^-(\d{8})$', (None, 'date')),
                  (r'^(\d{8})-$', ('date', None)),
                  (r'^(\d{8})-(\d{8})$', ('date', 'date')),
                  (r'^-(\d{10})$', (None, 'date')),
                  (r'^(\d{10})-$', ('date', None)),
                  (r'^(\d{10})-(\d{10})$', ('date', 'date')),
                  (r'^(-\d+)$', (None, 'line')),
                  (r'^(\d+)-$', ('line', None)),
                  (r'^(\d+)-(\d+)$', ('index', 'index'))]
        for rex, stype in syntax:
            # Apply the regular expression to text
            match = re.match(rex, text)
            if match:  # Regex has matched
                rvals = match.groups()
                index = 0
                start: int = 0
                stop: int = 0
                if stype[0]:
                    starts = rvals[index]
                    if stype[0] == 'date' and len(starts) == 8:
                        start = arrow.get(starts, 'YYYYMMDD').timestamp
                    else:
                        start = int(starts)
                    index += 1
                if stype[1]:
                    stops = rvals[index]
                    if stype[1] == 'date' and len(stops) == 8:
                        stop = arrow.get(stops, 'YYYYMMDD').timestamp
                    else:
                        stop = int(stops)
                return TimeRange(stype[0], stype[1], start, stop)
        raise Exception('Incorrect syntax for timerange "%s"' % text)

    @staticmethod
    def check_int_positive(value: str) -> int:
        try:
            uint = int(value)
            if uint <= 0:
                raise ValueError
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{value} is invalid for this parameter, should be a positive integer value"
            )
        return uint

    def common_scripts_options(self, subparser: argparse.ArgumentParser = None) -> None:
        """
        Parses arguments common for scripts.
        """
        parser = subparser or self.parser

        parser.add_argument(
            '-p', '--pairs',
            help='Show profits for only these pairs. Pairs are comma-separated.',
            dest='pairs',
        )

    def download_data_options(self) -> None:
        """
        Parses given arguments for testdata download script
        """
        parser = self.parser

        parser.add_argument(
            '--pairs-file',
            help='File containing a list of pairs to download.',
            dest='pairs_file',
            metavar='FILE',
        )
        parser.add_argument(
            '--days',
            help='Download data for given number of days.',
            dest='days',
            type=Arguments.check_int_positive,
            metavar='INT',
        )
        parser.add_argument(
            '--exchange',
            help=f'Exchange name (default: `{constants.DEFAULT_EXCHANGE}`). '
                 f'Only valid if no config is provided.',
            dest='exchange',
        )
        parser.add_argument(
            '-t', '--timeframes',
            help=f'Specify which tickers to download. Space-separated list. '
                 f'Default: `{constants.DEFAULT_DOWNLOAD_TICKER_INTERVALS}`.',
            choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',
                     '6h', '8h', '12h', '1d', '3d', '1w'],
            nargs='+',
            dest='timeframes',
        )
        parser.add_argument(
            '--erase',
            help='Clean all existing data for the selected exchange/pairs/timeframes.',
            dest='erase',
            action='store_true'
        )

    def plot_dataframe_options(self) -> None:
        """
        Parses given arguments for plot dataframe script
        """
        parser = self.parser

        parser.add_argument(
            '--indicators1',
            help='Set indicators from your strategy you want in the first row of the graph. '
                 'Comma-separated list. Example: `ema3,ema5`. Default: `%(default)s`.',
            default='sma,ema3,ema5',
            dest='indicators1',
        )

        parser.add_argument(
            '--indicators2',
            help='Set indicators from your strategy you want in the third row of the graph. '
                 'Comma-separated list. Example: `fastd,fastk`. Default: `%(default)s`.',
            default='macd,macdsignal',
            dest='indicators2',
        )
        parser.add_argument(
            '--plot-limit',
            help='Specify tick limit for plotting. Notice: too high values cause huge files. '
                 'Default: %(default)s.',
            dest='plot_limit',
            default=750,
            type=int,
        )
