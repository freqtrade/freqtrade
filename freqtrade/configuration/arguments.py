"""
This module contains the argument manager class
"""
import argparse
import os
import re
from typing import List, NamedTuple, Optional

import arrow
from freqtrade import __version__, constants


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


class Arg:
    # Optional CLI arguments
    def __init__(self, *args, **kwargs):
        self.cli = args
        self.kwargs = kwargs


# List of available command line options
AVAILABLE_CLI_OPTIONS = {
    # Common options
    "verbosity": Arg(
        '-v', '--verbose',
        help='Verbose mode (-vv for more, -vvv to get all messages).',
        action='count',
        default=0,
    ),
    "logfile": Arg(
        '--logfile',
        help='Log to the file specified.',
        metavar='FILE',
    ),
    "version": Arg(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}',
    ),
    "config": Arg(
        '-c', '--config',
        help=f'Specify configuration file (default: `{constants.DEFAULT_CONFIG}`). '
        f'Multiple --config options may be used. '
        f'Can be set to `-` to read config from stdin.',
        action='append',
        metavar='PATH',
    ),
    "datadir": Arg(
        '-d', '--datadir',
        help='Path to backtest data.',
        metavar='PATH',
    ),
    # Main options
    "strategy": Arg(
        '-s', '--strategy',
        help='Specify strategy class name (default: `%(default)s`).',
        metavar='NAME',
        default='DefaultStrategy',
    ),
    "strategy_path": Arg(
        '--strategy-path',
        help='Specify additional strategy lookup path.',
        metavar='PATH',
    ),
    "dynamic_whitelist": Arg(
        '--dynamic-whitelist',
        help='Dynamically generate and update whitelist '
        'based on 24h BaseVolume (default: %(const)s). '
        'DEPRECATED.',
        const=constants.DYNAMIC_WHITELIST,
        type=int,
        metavar='INT',
        nargs='?',
    ),
    "db_url": Arg(
        '--db-url',
        help=f'Override trades database URL, this is useful in custom deployments '
        f'(default: `{constants.DEFAULT_DB_PROD_URL}` for Live Run mode, '
        f'`{constants.DEFAULT_DB_DRYRUN_URL}` for Dry Run).',
        metavar='PATH',
    ),
    "sd_notify": Arg(
        '--sd-notify',
        help='Notify systemd service manager.',
        action='store_true',
    ),
    # Optimize common
    "ticker_interval": Arg(
        '-i', '--ticker-interval',
        help='Specify ticker interval (`1m`, `5m`, `30m`, `1h`, `1d`).',
    ),
    "timerange": Arg(
        '--timerange',
        help='Specify what timerange of data to use.',
    ),
    "max_open_trades": Arg(
        '--max_open_trades',
        help='Specify max_open_trades to use.',
        type=int,
        metavar='INT',
    ),
    "stake_amount": Arg(
        '--stake_amount',
        help='Specify stake_amount.',
        type=float,
    ),
    "refresh_pairs": Arg(
        '-r', '--refresh-pairs-cached',
        help='Refresh the pairs files in tests/testdata with the latest data from the '
        'exchange. Use it if you want to run your optimization commands with '
        'up-to-date data.',
        action='store_true',
    ),
    # Backtesting
    "position_stacking": Arg(
        '--eps', '--enable-position-stacking',
        help='Allow buying the same pair multiple times (position stacking).',
        action='store_true',
        default=False,
    ),
    "use_max_market_positions": Arg(
        '--dmmp', '--disable-max-market-positions',
        help='Disable applying `max_open_trades` during backtest '
        '(same as setting `max_open_trades` to a very high number).',
        action='store_false',
        default=True,
    ),
    "live": Arg(
        '-l', '--live',
        help='Use live data.',
        action='store_true',
    ),
    "strategy_list": Arg(
        '--strategy-list',
        help='Provide a comma-separated list of strategies to backtest. '
        'Please note that ticker-interval needs to be set either in config '
        'or via command line. When using this together with `--export trades`, '
        'the strategy-name is injected into the filename '
        '(so `backtest-data.json` becomes `backtest-data-DefaultStrategy.json`',
        nargs='+',
    ),
    "export": Arg(
        '--export',
        help='Export backtest results, argument are: trades. '
        'Example: `--export=trades`',
    ),
    "exportfilename": Arg(
        '--export-filename',
        help='Save backtest results to the file with this filename (default: `%(default)s`). '
        'Requires `--export` to be set as well. '
        'Example: `--export-filename=user_data/backtest_data/backtest_today.json`',
        metavar='PATH',
        default=os.path.join('user_data', 'backtest_data',
                             'backtest-result.json'),
    ),
    # Edge
    "stoploss_range": Arg(
        '--stoplosses',
        help='Defines a range of stoploss values against which edge will assess the strategy. '
        'The format is "min,max,step" (without any space). '
        'Example: `--stoplosses=-0.01,-0.1,-0.001`',
    ),
    # Hyperopt
    "hyperopt": Arg(
        '--customhyperopt',
        help='Specify hyperopt class name (default: `%(default)s`).',
        metavar='NAME',
        default=constants.DEFAULT_HYPEROPT,
    ),
    "epochs": Arg(
        '-e', '--epochs',
        help='Specify number of epochs (default: %(default)d).',
        type=check_int_positive,
        metavar='INT',
        default=constants.HYPEROPT_EPOCH,
    ),
    "spaces": Arg(
        '-s', '--spaces',
        help='Specify which parameters to hyperopt. Space-separated list. '
        'Default: `%(default)s`.',
        choices=['all', 'buy', 'sell', 'roi', 'stoploss'],
        nargs='+',
        default='all',
    ),
    "print_all": Arg(
        '--print-all',
        help='Print all results, not only the best ones.',
        action='store_true',
        default=False,
    ),
    "hyperopt_jobs": Arg(
        '-j', '--job-workers',
        help='The number of concurrently running jobs for hyperoptimization '
        '(hyperopt worker processes). '
        'If -1 (default), all CPUs are used, for -2, all CPUs but one are used, etc. '
        'If 1 is given, no parallel computing code is used at all.',
        type=int,
        metavar='JOBS',
        default=-1,
    ),
    "hyperopt_random_state": Arg(
        '--random-state',
        help='Set random state to some positive integer for reproducible hyperopt results.',
        type=check_int_positive,
        metavar='INT',
    ),
    "hyperopt_min_trades": Arg(
        '--min-trades',
        help="Set minimal desired number of trades for evaluations in the hyperopt "
        "optimization path (default: 1).",
        type=check_int_positive,
        metavar='INT',
        default=1,
    ),
    "hyperopt_continue": Arg(
        "--continue",
        help="Continue hyperopt from previous runs. "
        "By default, temporary files will be removed and hyperopt will start from scratch.",
        default=False,
        action='store_true',
    ),
    "loss_function": Arg(
        '--loss-function',
        help='Define the loss-function to use for hyperopt.'
        'Possibilities are `legacy`, and `custom` (providing a custom loss-function).'
        'Default: `%(default)s`.',
        choices=['legacy', 'sharpe', 'custom'],
        default='legacy',
    ),
    # List exchanges
    "print_one_column": Arg(
        '-1', '--one-column',
        help='Print exchanges in one column.',
        action='store_true',
    ),
    # Script options
    "pairs": Arg(
        '-p', '--pairs',
        help='Show profits for only these pairs. Pairs are comma-separated.',
    ),
    # Download data
    "pairs_file": Arg(
        '--pairs-file',
        help='File containing a list of pairs to download.',
        metavar='FILE',
    ),
    "days": Arg(
        '--days',
        help='Download data for given number of days.',
        type=check_int_positive,
        metavar='INT',
    ),
    "exchange": Arg(
        '--exchange',
        help=f'Exchange name (default: `{constants.DEFAULT_EXCHANGE}`). '
        f'Only valid if no config is provided.',
    ),
    "timeframes": Arg(
        '-t', '--timeframes',
        help=f'Specify which tickers to download. Space-separated list. '
        f'Default: `{constants.DEFAULT_DOWNLOAD_TICKER_INTERVALS}`.',
        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h',
                 '6h', '8h', '12h', '1d', '3d', '1w'],
        nargs='+',
    ),
    "erase": Arg(
        '--erase',
        help='Clean all existing data for the selected exchange/pairs/timeframes.',
        action='store_true',
    ),
    # Plot dataframe
    "indicators1": Arg(
        '--indicators1',
        help='Set indicators from your strategy you want in the first row of the graph. '
        'Comma-separated list. Example: `ema3,ema5`. Default: `%(default)s`.',
        default='sma,ema3,ema5',
    ),
    "indicators2": Arg(
        '--indicators2',
        help='Set indicators from your strategy you want in the third row of the graph. '
        'Comma-separated list. Example: `fastd,fastk`. Default: `%(default)s`.',
        default='macd,macdsignal',
    ),
    "plot_limit": Arg(
        '--plot-limit',
        help='Specify tick limit for plotting. Notice: too high values cause huge files. '
        'Default: %(default)s.',
        type=check_int_positive,
        metavar='INT',
        default=750,
    ),
    "trade_source": Arg(
        '--trade-source',
        help='Specify the source for trades (Can be DB or file (backtest file)) '
        'Default: %(default)s',
        choices=["DB", "file"],
        default="file",
    ),
}


ARGS_COMMON = ["verbosity", "logfile", "version", "config", "datadir"]

ARGS_STRATEGY = ["strategy", "strategy_path"]

ARGS_MAIN = ARGS_COMMON + ARGS_STRATEGY + ["dynamic_whitelist", "db_url", "sd_notify"]

ARGS_COMMON_OPTIMIZE = ["ticker_interval", "timerange",
                        "max_open_trades", "stake_amount", "refresh_pairs"]

ARGS_BACKTEST = ARGS_COMMON_OPTIMIZE + ["position_stacking", "use_max_market_positions",
                                        "live", "strategy_list", "export", "exportfilename"]

ARGS_HYPEROPT = ARGS_COMMON_OPTIMIZE + ["hyperopt", "position_stacking", "epochs", "spaces",
                                        "use_max_market_positions", "print_all", "hyperopt_jobs",
                                        "hyperopt_random_state", "hyperopt_min_trades",
                                        "hyperopt_continue", "loss_function"]

ARGS_EDGE = ARGS_COMMON_OPTIMIZE + ["stoploss_range"]

ARGS_LIST_EXCHANGES = ["print_one_column"]

ARGS_DOWNLOADER = ARGS_COMMON + ["pairs", "pairs_file", "days", "exchange", "timeframes", "erase"]

ARGS_PLOT_DATAFRAME = (ARGS_COMMON + ARGS_STRATEGY +
                       ["pairs", "indicators1", "indicators2", "plot_limit", "db_url",
                        "trade_source", "export", "exportfilename", "timerange",
                        "refresh_pairs", "live"])

ARGS_PLOT_PROFIT = (ARGS_COMMON + ARGS_STRATEGY +
                    ["pairs", "timerange", "export", "exportfilename", "db_url", "trade_source"])


class TimeRange(NamedTuple):
    """
    NamedTuple defining timerange inputs.
    [start/stop]type defines if [start/stop]ts shall be used.
    if *type is None, don't use corresponding startvalue.
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
        self.build_args(optionlist=ARGS_MAIN)

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

    def build_args(self, optionlist, parser=None):
        parser = parser or self.parser

        for val in optionlist:
            opt = AVAILABLE_CLI_OPTIONS[val]
            parser.add_argument(*opt.cli, dest=val, **opt.kwargs)

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
        self.build_args(optionlist=ARGS_BACKTEST, parser=backtesting_cmd)

        # Add edge subcommand
        edge_cmd = subparsers.add_parser('edge', help='Edge module.')
        edge_cmd.set_defaults(func=start_edge)
        self.build_args(optionlist=ARGS_EDGE, parser=edge_cmd)

        # Add hyperopt subcommand
        hyperopt_cmd = subparsers.add_parser('hyperopt', help='Hyperopt module.')
        hyperopt_cmd.set_defaults(func=start_hyperopt)
        self.build_args(optionlist=ARGS_HYPEROPT, parser=hyperopt_cmd)

        # Add list-exchanges subcommand
        list_exchanges_cmd = subparsers.add_parser(
            'list-exchanges',
            help='Print available exchanges.'
        )
        list_exchanges_cmd.set_defaults(func=start_list_exchanges)
        self.build_args(optionlist=ARGS_LIST_EXCHANGES, parser=list_exchanges_cmd)

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
