"""
Definition of cli arguments used in arguments.py
"""
from argparse import SUPPRESS, ArgumentTypeError

from freqtrade import __version__, constants
from freqtrade.constants import HYPEROPT_LOSS_BUILTIN
from freqtrade.enums import CandleType


def check_int_positive(value: str) -> int:
    try:
        uint = int(value)
        if uint <= 0:
            raise ValueError
    except ValueError:
        raise ArgumentTypeError(
            f"{value} is invalid for this parameter, should be a positive integer value"
        )
    return uint


def check_int_nonzero(value: str) -> int:
    try:
        uint = int(value)
        if uint == 0:
            raise ValueError
    except ValueError:
        raise ArgumentTypeError(
            f"{value} is invalid for this parameter, should be a non-zero integer value"
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
        '--logfile', '--log-file',
        help="Log to the file specified. Special values are: 'syslog', 'journald'. "
             "See the documentation for more details.",
        metavar='FILE',
    ),
    "version": Arg(
        '-V', '--version',
        action='version',
        version=f'%(prog)s {__version__}',
    ),
    "config": Arg(
        '-c', '--config',
        help=f'Specify configuration file (default: `userdir/{constants.DEFAULT_CONFIG}` '
        f'or `config.json` whichever exists). '
        f'Multiple --config options may be used. '
        f'Can be set to `-` to read config from stdin.',
        action='append',
        metavar='PATH',
    ),
    "datadir": Arg(
        '-d', '--datadir', '--data-dir',
        help='Path to directory with historical backtesting data.',
        metavar='PATH',
    ),
    "user_data_dir": Arg(
        '--userdir', '--user-data-dir',
        help='Path to userdata directory.',
        metavar='PATH',
    ),
    "reset": Arg(
        '--reset',
        help='Reset sample files to their original state.',
        action='store_true',
    ),
    "recursive_strategy_search": Arg(
        '--recursive-strategy-search',
        help='Recursively search for a strategy in the strategies folder.',
        action='store_true',
    ),
    # Main options
    "strategy": Arg(
        '-s', '--strategy',
        help='Specify strategy class name which will be used by the bot.',
        metavar='NAME',
    ),
    "strategy_path": Arg(
        '--strategy-path',
        help='Specify additional strategy lookup path.',
        metavar='PATH',
    ),
    "db_url": Arg(
        '--db-url',
        help=f'Override trades database URL, this is useful in custom deployments '
        f'(default: `{constants.DEFAULT_DB_PROD_URL}` for Live Run mode, '
        f'`{constants.DEFAULT_DB_DRYRUN_URL}` for Dry Run).',
        metavar='PATH',
    ),
    "db_url_from": Arg(
        '--db-url-from',
        help='Source db url to use when migrating a database.',
        metavar='PATH',
    ),
    "sd_notify": Arg(
        '--sd-notify',
        help='Notify systemd service manager.',
        action='store_true',
    ),
    "dry_run": Arg(
        '--dry-run',
        help='Enforce dry-run for trading (removes Exchange secrets and simulates trades).',
        action='store_true',
    ),
    "dry_run_wallet": Arg(
        '--dry-run-wallet', '--starting-balance',
        help='Starting balance, used for backtesting / hyperopt and dry-runs.',
        type=float,
    ),
    # Optimize common
    "timeframe": Arg(
        '-i', '--timeframe',
        help='Specify timeframe (`1m`, `5m`, `30m`, `1h`, `1d`).',
    ),
    "timerange": Arg(
        '--timerange',
        help='Specify what timerange of data to use.',
    ),
    "max_open_trades": Arg(
        '--max-open-trades',
        help='Override the value of the `max_open_trades` configuration setting.',
        type=int,
        metavar='INT',
    ),
    "stake_amount": Arg(
        '--stake-amount',
        help='Override the value of the `stake_amount` configuration setting.',
    ),
    # Backtesting
    "timeframe_detail": Arg(
        '--timeframe-detail',
        help='Specify detail timeframe for backtesting (`1m`, `5m`, `30m`, `1h`, `1d`).',
    ),
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
    "backtest_show_pair_list": Arg(
        '--show-pair-list',
        help='Show backtesting pairlist sorted by profit.',
        action='store_true',
        default=False,
    ),
    "enable_protections": Arg(
        '--enable-protections', '--enableprotections',
        help='Enable protections for backtesting.'
        'Will slow backtesting down by a considerable amount, but will include '
        'configured protections',
        action='store_true',
        default=False,
    ),
    "strategy_list": Arg(
        '--strategy-list',
        help='Provide a space-separated list of strategies to backtest. '
        'Please note that timeframe needs to be set either in config '
        'or via command line. When using this together with `--export trades`, '
        'the strategy-name is injected into the filename '
        '(so `backtest-data.json` becomes `backtest-data-SampleStrategy.json`',
        nargs='+',
    ),
    "export": Arg(
        '--export',
        help='Export backtest results (default: trades).',
        choices=constants.EXPORT_OPTIONS,
    ),
    "exportfilename": Arg(
        "--export-filename",
        "--backtest-filename",
        help="Use this filename for backtest results."
        "Requires `--export` to be set as well. "
        "Example: `--export-filename=user_data/backtest_results/backtest_today.json`",
        metavar="PATH",
    ),
    "disableparamexport": Arg(
        '--disable-param-export',
        help="Disable automatic hyperopt parameter export.",
        action='store_true',
    ),
    "fee": Arg(
        '--fee',
        help='Specify fee ratio. Will be applied twice (on trade entry and exit).',
        type=float,
        metavar='FLOAT',
    ),
    "backtest_breakdown": Arg(
        '--breakdown',
        help='Show backtesting breakdown per [day, week, month].',
        nargs='+',
        choices=constants.BACKTEST_BREAKDOWNS
    ),
    "backtest_cache": Arg(
        '--cache',
        help='Load a cached backtest result no older than specified age (default: %(default)s).',
        default=constants.BACKTEST_CACHE_DEFAULT,
        choices=constants.BACKTEST_CACHE_AGE,
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
        '--hyperopt',
        help=SUPPRESS,
        metavar='NAME',
        required=False,
    ),
    "hyperopt_path": Arg(
        '--hyperopt-path',
        help='Specify additional lookup path for Hyperopt Loss functions.',
        metavar='PATH',
    ),
    "epochs": Arg(
        '-e', '--epochs',
        help='Specify number of epochs (default: %(default)d).',
        type=check_int_positive,
        metavar='INT',
        default=constants.HYPEROPT_EPOCH,
    ),
    "spaces": Arg(
        '--spaces',
        help='Specify which parameters to hyperopt. Space-separated list.',
        choices=['all', 'buy', 'sell', 'roi', 'stoploss',
                 'trailing', 'protection', 'trades', 'default'],
        nargs='+',
        default='default',
    ),
    "analyze_per_epoch": Arg(
        '--analyze-per-epoch',
        help='Run populate_indicators once per epoch.',
        action='store_true',
        default=False,
    ),

    "print_all": Arg(
        '--print-all',
        help='Print all results, not only the best ones.',
        action='store_true',
        default=False,
    ),
    "print_colorized": Arg(
        '--no-color',
        help='Disable colorization of hyperopt results. May be useful if you are '
        'redirecting output to a file.',
        action='store_false',
        default=True,
    ),
    "print_json": Arg(
        '--print-json',
        help='Print output in JSON format.',
        action='store_true',
        default=False,
    ),
    "export_csv": Arg(
        '--export-csv',
        help='Export to CSV-File.'
        ' This will disable table print.'
        ' Example: --export-csv hyperopt.csv',
        metavar='FILE',
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
    "hyperopt_loss": Arg(
        '--hyperopt-loss', '--hyperoptloss',
        help='Specify the class name of the hyperopt loss function class (IHyperOptLoss). '
        'Different functions can generate completely different results, '
        'since the target for optimization is different. Built-in Hyperopt-loss-functions are: '
        f'{", ".join(HYPEROPT_LOSS_BUILTIN)}',
        metavar='NAME',
    ),
    "hyperoptexportfilename": Arg(
        '--hyperopt-filename',
        help='Hyperopt result filename.'
        'Example: `--hyperopt-filename=hyperopt_results_2020-09-27_16-20-48.pickle`',
        metavar='FILENAME',
    ),
    # List exchanges
    "print_one_column": Arg(
        '-1', '--one-column',
        help='Print output in one column.',
        action='store_true',
    ),
    "list_exchanges_all": Arg(
        '-a', '--all',
        help='Print all exchanges known to the ccxt library.',
        action='store_true',
    ),
    # List pairs / markets
    "list_pairs_all": Arg(
        '-a', '--all',
        help='Print all pairs or market symbols. By default only active '
             'ones are shown.',
        action='store_true',
    ),
    "print_list": Arg(
        '--print-list',
        help='Print list of pairs or market symbols. By default data is '
             'printed in the tabular format.',
        action='store_true',
    ),
    "list_pairs_print_json": Arg(
        '--print-json',
        help='Print list of pairs or market symbols in JSON format.',
        action='store_true',
        default=False,
    ),
    "print_csv": Arg(
        '--print-csv',
        help='Print exchange pair or market data in the csv format.',
        action='store_true',
    ),
    "quote_currencies": Arg(
        '--quote',
        help='Specify quote currency(-ies). Space-separated list.',
        nargs='+',
        metavar='QUOTE_CURRENCY',
    ),
    "base_currencies": Arg(
        '--base',
        help='Specify base currency(-ies). Space-separated list.',
        nargs='+',
        metavar='BASE_CURRENCY',
    ),
    "trading_mode": Arg(
        '--trading-mode', '--tradingmode',
        help='Select Trading mode',
        choices=constants.TRADING_MODES,
    ),
    "candle_types": Arg(
        '--candle-types',
        help='Select candle type to use',
        choices=[c.value for c in CandleType],
        nargs='+',
    ),
    # Script options
    "pairs": Arg(
        '-p', '--pairs',
        help='Limit command to these pairs. Pairs are space-separated.',
        nargs='+',
    ),
    # Download data
    "pairs_file": Arg(
        '--pairs-file',
        help='File containing a list of pairs. '
             'Takes precedence over --pairs or pairs configured in the configuration.',
        metavar='FILE',
    ),
    "days": Arg(
        '--days',
        help='Download data for given number of days.',
        type=check_int_positive,
        metavar='INT',
    ),
    "include_inactive": Arg(
        '--include-inactive-pairs',
        help='Also download data from inactive pairs.',
        action='store_true',
    ),
    "new_pairs_days": Arg(
        '--new-pairs-days',
        help='Download data of new pairs for given number of days. Default: `%(default)s`.',
        type=check_int_positive,
        metavar='INT',
    ),
    "download_trades": Arg(
        '--dl-trades',
        help='Download trades instead of OHLCV data. The bot will resample trades to the '
             'desired timeframe as specified as --timeframes/-t.',
        action='store_true',
    ),
    "format_from": Arg(
        '--format-from',
        help='Source format for data conversion.',
        choices=constants.AVAILABLE_DATAHANDLERS,
        required=True,
    ),
    "format_to": Arg(
        '--format-to',
        help='Destination format for data conversion.',
        choices=constants.AVAILABLE_DATAHANDLERS,
        required=True,
    ),
    "dataformat_ohlcv": Arg(
        '--data-format-ohlcv',
        help='Storage format for downloaded candle (OHLCV) data. (default: `json`).',
        choices=constants.AVAILABLE_DATAHANDLERS,
    ),
    "dataformat_trades": Arg(
        '--data-format-trades',
        help='Storage format for downloaded trades data. (default: `jsongz`).',
        choices=constants.AVAILABLE_DATAHANDLERS_TRADES,
    ),
    "show_timerange": Arg(
        '--show-timerange',
        help='Show timerange available for available data. (May take a while to calculate).',
        action='store_true',
    ),
    "exchange": Arg(
        '--exchange',
        help=f'Exchange name (default: `{constants.DEFAULT_EXCHANGE}`). '
        f'Only valid if no config is provided.',
    ),
    "timeframes": Arg(
        '-t', '--timeframes',
        help='Specify which tickers to download. Space-separated list. '
        'Default: `1m 5m`.',
        default=['1m', '5m'],
        nargs='+',
    ),
    "prepend_data": Arg(
        '--prepend',
        help='Allow data prepending. (Data-appending is disabled)',
        action='store_true',
    ),
    "erase": Arg(
        '--erase',
        help='Clean all existing data for the selected exchange/pairs/timeframes.',
        action='store_true',
    ),
    "erase_ui_only": Arg(
        '--erase',
        help="Clean UI folder, don't download new version.",
        action='store_true',
        default=False,
    ),
    "ui_version": Arg(
        '--ui-version',
        help=('Specify a specific version of FreqUI to install. '
              'Not specifying this installs the latest version.'),
        type=str,
    ),
    # Templating options
    "template": Arg(
        '--template',
        help='Use a template which is either `minimal`, '
        '`full` (containing multiple sample indicators) or `advanced`. Default: `%(default)s`.',
        choices=['full', 'minimal', 'advanced'],
        default='full',
    ),
    # Plot dataframe
    "indicators1": Arg(
        '--indicators1',
        help='Set indicators from your strategy you want in the first row of the graph. '
        "Space-separated list. Example: `ema3 ema5`. Default: `['sma', 'ema3', 'ema5']`.",
        nargs='+',
    ),
    "indicators2": Arg(
        '--indicators2',
        help='Set indicators from your strategy you want in the third row of the graph. '
        "Space-separated list. Example: `fastd fastk`. Default: `['macd', 'macdsignal']`.",
        nargs='+',
    ),
    "plot_limit": Arg(
        '--plot-limit',
        help='Specify tick limit for plotting. Notice: too high values cause huge files. '
        'Default: %(default)s.',
        type=check_int_positive,
        metavar='INT',
        default=750,
    ),
    "plot_auto_open": Arg(
        '--auto-open',
        help='Automatically open generated plot.',
        action='store_true',
    ),
    "no_trades": Arg(
        '--no-trades',
        help='Skip using trades from backtesting file and DB.',
        action='store_true',
    ),
    "trade_source": Arg(
        '--trade-source',
        help='Specify the source for trades (Can be DB or file (backtest file)) '
        'Default: %(default)s',
        choices=["DB", "file"],
        default="file",
    ),
    "trade_ids": Arg(
        '--trade-ids',
        help='Specify the list of trade ids.',
        nargs='+',
    ),
    # hyperopt-list, hyperopt-show
    "hyperopt_list_profitable": Arg(
        '--profitable',
        help='Select only profitable epochs.',
        action='store_true',
    ),
    "hyperopt_list_best": Arg(
        '--best',
        help='Select only best epochs.',
        action='store_true',
    ),
    "hyperopt_list_min_trades": Arg(
        '--min-trades',
        help='Select epochs with more than INT trades.',
        type=check_int_positive,
        metavar='INT',
    ),
    "hyperopt_list_max_trades": Arg(
        '--max-trades',
        help='Select epochs with less than INT trades.',
        type=check_int_positive,
        metavar='INT',
    ),
    "hyperopt_list_min_avg_time": Arg(
        '--min-avg-time',
        help='Select epochs above average time.',
        type=float,
        metavar='FLOAT',
    ),
    "hyperopt_list_max_avg_time": Arg(
        '--max-avg-time',
        help='Select epochs below average time.',
        type=float,
        metavar='FLOAT',
    ),
    "hyperopt_list_min_avg_profit": Arg(
        '--min-avg-profit',
        help='Select epochs above average profit.',
        type=float,
        metavar='FLOAT',
    ),
    "hyperopt_list_max_avg_profit": Arg(
        '--max-avg-profit',
        help='Select epochs below average profit.',
        type=float,
        metavar='FLOAT',
    ),
    "hyperopt_list_min_total_profit": Arg(
        '--min-total-profit',
        help='Select epochs above total profit.',
        type=float,
        metavar='FLOAT',
    ),
    "hyperopt_list_max_total_profit": Arg(
        '--max-total-profit',
        help='Select epochs below total profit.',
        type=float,
        metavar='FLOAT',
    ),
    "hyperopt_list_min_objective": Arg(
        '--min-objective',
        help='Select epochs above objective.',
        type=float,
        metavar='FLOAT',
    ),
    "hyperopt_list_max_objective": Arg(
        '--max-objective',
        help='Select epochs below objective.',
        type=float,
        metavar='FLOAT',
    ),
    "hyperopt_list_no_details": Arg(
        '--no-details',
        help='Do not print best epoch details.',
        action='store_true',
    ),
    "hyperopt_show_index": Arg(
        '-n', '--index',
        help='Specify the index of the epoch to print details for.',
        type=check_int_nonzero,
        metavar='INT',
    ),
    "hyperopt_show_no_header": Arg(
        '--no-header',
        help='Do not print epoch details header.',
        action='store_true',
    ),
    "hyperopt_ignore_missing_space": Arg(
        "--ignore-missing-spaces", "--ignore-unparameterized-spaces",
        help=("Suppress errors for any requested Hyperopt spaces "
              "that do not contain any parameters."),
        action="store_true",
    ),
    "analysis_groups": Arg(
        "--analysis-groups",
        help=("grouping output - "
              "0: simple wins/losses by enter tag, "
              "1: by enter_tag, "
              "2: by enter_tag and exit_tag, "
              "3: by pair and enter_tag, "
              "4: by pair, enter_ and exit_tag (this can get quite large), "
              "5: by exit_tag"),
        nargs='+',
        default=[],
        choices=['0', '1', '2', '3', '4', '5'],
    ),
    "enter_reason_list": Arg(
        "--enter-reason-list",
        help=("Space separated list of entry signals to analyse. Default: all. "
              "e.g. 'entry_tag_a entry_tag_b'"),
        nargs='+',
        default=['all'],
    ),
    "exit_reason_list": Arg(
        "--exit-reason-list",
        help=("Space separated list of exit signals to analyse. Default: all. "
              "e.g. 'exit_tag_a roi stop_loss trailing_stop_loss'"),
        nargs='+',
        default=['all'],
    ),
    "indicator_list": Arg(
        "--indicator-list",
        help=("Space separated list of indicators to analyse. "
              "e.g. 'close rsi bb_lowerband profit_abs'"),
        nargs='+',
        default=[],
    ),
    "analysis_rejected": Arg(
        '--rejected-signals',
        help='Analyse rejected signals',
        action='store_true',
    ),
    "analysis_to_csv": Arg(
        '--analysis-to-csv',
        help='Save selected analysis tables to individual CSVs',
        action='store_true',
    ),
    "analysis_csv_path": Arg(
        '--analysis-csv-path',
        help=("Specify a path to save the analysis CSVs "
              "if --analysis-to-csv is enabled. Default: user_data/basktesting_results/"),
    ),
    "freqaimodel": Arg(
        '--freqaimodel',
        help='Specify a custom freqaimodels.',
        metavar='NAME',
    ),
    "freqaimodel_path": Arg(
        '--freqaimodel-path',
        help='Specify additional lookup path for freqaimodels.',
        metavar='PATH',
    ),
    "freqai_backtest_live_models": Arg(
        '--freqai-backtest-live-models',
        help='Run backtest with ready models.',
        action='store_true'
    ),
    "minimum_trade_amount": Arg(
        '--minimum-trade-amount',
        help='Minimum trade amount for lookahead-analysis',
        type=check_int_positive,
        metavar='INT',
    ),
    "targeted_trade_amount": Arg(
        '--targeted-trade-amount',
        help='Targeted trade amount for lookahead analysis',
        type=check_int_positive,
        metavar='INT',
    ),
    "lookahead_analysis_exportfilename": Arg(
        '--lookahead-analysis-exportfilename',
        help="Use this csv-filename to store lookahead-analysis-results",
        type=str
    ),
}
