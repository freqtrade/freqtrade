``` output
usage: freqtrade backtesting-analysis [-h] [-v] [--no-color] [--logfile FILE]
                                      [-V] [-c PATH] [-d PATH]
                                      [--userdir PATH]
                                      [--backtest-filename PATH]
                                      [--backtest-directory PATH]
                                      [--analysis-groups {0,1,2,3,4,5} [{0,1,2,3,4,5} ...]]
                                      [--enter-reason-list ENTER_REASON_LIST [ENTER_REASON_LIST ...]]
                                      [--exit-reason-list EXIT_REASON_LIST [EXIT_REASON_LIST ...]]
                                      [--indicator-list INDICATOR_LIST [INDICATOR_LIST ...]]
                                      [--entry-only] [--exit-only]
                                      [--timerange TIMERANGE]
                                      [--rejected-signals] [--analysis-to-csv]
                                      [--analysis-csv-path ANALYSIS_CSV_PATH]

options:
  -h, --help            show this help message and exit
  --backtest-filename, --export-filename PATH
                        Use this filename for backtest results.Example:
                        `--backtest-
                        filename=backtest_results_2020-09-27_16-20-48.json`.
                        Assumes either `user_data/backtest_results/` or
                        `--export-directory` as base directory.
  --backtest-directory, --export-directory PATH
                        Directory to use for backtest results. Example:
                        `--export-directory=user_data/backtest_results/`.
  --analysis-groups {0,1,2,3,4,5} [{0,1,2,3,4,5} ...]
                        grouping output - 0: simple wins/losses by enter tag,
                        1: by enter_tag, 2: by enter_tag and exit_tag, 3: by
                        pair and enter_tag, 4: by pair, enter_ and exit_tag
                        (this can get quite large), 5: by exit_tag
  --enter-reason-list ENTER_REASON_LIST [ENTER_REASON_LIST ...]
                        Space separated list of entry signals to analyse.
                        Default: all. e.g. 'entry_tag_a entry_tag_b'
  --exit-reason-list EXIT_REASON_LIST [EXIT_REASON_LIST ...]
                        Space separated list of exit signals to analyse.
                        Default: all. e.g. 'exit_tag_a roi stop_loss
                        trailing_stop_loss'
  --indicator-list INDICATOR_LIST [INDICATOR_LIST ...]
                        Space separated list of indicators to analyse. e.g.
                        'close rsi bb_lowerband profit_abs'
  --entry-only          Only analyze entry signals.
  --exit-only           Only analyze exit signals.
  --timerange TIMERANGE
                        Specify what timerange of data to use.
  --rejected-signals    Analyse rejected signals
  --analysis-to-csv     Save selected analysis tables to individual CSVs
  --analysis-csv-path ANALYSIS_CSV_PATH
                        Specify a path to save the analysis CSVs if
                        --analysis-to-csv is enabled. Default:
                        user_data/basktesting_results/

Common arguments:
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --no-color            Disable colorization of hyperopt results. May be
                        useful if you are redirecting output to a file.
  --logfile, --log-file FILE
                        Log to the file specified. Special values are:
                        'syslog', 'journald'. See the documentation for more
                        details.
  -V, --version         show program's version number and exit
  -c, --config PATH     Specify configuration file (default:
                        `userdir/config.json` or `config.json` whichever
                        exists). Multiple --config options may be used. Can be
                        set to `-` to read config from stdin.
  -d, --datadir, --data-dir PATH
                        Path to the base directory of the exchange with
                        historical backtesting data. To see futures data, use
                        trading-mode additionally.
  --userdir, --user-data-dir PATH
                        Path to userdata directory.

```
