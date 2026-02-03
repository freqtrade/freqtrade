``` output
usage: freqtrade backtesting-show [-h] [-v] [--no-color] [--logfile FILE] [-V]
                                  [-c PATH] [-d PATH] [--userdir PATH]
                                  [--backtest-filename PATH]
                                  [--backtest-directory PATH]
                                  [--show-pair-list]
                                  [--breakdown {day,week,month,year,weekday} [{day,week,month,year,weekday} ...]]

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
  --show-pair-list      Show backtesting pairlist sorted by profit.
  --breakdown {day,week,month,year,weekday} [{day,week,month,year,weekday} ...]
                        Show backtesting breakdown per [day, week, month,
                        year, weekday].

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
