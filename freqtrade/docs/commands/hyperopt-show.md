``` output
usage: freqtrade hyperopt-show [-h] [-v] [--no-color] [--logfile FILE] [-V]
                               [-c PATH] [-d PATH] [--userdir PATH] [--best]
                               [--profitable] [-n INT] [--print-json]
                               [--hyperopt-filename FILENAME] [--no-header]
                               [--disable-param-export]
                               [--breakdown {day,week,month,year,weekday} [{day,week,month,year,weekday} ...]]

options:
  -h, --help            show this help message and exit
  --best                Select only best epochs.
  --profitable          Select only profitable epochs.
  -n, --index INT       Specify the index of the epoch to print details for.
  --print-json          Print output in JSON format.
  --hyperopt-filename FILENAME
                        Hyperopt result filename.Example: `--hyperopt-
                        filename=hyperopt_results_2020-09-27_16-20-48.pickle`
  --no-header           Do not print epoch details header.
  --disable-param-export
                        Disable automatic hyperopt parameter export.
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
