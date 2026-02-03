``` output
usage: freqtrade hyperopt-list [-h] [-v] [--no-color] [--logfile FILE] [-V]
                               [-c PATH] [-d PATH] [--userdir PATH] [--best]
                               [--profitable] [--min-trades INT]
                               [--max-trades INT] [--min-avg-time FLOAT]
                               [--max-avg-time FLOAT] [--min-avg-profit FLOAT]
                               [--max-avg-profit FLOAT]
                               [--min-total-profit FLOAT]
                               [--max-total-profit FLOAT]
                               [--min-objective FLOAT] [--max-objective FLOAT]
                               [--print-json] [--no-details]
                               [--hyperopt-filename FILENAME]
                               [--export-csv FILE]

options:
  -h, --help            show this help message and exit
  --best                Select only best epochs.
  --profitable          Select only profitable epochs.
  --min-trades INT      Select epochs with more than INT trades.
  --max-trades INT      Select epochs with less than INT trades.
  --min-avg-time FLOAT  Select epochs above average time.
  --max-avg-time FLOAT  Select epochs below average time.
  --min-avg-profit FLOAT
                        Select epochs above average profit.
  --max-avg-profit FLOAT
                        Select epochs below average profit.
  --min-total-profit FLOAT
                        Select epochs above total profit.
  --max-total-profit FLOAT
                        Select epochs below total profit.
  --min-objective FLOAT
                        Select epochs above objective.
  --max-objective FLOAT
                        Select epochs below objective.
  --print-json          Print output in JSON format.
  --no-details          Do not print best epoch details.
  --hyperopt-filename FILENAME
                        Hyperopt result filename.Example: `--hyperopt-
                        filename=hyperopt_results_2020-09-27_16-20-48.pickle`
  --export-csv FILE     Export to CSV-File. This will disable table print.
                        Example: --export-csv hyperopt.csv

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
