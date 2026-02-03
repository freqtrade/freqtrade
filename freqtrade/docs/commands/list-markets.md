``` output
usage: freqtrade list-markets [-h] [-v] [--no-color] [--logfile FILE] [-V]
                              [-c PATH] [-d PATH] [--userdir PATH]
                              [--exchange EXCHANGE] [--print-list]
                              [--print-json] [-1] [--print-csv]
                              [--base BASE_CURRENCY [BASE_CURRENCY ...]]
                              [--quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]]
                              [-a] [--trading-mode {spot,margin,futures}]

options:
  -h, --help            show this help message and exit
  --exchange EXCHANGE   Exchange name. Only valid if no config is provided.
  --print-list          Print list of pairs or market symbols. By default data
                        is printed in the tabular format.
  --print-json          Print list of pairs or market symbols in JSON format.
  -1, --one-column      Print output in one column.
  --print-csv           Print exchange pair or market data in the csv format.
  --base BASE_CURRENCY [BASE_CURRENCY ...]
                        Specify base currency(-ies). Space-separated list.
  --quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]
                        Specify quote currency(-ies). Space-separated list.
  -a, --all             Print all pairs or market symbols. By default only
                        active ones are shown.
  --trading-mode, --tradingmode {spot,margin,futures}
                        Select Trading mode

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
