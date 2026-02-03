``` output
usage: freqtrade download-data [-h] [-v] [--no-color] [--logfile FILE] [-V]
                               [-c PATH] [-d PATH] [--userdir PATH]
                               [-p PAIRS [PAIRS ...]] [--pairs-file FILE]
                               [--days INT] [--new-pairs-days INT]
                               [--include-inactive-pairs]
                               [--no-parallel-download]
                               [--timerange TIMERANGE] [--dl-trades]
                               [--convert] [--exchange EXCHANGE]
                               [-t TIMEFRAMES [TIMEFRAMES ...]] [--erase]
                               [--data-format-ohlcv {json,jsongz,feather,parquet}]
                               [--data-format-trades {json,jsongz,feather,parquet}]
                               [--trading-mode {spot,margin,futures}]
                               [--candle-types {spot,futures,mark,index,premiumIndex,funding_rate} [{spot,futures,mark,index,premiumIndex,funding_rate} ...]]
                               [--prepend]

options:
  -h, --help            show this help message and exit
  -p, --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  --pairs-file FILE     File containing a list of pairs. Takes precedence over
                        --pairs or pairs configured in the configuration.
  --days INT            Download data for given number of days.
  --new-pairs-days INT  Download data of new pairs for given number of days.
                        Default: `None`.
  --include-inactive-pairs
                        Also download data from inactive pairs.
  --no-parallel-download
                        Disable parallel startup download. Only use this if
                        you experience issues.
  --timerange TIMERANGE
                        Specify what timerange of data to use.
  --dl-trades           Download trades instead of OHLCV data.
  --convert             Convert downloaded trades to OHLCV data. Only
                        applicable in combination with `--dl-trades`. Will be
                        automatic for exchanges which don't have historic
                        OHLCV (e.g. Kraken). If not provided, use `trades-to-
                        ohlcv` to convert trades data to OHLCV data.
  --exchange EXCHANGE   Exchange name. Only valid if no config is provided.
  -t, --timeframes TIMEFRAMES [TIMEFRAMES ...]
                        Specify which tickers to download. Space-separated
                        list. Default: `1m 5m`.
  --erase               Clean all existing data for the selected
                        exchange/pairs/timeframes.
  --data-format-ohlcv {json,jsongz,feather,parquet}
                        Storage format for downloaded candle (OHLCV) data.
                        (default: `feather`).
  --data-format-trades {json,jsongz,feather,parquet}
                        Storage format for downloaded trades data. (default:
                        `feather`).
  --trading-mode, --tradingmode {spot,margin,futures}
                        Select Trading mode
  --candle-types {spot,futures,mark,index,premiumIndex,funding_rate} [{spot,futures,mark,index,premiumIndex,funding_rate} ...]
                        Select candle type to download. Defaults to the
                        necessary candles for the selected trading mode (e.g.
                        'spot' or ('futures', 'funding_rate' and 'mark') for
                        futures).
  --prepend             Allow data prepending. (Data-appending is disabled)

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
