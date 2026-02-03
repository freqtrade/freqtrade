``` output
usage: freqtrade hyperopt [-h] [-v] [--no-color] [--logfile FILE] [-V]
                          [-c PATH] [-d PATH] [--userdir PATH] [-s NAME]
                          [--strategy-path PATH] [--recursive-strategy-search]
                          [--freqaimodel NAME] [--freqaimodel-path PATH]
                          [-i TIMEFRAME] [--timerange TIMERANGE]
                          [--data-format-ohlcv {json,jsongz,feather,parquet}]
                          [--max-open-trades INT]
                          [--stake-amount STAKE_AMOUNT] [--fee FLOAT]
                          [-p PAIRS [PAIRS ...]] [--hyperopt-path PATH]
                          [--eps] [--enable-protections]
                          [--dry-run-wallet DRY_RUN_WALLET]
                          [--timeframe-detail TIMEFRAME_DETAIL] [-e INT]
                          [--spaces SPACES [SPACES ...]] [--print-all]
                          [--print-json] [-j JOBS] [--random-state INT]
                          [--min-trades INT] [--hyperopt-loss NAME]
                          [--disable-param-export] [--ignore-missing-spaces]
                          [--analyze-per-epoch] [--early-stop INT]

options:
  -h, --help            show this help message and exit
  -i, --timeframe TIMEFRAME
                        Specify timeframe (`1m`, `5m`, `30m`, `1h`, `1d`).
  --timerange TIMERANGE
                        Specify what timerange of data to use.
  --data-format-ohlcv {json,jsongz,feather,parquet}
                        Storage format for downloaded candle (OHLCV) data.
                        (default: `feather`).
  --max-open-trades INT
                        Override the value of the `max_open_trades`
                        configuration setting.
  --stake-amount STAKE_AMOUNT
                        Override the value of the `stake_amount` configuration
                        setting.
  --fee FLOAT           Specify fee ratio. Will be applied twice (on trade
                        entry and exit).
  -p, --pairs PAIRS [PAIRS ...]
                        Limit command to these pairs. Pairs are space-
                        separated.
  --hyperopt-path PATH  Specify additional lookup path for Hyperopt Loss
                        functions.
  --eps, --enable-position-stacking
                        Allow buying the same pair multiple times (position
                        stacking). Only applicable to backtesting and
                        hyperopt. Results archived by this cannot be
                        reproduced in dry/live trading.
  --enable-protections, --enableprotections
                        Enable protections for backtesting. Will slow
                        backtesting down by a considerable amount, but will
                        include configured protections
  --dry-run-wallet, --starting-balance DRY_RUN_WALLET
                        Starting balance, used for backtesting / hyperopt and
                        dry-runs.
  --timeframe-detail TIMEFRAME_DETAIL
                        Specify detail timeframe for backtesting (`1m`, `5m`,
                        `30m`, `1h`, `1d`).
  -e, --epochs INT      Specify number of epochs (default: 100).
  --spaces SPACES [SPACES ...]
                        Specify which parameters to hyperopt. Space-separated
                        list. Available builtin options (custom spaces will
                        not be listed here): default, all, buy, sell, enter,
                        exit, roi, stoploss, trailing, protection, trades.
                        Default: `default` - which includes all spaces except
                        for 'trailing', 'protection', and 'trades'.
  --print-all           Print all results, not only the best ones.
  --print-json          Print output in JSON format.
  -j, --job-workers JOBS
                        The number of concurrently running jobs for
                        hyperoptimization (hyperopt worker processes). If -1
                        (default), all CPUs are used, for -2, all CPUs but one
                        are used, etc. If 1 is given, no parallel computing
                        code is used at all.
  --random-state INT    Set random state to some positive integer for
                        reproducible hyperopt results.
  --min-trades INT      Set minimal desired number of trades for evaluations
                        in the hyperopt optimization path (default: 1).
  --hyperopt-loss, --hyperoptloss NAME
                        Specify the class name of the hyperopt loss function
                        class (IHyperOptLoss). Different functions can
                        generate completely different results, since the
                        target for optimization is different. Built-in
                        Hyperopt-loss-functions are:
                        ShortTradeDurHyperOptLoss, OnlyProfitHyperOptLoss,
                        SharpeHyperOptLoss, SharpeHyperOptLossDaily,
                        SortinoHyperOptLoss, SortinoHyperOptLossDaily,
                        CalmarHyperOptLoss, MaxDrawDownHyperOptLoss,
                        MaxDrawDownRelativeHyperOptLoss,
                        MaxDrawDownPerPairHyperOptLoss,
                        ProfitDrawDownHyperOptLoss, MultiMetricHyperOptLoss
  --disable-param-export
                        Disable automatic hyperopt parameter export.
  --ignore-missing-spaces, --ignore-unparameterized-spaces
                        Suppress errors for any requested Hyperopt spaces that
                        do not contain any parameters.
  --analyze-per-epoch   Run populate_indicators once per epoch.
  --early-stop INT      Early stop hyperopt if no improvement after (default:
                        0) epochs.

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

Strategy arguments:
  -s, --strategy NAME   Specify strategy class name which will be used by the
                        bot.
  --strategy-path PATH  Specify additional strategy lookup path.
  --recursive-strategy-search
                        Recursively search for a strategy in the strategies
                        folder.
  --freqaimodel NAME    Specify a custom freqaimodels.
  --freqaimodel-path PATH
                        Specify additional lookup path for freqaimodels.

```
