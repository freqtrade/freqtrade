# Utility Subcommands

Besides the Live-Trade and Dry-Run run modes, the `backtesting`, `edge` and `hyperopt` optimization subcommands, and the `download-data` subcommand which prepares historical data, the bot contains a number of utility subcommands. They are described in this section.

## List Exchanges

```
usage: freqtrade list-exchanges [-h] [-1] [-a]

optional arguments:
  -h, --help        show this help message and exit
  -1, --one-column  Print exchanges in one column.
  -a, --all         Print all exchanges known to the ccxt library.
```

## List Timeframes

```
usage: freqtrade list-timeframes [-h] [--exchange EXCHANGE] [-1]

optional arguments:
  -h, --help           show this help message and exit
  --exchange EXCHANGE  Exchange name (default: `bittrex`). Only valid if no
                       config is provided.
  -1, --one-column     Print exchanges in one column.

```
