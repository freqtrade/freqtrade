``` output
usage: freqtrade test-pairlist [-h] [--userdir PATH] [-v] [-c PATH]
                               [--quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]]
                               [-1] [--print-json] [--exchange EXCHANGE]

options:
  -h, --help            show this help message and exit
  --userdir, --user-data-dir PATH
                        Path to userdata directory.
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  -c, --config PATH     Specify configuration file (default:
                        `userdir/config.json` or `config.json` whichever
                        exists). Multiple --config options may be used. Can be
                        set to `-` to read config from stdin.
  --quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]
                        Specify quote currency(-ies). Space-separated list.
  -1, --one-column      Print output in one column.
  --print-json          Print list of pairs or market symbols in JSON format.
  --exchange EXCHANGE   Exchange name. Only valid if no config is provided.

```
