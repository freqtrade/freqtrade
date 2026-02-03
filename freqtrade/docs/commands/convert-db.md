``` output
usage: freqtrade convert-db [-h] [--db-url PATH] [--db-url-from PATH]

options:
  -h, --help          show this help message and exit
  --db-url PATH       Override trades database URL, this is useful in custom
                      deployments (default: `sqlite:///tradesv3.sqlite` for
                      Live Run mode, `sqlite:///tradesv3.dryrun.sqlite` for
                      Dry Run).
  --db-url-from PATH  Source db url to use when migrating a database.

```
