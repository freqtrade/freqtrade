#!/usr/bin/env python3

"""This script generate json data"""
import json
import sys
from pathlib import Path
import arrow

from freqtrade import arguments
from freqtrade.arguments import TimeRange
from freqtrade.exchange import Exchange
from freqtrade.data.history import download_backtesting_testdata
from freqtrade.configuration import set_loggers

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
set_loggers(0)

DEFAULT_DL_PATH = 'user_data/data'

arguments = arguments.Arguments(sys.argv[1:], 'download utility')
arguments.testdata_dl_options()
args = arguments.parse_args()

timeframes = args.timeframes

dl_path = Path(DEFAULT_DL_PATH).joinpath(args.exchange)
if args.export:
    dl_path = Path(args.export)

if not dl_path.is_dir():
    sys.exit(f'Directory {dl_path} does not exist.')

pairs_file = Path(args.pairs_file) if args.pairs_file else dl_path.joinpath('pairs.json')
if not pairs_file.exists():
    sys.exit(f'No pairs file found with path {pairs_file}.')

with pairs_file.open() as file:
    PAIRS = list(set(json.load(file)))

PAIRS.sort()


timerange = TimeRange()
if args.days:
    time_since = arrow.utcnow().shift(days=-args.days).strftime("%Y%m%d")
    timerange = arguments.parse_timerange(f'{time_since}-')


print(f'About to download pairs: {PAIRS} to {dl_path}')


# Init exchange
exchange = Exchange({'key': '',
                     'secret': '',
                     'stake_currency': '',
                     'dry_run': True,
                     'exchange': {
                         'name': args.exchange,
                         'pair_whitelist': [],
                         'ccxt_async_config': {
                             "enableRateLimit": False
                         }
                     }
                     })
pairs_not_available = []

for pair in PAIRS:
    if pair not in exchange._api.markets:
        pairs_not_available.append(pair)
        print(f"skipping pair {pair}")
        continue
    for tick_interval in timeframes:
        pair_print = pair.replace('/', '_')
        filename = f'{pair_print}-{tick_interval}.json'
        dl_file = dl_path.joinpath(filename)
        if args.erase and dl_file.exists():
            print(f'Deleting existing data for pair {pair}, interval {tick_interval}')
            dl_file.unlink()

        print(f'downloading pair {pair}, interval {tick_interval}')
        download_backtesting_testdata(dl_path, exchange=exchange,
                                      pair=pair,
                                      tick_interval=tick_interval,
                                      timerange=timerange)


if pairs_not_available:
    print(f"Pairs [{','.join(pairs_not_available)}] not availble.")
