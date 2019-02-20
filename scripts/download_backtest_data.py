#!/usr/bin/env python3

"""This script generate json data"""
import json
import sys
from pathlib import Path
import arrow
from typing import Any, Dict

from freqtrade.arguments import Arguments
from freqtrade.arguments import TimeRange
from freqtrade.exchange import Exchange
from freqtrade.data.history import download_pair_history
from freqtrade.configuration import Configuration, set_loggers
from freqtrade.misc import deep_merge_dicts

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
set_loggers(0)

DEFAULT_DL_PATH = 'user_data/data'

arguments = Arguments(sys.argv[1:], 'download utility')
arguments.testdata_dl_options()
args = arguments.parse_args()

timeframes = args.timeframes

if args.config:
    configuration = Configuration(args)

    config: Dict[str, Any] = {}
    # Now expecting a list of config filenames here, not a string
    for path in args.config:
        print('Using config: %s ...', path)
        # Merge config options, overwriting old values
        config = deep_merge_dicts(configuration._load_config_file(path), config)

    config['stake_currency'] = ''
    # Ensure we do not use Exchange credentials
    config['exchange']['key'] = ''
    config['exchange']['secret'] = ''
else:
    config = {'stake_currency': '',
              'dry_run': True,
              'exchange': {
                  'name': args.exchange,
                  'key': '',
                  'secret': '',
                  'pair_whitelist': [],
                  'ccxt_async_config': {
                      "enableRateLimit": False
                  }
              }
              }


dl_path = Path(DEFAULT_DL_PATH).joinpath(config['exchange']['name'])
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
exchange = Exchange(config)
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
        download_pair_history(datadir=dl_path, exchange=exchange,
                              pair=pair,
                              tick_interval=tick_interval,
                              timerange=timerange)


if pairs_not_available:
    print(f"Pairs [{','.join(pairs_not_available)}] not availble.")
