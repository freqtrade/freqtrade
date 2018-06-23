#!/usr/bin/env python3

"""This script generate json data from bittrex"""
import json
import sys
import os
import arrow

from freqtrade import (arguments, misc)
from freqtrade.exchange import Exchange

DEFAULT_DL_PATH = 'user_data/data'

arguments = arguments.Arguments(sys.argv[1:], 'download utility')
arguments.testdata_dl_options()
args = arguments.parse_args()

timeframes = args.timeframes

dl_path = os.path.join(DEFAULT_DL_PATH, args.exchange)
if args.export:
    dl_path = args.export

if not os.path.isdir(dl_path):
    sys.exit(f'Directory {dl_path} does not exist.')

pairs_file = args.pairs_file if args.pairs_file else os.path.join(dl_path, 'pairs.json')
if not os.path.isfile(pairs_file):
    sys.exit(f'No pairs file found with path {pairs_file}.')

with open(pairs_file) as file:
    PAIRS = list(set(json.load(file)))

PAIRS.sort()

since_time = None
if args.days:
    since_time = arrow.utcnow().shift(days=-args.days).timestamp * 1000


print(f'About to download pairs: {PAIRS} to {dl_path}')


# Init exchange
exchange = Exchange({'key': '',
                     'secret': '',
                     'stake_currency': '',
                     'dry_run': True,
                     'exchange': {
                        'name': args.exchange,
                        'pair_whitelist': []
                        }
                     })
pairs_not_available = []

for pair in PAIRS:
    if pair not in exchange._api.markets:
        pairs_not_available.append(pair)
        print(f"skipping pair {pair}")
        continue
    for tick_interval in timeframes:
        print(f'downloading pair {pair}, interval {tick_interval}')

        data = exchange.get_ticker_history(pair, tick_interval, since_ms=since_time)
        if not data:
            print('\tNo data was downloaded')
            break

        print('\tData was downloaded for period %s - %s' % (
            arrow.get(data[0][0] / 1000).format(),
            arrow.get(data[-1][0] / 1000).format()))

        # save data
        pair_print = pair.replace('/', '_')
        filename = f'{pair_print}-{tick_interval}.json'
        misc.file_dump_json(os.path.join(dl_path, filename), data)


if pairs_not_available:
    print(f"Pairs [{','.join(pairs_not_available)}] not availble.")
