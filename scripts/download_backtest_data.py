#!/usr/bin/env python3

"""This script generate json data from bittrex"""
import json
import sys
import os

from freqtrade import (exchange, arguments, misc)

DEFAULT_DL_PATH = 'freqtrade/tests/testdata'

arguments = arguments.Arguments(sys.argv[1:], 'download utility')
arguments.testdata_dl_options()
args = arguments.parse_args()

TICKER_INTERVALS = ['1m', '5m']
PAIRS = []

if args.pairs_file:
    with open(args.pairs_file) as file:
        PAIRS = json.load(file)
PAIRS = list(set(PAIRS))

dl_path = DEFAULT_DL_PATH
if args.export and os.path.exists(args.export):
    dl_path = args.export

print(f'About to download pairs: {PAIRS} to {dl_path}')

# Init Bittrex exchange
exchange._API = exchange.init_ccxt({'key': '',
                                    'secret': '',
                                    'name': 'bittrex'})

for pair in PAIRS:
    for tick_interval in TICKER_INTERVALS:
        print(f'downloading pair {pair}, interval {tick_interval}')
        data = exchange.get_ticker_history(pair, tick_interval)
        pair_print = pair.replace('/', '_')
        filename = f'{pair_print}-{tick_interval}.json'
        misc.file_dump_json(os.path.join(dl_path, filename), data)
