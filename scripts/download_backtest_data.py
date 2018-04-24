#!/usr/bin/env python3

"""This script generate json data from bittrex"""
import json
import sys
import os
import time
import datetime

DEFAULT_DL_PATH = 'freqtrade/tests/testdata'

arguments = arguments.Arguments(sys.argv[1:], 'download utility')
arguments.testdata_dl_options()
args = arguments.parse_args()

TICKER_INTERVALS = ['1m', '5m']
PAIRS = []
MIN_SECCONDS = 60
HOUR_SECCONDS = 60 * MIN_SECCONDS
DAY_SECCONDS = 24 * HOUR_SECCONDS

if args.pairs_file:
    with open(args.pairs_file) as file:
        PAIRS = json.load(file)
PAIRS = list(set(PAIRS))

dl_path = DEFAULT_DL_PATH
if args.export and os.path.exists(args.export):
    dl_path = args.export

print(f'About to download pairs: {PAIRS} to {dl_path}')

# Init exchange
exchange._API = exchange.init_ccxt({'key': '',
                                    'secret': '',
                                    'name': args.exchange})

for pair in PAIRS:
    for tick_interval in TICKER_INTERVALS:
        print(f'downloading pair {pair}, interval {tick_interval}')

        since_time = None
        if args.days:
            since_time = int((time.time() - args.days * DAY_SECCONDS) * 1000)

        # download data until it reaches today now time
        data = []
        while not since_time or since_time < (time.time() - 10 * MIN_SECCONDS) * 1000:
            data_part = exchange.get_ticker_history(pair, tick_interval, since=since_time)

            if not data_part:
                print('\tNo data since %s' % datetime.datetime.utcfromtimestamp(since_time / 1000).strftime('%Y-%m-%dT%H:%M:%S'))
                break
            
            print('\tData received for period %s - %s' % 
                    (datetime.datetime.utcfromtimestamp(data_part[0][0] / 1000).strftime('%Y-%m-%dT%H:%M:%S'),
                    datetime.datetime.utcfromtimestamp(data_part[-1][0] / 1000).strftime('%Y-%m-%dT%H:%M:%S')))

            data.extend(data_part)
            since_time = data[-1][0] + 1

        # save data
        pair_print = pair.replace('/', '_')
        filename = f'{pair_print}-{tick_interval}.json'
        misc.file_dump_json(os.path.join(dl_path, filename), data)
