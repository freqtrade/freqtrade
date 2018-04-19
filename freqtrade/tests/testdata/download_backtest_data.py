#!/usr/bin/env python3

"""This script generate json data from bittrex"""
import json
import sys

from freqtrade import exchange
from freqtrade import arguments
from freqtrade.exchange import ccxt

BASE_PATH = 'freqtrade/tests/testdata'

arguments = arguments.Arguments(sys.argv[1:], 'download utility')
arguments.scripts_options()
args = arguments.parse_args()

TICKER_INTERVALS = ['1m', '5m']
PAIRS = []

if args.pair:
    with open(args.pair) as file:
        PAIRS = json.load(file)
PAIRS = list(set(PAIRS))

print('About to download pairs:', PAIRS)

# Init Bittrex exchange
exchange._API = ccxt.bittrex({'key': '', 'secret': ''})

for pair in PAIRS:
    for tick_interval in TICKER_INTERVALS:
        print('downloading pair %s, interval %s' % (pair, tick_interval))
        data = exchange.get_ticker_history(pair, tick_interval)
        pair_print = pair.replace('/', '_')
        filename = '{}-{}.json'.format(pair_print, tick_interval)
        misc.file_dump_json(filename, data)
