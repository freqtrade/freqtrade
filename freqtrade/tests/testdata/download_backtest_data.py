#!/usr/bin/env python3

"""This script generate json data from bittrex"""
import json
import sys

from freqtrade import exchange
from freqtrade import misc
from freqtrade.exchange import ccxt

parser = misc.common_args_parser('download utility')
parser.add_argument(
    '-p', '--pair',
    help='JSON file containing pairs to download',
    dest='pair',
    default=None
)
args = parser.parse_args(sys.argv[1:])

TICKER_INTERVALS = ['1m', '5m']  # ticker interval in minutes (currently implemented: 1 and 5)
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
