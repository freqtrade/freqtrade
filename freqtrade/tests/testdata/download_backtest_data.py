#!/usr/bin/env python3

"""This script generate json data from bittrex"""
import json
from os import path

from freqtrade import exchange
from freqtrade.exchange import Bittrex

PAIRS = ['BTC-OK', 'BTC-NEO', 'BTC-DASH', 'BTC-ETC', 'BTC-ETH', 'BTC-SNT']
TICKER_INTERVAL = 1  # ticker interval in minutes (currently implemented: 1 and 5)
OUTPUT_DIR = path.dirname(path.realpath(__file__))

# Init Bittrex exchange
exchange._API = Bittrex({'key': '', 'secret': ''})

for pair in PAIRS:
    data = exchange.get_ticker_history(pair, TICKER_INTERVAL)
    filename = path.join(OUTPUT_DIR, '{}-{}m.json'.format(
        pair.lower(),
        TICKER_INTERVAL,
    ))
    with open(filename, 'w') as fp:
        json.dump(data, fp)
