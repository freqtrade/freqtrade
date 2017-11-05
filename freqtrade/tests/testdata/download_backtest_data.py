#!/usr/bin/env python3

"""This script generate json data from bittrex"""
import json
from os import path

from freqtrade import exchange
from freqtrade.exchange import Bittrex

PAIRS = ['BTC-OK', 'BTC-NEO', 'BTC-DASH', 'BTC-ETC', 'BTC-ETH', 'BTC-SNT']
OUTPUT_DIR = path.dirname(path.realpath(__file__))

# Init Bittrex exchange
exchange._API = Bittrex({'key': '', 'secret': ''})

for pair in PAIRS:
    data = exchange.get_ticker_history(pair)
    filename = path.join(OUTPUT_DIR, '{}.json'.format(pair.lower()))
    with open(filename, 'w') as fp:
        json.dump(data, fp)
