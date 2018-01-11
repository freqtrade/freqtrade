#!/usr/bin/env python3

"""This script generate json data from bittrex"""
from os import path

from freqtrade import exchange
from freqtrade.exchange import Bittrex
from freqtrade import misc

PAIRS = [
    'BTC_BCC', 'BTC_ETH', 'BTC_MER', 'BTC_POWR', 'BTC_ETC',
    'BTC_OK', 'BTC_NEO', 'BTC_EMC2', 'BTC_DASH', 'BTC_LSK',
    'BTC_LTC', 'BTC_XZC', 'BTC_OMG', 'BTC_STRAT', 'BTC_XRP',
    'BTC_QTUM', 'BTC_WAVES', 'BTC_VTC', 'BTC_XLM', 'BTC_MCO'
]
TICKER_INTERVAL = 5  # ticker interval in minutes (currently implemented: 1 and 5)
OUTPUT_DIR = path.dirname(path.realpath(__file__))

# Init Bittrex exchange
exchange._API = Bittrex({'key': '', 'secret': ''})

for pair in PAIRS:
    data = exchange.get_ticker_history(pair, TICKER_INTERVAL)
    filename = path.join(OUTPUT_DIR, '{}-{}.json'.format(
        pair,
        TICKER_INTERVAL,
    ))
    misc.file_dump_json(filename, data)
