#!/usr/bin/env python3

"""This script generate json data from bittrex"""
import json
from os import path
import urllib.request
import ssl


PAIRS = ['BTC-BCC', 'BTC-DASH', 'BTC-EDG', 'BTC-ETC', 'BTC-ETH', 'BTC-LTC', 'BTC-MTL', 'BTC-NEO',
         'BTC-OK', 'BTC-OMG', 'BTC-PAY', 'BTC-PIVX', 'BTC-QTUM', 'BTC-SNT', 'BTC-XMR', 'BTC-XRP', 'BTC-XZC', 'BTC-ZEC']

OUTPUT_DIR = path.dirname(path.realpath(__file__))


for pair in PAIRS:
    print('========== Generating', pair, ' ==========')

    filename = path.join(OUTPUT_DIR, '{}.json'.format(
        pair.lower(),
    ))
    print(filename)

    if path.isfile(filename):
        with open(filename) as fp:
            data = json.load(fp)
        print("Current Start:", data[1])
        print("Current End: ", data[-1:])
    else:
        data=[]
        print("Current Start: None")
        print("Current End: None")



    query = 'https://bittrex.com/Api/v2.0/pub/market/GetTicks?marketName=' + pair + '&tickInterval=oneMin'
    print("Sending query:", query)
    req = urllib.request.urlopen(url=query, timeout=60, context=ssl._create_unverified_context())
    new_data = json.loads(req.read())
    new_data = new_data['result']

    for row in new_data:
        if row not in data:
            data.append(row)
    #print("New Start:", data[1])
    print("New End: ", data[-1:])
    data = sorted(data, key=lambda data: data['T'])


    with open(filename, "w") as fp:
       json.dump(data, fp, indent=1)