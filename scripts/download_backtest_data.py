#!/usr/bin/env python3

"""This script generate json data from bittrex"""
import json
import sys
import os
import arrow

from freqtrade import (exchange, arguments, misc)

DEFAULT_DL_PATH = 'user_data/data'

arguments = arguments.Arguments(sys.argv[1:], 'download utility')
arguments.testdata_dl_options()
args = arguments.parse_args()

TICKER_INTERVALS = ['1m', '5m', '15m', '30m', '1h', '4h']
PAIRS = [
    "ETH/BTC",
    "LTC/BTC",
    "BNB/BTC",
    "NEO/BTC",
    "GAS/BTC",
    "MCO/BTC",
    "WTC/BTC",
    "QTUM/BTC",
    "OMG/BTC",
    "ZRX/BTC",
    "STRAT/BTC",
    "SNGLS/BTC",
    "BQX/BTC",
    "KNC/BTC",
    "FUN/BTC",
    "SNM/BTC",
    "LINK/BTC",
    "XVG/BTC",
    "SALT/BTC",
    "IOTA/BTC",
    "MDA/BTC",
    "MTL/BTC",
    "SUB/BTC",
    "EOS/BTC",
    "SNT/BTC",
    "ETC/BTC",
    "MTH/BTC",
    "ENG/BTC",
    "DNT/BTC",
    "BNT/BTC",
    "AST/BTC",
    "DASH/BTC",
    "ICN/BTC",
    "OAX/BTC",
    "BTG/BTC",
    "EVX/BTC",
    "REQ/BTC",
    "LRC/BTC",
    "VIB/BTC",
    "HSR/BTC",
    "TRX/BTC",
    "POWR/BTC",
    "ARK/BTC",
    "XRP/BTC",
    "MOD/BTC",
    "ENJ/BTC",
    "STORJ/BTC",
    "VEN/BTC",
    "KMD/BTC",
    "RCN/BTC",
    "NULS/BTC",
    "RDN/BTC",
    "XMR/BTC",
    "DLT/BTC",
    "AMB/BTC",
    "BAT/BTC",
    "ZEC/BTC",
    "BCPT/BTC",
    "ARN/BTC",
    "GVT/BTC",
    "CDT/BTC",
    "GXS/BTC",
    "POE/BTC",
    "QSP/BTC",
    "BTS/BTC",
    "XZC/BTC",
    "LSK/BTC",
    "TNT/BTC",
    "FUEL/BTC",
    "MANA/BTC",
    "BCD/BTC",
    "DGD/BTC",
    "ADX/BTC",
    "ADA/BTC",
    "PPT/BTC",
    "CMT/BTC",
    "XLM/BTC",
    "CND/BTC",
    "LEND/BTC",
    "WABI/BTC",
    "TNB/BTC",
    "WAVES/BTC",
    "ICX/BTC",
    "GTO/BTC",
    "OST/BTC",
    "ELF/BTC",
    "AION/BTC",
    "NEBL/BTC",
    "BRD/BTC",
    "EDO/BTC",
    "WINGS/BTC",
    "NAV/BTC",
    "LUN/BTC",
    "TRIG/BTC",
    "APPC/BTC",
    "VIBE/BTC",
    "RLC/BTC",
    "INS/BTC",
    "PIVX/BTC",
    "IOST/BTC",
    "CHAT/BTC",
    "STEEM/BTC",
    "VIA/BTC",
    "BLZ/BTC",
    "AE/BTC",
    "RPX/BTC",
    "NCASH/BTC",
    "POA/BTC",
    "ZIL/BTC",
    "ONT/BTC",
    "STORM/BTC",
    "XEM/BTC",
    "WAN/BTC",
    "QLC/BTC",
    "SYS/BTC",
    "WPR/BTC",
    "GRS/BTC",
    "CLOAK/BTC",
    "GNT/BTC",
    "LOOM/BTC",
    "BCN/BTC",
    "REP/BTC"
]

if args.pairs_file:
    with open(args.pairs_file) as file:
        PAIRS = json.load(file)
PAIRS = list(set(PAIRS))

dl_path = DEFAULT_DL_PATH
if args.export and os.path.exists(args.export):
    dl_path = args.export

since_time = None
if args.days:
    since_time = arrow.utcnow().shift(days=-args.days).timestamp * 1000

print(f'About to download pairs: {PAIRS} to {dl_path}')

# Init exchange
exchange._API = exchange.init_ccxt({'key': '',
                                    'secret': '',
                                    'name': args.exchange})

for pair in PAIRS:
    for tick_interval in TICKER_INTERVALS:
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
