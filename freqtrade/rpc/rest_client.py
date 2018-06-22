#!/usr/bin/env python3
"""
Simple command line client into RPC commands
Can be used as an alternate to Telegram
"""

import time
from requests import get
from sys import argv

# TODO - use argparse to clean this up
# TODO - use IP and Port from config.json not hardcode

if len(argv) == 1:
    print('\nThis script accepts the following arguments')
    print('- daily (int) - Where int is the number of days to report back. daily 3')
    print('- start  - this will start the trading thread')
    print('- stop  - this will start the trading thread')
    print('- there will be more....\n')

if len(argv) == 3 and argv[1] == "daily":
    if str.isnumeric(argv[2]):
        get_url = 'http://localhost:5002/daily?timescale=' + argv[2]
        d = get(get_url).json()
        print(d)
    else:
        print("\nThe second argument to daily must be an integer, 1,2,3 etc")

if len(argv) == 2 and argv[1] == "start":
    get_url = 'http://localhost:5002/start'
    d = get(get_url).text
    print(d)

    if "already" not in d:
        time.sleep(2)
        d = get(get_url).text
        print(d)

if len(argv) == 2 and argv[1] == "stop":
    get_url = 'http://localhost:5002/stop'
    d = get(get_url).text
    print(d)

    if "already" not in d:
        time.sleep(2)
        d = get(get_url).text
        print(d)
