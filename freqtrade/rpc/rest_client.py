#!/usr/bin/env python3
"""
Simple command line client into RPC commands
Can be used as an alternate to Telegram

Should not import anything from freqtrade,
so it can be used as a standalone script.
"""

import argparse
import logging
import time
from sys import argv

import click

from requests import get
from requests.exceptions import ConnectionError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ft_rest_client")

# TODO - use IP and Port from config.json not hardcode

COMMANDS_NO_ARGS = ["start",
                    "stop",
                    ]
COMMANDS_ARGS = ["daily",
                 ]

SERVER_URL = "http://localhost:5002"


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        help="Positional argument defining the command to execute.")
    args = parser.parse_args()
    # if len(argv) == 1:
    #     print('\nThis script accepts the following arguments')
    #     print('- daily (int) - Where int is the number of days to report back. daily 3')
    #     print('- start  - this will start the trading thread')
    #     print('- stop  - this will start the trading thread')
    #     print('- there will be more....\n')
    return vars(args)


def call_authorized(url):
    try:
        return get(url).json()
    except ConnectionError:
        logger.warning("Connection error")


def call_command_noargs(command):
    logger.info(f"Running command `{command}` at {SERVER_URL}")
    r = call_authorized(f"{SERVER_URL}/{command}")
    logger.info(r)


def main(args):

    # Call commands without arguments
    if args["command"] in COMMANDS_NO_ARGS:
        call_command_noargs(args["command"])

    if args["command"] == "daily":
        if str.isnumeric(argv[2]):
            get_url = SERVER_URL + '/daily?timescale=' + argv[2]
            d = get(get_url).json()
            print(d)
        else:
            print("\nThe second argument to daily must be an integer, 1,2,3 etc")


if __name__ == "__main__":
    args = add_arguments()
    main(args)
