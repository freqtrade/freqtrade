#!/usr/bin/env python3
"""
Simple command line client into RPC commands
Can be used as an alternate to Telegram

Should not import anything from freqtrade,
so it can be used as a standalone script.
"""

import argparse
import json
import logging
from sys import argv
from pathlib import Path

from requests import get
from requests.exceptions import ConnectionError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ft_rest_client")


COMMANDS_NO_ARGS = ["start",
                    "stop",
                    "stopbuy",
                    "reload_conf"
                    ]
COMMANDS_ARGS = ["daily",
                 ]


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        help="Positional argument defining the command to execute.")
    parser.add_argument('-c', '--config',
                        help='Specify configuration file (default: %(default)s). ',
                        dest='config',
                        type=str,
                        metavar='PATH',
                        default='config.json'
                        )
    args = parser.parse_args()
    # if len(argv) == 1:
    #     print('\nThis script accepts the following arguments')
    #     print('- daily (int) - Where int is the number of days to report back. daily 3')
    #     print('- start  - this will start the trading thread')
    #     print('- stop  - this will start the trading thread')
    #     print('- there will be more....\n')
    return vars(args)


def load_config(configfile):
    file = Path(configfile)
    if file.is_file():
        with file.open("r") as f:
            config = json.load(f)
    return config


def call_authorized(url):
    try:
        return get(url).json()
    except ConnectionError:
        logger.warning("Connection error")


def call_command_noargs(server_url, command):
    logger.info(f"Running command `{command}` at {server_url}")
    r = call_authorized(f"{server_url}/{command}")
    logger.info(r)


def main(args):

    config = load_config(args["config"])
    url = config.get("api_server", {}).get("server_url", "127.0.0.1")
    port = config.get("api_server", {}).get("listen_port", "8080")
    server_url = f"http://{url}:{port}"

    # Call commands without arguments
    if args["command"] in COMMANDS_NO_ARGS:
        call_command_noargs(server_url, args["command"])

    if args["command"] == "daily":
        if str.isnumeric(argv[2]):
            get_url = server_url + '/daily?timescale=' + argv[2]
            d = get(get_url).json()
            print(d)
        else:
            print("\nThe second argument to daily must be an integer, 1,2,3 etc")


if __name__ == "__main__":
    args = add_arguments()
    main(args)
