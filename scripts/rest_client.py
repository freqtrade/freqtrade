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
                    "reload_conf",
                    ]
INFO_COMMANDS = {"version": [],
                 "count": [],
                 "daily": ["timescale"],
                 "profit": [],
                 "status": [],
                 "balance": []
                 }


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        help="Positional argument defining the command to execute.")

    parser.add_argument("command_arguments",
                        help="Positional arguments for the parameters for [command]",
                        nargs="*",
                        default=[]
                        )

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
    return {}


def call_authorized(url):
    try:
        return get(url).json()
    except ConnectionError:
        logger.warning("Connection error")


def call_command_noargs(server_url, command):
    logger.info(f"Running command `{command}` at {server_url}")
    r = call_authorized(f"{server_url}/{command}")
    logger.info(r)


def call_info(server_url, command, command_args):
    logger.info(f"Running command `{command}` with parameters `{command_args}` at {server_url}")
    call = f"{server_url}/{command}?"
    args = INFO_COMMANDS[command]
    if len(args) < len(command_args):
        logger.error(f"Command {command} does only support {len(args)} arguments.")
        return
    for idx, arg in enumerate(command_args):

        call += f"{args[idx]}={arg}"
    logger.debug(call)
    r = call_authorized(call)

    logger.info(r)


def main(args):

    config = load_config(args["config"])
    url = config.get("api_server", {}).get("server_url", "127.0.0.1")
    port = config.get("api_server", {}).get("listen_port", "8080")
    server_url = f"http://{url}:{port}"

    # Call commands without arguments
    if args["command"] in COMMANDS_NO_ARGS:
        call_command_noargs(server_url, args["command"])

    if args["command"] in INFO_COMMANDS:
        call_info(server_url, args["command"], args["command_arguments"])


if __name__ == "__main__":
    args = add_arguments()
    main(args)
