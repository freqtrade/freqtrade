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
from urllib.parse import urlencode, urlparse, urlunparse
from pathlib import Path

import requests
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


class FtRestClient():

    def __init__(self, serverurl):

        self.serverurl = serverurl
        self.session = requests.Session()

    def _call(self, method, apipath, params: dict = None, data=None, files=None):

        if str(method).upper() not in ('GET', 'POST', 'PUT', 'DELETE'):
            raise ValueError('invalid method <{0}>'.format(method))
        basepath = f"{self.serverurl}/{apipath}"

        hd = {"Accept": "application/json",
              "Content-Type": "application/json"
              }

        # Split url
        schema, netloc, path, params, query, fragment = urlparse(basepath)
        # URLEncode query string
        query = urlencode(params)
        # recombine url
        url = urlunparse((schema, netloc, path, params, query, fragment))
        print(url)
        try:
            resp = self.session.request(method, url, headers=hd, data=data,
                                        # auth=self.session.auth
                                        )
            # return resp.text
            return resp.json()
        except ConnectionError:
            logger.warning("Connection error")

    def _call_command_noargs(self, command):
        logger.info(f"Running command `{command}` at {self.serverurl}")
        r = self._call("POST", command)
        logger.info(r)

    def _call_info(self, command, command_args):
        logger.info(
            f"Running command `{command}` with parameters `{command_args}` at {self.serverurl}")
        args = INFO_COMMANDS[command]
        if len(args) < len(command_args):
            logger.error(f"Command {command} does only support {len(args)} arguments.")
            return
        params = {}
        for idx, arg in enumerate(command_args):
            params[args[idx]] = arg

        logger.debug(params)
        r = self._call("GET", command, params)

        logger.info(r)


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


def main(args):

    config = load_config(args["config"])
    url = config.get("api_server", {}).get("server_url", "127.0.0.1")
    port = config.get("api_server", {}).get("listen_port", "8080")
    server_url = f"http://{url}:{port}"
    client = FtRestClient(server_url)

    # Call commands without arguments
    if args["command"] in COMMANDS_NO_ARGS:
        client._call_command_noargs(args["command"])

    if args["command"] in INFO_COMMANDS:
        client._call_info(args["command"], args["command_arguments"])


if __name__ == "__main__":
    args = add_arguments()
    main(args)
