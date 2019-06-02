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
import inspect
from urllib.parse import urlencode, urlparse, urlunparse
from pathlib import Path

import requests
from requests.exceptions import ConnectionError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ft_rest_client")


class FtRestClient():

    def __init__(self, serverurl, username=None, password=None):

        self._serverurl = serverurl
        self._session = requests.Session()
        self._session.auth = (username, password)

    def _call(self, method, apipath, params: dict = None, data=None, files=None):

        if str(method).upper() not in ('GET', 'POST', 'PUT', 'DELETE'):
            raise ValueError('invalid method <{0}>'.format(method))
        basepath = f"{self._serverurl}/api/v1/{apipath}"

        hd = {"Accept": "application/json",
              "Content-Type": "application/json"
              }

        # Split url
        schema, netloc, path, par, query, fragment = urlparse(basepath)
        # URLEncode query string
        query = urlencode(params) if params else ""
        # recombine url
        url = urlunparse((schema, netloc, path, par, query, fragment))

        try:
            resp = self._session.request(method, url, headers=hd, data=json.dumps(data))
            # return resp.text
            return resp.json()
        except ConnectionError:
            logger.warning("Connection error")

    def _get(self, apipath, params: dict = None):
        return self._call("GET", apipath, params=params)

    def _post(self, apipath, params: dict = None, data: dict = None):
        return self._call("POST", apipath, params=params, data=data)

    def start(self):
        """
        Start the bot if it's in stopped state.
        :returns: json object
        """
        return self._post("start")

    def stop(self):
        """
        Stop the bot. Use start to restart
        :returns: json object
        """
        return self._post("stop")

    def stopbuy(self):
        """
        Stop buying (but handle sells gracefully).
        use reload_conf to reset
        :returns: json object
        """
        return self._post("stopbuy")

    def reload_conf(self):
        """
        Reload configuration
        :returns: json object
        """
        return self._post("reload_conf")

    def balance(self):
        """
        Get the account balance
        :returns: json object
        """
        return self._get("balance")

    def count(self):
        """
        Returns the amount of open trades
        :returns: json object
        """
        return self._get("count")

    def daily(self, days=None):
        """
        Returns the amount of open trades
        :returns: json object
        """
        return self._get("daily", params={"timescale": days} if days else None)

    def edge(self):
        """
        Returns information about edge
        :returns: json object
        """
        return self._get("edge")

    def profit(self):
        """
        Returns the profit summary
        :returns: json object
        """
        return self._get("profit")

    def performance(self):
        """
        Returns the performance of the different coins
        :returns: json object
        """
        return self._get("performance")

    def status(self):
        """
        Get the status of open trades
        :returns: json object
        """
        return self._get("status")

    def version(self):
        """
        Returns the version of the bot
        :returns: json object containing the version
        """
        return self._get("version")

    def whitelist(self):
        """
        Show the current whitelist
        :returns: json object
        """
        return self._get("whitelist")

    def blacklist(self, *args):
        """
        Show the current blacklist
        :param add: List of coins to add (example: "BNB/BTC")
        :returns: json object
        """
        if not args:
            return self._get("blacklist")
        else:
            return self._post("blacklist", data={"blacklist": args})

    def forcebuy(self, pair, price=None):
        """
        Buy an asset
        :param pair: Pair to buy (ETH/BTC)
        :param price: Optional - price to buy
        :returns: json object of the trade
        """
        data = {"pair": pair,
                "price": price
                }
        return self._post("forcebuy", data=data)

    def forcesell(self, tradeid):
        """
        Force-sell a trade
        :param tradeid: Id of the trade (can be received via status command)
        :returns: json object
        """

        return self._post("forcesell", data={"tradeid": tradeid})


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        help="Positional argument defining the command to execute.")

    parser.add_argument('--show',
                        help='Show possible methods with this client',
                        dest='show',
                        action='store_true',
                        default=False
                        )

    parser.add_argument('-c', '--config',
                        help='Specify configuration file (default: %(default)s). ',
                        dest='config',
                        type=str,
                        metavar='PATH',
                        default='config.json'
                        )

    parser.add_argument("command_arguments",
                        help="Positional arguments for the parameters for [command]",
                        nargs="*",
                        default=[]
                        )

    args = parser.parse_args()
    return vars(args)


def load_config(configfile):
    file = Path(configfile)
    if file.is_file():
        with file.open("r") as f:
            config = json.load(f)
        return config
    return {}


def print_commands():
    # Print dynamic help for the different commands using the commands doc-strings
    client = FtRestClient(None)
    print("Possible commands:")
    for x, y in inspect.getmembers(client):
        if not x.startswith('_'):
            print(f"{x} {getattr(client, x).__doc__}")


def main(args):

    if args.get("help"):
        print_commands()

    config = load_config(args["config"])
    url = config.get("api_server", {}).get("server_url", "127.0.0.1")
    port = config.get("api_server", {}).get("listen_port", "8080")
    username = config.get("api_server", {}).get("username")
    password = config.get("api_server", {}).get("password")

    server_url = f"http://{url}:{port}"
    client = FtRestClient(server_url, username, password)

    m = [x for x, y in inspect.getmembers(client) if not x.startswith('_')]
    command = args["command"]
    if command not in m:
        logger.error(f"Command {command} not defined")
        print_commands()
        return

    print(getattr(client, command)(*args["command_arguments"]))


if __name__ == "__main__":
    args = add_arguments()
    main(args)
