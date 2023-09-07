#!/usr/bin/env python3
"""
Simple command line client into RPC commands
Can be used as an alternate to Telegram

Should not import anything from freqtrade,
so it can be used as a standalone script.
"""

import argparse
import inspect
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse, urlunparse

import rapidjson
import requests
from requests.exceptions import ConnectionError


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ft_rest_client")


class FtRestClient:

    def __init__(self, serverurl, username=None, password=None):

        self._serverurl = serverurl
        self._session = requests.Session()
        self._session.auth = (username, password)

    def _call(self, method, apipath, params: Optional[dict] = None, data=None, files=None):

        if str(method).upper() not in ('GET', 'POST', 'PUT', 'DELETE'):
            raise ValueError(f'invalid method <{method}>')
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

    def _get(self, apipath, params: Optional[dict] = None):
        return self._call("GET", apipath, params=params)

    def _delete(self, apipath, params: Optional[dict] = None):
        return self._call("DELETE", apipath, params=params)

    def _post(self, apipath, params: Optional[dict] = None, data: Optional[dict] = None):
        return self._call("POST", apipath, params=params, data=data)

    def start(self):
        """Start the bot if it's in the stopped state.

        :return: json object
        """
        return self._post("start")

    def stop(self):
        """Stop the bot. Use `start` to restart.

        :return: json object
        """
        return self._post("stop")

    def stopbuy(self):
        """Stop buying (but handle sells gracefully). Use `reload_config` to reset.

        :return: json object
        """
        return self._post("stopbuy")

    def reload_config(self):
        """Reload configuration.

        :return: json object
        """
        return self._post("reload_config")

    def balance(self):
        """Get the account balance.

        :return: json object
        """
        return self._get("balance")

    def count(self):
        """Return the amount of open trades.

        :return: json object
        """
        return self._get("count")

    def locks(self):
        """Return current locks

        :return: json object
        """
        return self._get("locks")

    def delete_lock(self, lock_id):
        """Delete (disable) lock from the database.

        :param lock_id: ID for the lock to delete
        :return: json object
        """
        return self._delete(f"locks/{lock_id}")

    def daily(self, days=None):
        """Return the profits for each day, and amount of trades.

        :return: json object
        """
        return self._get("daily", params={"timescale": days} if days else None)

    def weekly(self, weeks=None):
        """Return the profits for each week, and amount of trades.

        :return: json object
        """
        return self._get("weekly", params={"timescale": weeks} if weeks else None)

    def monthly(self, months=None):
        """Return the profits for each month, and amount of trades.

        :return: json object
        """
        return self._get("monthly", params={"timescale": months} if months else None)

    def edge(self):
        """Return information about edge.

        :return: json object
        """
        return self._get("edge")

    def profit(self):
        """Return the profit summary.

        :return: json object
        """
        return self._get("profit")

    def stats(self):
        """Return the stats report (durations, sell-reasons).

        :return: json object
        """
        return self._get("stats")

    def performance(self):
        """Return the performance of the different coins.

        :return: json object
        """
        return self._get("performance")

    def status(self):
        """Get the status of open trades.

        :return: json object
        """
        return self._get("status")

    def version(self):
        """Return the version of the bot.

        :return: json object containing the version
        """
        return self._get("version")

    def show_config(self):
        """ Returns part of the configuration, relevant for trading operations.
        :return: json object containing the version
        """
        return self._get("show_config")

    def ping(self):
        """simple ping"""
        configstatus = self.show_config()
        if not configstatus:
            return {"status": "not_running"}
        elif configstatus['state'] == "running":
            return {"status": "pong"}
        else:
            return {"status": "not_running"}

    def logs(self, limit=None):
        """Show latest logs.

        :param limit: Limits log messages to the last <limit> logs. No limit to get the entire log.
        :return: json object
        """
        return self._get("logs", params={"limit": limit} if limit else 0)

    def trades(self, limit=None, offset=None):
        """Return trades history, sorted by id

        :param limit: Limits trades to the X last trades. Max 500 trades.
        :param offset: Offset by this amount of trades.
        :return: json object
        """
        params = {}
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        return self._get("trades", params)

    def trade(self, trade_id):
        """Return specific trade

        :param trade_id: Specify which trade to get.
        :return: json object
        """
        return self._get(f"trade/{trade_id}")

    def delete_trade(self, trade_id):
        """Delete trade from the database.
        Tries to close open orders. Requires manual handling of this asset on the exchange.

        :param trade_id: Deletes the trade with this ID from the database.
        :return: json object
        """
        return self._delete(f"trades/{trade_id}")

    def cancel_open_order(self, trade_id):
        """Cancel open order for trade.

        :param trade_id: Cancels open orders for this trade.
        :return: json object
        """
        return self._delete(f"trades/{trade_id}/open-order")

    def whitelist(self):
        """Show the current whitelist.

        :return: json object
        """
        return self._get("whitelist")

    def blacklist(self, *args):
        """Show the current blacklist.

        :param add: List of coins to add (example: "BNB/BTC")
        :return: json object
        """
        if not args:
            return self._get("blacklist")
        else:
            return self._post("blacklist", data={"blacklist": args})

    def forcebuy(self, pair, price=None):
        """Buy an asset.

        :param pair: Pair to buy (ETH/BTC)
        :param price: Optional - price to buy
        :return: json object of the trade
        """
        data = {"pair": pair,
                "price": price
                }
        return self._post("forcebuy", data=data)

    def forceenter(self, pair, side, price=None):
        """Force entering a trade

        :param pair: Pair to buy (ETH/BTC)
        :param side: 'long' or 'short'
        :param price: Optional - price to buy
        :return: json object of the trade
        """
        data = {"pair": pair,
                "side": side,
                }
        if price:
            data['price'] = price
        return self._post("forceenter", data=data)

    def forceexit(self, tradeid, ordertype=None, amount=None):
        """Force-exit a trade.

        :param tradeid: Id of the trade (can be received via status command)
        :param ordertype: Order type to use (must be market or limit)
        :param amount: Amount to sell. Full sell if not given
        :return: json object
        """

        return self._post("forceexit", data={
            "tradeid": tradeid,
            "ordertype": ordertype,
            "amount": amount,
            })

    def strategies(self):
        """Lists available strategies

        :return: json object
        """
        return self._get("strategies")

    def strategy(self, strategy):
        """Get strategy details

        :param strategy: Strategy class name
        :return: json object
        """
        return self._get(f"strategy/{strategy}")

    def pairlists_available(self):
        """Lists available pairlist providers

        :return: json object
        """
        return self._get("pairlists/available")

    def plot_config(self):
        """Return plot configuration if the strategy defines one.

        :return: json object
        """
        return self._get("plot_config")

    def available_pairs(self, timeframe=None, stake_currency=None):
        """Return available pair (backtest data) based on timeframe / stake_currency selection

        :param timeframe: Only pairs with this timeframe available.
        :param stake_currency: Only pairs that include this timeframe
        :return: json object
        """
        return self._get("available_pairs", params={
            "stake_currency": stake_currency if timeframe else '',
            "timeframe": timeframe if timeframe else '',
        })

    def pair_candles(self, pair, timeframe, limit=None):
        """Return live dataframe for <pair><timeframe>.

        :param pair: Pair to get data for
        :param timeframe: Only pairs with this timeframe available.
        :param limit: Limit result to the last n candles.
        :return: json object
        """
        params = {
            "pair": pair,
            "timeframe": timeframe,
        }
        if limit:
            params['limit'] = limit
        return self._get("pair_candles", params=params)

    def pair_history(self, pair, timeframe, strategy, timerange=None, freqaimodel=None):
        """Return historic, analyzed dataframe

        :param pair: Pair to get data for
        :param timeframe: Only pairs with this timeframe available.
        :param strategy: Strategy to analyze and get values for
        :param freqaimodel: FreqAI model to use for analysis
        :param timerange: Timerange to get data for (same format than --timerange endpoints)
        :return: json object
        """
        return self._get("pair_history", params={
            "pair": pair,
            "timeframe": timeframe,
            "strategy": strategy,
            "freqaimodel": freqaimodel,
            "timerange": timerange if timerange else '',
        })

    def sysinfo(self):
        """Provides system information (CPU, RAM usage)

        :return: json object
        """
        return self._get("sysinfo")

    def health(self):
        """Provides a quick health check of the running bot.

        :return: json object
        """
        return self._get("health")


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        help="Positional argument defining the command to execute.",
                        nargs="?"
                        )

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
            config = rapidjson.load(f, parse_mode=rapidjson.PM_COMMENTS |
                                    rapidjson.PM_TRAILING_COMMAS)
        return config
    else:
        logger.warning(f"Could not load config file {file}.")
        sys.exit(1)


def print_commands():
    # Print dynamic help for the different commands using the commands doc-strings
    client = FtRestClient(None)
    print("Possible commands:\n")
    for x, y in inspect.getmembers(client):
        if not x.startswith('_'):
            doc = re.sub(':return:.*', '', getattr(client, x).__doc__, flags=re.MULTILINE).rstrip()
            print(f"{x}\n\t{doc}\n")


def main(args):

    if args.get("show"):
        print_commands()
        sys.exit()

    config = load_config(args['config'])
    url = config.get('api_server', {}).get('listen_ip_address', '127.0.0.1')
    port = config.get('api_server', {}).get('listen_port', '8080')
    username = config.get('api_server', {}).get('username')
    password = config.get('api_server', {}).get('password')

    server_url = f"http://{url}:{port}"
    client = FtRestClient(server_url, username, password)

    m = [x for x, y in inspect.getmembers(client) if not x.startswith('_')]
    command = args["command"]
    if command not in m:
        logger.error(f"Command {command} not defined")
        print_commands()
        return

    print(json.dumps(getattr(client, command)(*args["command_arguments"])))


if __name__ == "__main__":
    args = add_arguments()
    main(args)
