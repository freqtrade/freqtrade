import logging
import threading
from ipaddress import IPv4Address
from typing import Dict

from arrow import Arrow
from flask import Flask, jsonify, request
from flask.json import JSONEncoder

from freqtrade.__init__ import __version__
from freqtrade.rpc.rpc import RPC, RPCException

logger = logging.getLogger(__name__)


class ArrowJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, Arrow):
                return obj.for_json()
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return list(iterable)
        return JSONEncoder.default(self, obj)


app = Flask(__name__)
app.json_encoder = ArrowJSONEncoder


class ApiServer(RPC):
    """
    This class runs api server and provides rpc.rpc functionality to it

    This class starts a none blocking thread the api server runs within
    """

    def safe_rpc(func):

        def func_wrapper(self, *args, **kwargs):

            try:
                return func(self, *args, **kwargs)
            except RPCException as e:
                logger.exception("API Error calling %s: %s", func.__name__, e)
                return self.rest_error(f"Error querying {func.__name__}: {e}")

        return func_wrapper

    def __init__(self, freqtrade) -> None:
        """
        Init the api server, and init the super class RPC
        :param freqtrade: Instance of a freqtrade bot
        :return: None
        """
        super().__init__(freqtrade)

        self._config = freqtrade.config

        # Register application handling
        self.register_rest_other()
        self.register_rest_rpc_urls()

        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def cleanup(self) -> None:
        logger.info("Stopping API Server")
        # TODO: Gracefully shutdown - right now it'll fail on /reload_conf
        # since it's not terminated correctly.

    def send_msg(self, msg: Dict[str, str]) -> None:
        """
        We don't push to endpoints at the moment.
        Take a look at webhooks for that functionality.
        """
        pass

    def rest_dump(self, return_value):
        """ Helper function to jsonify object for a webserver """
        return jsonify(return_value)

    def rest_error(self, error_msg):
        return jsonify({"error": error_msg}), 502

    def register_rest_other(self):
        """
        Registers flask app URLs that are not calls to functionality in rpc.rpc.
        :return:
        """
        app.register_error_handler(404, self.page_not_found)
        app.add_url_rule('/', 'hello', view_func=self.hello, methods=['GET'])

    def register_rest_rpc_urls(self):
        """
        Registers flask app URLs that are calls to functonality in rpc.rpc.

        First two arguments passed are /URL and 'Label'
        Label can be used as a shortcut when refactoring
        :return:
        """
        # Actions to control the bot
        app.add_url_rule('/start', 'start', view_func=self._start, methods=['POST'])
        app.add_url_rule('/stop', 'stop', view_func=self._stop, methods=['POST'])
        app.add_url_rule('/stopbuy', 'stopbuy', view_func=self._stopbuy, methods=['POST'])
        app.add_url_rule('/reload_conf', 'reload_conf', view_func=self._reload_conf,
                         methods=['POST'])
        # Info commands
        app.add_url_rule('/balance', 'balance', view_func=self._balance, methods=['GET'])
        app.add_url_rule('/count', 'count', view_func=self._count, methods=['GET'])
        app.add_url_rule('/daily', 'daily', view_func=self._daily, methods=['GET'])
        app.add_url_rule('/edge', 'edge', view_func=self._edge, methods=['GET'])
        app.add_url_rule('/profit', 'profit', view_func=self._profit, methods=['GET'])
        app.add_url_rule('/performance', 'performance', view_func=self._performance,
                         methods=['GET'])
        app.add_url_rule('/status', 'status', view_func=self._status, methods=['GET'])
        app.add_url_rule('/version', 'version', view_func=self._version, methods=['GET'])

        # Combined actions and infos
        app.add_url_rule('/blacklist', 'blacklist', view_func=self._blacklist,
                         methods=['GET', 'POST'])
        app.add_url_rule('/whitelist', 'whitelist', view_func=self._whitelist,
                         methods=['GET'])
        app.add_url_rule('/forcebuy', 'forcebuy', view_func=self._forcebuy, methods=['POST'])
        app.add_url_rule('/forcesell', 'forcesell', view_func=self._forcesell, methods=['POST'])

        # TODO: Implement the following
        # help (?)

    def run(self):
        """ Method that runs flask app in its own thread forever """

        """
        Section to handle configuration and running of the Rest server
        also to check and warn if not bound to a loopback, warn on security risk.
        """
        rest_ip = self._config['api_server']['listen_ip_address']
        rest_port = self._config['api_server']['listen_port']

        logger.info('Starting HTTP Server at {}:{}'.format(rest_ip, rest_port))
        if not IPv4Address(rest_ip).is_loopback:
            logger.warning("SECURITY WARNING - Local Rest Server listening to external connections")
            logger.warning("SECURITY WARNING - This is insecure please set to your loopback,"
                        "e.g 127.0.0.1 in config.json")

        # Run the Server
        logger.info('Starting Local Rest Server')
        try:
            app.run(host=rest_ip, port=rest_port)
        except Exception:
            logger.exception("Api server failed to start, exception message is:")
        logger.info('Starting Local Rest Server_end')

    """
    Define the application methods here, called by app.add_url_rule
    each Telegram command should have a like local substitute
    """

    def page_not_found(self, error):
        """
        Return "404 not found", 404.
        """
        return self.rest_dump({
            'status': 'error',
            'reason': '''There's no API call for %s''' % request.base_url,
            'code': 404
        }), 404

    def hello(self):
        """
        None critical but helpful default index page.

        That lists URLs added to the flask server.
        This may be deprecated at any time.
        :return: index.html
        """
        rest_cmds = ('Commands implemented: <br>'
                     '<a href=/daily?timescale=7>Show 7 days of stats</a><br>'
                     '<a href=/stop>Stop the Trade thread</a><br>'
                     '<a href=/start>Start the Traded thread</a><br>'
                     '<a href=/profit>Show profit summary</a><br>'
                     '<a href=/status_table>Show status table - Open trades</a><br>'
                     '<a href=/paypal> 404 page does not exist</a><br>'
                     )
        return rest_cmds

    def _start(self):
        """
        Handler for /start.
        Starts TradeThread in bot if stopped.
        """
        msg = self._rpc_start()
        return self.rest_dump(msg)

    def _stop(self):
        """
        Handler for /stop.
        Stops TradeThread in bot if running
        """
        msg = self._rpc_stop()
        return self.rest_dump(msg)

    def _stopbuy(self):
        """
        Handler for /stopbuy.
        Sets max_open_trades to 0 and gracefully sells all open trades
        """
        msg = self._rpc_stopbuy()
        return self.rest_dump(msg)

    def _version(self):
        """
        Prints the bot's version
        """
        return self.rest_dump({"version": __version__})

    def _reload_conf(self):
        """
        Handler for /reload_conf.
        Triggers a config file reload
        """
        msg = self._rpc_reload_conf()
        return self.rest_dump(msg)

    @safe_rpc
    def _count(self):
        """
        Handler for /count.
        Returns the number of trades running
        """
        msg = self._rpc_count()
        return self.rest_dump(msg)

    @safe_rpc
    def _daily(self):
        """
        Returns the last X days trading stats summary.

        :return: stats
        """
        timescale = request.args.get('timescale', 7)
        timescale = int(timescale)

        stats = self._rpc_daily_profit(timescale,
                                       self._config['stake_currency'],
                                       self._config['fiat_display_currency']
                                       )

        return self.rest_dump(stats)

    @safe_rpc
    def _edge(self):
        """
        Returns information related to Edge.
        :return: edge stats
        """
        stats = self._rpc_edge()

        return self.rest_dump(stats)

    @safe_rpc
    def _profit(self):
        """
        Handler for /profit.

        Returns a cumulative profit statistics
        :return: stats
        """
        logger.info("LocalRPC - Profit Command Called")

        stats = self._rpc_trade_statistics(self._config['stake_currency'],
                                           self._config['fiat_display_currency']
                                           )

        return self.rest_dump(stats)

    @safe_rpc
    def _performance(self):
        """
        Handler for /performance.

        Returns a cumulative performance statistics
        :return: stats
        """
        logger.info("LocalRPC - performance Command Called")

        stats = self._rpc_performance()

        return self.rest_dump(stats)

    @safe_rpc
    def _status(self):
        """
        Handler for /status.

        Returns the current status of the trades in json format
        """
        results = self._rpc_trade_status()
        return self.rest_dump(results)

    @safe_rpc
    def _balance(self):
        """
        Handler for /balance.

        Returns the current status of the trades in json format
        """
        results = self._rpc_balance(self._config.get('fiat_display_currency', ''))
        return self.rest_dump(results)

    @safe_rpc
    def _whitelist(self):
        """
        Handler for /whitelist.
        """
        results = self._rpc_whitelist()
        return self.rest_dump(results)

    @safe_rpc
    def _blacklist(self):
        """
        Handler for /blacklist.
        """
        add = request.json.get("blacklist", None) if request.method == 'POST' else None
        results = self._rpc_blacklist(add)
        return self.rest_dump(results)

    @safe_rpc
    def _forcebuy(self):
        """
        Handler for /forcebuy.
        """
        asset = request.json.get("pair")
        price = request.json.get("price", None)
        trade = self._rpc_forcebuy(asset, price)
        return self.rest_dump(trade.to_json())

    @safe_rpc
    def _forcesell(self):
        """
        Handler for /forcesell.
        """
        tradeid = request.json.get("tradeid")
        results = self._rpc_forcesell(tradeid)
        return self.rest_dump(results)
