import logging
import threading
from ipaddress import IPv4Address
from typing import Dict

from flask import Flask, jsonify, request

from freqtrade.rpc.rpc import RPC, RPCException

logger = logging.getLogger(__name__)
app = Flask(__name__)


class ApiServer(RPC):
    """
    This class runs api server and provides rpc.rpc functionality to it

    This class starts a none blocking thread the api server runs within
    """

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
        """We don't push to endpoints at the moment. Look at webhooks for that."""
        pass

    def rest_dump(self, return_value):
        """ Helper function to jsonify object for a webserver """
        return jsonify(return_value)

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
        # TODO: actions should not be GET...
        app.add_url_rule('/start', 'start', view_func=self.start, methods=['GET'])
        app.add_url_rule('/stop', 'stop', view_func=self.stop, methods=['GET'])
        app.add_url_rule('/stopbuy', 'stopbuy', view_func=self.stopbuy, methods=['GET'])
        app.add_url_rule('/reload_conf', 'reload_conf', view_func=self.reload_conf,
                         methods=['GET'])
        app.add_url_rule('/count', 'count', view_func=self.count, methods=['GET'])
        app.add_url_rule('/daily', 'daily', view_func=self.daily, methods=['GET'])
        app.add_url_rule('/profit', 'profit', view_func=self.profit, methods=['GET'])
        app.add_url_rule('/status_table', 'status_table',
                         view_func=self.status_table, methods=['GET'])

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
            logger.info("SECURITY WARNING - Local Rest Server listening to external connections")
            logger.info("SECURITY WARNING - This is insecure please set to your loopback,"
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

    def daily(self):
        """
        Returns the last X days trading stats summary.

        :return: stats
        """
        try:
            timescale = request.args.get('timescale')
            logger.info("LocalRPC - Daily Command Called")
            timescale = int(timescale)

            stats = self._rpc_daily_profit(timescale,
                                           self._config['stake_currency'],
                                           self._config['fiat_display_currency']
                                           )

            return self.rest_dump(stats)
        except RPCException as e:
            logger.exception("API Error querying daily:", e)
            return "Error querying daily"

    def profit(self):
        """
        Handler for /profit.

        Returns a cumulative profit statistics
        :return: stats
        """
        try:
            logger.info("LocalRPC - Profit Command Called")

            stats = self._rpc_trade_statistics(self._config['stake_currency'],
                                               self._config['fiat_display_currency']
                                               )

            return self.rest_dump(stats)
        except RPCException as e:
            logger.exception("API Error calling profit", e)
            return "Error querying closed trades - maybe there are none"

    def status_table(self):
        """
        Handler for /status table.

        Returns the current TradeThread status in table format
        :return: results
        """
        try:
            results = self._rpc_trade_status()
            return self.rest_dump(results)

        except RPCException as e:
            logger.exception("API Error calling status table", e)
            return "Error querying open trades - maybe there are none."

    def start(self):
        """
        Handler for /start.

        Starts TradeThread in bot if stopped.
        """
        msg = self._rpc_start()
        return self.rest_dump(msg)

    def stop(self):
        """
        Handler for /stop.

        Stops TradeThread in bot if running
        """
        msg = self._rpc_stop()
        return self.rest_dump(msg)

    def stopbuy(self):
        """
        Handler for /stopbuy.

        Sets max_open_trades to 0 and gracefully sells all open trades
        """
        msg = self._rpc_stopbuy()
        return self.rest_dump(msg)

    def reload_conf(self):
        """
        Handler for /reload_conf.
        Triggers a config file reload
        """
        msg = self._rpc_reload_conf()
        return self.rest_dump(msg)

    def count(self):
        """
        Handler for /count.
        Returns the number of trades running
        """
        try:
            msg = self._rpc_count()
        except RPCException as e:
            msg = {"status": str(e)}
        return self.rest_dump(msg)
