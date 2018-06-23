import json
import threading
import logging

from flask import request
from json import dumps
from freqtrade.rpc.rpc import RPC, RPCException
from ipaddress import IPv4Address
from freqtrade.rpc.api_server_common import MyApiApp


logger = logging.getLogger(__name__)
"""
api server routes that do not need access to rpc.rpc
are held within api_server_common.api_server
"""
app = MyApiApp(__name__)


class ApiServer(RPC):
    """
    This class runs api server and provides rpc.rpc functionality to it

    This class starts a none blocking thread the api server runs within
    Any routes that require access to rpc.rpc defs are held within this
    class.

    Any routes that do not require access to rpc.rcp should be registered
    in api_server_common.MyApiApp
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
        self.register_rest_rpc_urls()

        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def register_rest_rpc_urls(self):
        """
        Registers flask app URLs that are calls to functonality in rpc.rpc.

        First two arguments passed are /URL and 'Label'
        Label can be used as a shortcut when refactoring
        :return:
        """
        app.add_url_rule('/stop', 'stop', view_func=self.stop, methods=['GET'])
        app.add_url_rule('/start', 'start', view_func=self.start, methods=['GET'])
        app.add_url_rule('/daily', 'daily', view_func=self.daily, methods=['GET'])

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

    def cleanup(self) -> None:
        pass

    def send_msg(self, msg: str) -> None:
        pass

    """
    Define the application methods here, called by app.add_url_rule
    each Telegram command should have a like local substitute
    """
    def stop_api(self):
        """ For calling shutdown_api_server over via api server HTTP"""
        self.shutdown_api_server()
        return 'Api Server shutting down... '

    def daily(self):
        """
        Returns the last X days trading stats summary.

        :return: stats
        """
        try:
            timescale = request.args.get('timescale')
            timescale = int(timescale)

            stats = self._rpc_daily_profit(timescale,
                                           self._config['stake_currency'],
                                           self._config['fiat_display_currency']
                                           )

            stats = dumps(stats, indent=4, sort_keys=True, default=str)
            return stats
        except RPCException as e:
            return e

    def start(self):
        """
        Handler for /start.

        Starts TradeThread in bot if stopped.
        """
        msg = self._rpc_start()
        return json.dumps(msg)

    def stop(self):
        """
        Handler for /stop.

        Stops TradeThread in bot if running
        """
        msg = self._rpc_stop()
        return json.dumps(msg)
