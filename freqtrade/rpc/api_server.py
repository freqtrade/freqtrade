import json
import threading
import logging
# import json

from flask import Flask, request, jsonify
# from flask_restful import Resource, Api
from json import dumps
from freqtrade.rpc.rpc import RPC, RPCException
from ipaddress import IPv4Address


logger = logging.getLogger(__name__)
app = Flask(__name__)


class ApiServer(RPC):
    """
    This class is for REST calls across api server
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

    def register_rest_other(self):
        """
        Registers flask app URLs that are not calls to functionality in rpc.rpc.
        :return:
        """
        app.register_error_handler(404, self.page_not_found)
        app.add_url_rule('/', 'hello', view_func=self.hello, methods=['GET'])
        app.add_url_rule('/stop_api', 'stop_api', view_func=self.stop_api, methods=['GET'])

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

    def send_msg(self, msg: str) -> None:
        pass

    def shutdown_api_server(self):
        """
        Stop the running flask application

        Records the shutdown in logger.info
        :return:
        """
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running the Flask Werkzeug Server')
        if func is not None:
            logger.info('Stopping the Local Rest Server')
            func()
            return

    def cleanup(self) -> None:
        """
        Stops the running application server

        Does not stop the thread,this may not be the desired outcome of cleanup. TBC
        :return:
        """
        self.shutdown_api_server()
    # def cleanup(self) -> None:
    #     pass

    """
    Define the application methods here, called by app.add_url_rule
    each Telegram command should have a like local substitute
    """
    def stop_api(self):
        """ For calling shutdown_api_server over via api server HTTP"""
        self.shutdown_api_server()
        return 'Api Server shutting down... '

    def page_not_found(self, error):
        # Return "404 not found", 404.
        return jsonify({'status': 'error',
                        'reason': '''There's no API call for %s''' % request.base_url,
                        'code': 404}), 404

    def hello(self):
        """
        None critical but helpful default index page.

        That lists URLs added to the flask server.
        This may be deprecated at any time.
        :return: index.html
        """
        rest_cmds = 'Commands implemented: <br>' \
                    '<a href=/daily?timescale=7>Show 7 days of stats</a>' \
                    '<br>' \
                    '<a href=/stop>Stop the Trade thread</a>' \
                    '<br>' \
                    '<a href=/start>Start the Traded thread</a>' \
                    '<br>' \
                    '<a href=/paypal> 404 page does not exist</a>' \
                    '<br>' \
                    '<br>' \
                    '<a href=/stop_api>Shut down the api server -  be sure</a>'
        return rest_cmds

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
