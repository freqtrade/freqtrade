import threading
import logging
# import json

from flask import Flask, request
# from flask_restful import Resource, Api
from json import dumps
from freqtrade.rpc.rpc import RPC, RPCException
from ipaddress import IPv4Address


logger = logging.getLogger(__name__)
app = Flask(__name__)


class ApiServerSuperWrap(RPC):
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

        self.interval = 1

        thread = threading.Thread(target=self.run, args=(freqtrade,)) # extra comma as ref ! Tuple
        thread.daemon = True                            # Daemonize thread
        thread.start()     # Start the execution


    def run(self, freqtrade):
        """ Method that runs forever """
        self._config = freqtrade.config

        """
        Define the application methods here, called by app.add_url_rule
        each Telegram command should have a like local substitute
        """
        # @app.route("/")
        def hello():
            # For simple rest server testing via browser
            # cmds = 'Try uri:/daily?timescale=7 /profit /balance /status
            #         /status /table /performance /count,
            #         /start /stop /help'

            rest_cmds ='Commands implemented: <br>' \
                       '<a href=/daily?timescale=7>/daily?timescale=7</a>' \
                       '<br>' \
                       '<a href=/stop>/stop</a>' \
                       '<br>' \
                       '<a href=/start>/start</a>'
            return rest_cmds

        def daily():
            try:
                timescale = request.args.get('timescale')
                logger.info("LocalRPC - Daily Command Called")
                timescale = int(timescale)

                stats = self._rpc_daily_profit(timescale,
                                               self._config['stake_currency'],
                                               self._config['fiat_display_currency']
                                               )

                stats = dumps(stats, indent=4, sort_keys=True, default=str)
                return stats
            except RPCException as e:
                return e

        def start():
            """
            Handler for /start.
            Starts TradeThread
            """
            msg = self._rpc_start()
            return msg

        def stop():
            """
            Handler for /stop.
            Stops TradeThread
            """
            msg = self._rpc_stop()
            return msg

        ## defines the url rules available on the api server
        '''
        First two arguments passed are /URL and 'Label'
        Label can be used as a shortcut when refactoring
        '''
        app.add_url_rule('/', 'hello', view_func=hello, methods=['GET'])
        app.add_url_rule('/stop', 'stop', view_func=stop, methods=['GET'])
        app.add_url_rule('/start', 'start', view_func=start, methods=['GET'])
        app.add_url_rule('/daily', 'daily', view_func=daily, methods=['GET'])


        """
        Section to handle configuration and running of the Rest server
        also to check and warn if not bound to a loopback, warn on security risk.
        """
        rest_ip = self._config['api_server']['listen_ip_address']
        rest_port = self._config['api_server']['listen_port']

        if not IPv4Address(rest_ip).is_loopback :
            logger.info("SECURITY WARNING - Local Rest Server listening to external connections")
            logger.info("SECURITY WARNING - This is insecure please set to your loopback, e.g 127.0.0.1 in config.json")

        # Run the Server
        logger.info('Starting Local Rest Server')
        try:
            app.run(host=rest_ip, port=rest_port)
        except:
            logger.exception("Api server failed to start, exception message is:")


