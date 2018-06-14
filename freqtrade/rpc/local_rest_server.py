import threading
import logging
import json

from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from freqtrade.rpc.rpc import RPC, RPCException


logger = logging.getLogger(__name__)

class LocalRestSuperWrap(RPC):
    """
    This class is for REST cmd line client
    """
    def __init__(self, freqtrade) -> None:
        """
        Init the LocalRestServer call, and init the super class RPC
        :param freqtrade: Instance of a freqtrade bot
        :return: None
        """
        super().__init__(freqtrade)
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.interval = int(1)

        thread = threading.Thread(target=self.run, args=(freqtrade,)) # extra comma as ref ! Tuple
        thread.daemon = True                            # Daemonize thread
        thread.start()     # Start the execution


    def run(self, freqtrade):
        """ Method that runs forever """
        self._config = freqtrade.config
        app = Flask(__name__)

        """
        Define the application routes here 
        each Telegram command should have a like local substitute 
        """
        @app.route("/")
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

        @app.route('/daily', methods=['GET'])
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

        @app.route('/start', methods=['GET'])
        def start():
            """
            Handler for /start.
            Starts TradeThread
            """
            msg = self._rpc_start()
            print("msg is", msg)
            return msg

        @app.route('/stop', methods=['GET'])
        def stop():
            """
            Handler for /stop.
            Stops TradeThread
            """
            msg = self._rpc_stop()
            print("msg is", msg)
            return msg

        """
        Section to handle configuration and running of the Rest serve
        also to check and warn if not bound to 127.0.0.1 as a security risk.
        """

        rest_ip = self._config['rest_cmd_line']['listen_ip_address']
        rest_port = self._config['rest_cmd_line']['listen_port']

        if rest_ip != "127.0.0.1":
            i=0
            while i < 10:
                logger.info("SECURITY WARNING - Local Rest Server listening to external connections")
                logger.info("SECURITY WARNING - This is insecure please set to 127.0.0.1 in config.json")
                i += 1

        # Run the Server
        logger.info('Starting Local Rest Server')
        app.run(host=rest_ip, port=rest_port)
