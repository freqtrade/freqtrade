import threading
import logging
import json

from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from freqtrade.rpc.rpc import RPC


logger = logging.getLogger(__name__)


class Daily(Resource):
    # called by http://127.0.0.1:/daily?timescale=7
    # where 7 is the number of days to report back with.

    def __init__(self, freqtrade) -> None:
        """
        Initializes all enabled rpc modules
        :param freqtrade: Instance of a freqtrade bot
        :return: None
        """
        self.freqtrade = freqtrade
        self._config = freqtrade.config


    def get(self):
        timescale = request.args.get('timescale')
        logger.info("LocalRPC - Daily Command Called")
        timescale = int(timescale)

        (error, stats) = RPC.rpc_daily_profit(self, timescale,
                                          self._config['stake_currency'],
                                          self._config['fiat_display_currency']
                                          )
        if error == False:
            stats = dumps(stats, indent=4, sort_keys=True, default=str)
            return stats
        else:
            json.dumps(error)
            return error


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
        thread.start()                                  # Start the execution

    def run(self, freqtrade):
        """ Method that runs forever """
        self._config = freqtrade.config


        # TODO add IP address / port to bind to in config.json and use in below.
        logger.info('Starting Local Rest Server')

        my_freqtrade = freqtrade
        app = Flask(__name__)
        api = Api(app)

        # Our resources for restful apps go here, pass freqtrade object across
        api.add_resource(Daily, '/daily', methods=['GET'],
                         resource_class_kwargs={'freqtrade': my_freqtrade})  # Route for returning daily

        #run the server
        app.run(port='5002')
