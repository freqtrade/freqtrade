import threading
import time
import zerorpc
import logging
import json

from freqtrade.rpc.rpc import RPC


logger = logging.getLogger(__name__)

class LocalRPCControls(object):
    """
    zeroRPC - allows local cmdline calls to super class in rpc.py
    as used by Telegram.py
    """

    def __init__(self, freqtrade) -> None:
        """
        Initializes all enabled rpc modules
        :param freqtrade: Instance of a freqtrade bot
        :return: None
        """
        self.freqtrade = freqtrade
        self._config = freqtrade.config

    # # Example of calling none serialed call
    # # without decorator - left if as template while in dev for me
    # def add_42(self, n):
    #     """ Add 42 to an integer argument to make it cooler, and return the
    #     result. """
    #     n = int(n)
    #     r = n + 42
    #     s = str(r)
    #     return s

    @zerorpc.stream
    def daily(self, timescale):
        logger.info("LocalRPC - Daily Command Called")
        timescale = int(timescale)

        (error, stats) = RPC.rpc_daily_profit(self, timescale,
                                          self._config['stake_currency'],
                                          self._config['fiat_display_currency']
                                          )

        #Everything in stats to a string, serialised, then back to client.
        stats = json.dumps(stats, indent=4, sort_keys=True, default=str)
        return(error, stats)

class LocalRPCSuperWrap(RPC):
    """
    Class to start thread with ZeroRPC running
    """
    def __init__(self, freqtrade) -> None:
        """
        Init the LocalRPCServer call, and init the super class RPC
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
        while True:
            # Do something
            logger.info('Starting Local RPC Listener')
            s = zerorpc.Server(LocalRPCControls(freqtrade))
            s.bind("tcp://0.0.0.0:4242")
            s.run()
            time.sleep(self.interval)
