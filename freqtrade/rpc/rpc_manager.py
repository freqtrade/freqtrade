"""
This module contains class to manage RPC communications (Telegram, Slack, ...)
"""

from freqtrade.logger import Logger
from freqtrade.rpc.telegram import Telegram


class RPCManager(object):
    """
    Class to manage RPC objects (Telegram, Slack, ...)
    """
    def __init__(self, freqtrade) -> None:
        """
        Initializes all enabled rpc modules
        :param config: config to use
        :return: None
        """
        self.freqtrade = freqtrade

        # Init the logger
        self.logger = Logger(
            name=__name__,
            level=self.freqtrade.config.get('loglevel')
        ).get_logger()

        self.registered_modules = []
        self.telegram = None
        self._init()

    def _init(self):
        """
        Init RPC modules
        :return:
        """
        if self.freqtrade.config['telegram'].get('enabled', False):
            self.logger.info('Enabling rpc.telegram ...')
            self.registered_modules.append('telegram')
            self.telegram = Telegram(self.freqtrade)

    def cleanup(self) -> None:
        """
        Stops all enabled rpc modules
        :return: None
        """
        if 'telegram' in self.registered_modules:
            self.logger.info('Cleaning up rpc.telegram ...')
            self.registered_modules.remove('telegram')
            self.telegram.cleanup()

    def send_msg(self, msg: str) -> None:
        """
        Send given markdown message to all registered rpc modules
        :param msg: message
        :return: None
        """
        self.logger.info(msg)
        if 'telegram' in self.registered_modules:
            self.telegram.send_msg(msg)
