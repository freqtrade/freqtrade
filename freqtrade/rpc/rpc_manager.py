"""
This module contains class to manage RPC communications (Telegram, Slack, ...)
"""
from typing import Any, List
import logging
import time

from freqtrade.rpc.telegram import Telegram
from freqtrade.rpc.local_rpc_server import LocalRPCSuperWrap


logger = logging.getLogger(__name__)


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

        self.registered_modules: List[str] = []
        self.telegram: Any = None
        self._init()

    def _init(self) -> None:
        """
        Init RPC modules
        :return:
        """
        if self.freqtrade.config['telegram'].get('enabled', False):
            logger.info('Enabling rpc.telegram ...')
            self.registered_modules.append('telegram')
            self.telegram = Telegram(self.freqtrade)

        # Added another RPC client - for cmdline local client.
        # Uses existing superclass RPC build for Telegram
        if self.freqtrade.config['localrpc'].get('enabled', False):
            self.localRPC = LocalRPCSuperWrap(self.freqtrade)
            time.sleep(1)

    def cleanup(self) -> None:
        """
        Stops all enabled rpc modules
        :return: None
        """
        if 'telegram' in self.registered_modules:
            logger.info('Cleaning up rpc.telegram ...')
            self.registered_modules.remove('telegram')
            self.telegram.cleanup()

    def send_msg(self, msg: str) -> None:
        """
        Send given markdown message to all registered rpc modules
        :param msg: message
        :return: None
        """
        logger.info(msg)
        if 'telegram' in self.registered_modules:
            self.telegram.send_msg(msg)
