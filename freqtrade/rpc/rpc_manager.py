"""
This module contains class to manage RPC communications (Telegram, Slack, ...)
"""
import logging
from typing import Any, Dict, List

from freqtrade.rpc import RPC, RPCHandler, RPCMessageType


logger = logging.getLogger(__name__)


class RPCManager:
    """
    Class to manage RPC objects (Telegram, Slack, ...)
    """
    def __init__(self, freqtrade) -> None:
        """ Initializes all enabled rpc modules """
        self.registered_modules: List[RPCHandler] = []
        self._rpc = RPC(freqtrade)
        config = freqtrade.config
        # Enable telegram
        if config.get('telegram', {}).get('enabled', False):
            logger.info('Enabling rpc.telegram ...')
            from freqtrade.rpc.telegram import Telegram
            self.registered_modules.append(Telegram(self._rpc, config))

        # Enable Webhook
        if config.get('webhook', {}).get('enabled', False):
            logger.info('Enabling rpc.webhook ...')
            from freqtrade.rpc.webhook import Webhook
            self.registered_modules.append(Webhook(self._rpc, config))

        # Enable local rest api server for cmd line control
        if config.get('api_server', {}).get('enabled', False):
            logger.info('Enabling rpc.api_server')
            from freqtrade.rpc.api_server import ApiServer

            self.registered_modules.append(ApiServer(self._rpc, config))

    def cleanup(self) -> None:
        """ Stops all enabled rpc modules """
        logger.info('Cleaning up rpc modules ...')
        while self.registered_modules:
            mod = self.registered_modules.pop()
            logger.debug('Cleaning up rpc.%s ...', mod.name)
            mod.cleanup()
            del mod

    def send_msg(self, msg: Dict[str, Any]) -> None:
        """
        Send given message to all registered rpc modules.
        A message consists of one or more key value pairs of strings.
        e.g.:
        {
            'status': 'stopping bot'
        }
        """
        logger.info('Sending rpc message: %s', msg)
        for mod in self.registered_modules:
            logger.debug('Forwarding message to rpc.%s', mod.name)
            try:
                mod.send_msg(msg)
            except NotImplementedError:
                logger.error(f"Message type '{msg['type']}' not implemented by handler {mod.name}.")

    def startup_messages(self, config: Dict[str, Any], pairlist, protections) -> None:
        if config['dry_run']:
            self.send_msg({
                'type': RPCMessageType.WARNING,
                'status': 'Dry run is enabled. All trades are simulated.'
            })
        stake_currency = config['stake_currency']
        stake_amount = config['stake_amount']
        minimal_roi = config['minimal_roi']
        stoploss = config['stoploss']
        trailing_stop = config['trailing_stop']
        timeframe = config['timeframe']
        exchange_name = config['exchange']['name']
        strategy_name = config.get('strategy', '')
        self.send_msg({
            'type': RPCMessageType.STARTUP,
            'status': f'*Exchange:* `{exchange_name}`\n'
                      f'*Stake per trade:* `{stake_amount} {stake_currency}`\n'
                      f'*Minimum ROI:* `{minimal_roi}`\n'
                      f'*{"Trailing " if trailing_stop else ""}Stoploss:* `{stoploss}`\n'
                      f'*Timeframe:* `{timeframe}`\n'
                      f'*Strategy:* `{strategy_name}`'
        })
        self.send_msg({
            'type': RPCMessageType.STARTUP,
            'status': f'Searching for {stake_currency} pairs to buy and sell '
                      f'based on {pairlist.short_desc()}'
        })
        if len(protections.name_list) > 0:
            prots = '\n'.join([p for prot in protections.short_desc() for k, p in prot.items()])
            self.send_msg({
                'type': RPCMessageType.STARTUP,
                'status': f'Using Protections: \n{prots}'
            })
