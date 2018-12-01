"""
This module contains class to manage RPC communications (Telegram, Slack, ...)
"""
import logging
from typing import List, Dict, Any

from freqtrade.rpc import RPC, RPCMessageType

logger = logging.getLogger(__name__)


class RPCManager(object):
    """
    Class to manage RPC objects (Telegram, Slack, ...)
    """
    def __init__(self, freqtrade) -> None:
        """ Initializes all enabled rpc modules """
        self.registered_modules: List[RPC] = []

        # Enable telegram
        if freqtrade.config['telegram'].get('enabled', False):
            logger.info('Enabling rpc.telegram ...')
            from freqtrade.rpc.telegram import Telegram
            self.registered_modules.append(Telegram(freqtrade))

        # Enable Webhook
        if freqtrade.config.get('webhook', {}).get('enabled', False):
            logger.info('Enabling rpc.webhook ...')
            from freqtrade.rpc.webhook import Webhook
            self.registered_modules.append(Webhook(freqtrade))

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
            mod.send_msg(msg)

    def startup_messages(self, config) -> None:
        if config.get('dry_run', False):
            self.send_msg({
                'type': RPCMessageType.WARNING_NOTIFICATION,
                'status': 'Dry run is enabled. All trades are simulated.'
            })
        stake_currency = config['stake_currency']
        stake_amount = config['stake_amount']
        minimal_roi = config['minimal_roi']
        ticker_interval = config['ticker_interval']
        exchange_name = config['exchange']['name']
        strategy_name = config.get('strategy', '')
        self.send_msg({
            'type': RPCMessageType.CUSTOM_NOTIFICATION,
            'status': f'*Exchange:* `{exchange_name}`\n'
                      f'*Stake per trade:* `{stake_amount} {stake_currency}`\n'
                      f'*Minimum ROI:* `{minimal_roi}`\n'
                      f'*Ticker Interval:* `{ticker_interval}`\n'
                      f'*Strategy:* `{strategy_name}`'
        })
        if config.get('dynamic_whitelist', False):
            top_pairs = 'top volume ' + str(config.get('dynamic_whitelist', 20))
            specific_pairs = ''
        else:
            top_pairs = 'whitelisted'
            specific_pairs = '\n' + ', '.join(config['exchange'].get('pair_whitelist', ''))
        self.send_msg({
            'type': RPCMessageType.STATUS_NOTIFICATION,
            'status': f'Searching for {top_pairs} {stake_currency} pairs to buy and sell...'
                      f'{specific_pairs}'
        })
