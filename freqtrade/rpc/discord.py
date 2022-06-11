import logging
from typing import Any, Dict

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.enums import RPCMessageType
from freqtrade.rpc import RPC
from freqtrade.rpc.webhook import Webhook


logger = logging.getLogger(__name__)


class Discord(Webhook):
    def __init__(self, rpc: 'RPC', config: Dict[str, Any]):
        # super().__init__(rpc, config)
        self.rpc = rpc
        self.config = config
        self.strategy = config.get('strategy', '')
        self.timeframe = config.get('timeframe', '')

        self._url = self.config['discord']['webhook_url']
        self._format = 'json'
        self._retries = 1
        self._retry_delay = 0.1

    def cleanup(self) -> None:
        """
        Cleanup pending module resources.
        This will do nothing for webhooks, they will simply not be called anymore
        """
        pass

    def send_msg(self, msg) -> None:
        logger.info(f"Sending discord message: {msg}")

        # TODO: handle other message types
        if msg['type'] == RPCMessageType.EXIT_FILL:
            profit_ratio = msg.get('profit_ratio')
            open_date = msg.get('open_date').strftime(DATETIME_PRINT_FORMAT)
            close_date = msg.get('close_date').strftime(
               DATETIME_PRINT_FORMAT) if msg.get('close_date') else ''

            embeds = [{
                'title': '{} Trade: {}'.format(
                    'Profit' if profit_ratio > 0 else 'Loss',
                    msg.get('pair')),
                'color': (0x00FF00 if profit_ratio > 0 else 0xFF0000),
                'fields': [
                    {'name': 'Trade ID', 'value': msg.get('trade_id'), 'inline': True},
                    {'name': 'Exchange', 'value': msg.get('exchange').capitalize(), 'inline': True},
                    {'name': 'Pair', 'value': msg.get('pair'), 'inline': True},
                    {'name': 'Direction', 'value': 'Short' if msg.get(
                        'is_short') else 'Long', 'inline': True},
                    {'name': 'Open rate', 'value': msg.get('open_rate'), 'inline': True},
                    {'name': 'Close rate', 'value': msg.get('close_rate'), 'inline': True},
                    {'name': 'Amount', 'value': msg.get('amount'), 'inline': True},
                    {'name': 'Open order', 'value': msg.get('open_order_id'), 'inline': True},
                    {'name': 'Open date', 'value': open_date, 'inline': True},
                    {'name': 'Close date', 'value': close_date, 'inline': True},
                    {'name': 'Profit', 'value': msg.get('profit_amount'), 'inline': True},
                    {'name': 'Profitability', 'value': f'{profit_ratio:.2%}', 'inline': True},
                    {'name': 'Stake currency', 'value': msg.get('stake_currency'), 'inline': True},
                    {'name': 'Fiat currency', 'value': msg.get('fiat_display_currency'),
                     'inline': True},
                    {'name': 'Buy Tag', 'value': msg.get('enter_tag'), 'inline': True},
                    {'name': 'Sell Reason', 'value': msg.get('exit_reason'), 'inline': True},
                    {'name': 'Strategy', 'value': self.strategy, 'inline': True},
                    {'name': 'Timeframe', 'value': self.timeframe, 'inline': True},
                ],
            }]

            # convert all value in fields to string for discord
            for embed in embeds:
                for field in embed['fields']:  # type: ignore
                    field['value'] = str(field['value'])

            # Send the message to discord channel
            payload = {'embeds': embeds}
            self._send_msg(payload)
