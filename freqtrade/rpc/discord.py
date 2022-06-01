import json
import logging
from typing import Dict, Any

import requests

from freqtrade.enums import RPCMessageType
from freqtrade.rpc import RPCHandler, RPC


class Discord(RPCHandler):
    def __init__(self, rpc: 'RPC', config: Dict[str, Any]):
        super().__init__(rpc, config)
        self.logger = logging.getLogger(__name__)
        self.strategy = config.get('strategy', '')
        self.timeframe = config.get('timeframe', '')
        self.config = config

    def send_msg(self, msg: Dict[str, str]) -> None:
        self._send_msg(msg)

    def _send_msg(self, msg):
        """
        msg = {
            'type': (RPCMessageType.EXIT_FILL if fill
                     else RPCMessageType.EXIT),
            'trade_id': trade.id,
            'exchange': trade.exchange.capitalize(),
            'pair': trade.pair,
            'leverage': trade.leverage,
            'direction': 'Short' if trade.is_short else 'Long',
            'gain': gain,
            'limit': profit_rate,
            'order_type': order_type,
            'amount': trade.amount,
            'open_rate': trade.open_rate,
            'close_rate': trade.close_rate,
            'current_rate': current_rate,
            'profit_amount': profit_trade,
            'profit_ratio': profit_ratio,
            'buy_tag': trade.enter_tag,
            'enter_tag': trade.enter_tag,
            'sell_reason': trade.exit_reason,  # Deprecated
            'exit_reason': trade.exit_reason,
            'open_date': trade.open_date,
            'close_date': trade.close_date or datetime.utcnow(),
            'stake_currency': self.config['stake_currency'],
            'fiat_currency': self.config.get('fiat_display_currency', None),
        }
        """
        self.logger.info(f"Sending discord message: {msg}")

        # TODO: handle other message types
        if msg['type'] == RPCMessageType.EXIT_FILL:
            profit_ratio = msg.get('profit_ratio')
            open_date = msg.get('open_date').strftime('%Y-%m-%d %H:%M:%S')
            close_date = msg.get('close_date').strftime('%Y-%m-%d %H:%M:%S') if msg.get('close_date') else ''

            embeds = [{
                'title': '{} Trade: {}'.format(
                    'Profit' if profit_ratio > 0 else 'Loss',
                    msg.get('pair')),
                'color': (0x00FF00 if profit_ratio > 0 else 0xFF0000),
                'fields': [
                    {'name': 'Trade ID', 'value': msg.get('id'), 'inline': True},
                    {'name': 'Exchange', 'value': msg.get('exchange').capitalize(), 'inline': True},
                    {'name': 'Pair', 'value': msg.get('pair'), 'inline': True},
                    {'name': 'Direction', 'value': 'Short' if msg.get('is_short') else 'Long', 'inline': True},
                    {'name': 'Open rate', 'value': msg.get('open_rate'), 'inline': True},
                    {'name': 'Close rate', 'value': msg.get('close_rate'), 'inline': True},
                    {'name': 'Amount', 'value': msg.get('amount'), 'inline': True},
                    {'name': 'Open order', 'value': msg.get('open_order_id'), 'inline': True},
                    {'name': 'Open date', 'value': open_date, 'inline': True},
                    {'name': 'Close date', 'value': close_date, 'inline': True},
                    {'name': 'Profit', 'value': msg.get('profit_amount'), 'inline': True},
                    {'name': 'Profitability', 'value': '{:.2f}%'.format(profit_ratio * 100), 'inline': True},
                    {'name': 'Stake currency', 'value': msg.get('stake_currency'), 'inline': True},
                    {'name': 'Fiat currency', 'value': msg.get('fiat_display_currency'), 'inline': True},
                    {'name': 'Buy Tag', 'value': msg.get('enter_tag'), 'inline': True},
                    {'name': 'Sell Reason', 'value': msg.get('exit_reason'), 'inline': True},
                    {'name': 'Strategy', 'value': self.strategy, 'inline': True},
                    {'name': 'Timeframe', 'value': self.timeframe, 'inline': True},
                ],
            }]

            # convert all value in fields to string for discord
            for embed in embeds:
                for field in embed['fields']:
                    field['value'] = str(field['value'])

            # Send the message to discord channel
            payload = {
                'embeds': embeds,
            }
            headers = {
                'Content-Type': 'application/json',
            }
            try:
                requests.post(self.config['discord']['webhook_url'], data=json.dumps(payload), headers=headers)
            except Exception as e:
                self.logger.error(f"Failed to send discord message: {e}")
