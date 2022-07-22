import logging
from typing import Any, Dict

from freqtrade.enums.rpcmessagetype import RPCMessageType
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

        if msg['type'].value in self.config['discord']:

            msg['strategy'] = self.strategy
            msg['timeframe'] = self.timeframe
            fields = self.config['discord'].get(msg['type'].value)
            color = 0x0000FF
            if msg['type'] in (RPCMessageType.EXIT, RPCMessageType.EXIT_FILL):
                profit_ratio = msg.get('profit_ratio')
                color = (0x00FF00 if profit_ratio > 0 else 0xFF0000)

            embeds = [{
                'title': f"Trade: {msg['pair']} {msg['type'].value}",
                'color': color,
                'fields': [],

            }]
            for f in fields:
                for k, v in f.items():
                    v = v.format(**msg)
                    embeds[0]['fields'].append(  # type: ignore
                        {'name': k, 'value': v, 'inline': True})

            # Send the message to discord channel
            payload = {'embeds': embeds}
            self._send_msg(payload)
