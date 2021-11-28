"""
This module manages webhook communication
"""
import logging
import time
from typing import Any, Dict

from requests import RequestException, post

from freqtrade.enums import RPCMessageType
from freqtrade.rpc import RPC, RPCHandler


logger = logging.getLogger(__name__)

logger.debug('Included module rpc.webhook ...')


class Webhook(RPCHandler):
    """  This class handles all webhook communication """

    def __init__(self, rpc: RPC, config: Dict[str, Any]) -> None:
        """
        Init the Webhook class, and init the super class RPCHandler
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        super().__init__(rpc, config)

        self._url = self._config['webhook']['url']
        self._format = self._config['webhook'].get('format', 'form')
        self._retries = self._config['webhook'].get('retries', 0)
        self._retry_delay = self._config['webhook'].get('retry_delay', 0.1)

        if self._retries < 0: self._retries = 0
        if self._retry_delay < 0: self._retry_delay = 0

        if not (self._format in ['form', 'json', 'raw']):
            raise NotImplementedError('Unknown webhook format `{}`, possible values are '
                                      '`form` (default), `json`, and `raw`'.format(self._format))

    def cleanup(self) -> None:
        """
        Cleanup pending module resources.
        This will do nothing for webhooks, they will simply not be called anymore
        """
        pass

    def send_msg(self, msg: Dict[str, Any]) -> None:
        """ Send a message to telegram channel """
        try:

            if msg['type'] == RPCMessageType.BUY:
                valuedict = self._config['webhook'].get('webhookbuy', None)
            elif msg['type'] == RPCMessageType.BUY_CANCEL:
                valuedict = self._config['webhook'].get('webhookbuycancel', None)
            elif msg['type'] == RPCMessageType.BUY_FILL:
                valuedict = self._config['webhook'].get('webhookbuyfill', None)
            elif msg['type'] == RPCMessageType.SELL:
                valuedict = self._config['webhook'].get('webhooksell', None)
            elif msg['type'] == RPCMessageType.SELL_FILL:
                valuedict = self._config['webhook'].get('webhooksellfill', None)
            elif msg['type'] == RPCMessageType.SELL_CANCEL:
                valuedict = self._config['webhook'].get('webhooksellcancel', None)
            elif msg['type'] in (RPCMessageType.STATUS,
                                 RPCMessageType.STARTUP,
                                 RPCMessageType.WARNING):
                valuedict = self._config['webhook'].get('webhookstatus', None)
            else:
                raise NotImplementedError('Unknown message type: {}'.format(msg['type']))
            if not valuedict:
                logger.info("Message type '%s' not configured for webhooks", msg['type'])
                return

            payload = {key: value.format(**msg) for (key, value) in valuedict.items()}
            self._send_msg(payload)
        except KeyError as exc:
            logger.exception("Problem calling Webhook. Please check your webhook configuration. "
                             "Exception: %s", exc)

    def _send_msg(self, payload: dict) -> None:
        """do the actual call to the webhook"""

        success = False
        attempts = 0
        while not success and attempts <= self._retries:
            if attempts:
                if self._retry_delay: time.sleep(self._retry_delay)
                logger.info("Retrying webhook...")

            attempts += 1

            try:
                if self._format == 'form':
                    response = post(self._url, data=payload)
                elif self._format == 'json':
                    response = post(self._url, json=payload)
                elif self._format == 'raw':
                    response = post(self._url, data=payload['data'], headers={'Content-Type': 'text/plain'})
                else:
                    raise NotImplementedError('Unknown format: {}'.format(self._format))
            
                # Throw a RequestException if the post was not successful
                response.raise_for_status()
                success = True

            except RequestException as exc:
                logger.warning("Could not call webhook url. Exception: %s", exc)
