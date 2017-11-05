import logging
from typing import List, Dict

import requests
from bittrex.bittrex import Bittrex as _Bittrex

from freqtrade.exchange.interface import Exchange

logger = logging.getLogger(__name__)

_API: _Bittrex = None
_EXCHANGE_CONF: dict = {}


class Bittrex(Exchange):
    """
    Bittrex API wrapper.
    """
    # Base URL and API endpoints
    BASE_URL: str = 'https://www.bittrex.com'
    TICKER_METHOD: str = BASE_URL + '/Api/v2.0/pub/market/GetTicks'
    PAIR_DETAIL_METHOD: str = BASE_URL + '/Market/Index'
    # Ticker inveral
    TICKER_INTERVAL: str = 'fiveMin'
    # Sleep time to avoid rate limits, used in the main loop
    SLEEP_TIME: float = 25

    @property
    def sleep_time(self) -> float:
        return self.SLEEP_TIME

    def __init__(self, config: dict) -> None:
        global _API, _EXCHANGE_CONF

        _EXCHANGE_CONF.update(config)
        _API = _Bittrex(api_key=_EXCHANGE_CONF['key'], api_secret=_EXCHANGE_CONF['secret'])

    @property
    def fee(self) -> float:
        # See https://bittrex.com/fees
        return 0.0025

    def buy(self, pair: str, rate: float, amount: float) -> str:
        data = _API.buy_limit(pair.replace('_', '-'), amount, rate)
        if not data['success']:
            raise RuntimeError('{message} params=({pair}, {rate}, {amount})'.format(
                message=data['message'],
                pair=pair,
                rate=rate,
                amount=amount))
        return data['result']['uuid']

    def sell(self, pair: str, rate: float, amount: float) -> str:
        data = _API.sell_limit(pair.replace('_', '-'), amount, rate)
        if not data['success']:
            raise RuntimeError('{message} params=({pair}, {rate}, {amount})'.format(
                message=data['message'],
                pair=pair,
                rate=rate,
                amount=amount))
        return data['result']['uuid']

    def get_balance(self, currency: str) -> float:
        data = _API.get_balance(currency)
        if not data['success']:
            raise RuntimeError('{message} params=({currency})'.format(
                message=data['message'],
                currency=currency))
        return float(data['result']['Balance'] or 0.0)

    def get_balances(self):
        data = _API.get_balances()
        if not data['success']:
            raise RuntimeError('{message}'.format(message=data['message']))
        return data['result']

    def get_ticker(self, pair: str) -> dict:
        data = _API.get_ticker(pair.replace('_', '-'))
        if not data['success']:
            raise RuntimeError('{message} params=({pair})'.format(
                message=data['message'],
                pair=pair))
        return {
            'bid': float(data['result']['Bid']),
            'ask': float(data['result']['Ask']),
            'last': float(data['result']['Last']),
        }

    def get_ticker_history(self, pair: str):
        url = self.TICKER_METHOD
        headers = {
            # TODO: Set as global setting
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
        }
        params = {
            'marketName': pair.replace('_', '-'),
            'tickInterval': self.TICKER_INTERVAL,
        }
        data = requests.get(url, params=params, headers=headers).json()
        if not data['success']:
            raise RuntimeError('{message} params=({pair})'.format(
                message=data['message'],
                pair=pair))
        return data['result']

    def get_order(self, order_id: str) -> Dict:
        data = _API.get_order(order_id)
        if not data['success']:
            raise RuntimeError('{message} params=({order_id})'.format(
                message=data['message'],
                order_id=order_id))
        data = data['result']
        return {
            'id': data['OrderUuid'],
            'type': data['Type'],
            'pair': data['Exchange'].replace('-', '_'),
            'opened': data['Opened'],
            'rate': data['PricePerUnit'],
            'amount': data['Quantity'],
            'remaining': data['QuantityRemaining'],
            'closed': data['Closed'],
        }

    def cancel_order(self, order_id: str) -> None:
        data = _API.cancel(order_id)
        if not data['success']:
            raise RuntimeError('{message} params=({order_id})'.format(
                message=data['message'],
                order_id=order_id))

    def get_pair_detail_url(self, pair: str) -> str:
        return self.PAIR_DETAIL_METHOD + '?MarketName={}'.format(pair.replace('_', '-'))

    def get_markets(self) -> List[str]:
        data = _API.get_markets()
        if not data['success']:
            raise RuntimeError('{message}'.format(message=data['message']))
        return [m['MarketName'].replace('-', '_') for m in data['result']]
