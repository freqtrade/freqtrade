import logging
from typing import List, Optional

import arrow
import requests
from bittrex.bittrex import Bittrex as _Bittrex

from freqtrade.exchanges import Exchange

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
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def sleep_time(self) -> float:
        return self.SLEEP_TIME

    def __init__(self, config: dict) -> None:
        global _API, _EXCHANGE_CONF

        _EXCHANGE_CONF.update(config)
        _API = _Bittrex(api_key=_EXCHANGE_CONF['key'], api_secret=_EXCHANGE_CONF['secret'])

        # Check if all pairs are available
        markets = self.get_markets()
        exchange_name = self.name
        for pair in _EXCHANGE_CONF['pair_whitelist']:
            if pair not in markets:
                raise RuntimeError('Pair {} is not available at {}'.format(pair, exchange_name))

    def buy(self, pair: str, rate: float, amount: float) -> str:
        data = _API.buy_limit(pair.replace('_', '-'), amount, rate)
        if not data['success']:
            raise RuntimeError('{}: {}'.format(self.name.upper(), data['message']))
        return data['result']['uuid']

    def sell(self, pair: str, rate: float, amount: float) -> str:
        data = _API.sell_limit(pair.replace('_', '-'), amount, rate)
        if not data['success']:
            raise RuntimeError('{}: {}'.format(self.name.upper(), data['message']))
        return data['result']['uuid']

    def get_balance(self, currency: str) -> float:
        data = _API.get_balance(currency)
        if not data['success']:
            raise RuntimeError('{}: {}'.format(self.name.upper(), data['message']))
        return float(data['result']['Balance'] or 0.0)

    def get_ticker(self, pair: str) -> dict:
        data = _API.get_ticker(pair.replace('_', '-'))
        if not data['success']:
            raise RuntimeError('{}: {}'.format(self.name.upper(), data['message']))
        return {
            'bid': float(data['result']['Bid']),
            'ask': float(data['result']['Ask']),
            'last': float(data['result']['Last']),
        }

    def get_ticker_history(self, pair: str, minimum_date: Optional[arrow.Arrow] = None):
        url = self.TICKER_METHOD
        headers = {
            # TODO: Set as global setting
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
        }
        params = {
            'marketName': pair.replace('_', '-'),
            'tickInterval': self.TICKER_INTERVAL,
            # TODO: Timestamp has no effect on API response
            '_': minimum_date.timestamp * 1000
        }
        data = requests.get(url, params=params, headers=headers).json()
        if not data['success']:
            raise RuntimeError('{}: {}'.format(self.name.upper(), data['message']))
        return data

    def cancel_order(self, order_id: str) -> None:
        data = _API.cancel(order_id)
        if not data['success']:
            raise RuntimeError('{}: {}'.format(self.name.upper(), data['message']))

    def get_open_orders(self, pair: str) -> List[dict]:
        data = _API.get_open_orders(pair.replace('_', '-'))
        if not data['success']:
            raise RuntimeError('{}: {}'.format(self.name.upper(), data['message']))
        return [{
            'id': entry['OrderUuid'],
            'type': entry['OrderType'],
            'opened': entry['Opened'],
            'rate': entry['PricePerUnit'],
            'amount': entry['Quantity'],
            'remaining': entry['QuantityRemaining'],
        } for entry in data['result']]

    def get_pair_detail_url(self, pair: str) -> str:
        return self.PAIR_DETAIL_METHOD + '?MarketName={}'.format(pair.replace('_', '-'))

    def get_markets(self) -> List[str]:
        data = _API.get_markets()
        if not data['success']:
            raise RuntimeError('{}: {}'.format(self.name.upper(), data['message']))
        return [m['MarketName'].replace('-', '_') for m in data['result']]
