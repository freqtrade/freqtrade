import logging
from typing import List, Optional, Dict

import arrow
from bittrex.bittrex import Bittrex as _Bittrex, API_V2_0, TICKINTERVAL_FIVEMIN, ORDERTYPE_LIMIT, \
    TIMEINEFFECT_GOOD_TIL_CANCELLED

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
    PAIR_DETAIL_METHOD: str = BASE_URL + '/Market/Index'
    # Ticker inveral
    TICKER_INTERVAL: str = TICKINTERVAL_FIVEMIN
    # Sleep time to avoid rate limits, used in the main loop
    SLEEP_TIME: float = 25

    @property
    def sleep_time(self) -> float:
        return self.SLEEP_TIME

    def __init__(self, config: dict) -> None:
        global _API, _EXCHANGE_CONF

        _EXCHANGE_CONF.update(config)
        _API = _Bittrex(
            api_key=_EXCHANGE_CONF['key'],
            api_secret=_EXCHANGE_CONF['secret'],
            api_version=API_V2_0,
        )

    def buy(self, pair: str, rate: float, amount: float) -> str:
        data = _API.trade_buy(
            market=pair.replace('_', '-'),
            order_type=ORDERTYPE_LIMIT,
            quantity=amount,
            rate=rate,
            time_in_effect=TIMEINEFFECT_GOOD_TIL_CANCELLED,
        )
        if not data['success']:
            raise RuntimeError('{}: {}'.format(self.name.upper(), data['message']))
        return data['result']['OrderId']

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

    def get_orderbook(self, pair: str, top_most: Optional[int] = None) -> Dict[str, List[Dict]]:
        data = _API.get_orderbook(pair.replace('_', '-'))
        if not data['success']:
            raise RuntimeError('{}: {}'.format(self.name.upper(), data['message']))
        return {
            'bid': data['result']['buy'][:top_most],
            'ask': data['result']['sell'][:top_most],
        }

    def get_ticker_history(self, pair: str, minimum_date: Optional[arrow.Arrow] = None):
        data = _API.get_candles(pair.replace('_', '-'), self.TICKER_INTERVAL)
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
