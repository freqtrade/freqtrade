import logging
from typing import List, Dict, Optional

from bittrex.bittrex import Bittrex as _Bittrex, API_V2_0, API_V1_1
from requests.exceptions import ContentDecodingError

from freqtrade import OperationalException
from freqtrade.exchange.interface import Exchange

logger = logging.getLogger(__name__)

_API: _Bittrex = None
_API_V2: _Bittrex = None
_EXCHANGE_CONF: dict = {}


class Bittrex(Exchange):
    """
    Bittrex API wrapper.
    """
    # Base URL and API endpoints
    BASE_URL: str = 'https://www.bittrex.com'
    PAIR_DETAIL_METHOD: str = BASE_URL + '/Market/Index'

    def __init__(self, config: dict) -> None:
        global _API, _API_V2, _EXCHANGE_CONF

        _EXCHANGE_CONF.update(config)
        _API = _Bittrex(
            api_key=_EXCHANGE_CONF['key'],
            api_secret=_EXCHANGE_CONF['secret'],
            calls_per_second=1,
            api_version=API_V1_1,
        )
        _API_V2 = _Bittrex(
            api_key=_EXCHANGE_CONF['key'],
            api_secret=_EXCHANGE_CONF['secret'],
            calls_per_second=1,
            api_version=API_V2_0,
        )
        self.cached_ticker = {}

    @staticmethod
    def _validate_response(response) -> None:
        """
        Validates the given bittrex response
        and raises a ContentDecodingError if a non-fatal issue happened.
        """
        temp_error_messages = [
            'NO_API_RESPONSE',
            'MIN_TRADE_REQUIREMENT_NOT_MET',
        ]
        if response['message'] in temp_error_messages:
            raise ContentDecodingError('Got {}'.format(response['message']))

    @property
    def fee(self) -> float:
        # 0.25 %: See https://bittrex.com/fees
        return 0.0025

    def buy(self, pair: str, rate: float, amount: float) -> str:
        data = _API.buy_limit(pair.replace('_', '-'), amount, rate)
        if not data['success']:
            Bittrex._validate_response(data)
            raise OperationalException('{message} params=({pair}, {rate}, {amount})'.format(
                message=data['message'],
                pair=pair,
                rate=rate,
                amount=amount))
        return data['result']['uuid']

    def sell(self, pair: str, rate: float, amount: float) -> str:
        data = _API.sell_limit(pair.replace('_', '-'), amount, rate)
        if not data['success']:
            Bittrex._validate_response(data)
            raise OperationalException('{message} params=({pair}, {rate}, {amount})'.format(
                message=data['message'],
                pair=pair,
                rate=rate,
                amount=amount))
        return data['result']['uuid']

    def get_balance(self, currency: str) -> float:
        data = _API.get_balance(currency)
        if not data['success']:
            Bittrex._validate_response(data)
            raise OperationalException('{message} params=({currency})'.format(
                message=data['message'],
                currency=currency))
        return float(data['result']['Balance'] or 0.0)

    def get_balances(self):
        data = _API.get_balances()
        if not data['success']:
            Bittrex._validate_response(data)
            raise OperationalException('{message}'.format(message=data['message']))
        return data['result']

    def get_ticker(self, pair: str, refresh: Optional[bool] = True) -> dict:
        data = _API.get_ticker(pair.replace('_', '-'), refresh)
        if refresh or pair not in self.cached_ticker.keys():
            if not data['success']:
                Bittrex._validate_response(data)
                raise OperationalException('{message} params=({pair})'.format(
                    message=data['message'],
                    pair=pair))

            if not data.get('result') \
                    or not data['result'].get('Bid') \
                    or not data['result'].get('Ask') \
                    or not data['result'].get('Last'):
                raise ContentDecodingError('{message} params=({pair})'.format(
                    message='Got invalid response from bittrex',
                    pair=pair))
            # Update the pair
            self.cached_ticker[pair] = {
                'bid': float(data['result']['Bid']),
                'ask': float(data['result']['Ask']),
                'last': float(data['result']['Last']),
            }
        return self.cached_ticker[pair]

    def get_ticker_history(self, pair: str, tick_interval: int) -> List[Dict]:
        if tick_interval == 1:
            interval = 'oneMin'
        elif tick_interval == 5:
            interval = 'fiveMin'
        else:
            raise ValueError('Cannot parse tick_interval: {}'.format(tick_interval))

        data = _API_V2.get_candles(pair.replace('_', '-'), interval)

        # These sanity check are necessary because bittrex cannot keep their API stable.
        if not data.get('result'):
            raise ContentDecodingError('{message} params=({pair})'.format(
                message='Got invalid response from bittrex',
                pair=pair))

        for prop in ['C', 'V', 'O', 'H', 'L', 'T']:
            for tick in data['result']:
                if prop not in tick.keys():
                    raise ContentDecodingError('{message} params=({pair})'.format(
                        message='Required property {} not present in response'.format(prop),
                        pair=pair))

        if not data['success']:
            Bittrex._validate_response(data)
            raise OperationalException('{message} params=({pair})'.format(
                message=data['message'],
                pair=pair))

        return data['result']

    def get_order(self, order_id: str) -> Dict:
        data = _API.get_order(order_id)
        if not data['success']:
            Bittrex._validate_response(data)
            raise OperationalException('{message} params=({order_id})'.format(
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
            Bittrex._validate_response(data)
            raise OperationalException('{message} params=({order_id})'.format(
                message=data['message'],
                order_id=order_id))

    def get_pair_detail_url(self, pair: str) -> str:
        return self.PAIR_DETAIL_METHOD + '?MarketName={}'.format(pair.replace('_', '-'))

    def get_markets(self) -> List[str]:
        data = _API.get_markets()
        if not data['success']:
            Bittrex._validate_response(data)
            raise OperationalException('{message}'.format(message=data['message']))
        return [m['MarketName'].replace('-', '_') for m in data['result']]

    def get_market_summaries(self) -> List[Dict]:
        data = _API.get_market_summaries()
        if not data['success']:
            Bittrex._validate_response(data)
            raise OperationalException('{message}'.format(message=data['message']))
        return data['result']

    def get_wallet_health(self) -> List[Dict]:
        data = _API_V2.get_wallet_health()
        if not data['success']:
            Bittrex._validate_response(data)
            raise OperationalException('{message}'.format(message=data['message']))
        return [{
            'Currency': entry['Health']['Currency'],
            'IsActive': entry['Health']['IsActive'],
            'LastChecked': entry['Health']['LastChecked'],
            'Notice': entry['Currency'].get('Notice'),
        } for entry in data['result']]
