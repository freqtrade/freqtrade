# pragma pylint: disable=missing-docstring,C0103

import pytest
from unittest.mock import MagicMock
from requests.exceptions import ContentDecodingError

from freqtrade.exchange.bittrex import Bittrex
import freqtrade.exchange.bittrex as btx


# Eat this flake8
# +------------------+
# |  bittrex.Bittrex |
# +------------------+
#         |
#       (mock Fake_bittrex)
#         |
# +-----------------------------+
# |  freqtrade.exchange.Bittrex |
# +-----------------------------+
# Call into Bittrex will flow up to the
# external package bittrex.Bittrex.
# By inserting a mock, we redirect those
# calls.
# The faked bittrex API is called just 'fb'
# The freqtrade.exchange.Bittrex is a
# wrapper, and is called 'wb'


def _stub_config():
    return {'key': '',
            'secret': ''}


class FakeBittrex():
    def __init__(self, success=True):
        self.success = True  # Believe in yourself
        self.result = None
        self.get_ticker_call_count = 0
        # This is really ugly, doing side-effect during instance creation
        # But we're allowed to in testing-code
        btx._API = MagicMock()
        btx._API.buy_limit = self.fake_buysell_limit
        btx._API.sell_limit = self.fake_buysell_limit
        btx._API.get_balance = self.fake_get_balance
        btx._API.get_balances = self.fake_get_balances
        btx._API.get_ticker = self.fake_get_ticker
        btx._API.get_order = self.fake_get_order
        btx._API.cancel = self.fake_cancel_order
        btx._API.get_markets = self.fake_get_markets
        btx._API.get_market_summaries = self.fake_get_market_summaries
        btx._API_V2 = MagicMock()
        btx._API_V2.get_candles = self.fake_get_candles
        btx._API_V2.get_wallet_health = self.fake_get_wallet_health

    def fake_buysell_limit(self, pair, amount, limit):
        return {'success': self.success,
                'result': {'uuid': '1234'},
                'message': 'barter'}

    def fake_get_balance(self, cur):
        return {'success': self.success,
                'result': {'Balance': 1234},
                'message': 'unbalanced'}

    def fake_get_balances(self):
        return {'success': self.success,
                'result': [{'BTC_ETH': 1234}],
                'message': 'no balances'}

    def fake_get_ticker(self, pair):
        self.get_ticker_call_count += 1
        return self.result or {'success': self.success,
                               'result': {'Bid': 1, 'Ask': 1, 'Last': 1},
                               'message': 'NO_API_RESPONSE'}

    def fake_get_candles(self, pair, interval):
        return self.result or {'success': self.success,
                               'result': [{'C': 0, 'V': 0, 'O': 0, 'H': 0, 'L': 0, 'T': 0}],
                               'message': 'candles lit'}

    def fake_get_order(self, uuid):
        return {'success': self.success,
                'result': {'OrderUuid': 'ABC123',
                           'Type': 'Type',
                           'Exchange': 'BTC_ETH',
                           'Opened': True,
                           'PricePerUnit': 1,
                           'Quantity': 1,
                           'QuantityRemaining': 1,
                           'Closed': True
                           },
                'message': 'lost'}

    def fake_cancel_order(self, uuid):
        return self.result or {'success': self.success,
                               'message': 'no such order'}

    def fake_get_markets(self):
        return self.result or {'success': self.success,
                               'message': 'market gone',
                               'result': [{'MarketName': '-_'}]}

    def fake_get_market_summaries(self):
        return self.result or {'success': self.success,
                               'message': 'no summary',
                               'result': ['sum']}

    def fake_get_wallet_health(self):
        return self.result or {'success': self.success,
                               'message': 'bad health',
                               'result': [{'Health': {'Currency': 'BTC_ETH',
                                                      'IsActive': True,
                                                      'LastChecked': 0},
                                           'Currency': {'Notice': True}}]}


# The freqtrade.exchange.bittrex is called wrap_bittrex
# to not confuse naming with bittrex.bittrex
def make_wrap_bittrex():
    conf = _stub_config()
    wb = btx.Bittrex(conf)
    return wb


def test_exchange_bittrex_class():
    conf = _stub_config()
    b = Bittrex(conf)
    assert isinstance(b, Bittrex)
    slots = dir(b)
    for name in ['fee', 'buy', 'sell', 'get_balance', 'get_balances',
                 'get_ticker', 'get_ticker_history', 'get_order',
                 'cancel_order', 'get_pair_detail_url', 'get_markets',
                 'get_market_summaries', 'get_wallet_health']:
        assert name in slots
        # FIX: ensure that the slot is also a method in the class
        # getattr(b, name) => bound method Bittrex.buy
        # type(getattr(b, name)) => class 'method'


def test_exchange_bittrex_fee():
    fee = Bittrex.fee.__get__(Bittrex)
    assert fee >= 0 and fee < 0.1  # Fee is 0-10 %


def test_exchange_bittrex_buy_good(mocker):
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    uuid = wb.buy('BTC_ETH', 1, 1)
    assert uuid == fb.fake_buysell_limit(1, 2, 3)['result']['uuid']

    fb.success = False
    with pytest.raises(btx.OperationalException, match=r'barter.*'):
        wb.buy('BAD', 1, 1)


def test_exchange_bittrex_sell_good(mocker):
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    uuid = wb.sell('BTC_ETH', 1, 1)
    assert uuid == fb.fake_buysell_limit(1, 2, 3)['result']['uuid']

    fb.success = False
    with pytest.raises(btx.OperationalException, match=r'barter.*'):
        uuid = wb.sell('BAD', 1, 1)


def test_exchange_bittrex_get_balance(mocker):
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    bal = wb.get_balance('BTC_ETH')
    assert bal == fb.fake_get_balance(1)['result']['Balance']

    fb.success = False
    with pytest.raises(btx.OperationalException, match=r'unbalanced'):
        wb.get_balance('BTC_ETH')


def test_exchange_bittrex_get_balances():
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    bals = wb.get_balances()
    assert bals == fb.fake_get_balances()['result']

    fb.success = False
    with pytest.raises(btx.OperationalException, match=r'no balances'):
        wb.get_balances()


def test_exchange_bittrex_get_ticker():
    wb = make_wrap_bittrex()
    fb = FakeBittrex()

    # Poll ticker, which updates the cache
    tick = wb.get_ticker('BTC_ETH')
    for x in ['bid', 'ask', 'last']:
        assert x in tick
    # Ensure the side-effect was made (update the ticker cache)
    assert 'BTC_ETH' in wb.cached_ticker.keys()

    # taint the cache, so we can recognize the cache wall utilized
    wb.cached_ticker['BTC_ETH']['bid'] = 1234
    # Poll again, getting the cached result
    fb.get_ticker_call_count = 0
    tick = wb.get_ticker('BTC_ETH', False)
    # Ensure the result was from the cache, and that we didn't call exchange
    assert wb.cached_ticker['BTC_ETH']['bid'] == 1234
    assert fb.get_ticker_call_count == 0


def test_exchange_bittrex_get_ticker_bad():
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    fb.result = {'success': True,
                 'result': {'Bid': 1, 'Ask': 0}}  # incomplete result

    with pytest.raises(ContentDecodingError, match=r'.*Got invalid response from bittrex params.*'):
        wb.get_ticker('BTC_ETH')
    fb.result = {'success': False,
                 'message': 'gone bad'
                 }
    with pytest.raises(btx.OperationalException, match=r'.*gone bad.*'):
        wb.get_ticker('BTC_ETH')

    fb.result = {'success': True,
                 'result': {}}  # incomplete result
    with pytest.raises(ContentDecodingError, match=r'.*Got invalid response from bittrex params.*'):
        wb.get_ticker('BTC_ETH')
    fb.result = {'success': False,
                 'message': 'gone bad'
                 }
    with pytest.raises(btx.OperationalException, match=r'.*gone bad.*'):
        wb.get_ticker('BTC_ETH')

    fb.result = {'success': True,
                 'result': {'Bid': 1, 'Ask': 0, 'Last': None}}  # incomplete result
    with pytest.raises(ContentDecodingError, match=r'.*Got invalid response from bittrex params.*'):
        wb.get_ticker('BTC_ETH')


def test_exchange_bittrex_get_ticker_history_one():
    wb = make_wrap_bittrex()
    FakeBittrex()
    assert wb.get_ticker_history('BTC_ETH', 1)


def test_exchange_bittrex_get_ticker_history():
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    assert wb.get_ticker_history('BTC_ETH', 5)
    with pytest.raises(ValueError, match=r'.*Cannot parse tick_interval.*'):
        wb.get_ticker_history('BTC_ETH', 2)

    fb.success = False
    with pytest.raises(btx.OperationalException, match=r'candles lit.*'):
        wb.get_ticker_history('BTC_ETH', 5)

    fb.success = True
    with pytest.raises(ContentDecodingError, match=r'.*Got invalid response from bittrex.*'):
        fb.result = {'bad': 0}
        wb.get_ticker_history('BTC_ETH', 5)

    with pytest.raises(ContentDecodingError, match=r'.*Required property C not present.*'):
        fb.result = {'success': True,
                     'result': [{'V': 0, 'O': 0, 'H': 0, 'L': 0, 'T': 0}],  # close is missing
                     'message': 'candles lit'}
        wb.get_ticker_history('BTC_ETH', 5)


def test_exchange_bittrex_get_order():
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    order = wb.get_order('someUUID')
    assert order['id'] == 'ABC123'
    fb.success = False
    with pytest.raises(btx.OperationalException, match=r'lost'):
        wb.get_order('someUUID')


def test_exchange_bittrex_cancel_order():
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    wb.cancel_order('someUUID')
    with pytest.raises(btx.OperationalException, match=r'no such order'):
        fb.success = False
        wb.cancel_order('someUUID')
    # Note: this can be a bug in exchange.bittrex._validate_response
    with pytest.raises(KeyError):
        fb.result = {'success': False}  # message is missing!
        wb.cancel_order('someUUID')
    with pytest.raises(btx.OperationalException, match=r'foo'):
        fb.result = {'success': False, 'message': 'foo'}
        wb.cancel_order('someUUID')


def test_exchange_get_pair_detail_url():
    wb = make_wrap_bittrex()
    assert wb.get_pair_detail_url('BTC_ETH')


def test_exchange_get_markets():
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    x = wb.get_markets()
    assert x == ['__']
    with pytest.raises(btx.OperationalException, match=r'market gone'):
        fb.success = False
        wb.get_markets()


def test_exchange_get_market_summaries():
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    assert ['sum'] == wb.get_market_summaries()
    with pytest.raises(btx.OperationalException, match=r'no summary'):
        fb.success = False
        wb.get_market_summaries()


def test_exchange_get_wallet_health():
    wb = make_wrap_bittrex()
    fb = FakeBittrex()
    x = wb.get_wallet_health()
    assert x[0]['Currency'] == 'BTC_ETH'
    with pytest.raises(btx.OperationalException, match=r'bad health'):
        fb.success = False
        wb.get_wallet_health()


def test_validate_response_success():
    response = {
        'message': '',
        'result': [],
    }
    Bittrex._validate_response(response)


def test_validate_response_no_api_response():
    response = {
        'message': 'NO_API_RESPONSE',
        'result': None,
    }
    with pytest.raises(ContentDecodingError, match=r'.*NO_API_RESPONSE.*'):
        Bittrex._validate_response(response)


def test_validate_response_min_trade_requirement_not_met():
    response = {
        'message': 'MIN_TRADE_REQUIREMENT_NOT_MET',
        'result': None,
    }
    with pytest.raises(ContentDecodingError, match=r'.*MIN_TRADE_REQUIREMENT_NOT_MET.*'):
        Bittrex._validate_response(response)


def test_custom_requests(mocker):
    mocker.patch('freqtrade.exchange.bittrex.requests', MagicMock())
    btx.custom_requests('http://', '')
