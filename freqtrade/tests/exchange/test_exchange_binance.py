# pragma pylint: disable=missing-docstring, C0103, protected-access, unused-argument

from unittest.mock import MagicMock
import pytest
import datetime
import dateutil
from freqtrade.exchange.binance import Binance
import freqtrade.exchange.binance as bin


# Eat this flake8
# +------------------+
# |  binance.Binance |
# +------------------+
#         |
#       (mock Fake_binance)
#         |
# +-----------------------------+
# |  freqtrade.exchange.Binance |
# +-----------------------------+
# Call into Binance will flow up to the
# external package binance.Binance.
# By inserting a mock, we redirect those
# calls.
# The faked binance API is called just 'fb'
# The freqtrade.exchange.Binance is a
# wrapper, and is called 'wb'


def _stub_config():
    return {'key': '',
            'secret': ''}


class FakeBinance():
    def __init__(self, success=True):
        self.success = True  # Believe in yourself
        self.result = None
        self.get_ticker_call_count = 0
        # This is really ugly, doing side-effect during instance creation
        # But we're allowed to in testing-code
        bin._API = MagicMock()
        bin._API.order_limit_buy = self.fake_order_limit_buy
        bin._API.order_limit_sell = self.fake_order_limit_sell
        bin._API.get_asset_balance = self.fake_get_asset_balance
        bin._API.get_account = self.fake_get_account
        bin._API.get_ticker = self.fake_get_ticker
        bin._API.get_klines = self.fake_get_klines
        bin._API.get_all_orders = self.fake_get_all_orders
        bin._API.cancel_order = self.fake_cancel_order
        bin._API.get_all_tickers = self.fake_get_all_tickers
        bin._API.get_exchange_info = self.fake_get_exchange_info
        bin._EXCHANGE_CONF = {'stake_currency': 'BTC'}

    def fake_order_limit_buy(self, symbol, quantity, price):
        return {"symbol": "BTCETH",
                "orderId": 42,
                "clientOrderId": "6gCrw2kRUAF9CvJDGP16IP",
                "transactTime": 1507725176595,
                "price": "0.00000000",
                "origQty": "10.00000000",
                "executedQty": "10.00000000",
                "status": "FILLED",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY"}

    def fake_order_limit_sell(self, symbol, quantity, price):
        return {"symbol": "BTCETH",
                "orderId": 42,
                "clientOrderId": "6gCrw2kRUAF9CvJDGP16IP",
                "transactTime": 1507725176595,
                "price": "0.00000000",
                "origQty": "10.00000000",
                "executedQty": "10.00000000",
                "status": "FILLED",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "SELL"}

    def fake_get_asset_balance(self, asset):
        return {
                   "asset": "BTC",
                   "free": "4723846.89208129",
                   "locked": "0.00000000"
               }

    def fake_get_account(self):
        return {
                   "makerCommission": 15,
                   "takerCommission": 15,
                   "buyerCommission": 0,
                   "sellerCommission": 0,
                   "canTrade": True,
                   "canWithdraw": True,
                   "canDeposit": True,
                   "balances": [
                       {
                           "asset": "BTC",
                           "free": "4723846.89208129",
                           "locked": "0.00000000"
                       },
                       {
                           "asset": "LTC",
                           "free": "4763368.68006011",
                           "locked": "0.00000000"
                       }
                   ]
               }

    def fake_get_ticker(self, symbol=None):
        self.get_ticker_call_count += 1
        t = {"symbol": "ETHBTC",
             "priceChange": "-94.99999800",
             "priceChangePercent": "-95.960",
             "weightedAvgPrice": "0.29628482",
             "prevClosePrice": "0.10002000",
             "lastPrice": "4.00000200",
             "bidPrice": "4.00000000",
             "askPrice": "4.00000200",
             "openPrice": "99.00000000",
             "highPrice": "100.00000000",
             "lowPrice": "0.10000000",
             "volume": "8913.30000000",
             "openTime": 1499783499040,
             "closeTime": 1499869899040,
             "fristId": 28385,
             "lastId": 28460,
             "count": 76}
        return t if symbol else [t]

    def fake_get_klines(self, symbol, interval):
        return [[0,
                 "0",
                 "0",
                 "0",
                 "0",
                 "0",
                 0,
                 "0",
                 0,
                 "0",
                 "0",
                 "0"]]

    def fake_get_all_orders(self, symbol, orderId):
        return [{"symbol": "LTCBTC",
                 "orderId": 42,
                 "clientOrderId": "myOrder1",
                 "price": "0.1",
                 "origQty": "1.0",
                 "executedQty": "0.0",
                 "status": "NEW",
                 "timeInForce": "GTC",
                 "type": "LIMIT",
                 "side": "BUY",
                 "stopPrice": "0.0",
                 "icebergQty": "0.0",
                 "status_code": "200",
                 "time": 1499827319559}]

    def fake_cancel_order(self, symbol, orderId):
        return {"symbol": "LTCBTC",
                "origClientOrderId": "myOrder1",
                "orderId": 42,
                "clientOrderId": "cancelMyOrder1"}

    def fake_get_all_tickers(self):
        return [{"symbol": "LTCBTC",
                 "price": "4.00000200"},
                {"symbol": "ETHBTC",
                 "price": "0.07946600"}]

    def fake_get_exchange_info(self):
        return {
                    "timezone": "UTC",
                    "serverTime": 1508631584636,
                    "rateLimits": [
                        {
                            "rateLimitType": "REQUESTS",
                            "interval": "MINUTE",
                            "limit": 1200
                        },
                        {
                            "rateLimitType": "ORDERS",
                            "interval": "SECOND",
                            "limit": 10
                        },
                        {
                            "rateLimitType": "ORDERS",
                            "interval": "DAY",
                            "limit": 100000
                        }
                    ],
                    "exchangeFilters": [],
                    "symbols": [
                        {
                            "symbol": "ETHBTC",
                            "status": "TRADING",
                            "baseAsset": "ETH",
                            "baseAssetPrecision": 8,
                            "quoteAsset": "BTC",
                            "quotePrecision": 8,
                            "orderTypes": ["LIMIT", "MARKET"],
                            "icebergAllowed": False,
                            "filters": [
                                {
                                    "filterType": "PRICE_FILTER",
                                    "minPrice": "0.00000100",
                                    "maxPrice": "100000.00000000",
                                    "tickSize": "0.00000100"
                                }, {
                                    "filterType": "LOT_SIZE",
                                    "minQty": "0.00100000",
                                    "maxQty": "100000.00000000",
                                    "stepSize": "0.00100000"
                                }, {
                                    "filterType": "MIN_NOTIONAL",
                                    "minNotional": "0.00100000"
                                }
                            ]
                        }
                    ]
                }


# The freqtrade.exchange.binance is called wrap_binance
# to not confuse naming with binance.binance
def make_wrap_binance():
    conf = _stub_config()
    wb = bin.Binance(conf)
    return wb


def test_exchange_binance_class():
    conf = _stub_config()
    b = Binance(conf)
    assert isinstance(b, Binance)
    slots = dir(b)
    for name in ['fee', 'buy', 'sell', 'get_balance', 'get_balances',
                 'get_ticker', 'get_ticker_history', 'get_order',
                 'cancel_order', 'get_pair_detail_url', 'get_markets',
                 'get_market_summaries', 'get_wallet_health']:
        assert name in slots
        # FIX: ensure that the slot is also a method in the class
        # getattr(b, name) => bound method Binance.buy
        # type(getattr(b, name)) => class 'method'


def test_exchange_binance_fee():
    fee = Binance.fee.__get__(Binance)
    assert fee >= 0 and fee < 0.1  # Fee is 0-10 %


def test_exchange_binance_buy_good():
    wb = make_wrap_binance()
    fb = FakeBinance()
    uuid = wb.buy('BTC_ETH', 1, 1)
    assert uuid == fb.fake_order_limit_buy(1, 2, 3)['orderId']

    with pytest.raises(IndexError, match=r'.*'):
        wb.buy('BAD', 1, 1)


def test_exchange_binance_sell_good():
    wb = make_wrap_binance()
    fb = FakeBinance()
    uuid = wb.sell('BTC_ETH', 1, 1)
    assert uuid == fb.fake_order_limit_sell(1, 2, 3)['orderId']

    with pytest.raises(IndexError, match=r'.*'):
        uuid = wb.sell('BAD', 1, 1)


def test_exchange_binance_get_balance():
    wb = make_wrap_binance()
    fb = FakeBinance()
    bal = wb.get_balance('BTC')
    assert str(bal) == fb.fake_get_asset_balance(1)['free']


def test_exchange_binance_get_balances():
    wb = make_wrap_binance()
    fb = FakeBinance()
    bals = wb.get_balances()
    assert len(bals) <= len(fb.fake_get_account()['balances'])


def test_exchange_binance_get_ticker():
    wb = make_wrap_binance()
    FakeBinance()

    # Poll ticker, which updates the cache
    tick = wb.get_ticker('BTC_ETH')
    for x in ['bid', 'ask', 'last']:
        assert x in tick


def test_exchange_binance_get_ticker_history_intervals():
    wb = make_wrap_binance()
    FakeBinance()
    for tick_interval in [1, 5]:
        h = wb.get_ticker_history('BTC_ETH', tick_interval)
        assert type(dateutil.parser.parse(h[0]['T'])) is datetime.datetime
        del h[0]['T']
        assert [{'O': 0.0, 'H': 0.0,
                  'L': 0.0, 'C': 0.0,
                  'V': 0.0, 'BV': 0.0}] == h


def test_exchange_binance_get_ticker_history():
    wb = make_wrap_binance()
    FakeBinance()
    assert wb.get_ticker_history('BTC_ETH', 5)


def test_exchange_binance_get_order():
    wb = make_wrap_binance()
    FakeBinance()
    order = wb.get_order('42', 'BTC_LTC')
    assert order['id'] == 42


def test_exchange_binance_cancel_order():
    wb = make_wrap_binance()
    FakeBinance()
    assert wb.cancel_order('42', 'BTC_LTC')['orderId'] == 42


def test_exchange_get_pair_detail_url():
    wb = make_wrap_binance()
    FakeBinance()
    assert wb.get_pair_detail_url('BTC_ETH')


def test_exchange_get_markets():
    wb = make_wrap_binance()
    FakeBinance()
    x = wb.get_markets()
    assert len(x) > 0


def test_exchange_get_market_summaries():
    wb = make_wrap_binance()
    FakeBinance()
    assert wb.get_market_summaries()


def test_exchange_get_wallet_health():
    wb = make_wrap_binance()
    FakeBinance()
    x = wb.get_wallet_health()
    assert x[0]['Currency'] == 'ETH'
