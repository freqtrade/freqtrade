from copy import deepcopy

import pytest

from freqtrade.enums import CandleType, RunMode
from freqtrade.exchange import Xcoin
from freqtrade.exchange.check_exchange import check_exchange
from freqtrade.exchange.xcoin_connector import XCoinClient
from freqtrade.resolvers.exchange_resolver import ExchangeResolver


def _xcoin_config(default_conf: dict, *, dry_run: bool = True) -> dict:
    conf = deepcopy(default_conf)
    conf.update(
        {
            "dry_run": dry_run,
            "stake_currency": "USDT",
            "stake_amount": 25,
            "timeframe": "5m",
            "trading_mode": "spot",
            "margin_mode": "",
        }
    )
    conf["exchange"].update(
        {
            "name": "xcoin",
            "pair_whitelist": ["BTC/USDT"],
            "pair_blacklist": [],
            "xcoin_live_trading_enabled": not dry_run,
        }
    )
    return conf


def _xcoin_response(method, path, params=None, data=None, private=False):
    if path == "/v2/public/symbols":
        return {
            "code": "0",
            "msg": "success",
            "data": [
                {
                    "businessType": "spot",
                    "symbol": "BTC-USDT",
                    "quoteCurrency": "USDT",
                    "baseCurrency": "BTC",
                    "settleCurrency": "USDT",
                    "tickSize": "0.01",
                    "status": "trading",
                    "pricePrecision": "2",
                    "quantityPrecision": "6",
                    "orderParameters": {
                        "minOrderQty": "0.00001",
                        "minOrderAmt": "5",
                        "maxLmtOrderQty": "100",
                        "maxLmtOrderAmt": "1000000",
                    },
                }
            ],
            "ts": "1732193257273",
        }
    if path == "/v1/market/ticker/mini":
        return {
            "code": "0",
            "msg": "success",
            "data": [
                {
                    "symbol": "BTC-USDT",
                    "priceChange": "10",
                    "priceChangePercent": "0.01",
                    "lastPrice": "80000",
                    "fillQty": "1.2",
                    "fillAmount": "96000",
                    "baseCurrency": "BTC",
                }
            ],
            "ts": "1732193257273",
        }
    if path == "/v1/market/depth":
        return {
            "code": "0",
            "msg": "success",
            "data": {
                "bids": [["79999.99", "0.5"]],
                "asks": [["80000.01", "0.4"]],
                "lastUpdateId": "5001",
            },
            "ts": "1732193257273",
        }
    if path == "/v1/market/kline":
        return {
            "code": "0",
            "msg": "success",
            "data": [
                [
                    "5m",
                    "1732193400000",
                    "1732193699999",
                    "80100",
                    "80200",
                    "80300",
                    "80000",
                    "2.0",
                    "160400",
                    "10",
                    "100",
                    "0.001",
                ],
                [
                    "5m",
                    "1732193100000",
                    "1732193399999",
                    "80000",
                    "80100",
                    "80200",
                    "79900",
                    "1.5",
                    "120150",
                    "8",
                    "100",
                    "0.001",
                ],
            ],
            "ts": "1732193700000",
        }

    assert private
    if path == "/v1/account/balance":
        return {
            "code": "0",
            "msg": "success",
            "data": {
                "details": [
                    {
                        "currency": "USDT",
                        "totalBalance": "100",
                        "cashBalance": "80",
                        "frozen": "20",
                    },
                    {
                        "currency": "BTC",
                        "totalBalance": "0.5",
                        "cashBalance": "0.4",
                        "frozen": "0.1",
                    },
                ]
            },
            "ts": "1732193257273",
        }
    if path == "/v2/trade/order":
        assert method == "POST"
        assert data["symbol"] == "BTC-USDT"
        assert data["side"] == "buy"
        assert data["marketUnit"] == "baseCoin"
        if data["orderType"] == "limit":
            assert float(data["price"]) == 80000
            assert data["timeInForce"] == "gtc"
        elif data["orderType"] == "market":
            assert "price" not in data
            assert data["timeInForce"] == "ioc"
        else:
            raise AssertionError(f"Unexpected XCoin order type: {data['orderType']}")
        return {
            "code": "0",
            "msg": "success",
            "data": {"orderId": "1322590062927904769", "clientOrderId": "Client1001"},
            "ts": "1732193257273",
        }
    if path == "/v1/trade/cancelOrder":
        assert method == "POST"
        assert data["orderId"] == "1322590062927904769"
        return {
            "code": "0",
            "msg": "success",
            "data": {"orderId": "1322590062927904769"},
            "ts": "1732193257273",
        }
    if path == "/v2/trade/order/info":
        return {
            "code": "0",
            "msg": "success",
            "data": {
                "businessType": "spot",
                "symbol": "BTC-USDT",
                "orderId": "1322590062927904769",
                "clientOrderId": "Client1001",
                "price": "80000",
                "qty": "0.01",
                "quoteQty": "400",
                "orderType": "limit",
                "side": "buy",
                "totalFillQty": "0.005",
                "avgPrice": "80000",
                "status": "partially_filled",
                "baseFee": "0",
                "quoteFee": "-0.4",
                "timeInForce": "gtc",
                "createTime": "1732193257000",
                "updateTime": "1732193257273",
            },
            "ts": "1732193257273",
        }
    if path == "/v2/trade/openOrders":
        return {
            "code": "0",
            "msg": "success",
            "data": [
                {
                    "businessType": "spot",
                    "symbol": "BTC-USDT",
                    "orderId": "1322590062927904769",
                    "clientOrderId": "Client1001",
                    "price": "80000",
                    "qty": "0.01",
                    "quoteQty": "400",
                    "orderType": "limit",
                    "side": "buy",
                    "totalFillQty": "0.005",
                    "avgPrice": "80000",
                    "status": "partially_filled",
                    "timeInForce": "gtc",
                    "createTime": "1732193257000",
                    "updateTime": "1732193257273",
                }
            ],
            "ts": "1732193257273",
        }
    raise AssertionError(f"Unhandled XCoin mock path: {method} {path}")


def _patch_xcoin_request(mocker):
    calls = []

    def fake_request(self, method, path, *, params=None, data=None, private=False):
        calls.append((method, path, params, data, private))
        return _xcoin_response(method, path, params=params, data=data, private=private)

    mocker.patch.object(XCoinClient, "request", fake_request)
    return calls


def test_check_exchange_accepts_native_xcoin(default_conf):
    conf = _xcoin_config(default_conf)
    conf["runmode"] = RunMode.DRY_RUN
    assert check_exchange(conf)


def test_xcoin_resolver_loads_native_exchange(default_conf, mocker, monkeypatch):
    monkeypatch.setenv("FREQTRADE__EXCHANGE__KEY", "env-key")
    monkeypatch.setenv("FREQTRADE__EXCHANGE__SECRET", "env-secret")
    _patch_xcoin_request(mocker)

    exchange = ExchangeResolver.load_exchange(_xcoin_config(default_conf))

    assert isinstance(exchange, Xcoin)
    assert exchange.name == "XCoin"
    assert exchange.markets["BTC/USDT"]["id"] == "BTC-USDT"
    assert exchange.markets["BTC/USDT"]["limits"]["amount"]["min"] == 0.00001


def test_xcoin_credentials_are_read_from_environment(default_conf, mocker, monkeypatch):
    monkeypatch.setenv("FREQTRADE__EXCHANGE__KEY", "env-key")
    monkeypatch.setenv("FREQTRADE__EXCHANGE__SECRET", "env-secret")
    _patch_xcoin_request(mocker)
    conf = _xcoin_config(default_conf)
    conf["exchange"]["key"] = "config-key"
    conf["exchange"]["secret"] = "config-secret"
    conf["exchange"]["ccxt_config"] = {"apiKey": "ccxt-key", "secret": "ccxt-secret"}

    exchange = ExchangeResolver.load_exchange(conf)

    assert exchange._api.apiKey == "env-key"
    assert exchange._api.secret == "env-secret"


def test_xcoin_public_market_data(default_conf, mocker, monkeypatch):
    monkeypatch.setenv("FREQTRADE__EXCHANGE__KEY", "env-key")
    monkeypatch.setenv("FREQTRADE__EXCHANGE__SECRET", "env-secret")
    _patch_xcoin_request(mocker)
    exchange = ExchangeResolver.load_exchange(_xcoin_config(default_conf))

    ticker = exchange.fetch_ticker("BTC/USDT")
    assert ticker["last"] == 80000
    assert ticker["bid"] == 79999.99
    assert ticker["ask"] == 80000.01

    candles = exchange.refresh_latest_ohlcv(
        [("BTC/USDT", "5m", CandleType.SPOT)], cache=False, drop_incomplete=False
    )
    candle_df = candles[("BTC/USDT", "5m", CandleType.SPOT)]
    assert len(candle_df) == 2
    assert candle_df.iloc[0]["open"] == 80000
    assert candle_df.iloc[1]["close"] == 80200


def test_xcoin_private_account_and_order_methods(default_conf, mocker, monkeypatch):
    monkeypatch.setenv("FREQTRADE__EXCHANGE__KEY", "env-key")
    monkeypatch.setenv("FREQTRADE__EXCHANGE__SECRET", "env-secret")
    _patch_xcoin_request(mocker)
    exchange = ExchangeResolver.load_exchange(_xcoin_config(default_conf, dry_run=False))

    balances = exchange.get_balances()
    assert balances["USDT"] == {"free": 80.0, "used": 20.0, "total": 100.0}

    order = exchange.create_order(
        pair="BTC/USDT",
        ordertype="limit",
        side="buy",
        amount=0.01,
        rate=80000,
        leverage=1,
    )
    assert order["id"] == "1322590062927904769"
    assert order["status"] == "open"
    assert order["type"] == "limit"

    fetched = exchange.fetch_order("1322590062927904769", "BTC/USDT")
    assert fetched["status"] == "open"
    assert fetched["filled"] == 0.005
    assert fetched["remaining"] == 0.005

    canceled = exchange.cancel_order("1322590062927904769", "BTC/USDT")
    assert canceled["status"] == "canceled"

    open_orders = exchange._api.fetch_open_orders("BTC/USDT")
    assert len(open_orders) == 1
    assert open_orders[0]["status"] == "open"


def test_xcoin_private_market_order(default_conf, mocker, monkeypatch):
    monkeypatch.setenv("FREQTRADE__EXCHANGE__KEY", "env-key")
    monkeypatch.setenv("FREQTRADE__EXCHANGE__SECRET", "env-secret")
    calls = _patch_xcoin_request(mocker)
    exchange = ExchangeResolver.load_exchange(_xcoin_config(default_conf, dry_run=False))

    order = exchange.create_order(
        pair="BTC/USDT",
        ordertype="market",
        side="buy",
        amount=0.01,
        rate=80000,
        leverage=1,
    )

    assert order["id"] == "1322590062927904769"
    assert order["status"] == "open"
    assert order["type"] == "market"
    live_order_calls = [call for call in calls if call[1] == "/v2/trade/order"]
    assert live_order_calls[-1][3]["orderType"] == "market"
    assert live_order_calls[-1][3]["timeInForce"] == "ioc"


def test_xcoin_dry_run_order_does_not_call_live_order(default_conf, mocker, monkeypatch):
    monkeypatch.setenv("FREQTRADE__EXCHANGE__KEY", "env-key")
    monkeypatch.setenv("FREQTRADE__EXCHANGE__SECRET", "env-secret")
    calls = _patch_xcoin_request(mocker)
    exchange = ExchangeResolver.load_exchange(_xcoin_config(default_conf))

    order = exchange.create_order(
        pair="BTC/USDT",
        ordertype="limit",
        side="buy",
        amount=0.01,
        rate=70000,
        leverage=1,
    )

    assert order["id"].startswith("dry_run_buy_BTC/USDT")
    assert all(call[1] != "/v2/trade/order" for call in calls)


def test_xcoin_live_requires_explicit_enable(default_conf, mocker, monkeypatch):
    monkeypatch.setenv("FREQTRADE__EXCHANGE__KEY", "env-key")
    monkeypatch.setenv("FREQTRADE__EXCHANGE__SECRET", "env-secret")
    _patch_xcoin_request(mocker)
    conf = _xcoin_config(default_conf, dry_run=False)
    conf["exchange"]["xcoin_live_trading_enabled"] = False

    with pytest.raises(Exception, match="XCoin live trading is disabled"):
        ExchangeResolver.load_exchange(conf)
