from copy import deepcopy
from pathlib import Path

import pytest

from freqtrade.constants import Config
from freqtrade.exchange.exchange import Exchange
from freqtrade.resolvers.exchange_resolver import ExchangeResolver
from tests.conftest import EXMS, get_default_conf_usdt


EXCHANGE_FIXTURE_TYPE = tuple[Exchange, str]
EXCHANGE_WS_FIXTURE_TYPE = tuple[Exchange, str, str]


# Exchanges that should be tested online
EXCHANGES = {
    "binance": {
        "pair": "BTC/USDT",
        "stake_currency": "USDT",
        "use_ci_proxy": True,
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 1000,
        "futures": True,
        "futures_pair": "BTC/USDT:USDT",
        "hasQuoteVolumeFutures": True,
        "leverage_tiers_public": False,
        "leverage_in_spot_market": False,
        "trades_lookback_hours": 4,
        "private_methods": [
            "fapiPrivateGetPositionSideDual",
            "fapiPrivateGetMultiAssetsMargin",
            "sapi_get_spot_delist_schedule",
        ],
        "sample_order": [
            {
                "exchange_response": {
                    "symbol": "SOLUSDT",
                    "orderId": 3551312894,
                    "orderListId": -1,
                    "clientOrderId": "x-R4DD3S8297c73a11ccb9dc8f2811ba",
                    "transactTime": 1674493798550,
                    "price": "15.50000000",
                    "origQty": "1.10000000",
                    "executedQty": "0.00000000",
                    "cummulativeQuoteQty": "0.00000000",
                    "status": "NEW",
                    "timeInForce": "GTC",
                    "type": "LIMIT",
                    "side": "BUY",
                    "workingTime": 1674493798550,
                    "fills": [],
                    "selfTradePreventionMode": "NONE",
                },
                "pair": "SOL/USDT",
                "expected": {
                    "symbol": "SOL/USDT",
                    "orderId": "3551312894",
                    "timestamp": 1674493798550,
                    "datetime": "2023-03-25T15:49:58.550Z",
                    "price": 15.5,
                    "status": "open",
                    "amount": 1.1,
                },
            },
            {
                "exchange_response": {
                    "symbol": "SOLUSDT",
                    "orderId": 3551312894,
                    "orderListId": -1,
                    "clientOrderId": "x-R4DD3S8297c73a11ccb9dc8f2811ba",
                    "transactTime": 1674493798550,
                    "price": "15.50000000",
                    "origQty": "1.10000000",
                    "executedQty": "1.10000000",
                    "cummulativeQuoteQty": "17.05",
                    "status": "FILLED",
                    "timeInForce": "GTC",
                    "type": "LIMIT",
                    "side": "BUY",
                    "workingTime": 1674493798550,
                    "fills": [],
                    "selfTradePreventionMode": "NONE",
                },
                "pair": "SOL/USDT",
                "expected": {
                    "symbol": "SOL/USDT",
                    "orderId": "3551312894",
                    "timestamp": 1674493798550,
                    "datetime": "2023-03-25T15:49:58.550Z",
                    "price": 15.5,
                    "status": "open",
                    "amount": 1.1,
                },
            },
        ],
    },
    "binanceus": {
        "pair": "BTC/USDT",
        "stake_currency": "USDT",
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 1000,
        "futures": False,
        "skip_ws_tests": True,
        "sample_order": [
            {
                "exchange_response": {
                    "symbol": "SOLUSDT",
                    "orderId": 3551312894,
                    "orderListId": -1,
                    "clientOrderId": "x-R4DD3S8297c73a11ccb9dc8f2811ba",
                    "transactTime": 1674493798550,
                    "price": "15.50000000",
                    "origQty": "1.10000000",
                    "executedQty": "0.00000000",
                    "cummulativeQuoteQty": "0.00000000",
                    "status": "NEW",
                    "timeInForce": "GTC",
                    "type": "LIMIT",
                    "side": "BUY",
                    "workingTime": 1674493798550,
                    "fills": [],
                    "selfTradePreventionMode": "NONE",
                },
                "pair": "SOL/USDT",
                "expected": {
                    "symbol": "SOL/USDT",
                    "orderId": "3551312894",
                    "timestamp": 1674493798550,
                    "datetime": "2023-03-25T15:49:58.550Z",
                    "price": 15.5,
                    "status": "open",
                    "amount": 1.1,
                },
            }
        ],
    },
    "kraken": {
        "pair": "BTC/USD",
        "stake_currency": "USD",
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 720,
        "leverage_tiers_public": False,
        "leverage_in_spot_market": True,
        "trades_lookback_hours": 12,
        "sample_balances": {
            "exchange_response": {
                "result": {
                    "ADA": {"balance": "0.00000000", "hold_trade": "0.00000000"},
                    "ADA.F": {"balance": "2.00000000", "hold_trade": "0.00000000"},
                    "XBT": {"balance": "0.00060000", "hold_trade": "0.00000000"},
                    "XBT.F": {"balance": "0.00100000", "hold_trade": "0.00000000"},
                    "ZEUR": {"balance": "1000.00000000", "hold_trade": "0.00000000"},
                    "ZUSD": {"balance": "1000.00000000", "hold_trade": "0.00000000"},
                }
            },
            "expected": {
                "ADA": {"free": 0.0, "total": 0.0, "used": 0.0},
                "ADA.F": {"free": 2.0, "total": 2.0, "used": 0.0},
                "BTC": {"free": 0.0006, "total": 0.0006, "used": 0.0},
                # XBT.F should be mapped to BTC.F
                "BTC.F": {"free": 0.001, "total": 0.001, "used": 0.0},
                "EUR": {"free": 1000.0, "total": 1000.0, "used": 0.0},
                "USD": {"free": 1000.0, "total": 1000.0, "used": 0.0},
            },
        },
    },
    "kucoin": {
        "pair": "XRP/USDT",
        "stake_currency": "USDT",
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 1500,
        "leverage_tiers_public": False,
        "leverage_in_spot_market": True,
        "sample_order": [
            {
                "exchange_response": {"id": "63d6742d0adc5570001d2bbf7"},
                "pair": "SOL/USDT",
                "expected": {
                    "symbol": "SOL/USDT",
                    "orderId": "3551312894",
                    "timestamp": 1674493798550,
                    "datetime": "2023-03-25T15:49:58.550Z",
                    "price": 15.5,
                    "status": "open",
                    "amount": 1.1,
                },
            },  # create order
            {
                "exchange_response": {
                    "id": "63d6742d0adc5570001d2bbf7",
                    "symbol": "SOL-USDT",
                    "opType": "DEAL",
                    "type": "limit",
                    "side": "buy",
                    "price": "15.5",
                    "size": "1.1",
                    "funds": "0",
                    "dealFunds": "17.05",
                    "dealSize": "1.1",
                    "fee": "0.000065252",
                    "feeCurrency": "USDT",
                    "stp": "",
                    "stop": "",
                    "stopTriggered": False,
                    "stopPrice": "0",
                    "timeInForce": "GTC",
                    "postOnly": False,
                    "hidden": False,
                    "iceberg": False,
                    "visibleSize": "0",
                    "cancelAfter": 0,
                    "channel": "API",
                    "clientOid": "0a053870-11bf-41e5-be61-b272a4cb62e1",
                    "remark": None,
                    "tags": "partner:ccxt",
                    "isActive": False,
                    "cancelExist": False,
                    "createdAt": 1674493798550,
                    "tradeType": "TRADE",
                },
                "pair": "SOL/USDT",
                "expected": {
                    "symbol": "SOL/USDT",
                    "orderId": "3551312894",
                    "timestamp": 1674493798550,
                    "datetime": "2023-03-25T15:49:58.550Z",
                    "price": 15.5,
                    "status": "open",
                    "amount": 1.1,
                },
            },
        ],
    },
    "gate": {
        "pair": "BTC/USDT",
        "stake_currency": "USDT",
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 1000,
        "futures": True,
        "futures_pair": "BTC/USDT:USDT",
        "hasQuoteVolumeFutures": True,
        "leverage_tiers_public": True,
        "leverage_in_spot_market": True,
        "sample_order": [
            {
                "exchange_response": {
                    "id": "276266139423",
                    "text": "apiv4",
                    "create_time": "1674493798",
                    "update_time": "1674493798",
                    "create_time_ms": "1674493798550",
                    "update_time_ms": "1674493798550",
                    "status": "closed",
                    "currency_pair": "SOL_USDT",
                    "type": "limit",
                    "account": "spot",
                    "side": "buy",
                    "amount": "1.1",
                    "price": "15.5",
                    "time_in_force": "gtc",
                    "iceberg": "0",
                    "left": "0",
                    "fill_price": "17.05",
                    "filled_total": "17.05",
                    "avg_deal_price": "15.5",
                    "fee": "0.0000018",
                    "fee_currency": "SOL",
                    "point_fee": "0",
                    "gt_fee": "0",
                    "gt_maker_fee": "0",
                    "gt_taker_fee": "0.0015",
                    "gt_discount": True,
                    "rebated_fee": "0",
                    "rebated_fee_currency": "USDT",
                },
                "pair": "SOL/USDT",
                "expected": {
                    "symbol": "SOL/USDT",
                    "orderId": "3551312894",
                    "timestamp": 1674493798550,
                    "datetime": "2023-03-25T15:49:58.550Z",
                    "price": 15.5,
                    "status": "open",
                    "amount": 1.1,
                },
            },
            {
                "exchange_response": {
                    # market order
                    "id": "276401180529",
                    "text": "apiv4",
                    "create_time": "1674493798",
                    "update_time": "1674493798",
                    "create_time_ms": "1674493798550",
                    "update_time_ms": "1674493798550",
                    "status": "cancelled",
                    "currency_pair": "SOL_USDT",
                    "type": "market",
                    "account": "spot",
                    "side": "buy",
                    "amount": "17.05",
                    "price": "0",
                    "time_in_force": "ioc",
                    "iceberg": "0",
                    "left": "0.0000000016228",
                    "fill_price": "17.05",
                    "filled_total": "17.05",
                    "avg_deal_price": "15.5",
                    "fee": "0",
                    "fee_currency": "SOL",
                    "point_fee": "0.0199999999967544",
                    "gt_fee": "0",
                    "gt_maker_fee": "0",
                    "gt_taker_fee": "0",
                    "gt_discount": False,
                    "rebated_fee": "0",
                    "rebated_fee_currency": "USDT",
                },
                "pair": "SOL/USDT",
                "expected": {
                    "symbol": "SOL/USDT",
                    "orderId": "3551312894",
                    "timestamp": 1674493798550,
                    "datetime": "2023-03-25T15:49:58.550Z",
                    "price": 15.5,
                    "status": "open",
                    "amount": 1.1,
                },
            },
        ],
        "sample_my_trades": [
            {
                "id": "123412341234",
                "create_time": "167997798",
                "create_time_ms": "167997798825.566200",
                "currency_pair": "SOL_USDT",
                "side": "sell",
                "role": "taker",
                "amount": "0.0115",
                "price": "1712.63",
                "order_id": "1234123412",
                "fee": "0.0",
                "fee_currency": "USDT",
                "point_fee": "0.03939049",
                "gt_fee": "0.0",
                "amend_text": "-",
            }
        ],
    },
    "okx": {
        "pair": "BTC/USDT",
        "stake_currency": "USDT",
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 300,
        "futures": True,
        "futures_pair": "BTC/USDT:USDT",
        "hasQuoteVolumeFutures": False,
        "leverage_tiers_public": True,
        "leverage_in_spot_market": True,
        "private_methods": ["fetch_accounts"],
    },
    "bybit": {
        "pair": "BTC/USDT",
        "stake_currency": "USDT",
        "hasQuoteVolume": True,
        "use_ci_proxy": True,
        "timeframe": "1h",
        "candle_count": 1000,
        "futures_pair": "BTC/USDT:USDT",
        "futures": True,
        "orderbook_max_entries": 50,
        "leverage_tiers_public": True,
        "leverage_in_spot_market": True,
        "sample_order": [
            {
                "exchange_response": {
                    "orderId": "1274754916287346280",
                    "orderLinkId": "1666798627015730",
                    "symbol": "SOLUSDT",
                    "createdTime": "1674493798550",
                    "price": "15.5",
                    "qty": "1.1",
                    "orderType": "Limit",
                    "side": "Buy",
                    "orderStatus": "New",
                    "timeInForce": "GTC",
                    "accountId": "5555555",
                    "execQty": "0",
                    "orderCategory": "0",
                },
                "pair": "SOL/USDT",
                "expected": {
                    "symbol": "SOL/USDT",
                    "orderId": "3551312894",
                    "timestamp": 1674493798550,
                    "datetime": "2023-03-25T15:49:58.550Z",
                    "price": 15.5,
                    "status": "open",
                    "amount": 1.1,
                },
            }
        ],
    },
    "bitmart": {
        "pair": "BTC/USDT",
        "stake_currency": "USDT",
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 200,
        "orderbook_max_entries": 50,
    },
    "bitget": {
        "pair": "BTC/USDT",
        "stake_currency": "USDT",
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 1000,
        "futures": True,
        "futures_pair": "BTC/USDT:USDT",
        "leverage_tiers_public": True,
        "leverage_in_spot_market": True,
    },
    "coinex": {
        "pair": "BTC/USDT",
        "stake_currency": "USDT",
        "hasQuoteVolume": False,
        "timeframe": "1h",
        "candle_count": 1000,
        "orderbook_max_entries": 50,
    },
    "htx": {
        "pair": "ETH/BTC",
        "stake_currency": "BTC",
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 1000,
    },
    "bitvavo": {
        "pair": "BTC/EUR",
        "stake_currency": "EUR",
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 1440,
        "leverage_tiers_public": False,
        "leverage_in_spot_market": False,
    },
    "bingx": {
        "pair": "BTC/USDT",
        "stake_currency": "USDT",
        "hasQuoteVolume": True,
        "timeframe": "1h",
        "candle_count": 1000,
        "futures": False,
        "sample_order": [
            {
                "exchange_response": {
                    "symbol": "SOL-USDT",
                    "orderId": "1762393630149869568",
                    "transactTime": "1674493798550",
                    "price": "15.5",
                    "stopPrice": "0",
                    "origQty": "1.1",
                    "executedQty": "1.1",
                    "cummulativeQuoteQty": "17.05",
                    "status": "FILLED",
                    "type": "LIMIT",
                    "side": "BUY",
                    "clientOrderID": "",
                },
                "pair": "SOL/USDT",
                "expected": {
                    "symbol": "SOL/USDT",
                    "orderId": "3551312894",
                    "timestamp": 1674493798550,
                    "datetime": "2023-03-25T15:49:58.550Z",
                    "price": 15.5,
                    "status": "open",
                    "amount": 1.1,
                },
            },
            {
                "exchange_response": {
                    "symbol": "SOL-USDT",
                    "orderId": "1762393630149869568",
                    "transactTime": "1674493798550",
                    "price": "15.5",
                    "stopPrice": "0",
                    "origQty": "1.1",
                    "executedQty": "1.1",
                    "cummulativeQuoteQty": "17.05",
                    "status": "FILLED",
                    "type": "MARKET",
                    "side": "BUY",
                    "clientOrderID": "",
                },
                "pair": "SOL/USDT",
                "expected": {
                    "symbol": "SOL/USDT",
                    "orderId": "3551312894",
                    "timestamp": 1674493798550,
                    "datetime": "2023-03-25T15:49:58.550Z",
                    "price": 15.5,
                    "status": "open",
                    "amount": 1.1,
                },
            },
        ],
    },
    "hyperliquid": {
        "pair": "BTC/USDC",
        "stake_currency": "USDC",
        "hasQuoteVolume": False,
        "timeframe": "30m",
        "futures": True,
        "candle_count": 5000,
        "orderbook_max_entries": 20,
        "futures_pair": "BTC/USDC:USDC",
        # Assert that HIP3 pairs are fetched as part of load_markets
        "futures_alt_pairs": ["XYZ-NVDA/USDC:USDC", "VNTL-ANTHROPIC/USDH:USDH"],
        "hasQuoteVolumeFutures": True,
        "leverage_tiers_public": False,
        "leverage_in_spot_market": False,
        # TODO: re-enable hyperliquid websocket tests
        "skip_ws_tests": True,
    },
}


@pytest.fixture(scope="class")
def exchange_conf():
    config = get_default_conf_usdt((Path(__file__).parent / "testdata").resolve())
    config["exchange"]["pair_whitelist"] = []
    config["exchange"]["key"] = ""
    config["exchange"]["secret"] = ""
    config["dry_run"] = False
    config["entry_pricing"]["use_order_book"] = True
    config["exit_pricing"]["use_order_book"] = True
    return config


def set_test_proxy(config: Config, use_proxy: bool) -> Config:
    # Set proxy to test in CI.
    import os

    if use_proxy and (proxy := os.environ.get("CI_WEB_PROXY")):
        config1 = deepcopy(config)
        config1["exchange"]["ccxt_config"] = {
            "httpsProxy": proxy,
            "wsProxy": proxy,
        }
        return config1

    return config


def get_exchange(exchange_name, exchange_conf):
    exchange_conf = set_test_proxy(
        exchange_conf, EXCHANGES[exchange_name].get("use_ci_proxy", False)
    )
    exchange_conf["exchange"]["name"] = exchange_name
    exchange_conf["stake_currency"] = EXCHANGES[exchange_name]["stake_currency"]
    exchange = ExchangeResolver.load_exchange(
        exchange_conf, validate=True, load_leverage_tiers=True
    )

    return exchange, exchange_name


def get_futures_exchange(exchange_name, exchange_conf, class_mocker):
    if EXCHANGES[exchange_name].get("futures") is not True:
        pytest.skip(f"Exchange {exchange_name} does not support futures.")
    else:
        exchange_conf = deepcopy(exchange_conf)
        exchange_conf = set_test_proxy(
            exchange_conf, EXCHANGES[exchange_name].get("use_ci_proxy", False)
        )
        exchange_conf["trading_mode"] = "futures"
        exchange_conf["margin_mode"] = "isolated"

        class_mocker.patch("freqtrade.exchange.binance.Binance.fill_leverage_tiers")
        class_mocker.patch(f"{EXMS}.fetch_trading_fees")
        class_mocker.patch(f"{EXMS}.ft_additional_exchange_init")
        class_mocker.patch(f"{EXMS}.load_cached_leverage_tiers", return_value=None)
        class_mocker.patch(f"{EXMS}.cache_leverage_tiers")

        return get_exchange(exchange_name, exchange_conf)


@pytest.fixture(params=EXCHANGES, scope="class")
def exchange(request, exchange_conf, class_mocker):
    class_mocker.patch(f"{EXMS}.ft_additional_exchange_init")
    exchange, name = get_exchange(request.param, exchange_conf)
    yield exchange, name
    exchange.close()


@pytest.fixture(params=EXCHANGES, scope="class")
def exchange_futures(request, exchange_conf, class_mocker):
    exchange, name = get_futures_exchange(request.param, exchange_conf, class_mocker)
    yield exchange, name
    exchange.close()


@pytest.fixture(params=["spot", "futures"], scope="class")
def exchange_mode(request):
    return request.param


@pytest.fixture(params=EXCHANGES, scope="class")
def exchange_ws(request, exchange_conf, exchange_mode, class_mocker):
    class_mocker.patch("freqtrade.exchange.bybit.Bybit.additional_exchange_init")
    exchange_conf["exchange"]["enable_ws"] = True
    exchange_param = EXCHANGES[request.param]
    if exchange_param.get("skip_ws_tests"):
        pytest.skip(f"{request.param} does not support websocket tests.")
    if exchange_mode == "spot":
        exchange, name = get_exchange(request.param, exchange_conf)
        pair = exchange_param["pair"]
    elif exchange_param.get("futures"):
        exchange, name = get_futures_exchange(
            request.param, exchange_conf, class_mocker=class_mocker
        )
        pair = exchange_param["futures_pair"]
    else:
        pytest.skip("Exchange does not support futures.")

    if not exchange._exchange_ws:
        pytest.skip("Exchange does not support watch_ohlcv.")
    yield exchange, name, pair
    exchange.close()
