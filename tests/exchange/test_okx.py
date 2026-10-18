from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import ccxt
import pytest

from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.exceptions import RetryableOrderError, TemporaryError
from freqtrade.exchange.common import API_RETRY_COUNT
from freqtrade.exchange.exchange import timeframe_to_minutes
from tests.conftest import EXMS, get_markets, get_patched_exchange, log_has
from tests.exchange.test_exchange import ccxt_exceptionhandlers


# Real Okx data (first 3 tiers of fetch_market_leverage_tiers()).
# minSz / maxSz are in contracts - ccxt maps them to minNotional / maxNotional unchanged.
# Tickers and contract sizes were fetched at the same time.
@pytest.fixture
def okx_tickers():
    return {
        "ETH/USDT:USDT": {"symbol": "ETH/USDT:USDT", "last": 2409.27},
        "ADA/USDT:USDT": {"symbol": "ADA/USDT:USDT", "last": 0.2196},
    }


@pytest.fixture
def okx_leverage_tiers():
    return {
        "ETH/USDT:USDT": [
            {
                "tier": 1,
                "symbol": "ETH/USDT:USDT",
                "currency": "USDT",
                "minNotional": 0.0,
                "maxNotional": 5000.0,
                "maintenanceMarginRate": 0.004,
                "maxLeverage": 100.0,
                "info": {
                    "baseMaxLoan": "",
                    "imr": "0.01",
                    "instFamily": "ETH-USDT",
                    "instId": "",
                    "maxLever": "100",
                    "maxSz": "5000",
                    "minSz": "0",
                    "mmr": "0.004",
                    "optMgnFactor": "0",
                    "quoteMaxLoan": "",
                    "tier": "1",
                    "uly": "ETH-USDT",
                },
            },
            {
                "tier": 2,
                "symbol": "ETH/USDT:USDT",
                "currency": "USDT",
                "minNotional": 5000.01,
                "maxNotional": 10000.0,
                "maintenanceMarginRate": 0.005,
                "maxLeverage": 66.66,
                "info": {
                    "baseMaxLoan": "",
                    "imr": "0.015",
                    "instFamily": "ETH-USDT",
                    "instId": "",
                    "maxLever": "66.66",
                    "maxSz": "10000",
                    "minSz": "5000.01",
                    "mmr": "0.005",
                    "optMgnFactor": "0",
                    "quoteMaxLoan": "",
                    "tier": "2",
                    "uly": "ETH-USDT",
                },
            },
            {
                "tier": 3,
                "symbol": "ETH/USDT:USDT",
                "currency": "USDT",
                "minNotional": 10000.01,
                "maxNotional": 25000.0,
                "maintenanceMarginRate": 0.0075,
                "maxLeverage": 50.0,
                "info": {
                    "baseMaxLoan": "",
                    "imr": "0.02",
                    "instFamily": "ETH-USDT",
                    "instId": "",
                    "maxLever": "50",
                    "maxSz": "25000",
                    "minSz": "10000.01",
                    "mmr": "0.0075",
                    "optMgnFactor": "0",
                    "quoteMaxLoan": "",
                    "tier": "3",
                    "uly": "ETH-USDT",
                },
            },
        ],
        "ADA/USDT:USDT": [
            {
                "tier": 1,
                "symbol": "ADA/USDT:USDT",
                "currency": "USDT",
                "minNotional": 0.0,
                "maxNotional": 2200.0,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50.0,
                "info": {
                    "baseMaxLoan": "",
                    "imr": "0.02",
                    "instFamily": "ADA-USDT",
                    "instId": "",
                    "maxLever": "50",
                    "maxSz": "2200",
                    "minSz": "0",
                    "mmr": "0.01",
                    "optMgnFactor": "0",
                    "quoteMaxLoan": "",
                    "tier": "1",
                    "uly": "ADA-USDT",
                },
            },
            {
                "tier": 2,
                "symbol": "ADA/USDT:USDT",
                "currency": "USDT",
                "minNotional": 2200.1,
                "maxNotional": 4500.0,
                "maintenanceMarginRate": 0.015,
                "maxLeverage": 40.0,
                "info": {
                    "baseMaxLoan": "",
                    "imr": "0.025",
                    "instFamily": "ADA-USDT",
                    "instId": "",
                    "maxLever": "40",
                    "maxSz": "4500",
                    "minSz": "2200.1",
                    "mmr": "0.015",
                    "optMgnFactor": "0",
                    "quoteMaxLoan": "",
                    "tier": "2",
                    "uly": "ADA-USDT",
                },
            },
            {
                "tier": 3,
                "symbol": "ADA/USDT:USDT",
                "currency": "USDT",
                "minNotional": 4500.1,
                "maxNotional": 10000.0,
                "maintenanceMarginRate": 0.03,
                "maxLeverage": 20.0,
                "info": {
                    "baseMaxLoan": "",
                    "imr": "0.05",
                    "instFamily": "ADA-USDT",
                    "instId": "",
                    "maxLever": "20",
                    "maxSz": "10000",
                    "minSz": "4500.1",
                    "mmr": "0.03",
                    "optMgnFactor": "0",
                    "quoteMaxLoan": "",
                    "tier": "3",
                    "uly": "ADA-USDT",
                },
            },
        ],
    }


@pytest.fixture
def okx_markets():
    markets = get_markets()
    okx_contract_sizes = {
        "ETH/USDT:USDT": 0.1,
        "ADA/USDT:USDT": 100.0,
    }
    for pair, contract_size in okx_contract_sizes.items():
        markets[pair]["contractSize"] = contract_size
    return markets


def test_okx_ohlcv_candle_limit(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    timeframes = ("1m", "5m", "1h")
    start_time = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp() * 1000)

    for timeframe in timeframes:
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE) == 100

        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 100
        one_call = int(
            (
                datetime.now(UTC) - timedelta(minutes=290 * timeframe_to_minutes(timeframe))
            ).timestamp()
            * 1000
        )

        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK, one_call) == 100

        one_call = int(
            (
                datetime.now(UTC) - timedelta(minutes=320 * timeframe_to_minutes(timeframe))
            ).timestamp()
            * 1000
        )
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK, one_call) == 100


def test_get_maintenance_ratio_and_amt_okx(
    default_conf, mocker, okx_tickers, okx_leverage_tiers, okx_markets
):
    api_mock = MagicMock()
    api_mock.fetch_tickers = MagicMock(return_value=okx_tickers)
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["dry_run"] = False
    mocker.patch.multiple(
        "freqtrade.exchange.okx.Okx",
        exchange_has=MagicMock(return_value=True),
        load_leverage_tiers=MagicMock(
            return_value={
                "ETH/USDT:USDT": okx_leverage_tiers["ETH/USDT:USDT"],
                "ADA/USDT:USDT": okx_leverage_tiers["ADA/USDT:USDT"],
            }
        ),
    )

    exchange = get_patched_exchange(
        mocker, default_conf, api_mock, exchange="okx", mock_markets=okx_markets
    )
    # ETH: maxSz * 0.1 ETH * 2409.27 - tiers end at 1204635, 2409270 and 6023175 USDT
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 1_000_000) == (0.004, None)
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 2_000_000) == (0.005, None)
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 5_000_000) == (0.0075, None)

    # ADA: maxSz * 100 ADA * 0.2196 - tiers end at 48312, 98820 and 219600 USDT
    assert exchange.get_maintenance_ratio_and_amt("ADA/USDT:USDT", 1) == (0.01, None)
    assert exchange.get_maintenance_ratio_and_amt("ADA/USDT:USDT", 40_000) == (0.01, None)
    assert exchange.get_maintenance_ratio_and_amt("ADA/USDT:USDT", 50_000) == (0.015, None)
    assert exchange.get_maintenance_ratio_and_amt("ADA/USDT:USDT", 200_000) == (0.03, None)


def test_get_max_leverage_okx_contract_tiers(
    default_conf, mocker, okx_tickers, okx_leverage_tiers, okx_markets
):
    api_mock = MagicMock()
    api_mock.fetch_tickers = MagicMock(return_value=okx_tickers)
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    mocker.patch.multiple(
        "freqtrade.exchange.okx.Okx",
        exchange_has=MagicMock(return_value=True),
        load_leverage_tiers=MagicMock(
            return_value={
                "ETH/USDT:USDT": okx_leverage_tiers["ETH/USDT:USDT"],
                # No ticker available
                "TTT/USDT:USDT": okx_leverage_tiers["ETH/USDT:USDT"],
            }
        ),
    )
    exchange = get_patched_exchange(
        mocker, default_conf, api_mock, exchange="okx", mock_markets=okx_markets
    )
    # Tier 1: 5000 contracts * 0.1 ETH * 2409.27 = 1204635 USDT -> max stake 12046.35 at 100x.
    # Used to fail with "Stake amount 12000 too high", as maxSz was taken as USDT.
    assert exchange.get_max_leverage("ETH/USDT:USDT", 12000) == 100.0
    assert exchange.get_max_leverage("ETH/USDT:USDT", 13000) == 66.66
    # Last tier: 25000 contracts * 0.1 ETH * 2409.27 = 6023175 USDT
    assert exchange.get_max_pair_stake_amount("ETH/USDT:USDT", 2409.27) == pytest.approx(6023175)
    assert "TTT/USDT:USDT" not in exchange._leverage_tiers


def test_get_max_pair_stake_amount_okx(default_conf, mocker, leverage_tiers):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    assert exchange.get_max_pair_stake_amount("BNB/BUSD", 1.0) == float("inf")

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    exchange._leverage_tiers = leverage_tiers

    assert exchange.get_max_pair_stake_amount("XRP/USDT:USDT", 1.0) == 30000000
    assert exchange.get_max_pair_stake_amount("BNB/USDT:USDT", 1.0) == 50000000
    assert exchange.get_max_pair_stake_amount("BTC/USDT:USDT", 1.0) == 1000000000
    assert exchange.get_max_pair_stake_amount("BTC/USDT:USDT", 1.0, 10.0) == 100000000

    assert exchange.get_max_pair_stake_amount("TTT/USDT:USDT", 1.0) == float("inf")  # Not in tiers


@pytest.mark.parametrize(
    "mode,side,reduceonly,result",
    [
        ("net", "buy", False, "net"),
        ("net", "sell", True, "net"),
        ("net", "sell", False, "net"),
        ("net", "buy", True, "net"),
        ("longshort", "buy", False, "long"),
        ("longshort", "sell", True, "long"),
        ("longshort", "sell", False, "short"),
        ("longshort", "buy", True, "short"),
    ],
)
def test__get_posSide(default_conf, mocker, mode, side, reduceonly, result):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    exchange.net_only = mode == "net"
    assert exchange._get_posSide(side, reduceonly) == result


def test_additional_exchange_init_okx(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fetch_accounts = MagicMock(
        return_value=[
            {
                "id": "2555",
                "type": "2",
                "currency": None,
                "info": {
                    "acctLv": "2",
                    "autoLoan": False,
                    "ctIsoMode": "automatic",
                    "greeksType": "PA",
                    "level": "Lv1",
                    "levelTmp": "",
                    "mgnIsoMode": "automatic",
                    "posMode": "long_short_mode",
                    "uid": "2555",
                },
            }
        ]
    )
    default_conf["dry_run"] = False
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx", api_mock=api_mock)
    assert api_mock.fetch_accounts.call_count == 0
    exchange.trading_mode = TradingMode.FUTURES
    # Default to netOnly
    assert exchange.net_only
    exchange.additional_exchange_init()
    assert api_mock.fetch_accounts.call_count == 1
    assert not exchange.net_only

    api_mock.fetch_accounts = MagicMock(
        return_value=[
            {
                "id": "2555",
                "type": "2",
                "currency": None,
                "info": {
                    "acctLv": "2",
                    "autoLoan": False,
                    "ctIsoMode": "automatic",
                    "greeksType": "PA",
                    "level": "Lv1",
                    "levelTmp": "",
                    "mgnIsoMode": "automatic",
                    "posMode": "net_mode",
                    "uid": "2555",
                },
            }
        ]
    )
    exchange.additional_exchange_init()
    assert api_mock.fetch_accounts.call_count == 1
    assert exchange.net_only
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, "okx", "additional_exchange_init", "fetch_accounts"
    )


def test_load_leverage_tiers_okx(
    default_conf,
    mocker,
    markets,
    tmp_path,
    caplog,
    time_machine,
    okx_tickers,
    okx_leverage_tiers,
    okx_markets,
):
    default_conf["datadir"] = tmp_path
    # fd_mock = mocker.patch('freqtrade.exchange.exchange.file_dump_json')
    api_mock = MagicMock()
    type(api_mock).has = PropertyMock(
        return_value={
            "fetchLeverageTiers": False,
            "fetchMarketLeverageTiers": True,
            "fetchTickers": True,
        }
    )
    api_mock.fetch_tickers = MagicMock(return_value=okx_tickers)
    api_mock.fetch_market_leverage_tiers = AsyncMock(
        side_effect=[
            okx_leverage_tiers["ADA/USDT:USDT"],
            TemporaryError("this Failed"),
            okx_leverage_tiers["ETH/USDT:USDT"],
        ]
    )
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["stake_currency"] = "USDT"
    exchange = get_patched_exchange(
        mocker, default_conf, api_mock, exchange="okx", mock_markets=okx_markets
    )
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.ISOLATED
    exchange.markets = markets
    # Initialization of load_leverage_tiers happens as part of exchange init.
    # Tiers are converted from contracts to quote notional (minSz * contractSize * price)
    assert exchange._leverage_tiers == {
        "ADA/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": pytest.approx(48312),
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 50,
                "maintAmt": None,
            },
            {
                "minNotional": pytest.approx(48314.196),
                "maxNotional": pytest.approx(98820),
                "maintenanceMarginRate": 0.015,
                "maxLeverage": 40,
                "maintAmt": None,
            },
            {
                "minNotional": pytest.approx(98822.196),
                "maxNotional": pytest.approx(219600),
                "maintenanceMarginRate": 0.03,
                "maxLeverage": 20,
                "maintAmt": None,
            },
        ],
        "ETH/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": pytest.approx(1204635),
                "maintenanceMarginRate": 0.004,
                "maxLeverage": 100,
                "maintAmt": None,
            },
            {
                "minNotional": pytest.approx(1204637.40927),
                "maxNotional": pytest.approx(2409270),
                "maintenanceMarginRate": 0.005,
                "maxLeverage": 66.66,
                "maintAmt": None,
            },
            {
                "minNotional": pytest.approx(2409272.40927),
                "maxNotional": pytest.approx(6023175),
                "maintenanceMarginRate": 0.0075,
                "maxLeverage": 50,
                "maintAmt": None,
            },
        ],
    }
    filename = (
        default_conf["datadir"] / f"futures/leverage_tiers_{default_conf['stake_currency']}.json"
    )
    assert filename.is_file()

    logmsg = "Cached leverage tiers are outdated. Will update."
    assert not log_has(logmsg, caplog)

    api_mock.fetch_market_leverage_tiers.reset_mock()

    exchange.load_leverage_tiers()
    assert not log_has(logmsg, caplog)

    assert api_mock.fetch_market_leverage_tiers.call_count == 0
    # 2 day passes ...
    time_machine.move_to(datetime.now() + timedelta(weeks=5))
    exchange.load_leverage_tiers()

    assert log_has(logmsg, caplog)


def test__set_leverage_okx(mocker, default_conf):
    api_mock = MagicMock()
    api_mock.set_leverage = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setLeverage": True})
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    exchange._lev_prep("BTC/USDT:USDT", 3.2, "buy")
    assert api_mock.set_leverage.call_count == 1
    # Leverage is rounded to 3.
    assert api_mock.set_leverage.call_args_list[0][1]["leverage"] == 3.2
    assert api_mock.set_leverage.call_args_list[0][1]["symbol"] == "BTC/USDT:USDT"
    assert api_mock.set_leverage.call_args_list[0][1]["params"] == {
        "mgnMode": "isolated",
        "posSide": "net",
    }
    api_mock.set_leverage = MagicMock(side_effect=ccxt.NetworkError())
    exchange._lev_prep("BTC/USDT:USDT", 3.2, "buy")
    assert api_mock.fetch_leverage.call_count == 1

    api_mock.fetch_leverage = MagicMock(side_effect=ccxt.NetworkError())
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "okx",
        "_lev_prep",
        "set_leverage",
        pair="XRP/USDT:USDT",
        leverage=5.0,
        side="buy",
    )


@pytest.mark.usefixtures("init_persistence")
def test_fetch_stoploss_order_okx(default_conf, mocker):
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.common.time.sleep")
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock()

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")

    exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_order.call_count == 1
    assert api_mock.fetch_order.call_args_list[0][0][0] == "1234"
    assert api_mock.fetch_order.call_args_list[0][0][1] == "ETH/BTC"
    assert api_mock.fetch_order.call_args_list[0][1]["params"] == {"stop": True}

    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound)
    api_mock.fetch_open_orders = MagicMock(return_value=[])
    api_mock.fetch_closed_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_orders = MagicMock(creturn_value=[])

    with pytest.raises(RetryableOrderError):
        exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_order.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_open_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_closed_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_canceled_orders.call_count == API_RETRY_COUNT + 1

    api_mock.fetch_order.reset_mock()
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    api_mock.fetch_canceled_orders.reset_mock()

    api_mock.fetch_closed_orders = MagicMock(
        return_value=[{"id": "1234", "status": "closed", "info": {"ordId": "123455"}}]
    )
    mocker.patch(f"{EXMS}.fetch_order", MagicMock(return_value={"id": "123455"}))
    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_order.call_count == 1
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 1
    assert api_mock.fetch_canceled_orders.call_count == 0

    assert resp["id"] == "1234"
    assert resp["id_stop"] == "123455"
    assert resp["type"] == "stoploss"

    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    dro_mock = mocker.patch(f"{EXMS}.fetch_dry_run_order", MagicMock(return_value={"id": "123455"}))

    api_mock.fetch_order.reset_mock()
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    api_mock.fetch_canceled_orders.reset_mock()
    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")

    assert api_mock.fetch_order.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_closed_orders.call_count == 0
    assert api_mock.fetch_canceled_orders.call_count == 0
    assert dro_mock.call_count == 1


def test_fetch_stoploss_order_okx_exceptions(default_conf_usdt, mocker):
    default_conf_usdt["dry_run"] = False
    api_mock = MagicMock()
    ccxt_exceptionhandlers(
        mocker,
        default_conf_usdt,
        api_mock,
        "okx",
        "fetch_stoploss_order",
        "fetch_order",
        retries=API_RETRY_COUNT + 1,
        order_id="12345",
        pair="ETH/USDT",
    )

    # Test 2nd part of the function
    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound())
    api_mock.fetch_closed_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_orders = MagicMock(return_value=[])

    ccxt_exceptionhandlers(
        mocker,
        default_conf_usdt,
        api_mock,
        "okx",
        "fetch_stoploss_order",
        "fetch_open_orders",
        retries=API_RETRY_COUNT + 1,
        order_id="12345",
        pair="ETH/USDT",
    )


@pytest.mark.parametrize(
    "sl1,sl2,sl3,side", [(1501, 1499, 1501, "sell"), (1499, 1501, 1499, "buy")]
)
def test_stoploss_adjust_okx(mocker, default_conf, sl1, sl2, sl3, side):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    order = {
        "type": "stoploss",
        "price": 1500,
        "stopLossPrice": 1500,
    }
    assert exchange.stoploss_adjust(sl1, order, side=side)
    assert not exchange.stoploss_adjust(sl2, order, side=side)


def test_stoploss_cancel_okx(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    co_mock = mocker.patch.object(exchange, "cancel_order", autospec=True)

    exchange.cancel_stoploss_order("1234", "ETH/USDT")
    assert co_mock.call_count == 1
    args, _ = co_mock.call_args
    assert args[0] == "1234"
    assert args[1] == "ETH/USDT"
    assert args[2] == {"stop": True}


@pytest.mark.parametrize(
    "trading_mode,margin_mode,expected",
    [
        (TradingMode.SPOT, MarginMode.NONE, {"stopLossPrice": 1500, "tdMode": "cash"}),
        (
            TradingMode.FUTURES,
            MarginMode.ISOLATED,
            {"stopLossPrice": 1500, "tdMode": "isolated", "posSide": "net"},
        ),
    ],
)
def test__get_stop_params_okx(mocker, default_conf, trading_mode, margin_mode, expected):
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = margin_mode
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    params = exchange._get_stop_params("sell", "market", 1500)

    assert params == expected


def test_fetch_orders_okx(default_conf, mocker, limit_order):
    api_mock = MagicMock()
    api_mock.fetch_orders = MagicMock(
        return_value=[
            limit_order["buy"],
            limit_order["sell"],
        ]
    )
    api_mock.fetch_open_orders = MagicMock(return_value=[limit_order["buy"]])
    api_mock.fetch_closed_orders = MagicMock(return_value=[limit_order["buy"]])

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    start_time = datetime.now(UTC) - timedelta(days=20)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    # Not available in dry-run
    assert exchange.fetch_orders("mocked", start_time) == []
    assert api_mock.fetch_orders.call_count == 0
    default_conf["dry_run"] = False

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")

    def has_resp(_, endpoint):
        if endpoint == "fetchOrders":
            return False
        if endpoint == "fetchClosedOrders":
            return True
        if endpoint == "fetchOpenOrders":
            return True

    mocker.patch(f"{EXMS}.exchange_has", has_resp)

    history_params = {"method": "privateGetTradeOrdersHistoryArchive"}

    # happy path without fetchOrders
    exchange.fetch_orders("mocked", start_time)
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 2
    assert "params" not in api_mock.fetch_closed_orders.call_args_list[0][1]
    assert api_mock.fetch_closed_orders.call_args_list[1][1]["params"] == history_params

    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()

    # regular closed_orders endpoint only has history for 7 days.
    exchange.fetch_orders("mocked", datetime.now(UTC) - timedelta(days=6))
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 1
    assert "params" not in api_mock.fetch_closed_orders.call_args_list[0][1]

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)

    # Unhappy path - first fetch-orders call fails.
    api_mock.fetch_orders = MagicMock(side_effect=ccxt.NotSupported())
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()

    exchange.fetch_orders("mocked", start_time)

    assert api_mock.fetch_orders.call_count == 1
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 2
    assert "params" not in api_mock.fetch_closed_orders.call_args_list[0][1]
    assert api_mock.fetch_closed_orders.call_args_list[1][1]["params"] == history_params
