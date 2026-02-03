from copy import deepcopy
from datetime import timedelta
from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.enums import CandleType, MarginMode, RunMode, TradingMode
from freqtrade.exceptions import OperationalException, RetryableOrderError
from freqtrade.exchange.common import API_RETRY_COUNT
from freqtrade.util import dt_now, dt_ts, dt_utc
from tests.conftest import EXMS, get_patched_exchange
from tests.exchange.test_exchange import ccxt_exceptionhandlers


@pytest.mark.usefixtures("init_persistence")
def test_fetch_stoploss_order_bitget(default_conf, mocker):
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.common.time.sleep")
    api_mock = MagicMock()

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bitget")

    api_mock.fetch_open_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_and_closed_orders = MagicMock(return_value=[])

    with pytest.raises(RetryableOrderError):
        exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_open_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_canceled_and_closed_orders.call_count == API_RETRY_COUNT + 1

    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_canceled_and_closed_orders.reset_mock()

    api_mock.fetch_canceled_and_closed_orders = MagicMock(
        return_value=[{"id": "1234", "status": "closed", "clientOrderId": "123455"}]
    )
    api_mock.fetch_open_orders = MagicMock(return_value=[{"id": "50110", "clientOrderId": "1234"}])

    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_open_orders.call_count == 2
    assert api_mock.fetch_canceled_and_closed_orders.call_count == 2

    assert resp["id"] == "1234"
    assert resp["id_stop"] == "50110"
    assert resp["type"] == "stoploss"

    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bitget")
    dro_mock = mocker.patch(f"{EXMS}.fetch_dry_run_order", MagicMock(return_value={"id": "123455"}))

    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_canceled_and_closed_orders.reset_mock()
    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")

    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_canceled_and_closed_orders.call_count == 0
    assert dro_mock.call_count == 1


def test_fetch_stoploss_order_bitget_exceptions(default_conf_usdt, mocker):
    default_conf_usdt["dry_run"] = False
    api_mock = MagicMock()

    # Test emulation of the stoploss getters
    api_mock.fetch_canceled_and_closed_orders = MagicMock(return_value=[])

    ccxt_exceptionhandlers(
        mocker,
        default_conf_usdt,
        api_mock,
        "bitget",
        "fetch_stoploss_order",
        "fetch_open_orders",
        retries=API_RETRY_COUNT + 1,
        order_id="12345",
        pair="ETH/USDT",
    )


def test_bitget_ohlcv_candle_limit(mocker, default_conf_usdt):
    # This test is also a live test - so we're sure our limits are correct.
    api_mock = MagicMock()
    api_mock.options = {
        "fetchOHLCV": {
            "maxRecentDaysPerTimeframe": {
                "1m": 30,
                "5m": 30,
                "15m": 30,
                "30m": 30,
                "1h": 60,
                "4h": 60,
                "1d": 60,
            }
        }
    }

    exch = get_patched_exchange(mocker, default_conf_usdt, api_mock, exchange="bitget")
    timeframes = ("1m", "5m", "1h")

    for timeframe in timeframes:
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE) == 200

        start_time = dt_ts(dt_now() - timedelta(days=17))
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200
        start_time = dt_ts(dt_now() - timedelta(days=48))
        length = 200 if timeframe in ("1m", "5m") else 1000
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200

        start_time = dt_ts(dt_now() - timedelta(days=61))
        length = 200
        assert exch.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == length
        assert exch.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 200


def test_additional_exchange_init_bitget(default_conf, mocker):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.set_position_mode = MagicMock(return_value={})

    get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)
    assert api_mock.set_position_mode.call_count == 1

    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, "bitget", "additional_exchange_init", "set_position_mode"
    )


def test_dry_run_liquidation_price_cross_bitget(default_conf, mocker):
    default_conf["dry_run"] = True
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.CROSS
    api_mock = MagicMock()
    mocker.patch(f"{EXMS}.get_maintenance_ratio_and_amt", MagicMock(return_value=(0.005, 0.0)))
    exchange = get_patched_exchange(mocker, default_conf, exchange="bitget", api_mock=api_mock)

    with pytest.raises(
        OperationalException, match="Freqtrade currently only supports isolated futures for bitget"
    ):
        exchange.dry_run_liquidation_price(
            "ETH/USDT:USDT",
            100_000,
            False,
            0.1,
            100,
            10,
            100,
            [],
        )


def test__lev_prep_bitget(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.set_margin_mode = MagicMock()
    api_mock.set_leverage = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setMarginMode": True, "setLeverage": True})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bitget")
    exchange._lev_prep("BTC/USDC:USDC", 3.2, "buy")

    assert api_mock.set_margin_mode.call_count == 0
    assert api_mock.set_leverage.call_count == 0

    # test in futures mode
    api_mock.set_margin_mode.reset_mock()
    api_mock.set_leverage.reset_mock()
    default_conf["dry_run"] = False

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bitget")
    exchange._lev_prep("BTC/USDC:USDC", 3.2, "buy")

    assert api_mock.set_margin_mode.call_count == 0
    assert api_mock.set_leverage.call_count == 1
    api_mock.set_leverage.assert_called_with(symbol="BTC/USDC:USDC", leverage=3.2)

    api_mock.reset_mock()

    exchange._lev_prep("BTC/USDC:USDC", 19.99, "sell")

    assert api_mock.set_margin_mode.call_count == 0
    assert api_mock.set_leverage.call_count == 1
    api_mock.set_leverage.assert_called_with(symbol="BTC/USDC:USDC", leverage=19.99)


def test_check_delisting_time_bitget(default_conf_usdt, mocker):
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="bitget")
    exchange._config["runmode"] = RunMode.BACKTEST
    delist_fut_mock = MagicMock(return_value=None)
    mocker.patch.object(exchange, "_check_delisting_futures", delist_fut_mock)

    # Invalid run mode
    resp = exchange.check_delisting_time("BTC/USDT")
    assert resp is None
    assert delist_fut_mock.call_count == 0

    # Delist spot called
    exchange._config["runmode"] = RunMode.DRY_RUN
    resp1 = exchange.check_delisting_time("BTC/USDT")
    assert resp1 is None
    assert delist_fut_mock.call_count == 0

    # Delist futures called
    exchange.trading_mode = TradingMode.FUTURES
    resp1 = exchange.check_delisting_time("BTC/USDT:USDT")
    assert resp1 is None
    assert delist_fut_mock.call_count == 1


def test__check_delisting_futures_bitget(default_conf_usdt, mocker, markets):
    markets["BTC/USDT:USDT"] = deepcopy(markets["SOL/BUSD:BUSD"])
    markets["BTC/USDT:USDT"]["info"]["limitOpenTime"] = "-1"
    markets["SOL/BUSD:BUSD"]["info"]["limitOpenTime"] = "-1"
    markets["ADA/USDT:USDT"]["info"]["limitOpenTime"] = "1760745600000"  # 2025-10-18
    exchange = get_patched_exchange(mocker, default_conf_usdt, exchange="bitget")
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))

    resp_sol = exchange._check_delisting_futures("SOL/BUSD:BUSD")
    # No delisting date
    assert resp_sol is None
    # Has a delisting date
    resp_ada = exchange._check_delisting_futures("ADA/USDT:USDT")
    assert resp_ada == dt_utc(2025, 10, 18)
