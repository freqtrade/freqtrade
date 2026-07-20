from unittest.mock import MagicMock

import pytest

from freqtrade.enums import MarginMode, TradingMode
from tests.conftest import get_patched_exchange


def test_bingx_get_params(default_conf, mocker):
    api_mock = MagicMock()
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bingx")

    # Open LONG position
    params = exchange._get_params(
        side="buy",
        ordertype="limit",
        leverage=2.0,
        reduceOnly=False,
    )
    assert params["positionSide"] == "LONG"

    # Close LONG position
    params = exchange._get_params(
        side="sell",
        ordertype="limit",
        leverage=2.0,
        reduceOnly=True,
    )
    assert params["positionSide"] == "LONG"

    # Open SHORT position
    params = exchange._get_params(
        side="sell",
        ordertype="limit",
        leverage=2.0,
        reduceOnly=False,
    )
    assert params["positionSide"] == "SHORT"

    # Close SHORT position
    params = exchange._get_params(
        side="buy",
        ordertype="limit",
        leverage=2.0,
        reduceOnly=True,
    )
    assert params["positionSide"] == "SHORT"


def test_bingx_lev_prep(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.set_leverage = MagicMock(return_value={})

    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bingx")

    # Initially hedge mode is None
    assert exchange._bingx_hedge_mode is None

    exchange._lev_prep("DOGE/USDT:USDT", 2.0, "buy")
    # Should be set to integer and BOTH first
    api_mock.set_leverage.assert_called_with(
        leverage=2, symbol="DOGE/USDT:USDT", params={"side": "BOTH"}
    )
    assert exchange._bingx_hedge_mode is False

    # Simulate Hedge mode error
    api_mock.set_leverage.side_effect = [Exception("109400 Hedge mode"), {}, {}]
    exchange._bingx_hedge_mode = None
    exchange._lev_prep("DOGE/USDT:USDT", 3.0, "buy")
    assert exchange._bingx_hedge_mode is True
    # In hedge mode, it should call for LONG and SHORT
    assert api_mock.set_leverage.call_count >= 3


def test_bingx_dry_run_liquidation_price(default_conf, mocker):
    api_mock = MagicMock()
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bingx")

    # LONG
    liq_long = exchange.dry_run_liquidation_price(
        pair="DOGE/USDT:USDT",
        open_rate=100.0,
        is_short=False,
        amount=10,
        stake_amount=100,
        leverage=10.0,
        wallet_balance=1000,
        open_trades=[],
    )
    # open_rate * (1 - 1 / leverage + mm_ratio)
    # 100 * (1 - 0.1 + 0.005) = 100 * 0.905 = 90.5
    assert liq_long == pytest.approx(90.5)

    # SHORT
    liq_short = exchange.dry_run_liquidation_price(
        pair="DOGE/USDT:USDT",
        open_rate=100.0,
        is_short=True,
        amount=10,
        stake_amount=100,
        leverage=10.0,
        wallet_balance=1000,
        open_trades=[],
    )
    # open_rate * (1 + 1 / leverage - mm_ratio)
    # 100 * (1 + 0.1 - 0.005) = 100 * 1.095 = 109.5
    assert liq_short == pytest.approx(109.5)


def test_bingx_load_leverage_tiers(default_conf, mocker):
    api_mock = MagicMock()
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bingx")

    tiers = exchange.load_leverage_tiers()
    assert "ETH/USDT:USDT" in tiers
    assert "ETH/USDT" not in tiers
    assert tiers["ETH/USDT:USDT"][0]["maintenanceMarginRate"] == 0.005


def test_bingx_fetch_l2_order_book(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fetch_l2_order_book = MagicMock(return_value={})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="bingx")

    exchange.fetch_l2_order_book("DOGE/USDT:USDT", 1)
    api_mock.fetch_l2_order_book.assert_called_with("DOGE/USDT:USDT", 5)

    exchange.fetch_l2_order_book("DOGE/USDT:USDT", 10)
    api_mock.fetch_l2_order_book.assert_called_with("DOGE/USDT:USDT", 10)
