"""Tests for Kraken Futures exchange class"""

from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import MagicMock, PropertyMock

import ccxt
import pytest

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import (
    DDosProtection,
    ExchangeError,
    InvalidOrderException,
    OperationalException,
    TemporaryError,
)
from freqtrade.exchange.krakenfutures import Krakenfutures
from tests.conftest import EXMS, get_patched_exchange


ExchangeBase = Krakenfutures.__mro__[1]  # freqtrade.exchange.exchange.Exchange


# --- _ft_has and OHLCV tests ---


def test_krakenfutures_ft_has_overrides():
    """Test that _ft_has contains Kraken Futures stoploss settings."""
    ft_has = Krakenfutures._ft_has
    assert ft_has["stoploss_on_exchange"] is True
    assert ft_has["stoploss_order_types"] == {"limit": "limit", "market": "market"}
    assert ft_has["stoploss_query_requires_stop_flag"] is True
    assert ft_has["stop_price_prop"] == "stopPrice"
    assert ft_has["stop_price_param"] == "triggerPrice"
    assert ft_has["stop_price_type_field"] == "triggerSignal"


# --- _adjust_krakenfutures_order average price tests ---


def test_krakenfutures_adjust_order_computes_average_from_trades(mocker, default_conf):
    """Compute VWAP average price from trades when CCXT returns None."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    order = {
        "id": "abc",
        "symbol": "BTC/USD:USD",
        "status": "closed",
        "filled": 0.0004,
        "average": None,
        "timestamp": 1771354195241,
    }
    trades = [
        {
            "amount": 0.0002,
            "price": 67800.0,
            "cost": 13.56,
            "takerOrMaker": "taker",
            "symbol": "BTC/USD:USD",
            "fee": None,
        },
        {
            "amount": 0.0002,
            "price": 67900.0,
            "cost": 13.58,
            "takerOrMaker": "taker",
            "symbol": "BTC/USD:USD",
            "fee": None,
        },
    ]
    mocker.patch.object(ex, "get_trades_for_order", return_value=trades)

    result = ex._adjust_krakenfutures_order(order)
    assert result["average"] == pytest.approx(67850.0)
    assert result["cost"] == pytest.approx(27.14)


def test_krakenfutures_adjust_order_skips_open_orders(mocker, default_conf):
    """Don't fetch trades for open orders."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    order = {
        "id": "abc",
        "symbol": "BTC/USD:USD",
        "status": "open",
        "filled": 0,
        "average": None,
        "timestamp": 1771354195241,
    }
    trades_mock = mocker.patch.object(ex, "get_trades_for_order")

    result = ex._adjust_krakenfutures_order(order)
    assert result["average"] is None
    trades_mock.assert_not_called()


def test_krakenfutures_adjust_order_handles_none_filled(mocker, default_conf):
    """Don't crash or fetch trades when filled is None."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    order = {
        "id": "abc",
        "symbol": "BTC/USD:USD",
        "status": "closed",
        "filled": None,
        "average": None,
        "timestamp": 1771354195241,
    }
    trades_mock = mocker.patch.object(ex, "get_trades_for_order")

    result = ex._adjust_krakenfutures_order(order)
    assert result["average"] is None
    trades_mock.assert_not_called()


def test_krakenfutures_adjust_order_recomputes_existing_average(mocker, default_conf):
    """Recompute average from fills even when CCXT already provided one."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    order = {
        "id": "abc",
        "symbol": "BTC/USD:USD",
        "status": "closed",
        "filled": 0.0004,
        "average": 67843.0,
        "timestamp": 1771354195241,
    }
    trades = [
        {
            "amount": 0.0002,
            "price": 67800.0,
            "cost": 13.56,
            "takerOrMaker": "taker",
            "symbol": "BTC/USD:USD",
            "fee": None,
        },
        {
            "amount": 0.0002,
            "price": 67900.0,
            "cost": 13.58,
            "takerOrMaker": "taker",
            "symbol": "BTC/USD:USD",
            "fee": None,
        },
    ]
    trades_mock = mocker.patch.object(ex, "get_trades_for_order", return_value=trades)

    result = ex._adjust_krakenfutures_order(order)
    assert result["average"] == pytest.approx(67850.0)
    assert result["cost"] == pytest.approx(27.14)
    trades_mock.assert_called_once()


def test_krakenfutures_adjust_order_no_trades_found(mocker, default_conf):
    """Leave average as None when no trades are found."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    order = {
        "id": "abc",
        "symbol": "BTC/USD:USD",
        "status": "closed",
        "filled": 0.0004,
        "average": None,
        "timestamp": 1771354195241,
    }
    mocker.patch.object(ex, "get_trades_for_order", return_value=[])

    result = ex._adjust_krakenfutures_order(order)
    assert result["average"] is None


# --- get_trades_for_order fee enrichment tests ---


def test_krakenfutures_get_trades_enriches_fees(mocker, default_conf):
    """Calculate fees from market fee schedule when CCXT returns fee: None."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    raw_trades = [
        {
            "amount": 0.0004,
            "price": 67843.0,
            "cost": 27.14,
            "order": "abc",
            "symbol": "BTC/USD:USD",
            "takerOrMaker": "taker",
            "fee": {"cost": None, "currency": None},
        },
    ]
    mocker.patch.object(
        ExchangeBase,
        "get_trades_for_order",
        return_value=raw_trades,
    )
    # Re-patch markets property with fee rates for BTC/USD:USD
    kf_markets = {"BTC/USD:USD": {"taker": 0.0005, "maker": 0.0002, "quote": "USD"}}
    mocker.patch.object(type(ex), "markets", PropertyMock(return_value=kf_markets))

    result = ex.get_trades_for_order("abc", "BTC/USD:USD", since=MagicMock())
    assert len(result) == 1
    assert result[0]["fee"]["cost"] == pytest.approx(27.14 * 0.0005)
    assert result[0]["fee"]["currency"] == "USD"
    assert result[0]["fee"]["rate"] == 0.0005


def test_krakenfutures_get_trades_uses_maker_rate(mocker, default_conf):
    """Use maker fee rate when fillType is maker."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    raw_trades = [
        {
            "amount": 0.0004,
            "price": 67843.0,
            "cost": 27.14,
            "order": "abc",
            "symbol": "BTC/USD:USD",
            "takerOrMaker": "maker",
            "fee": None,
        },
    ]
    mocker.patch.object(
        ExchangeBase,
        "get_trades_for_order",
        return_value=raw_trades,
    )
    kf_markets = {"BTC/USD:USD": {"taker": 0.0005, "maker": 0.0002, "quote": "USD"}}
    mocker.patch.object(type(ex), "markets", PropertyMock(return_value=kf_markets))

    result = ex.get_trades_for_order("abc", "BTC/USD:USD", since=MagicMock())
    assert result[0]["fee"]["cost"] == pytest.approx(27.14 * 0.0002)
    assert result[0]["fee"]["rate"] == 0.0002


def test_krakenfutures_get_trades_preserves_existing_fees(mocker, default_conf):
    """Don't overwrite fees if CCXT already provided them."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    existing_fee = {"cost": 0.01, "currency": "USD", "rate": 0.0005}
    raw_trades = [
        {
            "amount": 0.0004,
            "price": 67843.0,
            "cost": 27.14,
            "order": "abc",
            "symbol": "BTC/USD:USD",
            "takerOrMaker": "taker",
            "fee": existing_fee,
        },
    ]
    mocker.patch.object(
        ExchangeBase,
        "get_trades_for_order",
        return_value=raw_trades,
    )
    kf_markets = {"BTC/USD:USD": {"taker": 0.0005, "maker": 0.0002, "quote": "USD"}}
    mocker.patch.object(type(ex), "markets", PropertyMock(return_value=kf_markets))

    result = ex.get_trades_for_order("abc", "BTC/USD:USD", since=MagicMock())
    # Should keep existing fee, not recalculate
    assert result[0]["fee"] == existing_fee


# --- fetch_order fallback tests ---


def test_krakenfutures_fetch_order_falls_back_to_closed_orders(mocker, default_conf):
    """Fallback to fetch_closed_orders when fetch_order can't find the order."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    mocker.patch.object(ex._api, "fetch_order", side_effect=ccxt.OrderNotFound("not found"))
    open_fetch = mocker.patch.object(ex._api, "fetch_open_orders", return_value=[], create=True)
    open_fetch.__name__ = "fetch_open_orders"
    closed_fetch = mocker.patch.object(
        ex._api,
        "fetch_closed_orders",
        return_value=[{"id": "abc", "symbol": "BTC/USD:USD", "status": "closed"}],
        create=True,
    )
    closed_fetch.__name__ = "fetch_closed_orders"

    res = ex.fetch_order("abc", "BTC/USD:USD")
    assert res["id"] == "abc"


def test_krakenfutures_fetch_order_falls_back_to_canceled_orders(mocker, default_conf):
    """Fallback to fetch_canceled_orders when closed orders don't contain the order."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    mocker.patch.object(ex._api, "fetch_order", side_effect=ccxt.ExchangeError("UUID too large"))
    open_fetch = mocker.patch.object(ex._api, "fetch_open_orders", return_value=[], create=True)
    open_fetch.__name__ = "fetch_open_orders"
    closed_fetch = mocker.patch.object(ex._api, "fetch_closed_orders", return_value=[], create=True)
    closed_fetch.__name__ = "fetch_closed_orders"
    canceled_fetch = mocker.patch.object(
        ex._api,
        "fetch_canceled_orders",
        return_value=[{"id": "def", "symbol": "BTC/USD:USD", "status": "canceled"}],
        create=True,
    )
    canceled_fetch.__name__ = "fetch_canceled_orders"

    res = ex.fetch_order("def", "BTC/USD:USD")
    assert res["id"] == "def"


def test_krakenfutures_fetch_order_returns_direct_ccxt_result(mocker, default_conf):
    """Use direct CCXT fetch_order result when available."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    ccxt_order = {"id": "live-123", "symbol": "BTC/USD:USD", "status": "open"}
    converted = {"id": "live-123", "status": "open"}
    mocker.patch.object(ex._api, "fetch_order", return_value=ccxt_order)
    converter = mocker.patch.object(ex, "_order_contracts_to_amount", return_value=converted)
    fallback = mocker.patch.object(ex, "_fetch_order_fallback")

    res = ex.fetch_order("live-123", "BTC/USD:USD")

    assert res == converted
    converter.assert_called_once_with(ccxt_order)
    fallback.assert_not_called()


def test_krakenfutures_fetch_order_strips_stop_from_status_query(mocker, default_conf):
    """Direct CCXT fetch_order status lookup should not receive stop/trigger params."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    ccxt_order = {"id": "order-1", "symbol": "BTC/USD:USD", "status": "open"}
    fetch_order = mocker.patch.object(ex._api, "fetch_order", return_value=ccxt_order)

    # Simulate call from base fetch_stoploss_order which adds stop=True
    ex.fetch_order("order-1", "BTC/USD:USD", params={"stop": True})

    # stop should be stripped from the direct CCXT status call
    fetch_order.assert_called_once_with("order-1", "BTC/USD:USD", params={})


def test_krakenfutures_fetch_order_raises_invalid_when_not_found(mocker, default_conf):
    """Raise InvalidOrderException (non-retrying) when order is not in any endpoint."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    mocker.patch.object(ex._api, "fetch_order", side_effect=ccxt.OrderNotFound("not found"))
    mocker.patch.object(ex, "_fetch_order_fallback", return_value=None)

    with pytest.raises(InvalidOrderException, match="Order not found in any endpoint"):
        ex.fetch_order("abc", "BTC/USD:USD", count=0)


def test_krakenfutures_fetch_order_invalid_order_maps_exception(mocker, default_conf):
    """Map ccxt.InvalidOrder to InvalidOrderException."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")
    mocker.patch.object(ex._api, "fetch_order", side_effect=ccxt.InvalidOrder("bad order"))

    with pytest.raises(InvalidOrderException, match="bad order"):
        ex.fetch_order("abc", "BTC/USD:USD", count=0)


def test_krakenfutures_fetch_order_ddos_maps_exception(mocker, default_conf):
    """Map ccxt.DDoSProtection to DDosProtection."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")
    mocker.patch.object(ex._api, "fetch_order", side_effect=ccxt.DDoSProtection("ratelimit"))

    with pytest.raises(DDosProtection):
        ex.fetch_order("abc", "BTC/USD:USD", count=0)


def test_krakenfutures_fetch_order_baseerror_maps_exception(mocker, default_conf):
    """Map generic ccxt.BaseError to OperationalException."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")
    mocker.patch.object(ex._api, "fetch_order", side_effect=ccxt.BaseError("unexpected"))

    with pytest.raises(OperationalException):
        ex.fetch_order("abc", "BTC/USD:USD", count=0)


def test_krakenfutures_fetch_order_fallback_returns_none(mocker, default_conf):
    """Return None when order is not found in any endpoint."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    open_fetch = mocker.patch.object(ex._api, "fetch_open_orders", return_value=[], create=True)
    open_fetch.__name__ = "fetch_open_orders"
    closed_fetch = mocker.patch.object(ex._api, "fetch_closed_orders", return_value=[], create=True)
    closed_fetch.__name__ = "fetch_closed_orders"
    canceled_fetch = mocker.patch.object(
        ex._api, "fetch_canceled_orders", return_value=[], create=True
    )
    canceled_fetch.__name__ = "fetch_canceled_orders"

    res = ex._fetch_order_fallback("abc", "BTC/USD:USD", {})
    assert res is None


def test_krakenfutures_fetch_order_fallback_returns_open_order_first(mocker, default_conf):
    """Return immediately when order is found in open orders."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    open_fetch = mocker.patch.object(
        ex._api,
        "fetch_open_orders",
        return_value=[{"id": "abc", "symbol": "BTC/USD:USD", "status": "open"}],
        create=True,
    )
    open_fetch.__name__ = "fetch_open_orders"
    closed_fetch = mocker.patch.object(ex._api, "fetch_closed_orders", return_value=[], create=True)
    closed_fetch.__name__ = "fetch_closed_orders"
    canceled_fetch = mocker.patch.object(
        ex._api, "fetch_canceled_orders", return_value=[], create=True
    )
    canceled_fetch.__name__ = "fetch_canceled_orders"

    res = ex._fetch_order_fallback("abc", "BTC/USD:USD", {})

    assert res is not None
    assert res["id"] == "abc"
    open_fetch.assert_called_once()
    closed_fetch.assert_not_called()
    canceled_fetch.assert_not_called()


def test_krakenfutures_fetch_order_dry_run(mocker, default_conf):
    """Test fetch_order uses dry_run order in dry_run mode."""
    conf = dict(default_conf)
    conf["dry_run"] = True
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    dry_order = {"id": "dry-123", "status": "open"}
    mocker.patch.object(ex, "fetch_dry_run_order", return_value=dry_order)

    res = ex.fetch_order("dry-123", "BTC/USD:USD")
    assert res["id"] == "dry-123"


def test_krakenfutures_fetch_order_finds_stoploss_via_stop_param(mocker, default_conf):
    """Test fetch_order finds stoploss orders via closed orders fallback with stop=True."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    mocker.patch.object(ex._api, "fetch_order", side_effect=ccxt.OrderNotFound("not found"))
    open_fetch = mocker.patch.object(ex._api, "fetch_open_orders", return_value=[], create=True)
    open_fetch.__name__ = "fetch_open_orders"
    # With stop=True, CCXT queries trigger history endpoint
    closed_fetch = mocker.patch.object(
        ex._api,
        "fetch_closed_orders",
        return_value=[
            {"id": "trigger-123", "symbol": "BTC/USD:USD", "status": "closed"},
        ],
        create=True,
    )
    closed_fetch.__name__ = "fetch_closed_orders"

    # Simulate what base class fetch_stoploss_order does (adds stop=True)
    res = ex.fetch_order("trigger-123", "BTC/USD:USD", params={"stop": True})
    assert res["id"] == "trigger-123"


def test_krakenfutures_fetch_order_fallback_passes_stop_to_history(mocker, default_conf):
    """Stoploss query (stop=True) should pass through to closed/canceled endpoints."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    open_fetch = mocker.patch.object(ex._api, "fetch_open_orders", return_value=[], create=True)
    open_fetch.__name__ = "fetch_open_orders"

    closed_order = {"id": "sl-123", "symbol": "BTC/USD:USD", "status": "closed"}
    closed_fetch = mocker.patch.object(
        ex._api,
        "fetch_closed_orders",
        return_value=[closed_order],
        create=True,
    )
    closed_fetch.__name__ = "fetch_closed_orders"

    res = ex._fetch_order_fallback("sl-123", "BTC/USD:USD", {"stop": True})

    assert res is not None
    assert res["id"] == "sl-123"
    # Verify stop=True was passed to closed orders (CCXT maps stop→trigger)
    closed_fetch.assert_called_once_with("BTC/USD:USD", params={"stop": True})


def test_krakenfutures_fetch_order_fallback_strips_stop_from_open_orders(mocker, default_conf):
    """Open orders query should not receive stop/trigger flags."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    open_fetch = mocker.patch.object(ex._api, "fetch_open_orders", return_value=[], create=True)
    open_fetch.__name__ = "fetch_open_orders"
    closed_fetch = mocker.patch.object(ex._api, "fetch_closed_orders", return_value=[], create=True)
    closed_fetch.__name__ = "fetch_closed_orders"
    canceled_fetch = mocker.patch.object(
        ex._api, "fetch_canceled_orders", return_value=[], create=True
    )
    canceled_fetch.__name__ = "fetch_canceled_orders"

    ex._fetch_order_fallback("abc", "BTC/USD:USD", {"stop": True})

    # stop should be stripped from open orders call
    open_fetch.assert_called_once_with("BTC/USD:USD", params={})


def test_krakenfutures_fetch_order_propagates_exchange_errors_from_fallback(mocker, default_conf):
    """Fallback list fetch should not hide exchange-level failures."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    mocker.patch.object(ex._api, "fetch_order", side_effect=ccxt.OrderNotFound("not found"))
    open_fetch = mocker.patch.object(
        ex._api, "fetch_open_orders", side_effect=ccxt.ExchangeError("service unavailable")
    )
    open_fetch.__name__ = "fetch_open_orders"

    with pytest.raises(TemporaryError):
        ex.fetch_order("abc", "BTC/USD:USD", count=0)


def test_krakenfutures_fetch_order_exchangeerror_uses_fallback(mocker, default_conf):
    """ExchangeError from fetch_order should trigger fallback lookup."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    fallback_order = {"id": "abc", "symbol": "BTC/USD:USD", "status": "closed"}
    mocker.patch.object(ex._api, "fetch_order", side_effect=ccxt.ExchangeError("temporary"))
    fallback = mocker.patch.object(ex, "_fetch_order_fallback", return_value=fallback_order)

    result = ex.fetch_order("abc", "BTC/USD:USD", count=0)

    assert result == fallback_order
    fallback.assert_called_once_with("abc", "BTC/USD:USD", {})


def test_krakenfutures_find_order_in_list_handles_ordernotfound(mocker, default_conf):
    """OrderNotFound in list fetch is treated as a missing order."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    def raise_order_not_found(_symbol, params=None):
        raise ccxt.OrderNotFound("missing")

    assert ex._find_order_in_list(raise_order_not_found, "BTC/USD:USD", {}, "abc") is None


def test_krakenfutures_find_order_in_list_maps_ddos(mocker, default_conf):
    """DDoS errors from list fetch are mapped to DDosProtection."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    def raise_ddos(_symbol, params=None):
        raise ccxt.DDoSProtection("ratelimit")

    with pytest.raises(DDosProtection):
        ex._find_order_in_list(raise_ddos, "BTC/USD:USD", {}, "abc")


def test_krakenfutures_find_order_in_list_maps_temporary(mocker, default_conf):
    """OperationFailed/ExchangeError from list fetch map to TemporaryError."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    def raise_temp(_symbol, params=None):
        raise ccxt.OperationFailed("temporary")

    with pytest.raises(TemporaryError):
        ex._find_order_in_list(raise_temp, "BTC/USD:USD", {}, "abc")


def test_krakenfutures_find_order_in_list_maps_operational(mocker, default_conf):
    """Unexpected BaseError from list fetch maps to OperationalException."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    def raise_base(_symbol, params=None):
        raise ccxt.BaseError("unexpected")

    with pytest.raises(OperationalException):
        ex._find_order_in_list(raise_base, "BTC/USD:USD", {}, "abc")


# --- Stoploss tests ---


def test_krakenfutures_create_stoploss_uses_trigger_price_type(mocker, default_conf):
    """Test create_stoploss uses triggerPrice, triggerSignal, and reduceOnly."""
    api_mock = MagicMock()
    api_mock.create_order = MagicMock(return_value={"id": "order-id", "info": {"foo": "bar"}})

    conf = deepcopy(default_conf)
    conf["dry_run"] = False
    conf["trading_mode"] = TradingMode.FUTURES
    conf["margin_mode"] = MarginMode.ISOLATED

    mocker.patch(f"{EXMS}.amount_to_precision", lambda s, x, y: y)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y, **kwargs: y)

    ex = get_patched_exchange(mocker, conf, api_mock, exchange="krakenfutures")

    ex.create_stoploss(
        pair="ETH/BTC",
        amount=1,
        stop_price=90000.0,
        side="sell",
        order_types={"stoploss": "market", "stoploss_price_type": "mark"},
        leverage=1.0,
    )

    call_args = api_mock.create_order.call_args
    params = call_args[1].get("params") if call_args[1] else call_args[0][5]

    assert params["triggerPrice"] == 90000.0
    assert params["triggerSignal"] == "mark"
    assert params["reduceOnly"] is True


# --- Funding fees tests ---


def test_krakenfutures_get_funding_fees_futures_success(mocker, default_conf):
    """Use funding fee helper in futures mode."""
    conf = dict(default_conf)
    conf["trading_mode"] = TradingMode.FUTURES
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    helper = mocker.patch.object(ex, "_fetch_and_calculate_funding_fees", return_value=1.23)
    open_date = datetime.now(UTC)

    assert ex.get_funding_fees("BTC/USD:USD", 0.1, False, open_date) == 1.23
    helper.assert_called_once_with("BTC/USD:USD", 0.1, False, open_date)


def test_krakenfutures_get_funding_fees_futures_exchange_error(mocker, default_conf):
    """Return 0.0 when funding fee retrieval fails."""
    conf = dict(default_conf)
    conf["trading_mode"] = TradingMode.FUTURES
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    mocker.patch.object(ex, "_fetch_and_calculate_funding_fees", side_effect=ExchangeError("fail"))

    assert ex.get_funding_fees("BTC/USD:USD", 0.1, False, None) == 0.0


def test_krakenfutures_get_funding_fees_spot_returns_zero(mocker, default_conf):
    """Return 0.0 outside futures mode without calling the helper."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    helper = mocker.patch.object(ex, "_fetch_and_calculate_funding_fees")

    assert ex.get_funding_fees("BTC/USD:USD", 0.1, False, None) == 0.0
    helper.assert_not_called()


# --- Balance tests (flex account USD synthesis) ---


def test_krakenfutures_get_balances_flex_account_synthesizes_usd(mocker, default_conf):
    """Test that flex account availableMargin/portfolioValue are synthesized as USD balance."""
    default_conf["stake_currency"] = "USD"
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    flex_response = {
        "EUR": {"free": 100.0, "used": 0.0, "total": 100.0},
        "info": {
            "accounts": {
                "flex": {
                    "availableMargin": "950.50",
                    "marginEquity": "1000.00",
                    "portfolioValue": "1050.00",  # Should be ignored, marginEquity preferred
                    "currencies": {"EUR": {"quantity": "100", "value": "105.00"}},
                }
            }
        },
        "free": {"EUR": 100.0},
        "used": {"EUR": 0.0},
        "total": {"EUR": 100.0},
    }
    mocker.patch.object(ex._api, "fetch_balance", return_value=flex_response)

    balances = ex.get_balances()

    # USD should be synthesized from flex account
    assert "USD" in balances
    assert balances["USD"]["free"] == 950.50
    assert balances["USD"]["total"] == 1000.00
    # used = total - free = 1000.00 - 950.50 = 49.50
    assert balances["USD"]["used"] == 49.50
    # EUR should still be present
    assert "EUR" in balances
    # info, free, total, used dicts should be removed
    assert "info" not in balances
    assert "free" not in balances
    assert "total" not in balances
    assert "used" not in balances


def test_krakenfutures_get_balances_no_flex_account(mocker, default_conf):
    """Test that non-flex accounts work without USD synthesis."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    standard_response = {
        "USD": {"free": 500.0, "used": 100.0, "total": 600.0},
        "info": {"type": "cashAccount"},
        "free": {"USD": 500.0},
        "used": {"USD": 100.0},
        "total": {"USD": 600.0},
    }
    mocker.patch.object(ex._api, "fetch_balance", return_value=standard_response)

    balances = ex.get_balances()

    # USD should be preserved as-is
    assert balances["USD"]["free"] == 500.0
    assert balances["USD"]["total"] == 600.0
    # info, free, total, used dicts should be removed
    assert "info" not in balances


def test_krakenfutures_get_balances_flex_fallback_chain(mocker, default_conf):
    """Test fallback chain: marginEquity -> portfolioValue -> balanceValue."""
    default_conf["stake_currency"] = "USD"
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    # Test fallback to balanceValue (no marginEquity or portfolioValue)
    flex_response = {
        "info": {
            "accounts": {
                "flex": {
                    "availableMargin": "800.00",
                    "balanceValue": "850.00",
                }
            }
        },
        "free": {},
        "used": {},
        "total": {},
    }
    mocker.patch.object(ex._api, "fetch_balance", return_value=flex_response)

    balances = ex.get_balances()

    assert balances["USD"]["free"] == 800.00
    assert balances["USD"]["total"] == 850.00
    # used = total - free = 850.00 - 800.00 = 50.00
    assert balances["USD"]["used"] == 50.00


def test_krakenfutures_get_balances_flex_zero_free_calculates_used(mocker, default_conf):
    """Test used margin is correct when availableMargin is 0.0."""
    default_conf["stake_currency"] = "USD"
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    flex_response = {
        "info": {
            "accounts": {
                "flex": {
                    "availableMargin": "0.00",
                    "marginEquity": "125.00",
                }
            }
        },
        "free": {},
        "used": {},
        "total": {},
    }
    mocker.patch.object(ex._api, "fetch_balance", return_value=flex_response)

    balances = ex.get_balances()

    assert balances["USD"]["free"] == 0.00
    assert balances["USD"]["total"] == 125.00
    assert balances["USD"]["used"] == 125.00


def test_krakenfutures_get_balances_flex_missing_free_uses_total(mocker, default_conf):
    """When availableMargin is missing, free falls back to total and used is 0.0."""
    default_conf["stake_currency"] = "USD"
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    flex_response = {
        "info": {
            "accounts": {
                "flex": {
                    "marginEquity": "250.00",
                }
            }
        },
        "free": {},
        "used": {},
        "total": {},
    }
    mocker.patch.object(ex._api, "fetch_balance", return_value=flex_response)

    balances = ex.get_balances()

    assert balances["USD"]["free"] == 250.00
    assert balances["USD"]["total"] == 250.00
    assert balances["USD"]["used"] == 0.00


def test_krakenfutures_get_balances_skips_synthesis_for_non_usd_stake(mocker, default_conf):
    """Test that USD synthesis is skipped when stake_currency is not USD."""
    default_conf["stake_currency"] = "EUR"
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")

    flex_response = {
        "EUR": {"free": 100.0, "used": 0.0, "total": 100.0},
        "info": {
            "accounts": {
                "flex": {
                    "availableMargin": "950.50",
                    "portfolioValue": "1000.00",
                }
            }
        },
        "free": {"EUR": 100.0},
        "used": {"EUR": 0.0},
        "total": {"EUR": 100.0},
    }
    mocker.patch.object(ex._api, "fetch_balance", return_value=flex_response)

    balances = ex.get_balances()

    # USD should NOT be synthesized since stake_currency is EUR
    assert "USD" not in balances
    # EUR should still be present
    assert "EUR" in balances
    assert balances["EUR"]["free"] == 100.0


def test_krakenfutures_get_balances_maps_ddos(mocker, default_conf):
    """Map ccxt.DDoSProtection from fetch_balance to DDosProtection."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    mocker.patch.object(ex._api, "fetch_balance", side_effect=ccxt.DDoSProtection("ratelimit"))

    with pytest.raises(DDosProtection):
        ex.get_balances(count=0)


def test_krakenfutures_get_balances_maps_temporary(mocker, default_conf):
    """Map ccxt.OperationFailed/ExchangeError from fetch_balance to TemporaryError."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    mocker.patch.object(ex._api, "fetch_balance", side_effect=ccxt.OperationFailed("temporary"))

    with pytest.raises(TemporaryError):
        ex.get_balances(count=0)


def test_krakenfutures_get_balances_maps_operational(mocker, default_conf):
    """Map unexpected ccxt.BaseError from fetch_balance to OperationalException."""
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    mocker.patch.object(ex._api, "fetch_balance", side_effect=ccxt.BaseError("unexpected"))

    with pytest.raises(OperationalException):
        ex.get_balances(count=0)


def test_krakenfutures_safe_float():
    """Test _safe_float handles various input types."""
    assert Krakenfutures._safe_float("123.45") == 123.45
    assert Krakenfutures._safe_float(100) == 100.0
    assert Krakenfutures._safe_float(None) is None
    assert Krakenfutures._safe_float("invalid") is None
    assert Krakenfutures._safe_float({}) is None


# --- Stoploss via base class (stoploss_query_requires_stop_flag) ---


def test_krakenfutures_fetch_stoploss_order_uses_base_class(mocker, default_conf):
    """Base class fetch_stoploss_order should add stop=True and delegate to fetch_order."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    expected_order = {"id": "sl-order-1", "status": "open", "info": {}}
    fetch_order = mocker.patch.object(ex, "fetch_order", return_value=expected_order)

    result = ex.fetch_stoploss_order("sl-order-1", "BTC/USD:USD")

    assert result["id"] == "sl-order-1"
    # Base class should pass stop=True
    fetch_order.assert_called_once()
    call_params = fetch_order.call_args[0][2] if len(fetch_order.call_args[0]) > 2 else {}
    assert call_params.get("stop") is True


def test_krakenfutures_cancel_stoploss_order_uses_base_class(mocker, default_conf):
    """Base class cancel_stoploss_order should add stop=True and delegate to cancel_order."""
    conf = dict(default_conf)
    conf["dry_run"] = False
    ex = get_patched_exchange(mocker, conf, exchange="krakenfutures")

    expected_order = {"id": "sl-cancel-1", "status": "canceled"}
    cancel_order = mocker.patch.object(ex, "cancel_order", return_value=expected_order)

    result = ex.cancel_stoploss_order("sl-cancel-1", "BTC/USD:USD")

    assert result["id"] == "sl-cancel-1"
    # Base class should pass stop=True
    cancel_order.assert_called_once()
    call_params = cancel_order.call_args[0][2] if len(cancel_order.call_args[0]) > 2 else {}
    assert call_params.get("stop") is True
