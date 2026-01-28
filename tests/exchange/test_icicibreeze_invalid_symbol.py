import pytest
import os
from unittest import mock
from freqtrade.exceptions import OperationalException
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT


@pytest.fixture
def exchange_for_resilience():
    exchange = BreezeCCXT()
    exchange.breeze = mock.Mock()  # Mock session
    return exchange


def test_fetch_ohlcv_invalid_symbol_raises_cleanly(exchange_for_resilience):
    # Mock _load_security_master to return empty master, so any symbol is not found
    with mock.patch.object(
        exchange_for_resilience,
        "_load_security_master",
        return_value={"nfo": {}, "nse": {"by_symbol": {}}},
    ):
        # "INVALID/INR" -> Underlying=INVALID, Quote=INR -> CASH
        # Lookup INVALID in nse master -> None -> Raise "Cash symbol not found"
        with pytest.raises(OperationalException, match="Cash symbol not found"):
            exchange_for_resilience.fetch_ohlcv("INVALID/INR", "5m")


def test_fetch_ohlcv_api_error_returns_empty(exchange_for_resilience):
    # Case 2: Breeze SDK returns Generic Exception
    symbol = "RELIANCE/INR"

    # Mock master to find symbol
    mock_master = {"nse": {"by_symbol": {"RELIANCE": {"token": "123"}}}, "nfo": {}}

    with mock.patch.object(
        exchange_for_resilience, "_load_security_master", return_value=mock_master
    ):
        with mock.patch.object(
            exchange_for_resilience,
            "_parse_symbol",
            return_value={"stock_code": "REL", "exchange_code": "NSE", "product_type": "cash"},
        ):
            # Mock Breeze SDK to raise Exception
            exchange_for_resilience.breeze.get_historical_data_v2.side_effect = Exception(
                "API Error"
            )

            # Force real mode so we hit the breeze SDK call
            with mock.patch.object(exchange_for_resilience, "_is_mock_mode", return_value=False):
                # Should NOT raise, but return empty list (resilience)
                res = exchange_for_resilience.fetch_ohlcv(symbol, "5m")
                assert res == []
