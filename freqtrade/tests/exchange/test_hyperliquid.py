from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.exceptions import ConfigurationError
from tests.conftest import EXMS, get_mock_coro, get_patched_exchange, log_has_re


@pytest.fixture
def markets_hip3():
    markets = {
        "BTC/USDC:USDC": {
            "quote": "USDC",
            "base": "BTC",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 50}},
            "info": {},
        },
        "ETH/USDC:USDC": {
            "quote": "USDC",
            "base": "ETH",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 50}},
            "info": {},
        },
        "SOL/USDC:USDC": {
            "quote": "USDC",
            "base": "SOL",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 20}},
            "info": {},
        },
        "DOGE/USDC:USDC": {
            "quote": "USDC",
            "base": "DOGE",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 20}},
            "info": {},
        },
        "XYZ-AAPL/USDC:USDC": {
            "quote": "USDC",
            "base": "XYZ-AAPL",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 10}},
            "info": {"hip3": True, "dex": "xyz"},
        },
        "XYZ-TSLA/USDC:USDC": {
            "quote": "USDC",
            "base": "XYZ-TSLA",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 10}},
            "info": {"hip3": True, "dex": "xyz"},
        },
        "XYZ-GOOGL/USDC:USDC": {
            "quote": "USDC",
            "base": "XYZ-GOOGL",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 10}},
            "info": {"hip3": True, "dex": "xyz"},
        },
        "XYZ-NVDA/USDC:USDC": {
            "quote": "USDC",
            "base": "XYZ-NVDA",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 10}},
            "info": {"hip3": True, "dex": "xyz"},
        },
        "VNTL-SPACEX/USDH:USDH": {
            "quote": "USDH",
            "base": "VNTL-SPACEX",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 3}},
            "info": {"hip3": True, "dex": "vntl"},
        },
        "VNTL-ANTHROPIC/USDH:USDH": {
            "quote": "USDH",
            "base": "VNTL-ANTHROPIC",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 3}},
            "info": {"hip3": True, "dex": "vntl"},
        },
        "FLX-TOKEN/USDC:USDC": {
            "quote": "USDC",
            "base": "FLX-TOKEN",
            "type": "swap",
            "swap": True,
            "linear": True,
            "limits": {"leverage": {"max": 3}},
            "info": {"hip3": True, "dex": "flx"},
        },
    }

    return markets


@pytest.mark.parametrize("margin_mode", ["isolated", "cross"])
def test_hyperliquid_dry_run_liquidation_price(default_conf, markets_hip3, mocker, margin_mode):
    # test if liq price calculated by dry_run_liquidation_price() is close to ccxt liq price
    # testing different pairs with large/small prices, different leverages, long, short

    positions = [
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2458.5,
            "side": "long",
            "contracts": 0.015,
            "collateral": 36.864593,
            "leverage": 1.0,
            "liquidationPrice": 0.86915825,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 63287.0,
            "side": "long",
            "contracts": 0.00039,
            "collateral": 24.673292,
            "leverage": 1.0,
            "liquidationPrice": 22.37166537,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 146.82,
            "side": "long",
            "contracts": 0.16,
            "collateral": 23.482979,
            "leverage": 1.0,
            "liquidationPrice": 0.05269872,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 145.83,
            "side": "long",
            "contracts": 0.33,
            "collateral": 24.045107,
            "leverage": 2.0,
            "liquidationPrice": 74.83696193,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2459.5,
            "side": "long",
            "contracts": 0.0199,
            "collateral": 24.454895,
            "leverage": 2.0,
            "liquidationPrice": 1243.0411908,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 62739.0,
            "side": "long",
            "contracts": 0.00077,
            "collateral": 24.137992,
            "leverage": 2.0,
            "liquidationPrice": 31708.03843631,
        },
        {
            "symbol": "DOGE/USDC:USDC",
            "entryPrice": 0.11586,
            "side": "long",
            "contracts": 437.0,
            "collateral": 25.29769,
            "leverage": 2.0,
            "liquidationPrice": 0.05945697,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2642.8,
            "side": "short",
            "contracts": 0.019,
            "collateral": 25.091876,
            "leverage": 2.0,
            "liquidationPrice": 3924.18322043,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 155.89,
            "side": "short",
            "contracts": 0.32,
            "collateral": 24.924941,
            "leverage": 2.0,
            "liquidationPrice": 228.07847866,
        },
        {
            "symbol": "DOGE/USDC:USDC",
            "entryPrice": 0.14333,
            "side": "short",
            "contracts": 351.0,
            "collateral": 25.136807,
            "leverage": 2.0,
            "liquidationPrice": 0.20970228,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 68595.0,
            "side": "short",
            "contracts": 0.00069,
            "collateral": 23.64871,
            "leverage": 2.0,
            "liquidationPrice": 101849.99354283,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 65536.0,
            "side": "short",
            "contracts": 0.00099,
            "collateral": 21.604172,
            "leverage": 3.0,
            "liquidationPrice": 86493.46174617,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 173.06,
            "side": "long",
            "contracts": 0.6,
            "collateral": 20.735658,
            "leverage": 5.0,
            "liquidationPrice": 142.05186667,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2545.5,
            "side": "long",
            "contracts": 0.0329,
            "collateral": 20.909894,
            "leverage": 4.0,
            "liquidationPrice": 1929.23322895,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 67400.0,
            "side": "short",
            "contracts": 0.00031,
            "collateral": 20.887308,
            "leverage": 1.0,
            "liquidationPrice": 133443.97317151,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2552.0,
            "side": "short",
            "contracts": 0.0327,
            "collateral": 20.833393,
            "leverage": 4.0,
            "liquidationPrice": 3157.53150453,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 66930.0,
            "side": "long",
            "contracts": 0.0015,
            "collateral": 20.043862,
            "leverage": 5.0,
            "liquidationPrice": 54108.51043771,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 67033.0,
            "side": "long",
            "contracts": 0.00121,
            "collateral": 20.251817,
            "leverage": 4.0,
            "liquidationPrice": 50804.00091827,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2521.9,
            "side": "long",
            "contracts": 0.0237,
            "collateral": 19.902091,
            "leverage": 3.0,
            "liquidationPrice": 1699.14071943,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 68139.0,
            "side": "short",
            "contracts": 0.00145,
            "collateral": 19.72573,
            "leverage": 5.0,
            "liquidationPrice": 80933.61590987,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 178.29,
            "side": "short",
            "contracts": 0.11,
            "collateral": 19.605036,
            "leverage": 1.0,
            "liquidationPrice": 347.82205322,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 176.23,
            "side": "long",
            "contracts": 0.33,
            "collateral": 19.364946,
            "leverage": 3.0,
            "liquidationPrice": 120.56240404,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 173.08,
            "side": "short",
            "contracts": 0.33,
            "collateral": 19.01881,
            "leverage": 3.0,
            "liquidationPrice": 225.08561715,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 68240.0,
            "side": "short",
            "contracts": 0.00105,
            "collateral": 17.887922,
            "leverage": 4.0,
            "liquidationPrice": 84431.79820839,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2518.4,
            "side": "short",
            "contracts": 0.007,
            "collateral": 17.62263,
            "leverage": 1.0,
            "liquidationPrice": 4986.05799151,
        },
        {
            "symbol": "ETH/USDC:USDC",
            "entryPrice": 2533.2,
            "side": "long",
            "contracts": 0.0347,
            "collateral": 17.555195,
            "leverage": 5.0,
            "liquidationPrice": 2047.7642302,
        },
        {
            "symbol": "DOGE/USDC:USDC",
            "entryPrice": 0.13284,
            "side": "long",
            "contracts": 360.0,
            "collateral": 15.943218,
            "leverage": 3.0,
            "liquidationPrice": 0.09082388,
        },
        {
            "symbol": "SOL/USDC:USDC",
            "entryPrice": 163.11,
            "side": "short",
            "contracts": 0.48,
            "collateral": 15.650731,
            "leverage": 5.0,
            "liquidationPrice": 190.94213618,
        },
        {
            "symbol": "BTC/USDC:USDC",
            "entryPrice": 67141.0,
            "side": "long",
            "contracts": 0.00067,
            "collateral": 14.979079,
            "leverage": 3.0,
            "liquidationPrice": 45236.52992613,
        },
        {
            "symbol": "XYZ-AAPL/USDC:USDC",
            "entryPrice": 250.0,
            "side": "long",
            "contracts": 0.5,
            "collateral": 25.0,
            "leverage": 5.0,
            "liquidationPrice": 210.5263157894737,
        },
        {
            "symbol": "XYZ-GOOGL/USDC:USDC",
            "entryPrice": 190.0,
            "side": "short",
            "contracts": 0.5,
            "collateral": 9.5,
            "leverage": 10.0,
            "liquidationPrice": 199.04761904761904,
        },
        {
            "symbol": "XYZ-TSLA/USDC:USDC",
            "entryPrice": 350.0,
            "side": "long",
            "contracts": 1.0,
            "collateral": 50.0,
            "leverage": 7.0,
            "liquidationPrice": 315.7894736842105,
        },
    ]

    api_mock = MagicMock()
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = margin_mode
    default_conf["stake_currency"] = "USDC"
    api_mock.load_markets = get_mock_coro()
    api_mock.markets = markets_hip3
    exchange = get_patched_exchange(
        mocker, default_conf, api_mock, exchange="hyperliquid", mock_markets=False
    )

    for position in positions:
        is_short = True if position["side"] == "short" else False
        liq_price_returned = position["liquidationPrice"]
        liq_price_calculated = exchange.dry_run_liquidation_price(
            position["symbol"],
            position["entryPrice"],
            is_short,
            position["contracts"],
            position["collateral"],
            position["leverage"],
            # isolated doesn't use wallet-balance
            wallet_balance=0.0 if margin_mode == "isolated" else position["collateral"],
            open_trades=[],
        )
        # Assume full position size is the wallet balance
        assert pytest.approx(liq_price_returned, rel=0.0001) == liq_price_calculated

        if margin_mode == "cross":
            # test with larger wallet balance
            liq_price_calculated_cross = exchange.dry_run_liquidation_price(
                position["symbol"],
                position["entryPrice"],
                is_short,
                position["contracts"],
                position["collateral"],
                position["leverage"],
                wallet_balance=position["collateral"] * 2,
                open_trades=[],
            )
            # Assume full position size is the wallet balance
            # This
            if position["side"] == "long":
                assert liq_price_returned > liq_price_calculated_cross < position["entryPrice"]
            else:
                assert liq_price_returned < liq_price_calculated_cross > position["entryPrice"]


def test_hyperliquid_get_funding_fees(default_conf, mocker):
    now = datetime.now(UTC)
    exchange = get_patched_exchange(mocker, default_conf, exchange="hyperliquid")
    exchange._fetch_and_calculate_funding_fees = MagicMock()

    # Spot mode - no funding fees
    exchange.get_funding_fees("BTC/USDC:USDC", 1, False, now)
    assert exchange._fetch_and_calculate_funding_fees.call_count == 0

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["exchange"]["hip3_dexes"] = ["xyz", "vntl"]

    # Mock validate_config to skip validation
    mocker.patch("freqtrade.exchange.hyperliquid.Hyperliquid.validate_config")

    exchange = get_patched_exchange(mocker, default_conf, exchange="hyperliquid")
    exchange._fetch_and_calculate_funding_fees = MagicMock()

    # Normal market
    exchange.get_funding_fees("BTC/USDC:USDC", 1, False, now)
    assert exchange._fetch_and_calculate_funding_fees.call_count == 1

    # HIP-3 XYZ market
    exchange._fetch_and_calculate_funding_fees.reset_mock()
    exchange.get_funding_fees("XYZ-TSLA/USDC:USDC", 1, False, now)
    assert exchange._fetch_and_calculate_funding_fees.call_count == 1

    # HIP-3 VNTL market
    exchange._fetch_and_calculate_funding_fees.reset_mock()
    exchange.get_funding_fees("VNTL-SPACEX/USDH:USDH", 1, True, now)
    assert exchange._fetch_and_calculate_funding_fees.call_count == 1


def test_hyperliquid_get_max_leverage(default_conf, mocker, markets_hip3):
    exchange = get_patched_exchange(mocker, default_conf, exchange="hyperliquid")
    assert exchange.get_max_leverage("BTC/USDC:USDC", 1) == 1.0

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["exchange"]["hip3_dexes"] = ["xyz", "vntl"]

    # Mock validate_config to skip validation
    mocker.patch("freqtrade.exchange.hyperliquid.Hyperliquid.validate_config")

    exchange = get_patched_exchange(mocker, default_conf, exchange="hyperliquid")
    mocker.patch.multiple(EXMS, markets=PropertyMock(return_value=markets_hip3))

    # Normal markets
    assert exchange.get_max_leverage("BTC/USDC:USDC", 1) == 50
    assert exchange.get_max_leverage("ETH/USDC:USDC", 20) == 50
    assert exchange.get_max_leverage("SOL/USDC:USDC", 50) == 20
    assert exchange.get_max_leverage("DOGE/USDC:USDC", 3) == 20

    # HIP-3 markets
    assert exchange.get_max_leverage("XYZ-TSLA/USDC:USDC", 1) == 10
    assert exchange.get_max_leverage("XYZ-NVDA/USDC:USDC", 5) == 10
    assert exchange.get_max_leverage("VNTL-SPACEX/USDH:USDH", 2) == 3
    assert exchange.get_max_leverage("VNTL-ANTHROPIC/USDH:USDH", 1) == 3


def test_hyperliquid__lev_prep(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.set_margin_mode = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setMarginMode": True})
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="hyperliquid")
    exchange._lev_prep("BTC/USDC:USDC", 3.2, "buy")

    assert api_mock.set_margin_mode.call_count == 0

    # test in futures mode
    api_mock.set_margin_mode.reset_mock()
    default_conf["dry_run"] = False

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["exchange"]["hip3_dexes"] = ["xyz", "vntl"]

    # Mock validate_config to skip validation
    mocker.patch("freqtrade.exchange.hyperliquid.Hyperliquid.validate_config")

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="hyperliquid")

    # Normal market
    exchange._lev_prep("BTC/USDC:USDC", 3.2, "buy")
    assert api_mock.set_margin_mode.call_count == 1
    api_mock.set_margin_mode.assert_called_with("isolated", "BTC/USDC:USDC", {"leverage": 3})

    api_mock.reset_mock()
    exchange._lev_prep("BTC/USDC:USDC", 19.99, "sell")
    assert api_mock.set_margin_mode.call_count == 1
    api_mock.set_margin_mode.assert_called_with("isolated", "BTC/USDC:USDC", {"leverage": 19})

    # HIP-3 XYZ market
    api_mock.reset_mock()
    exchange._lev_prep("XYZ-TSLA/USDC:USDC", 5.7, "buy")
    assert api_mock.set_margin_mode.call_count == 1
    api_mock.set_margin_mode.assert_called_with("isolated", "XYZ-TSLA/USDC:USDC", {"leverage": 5})

    api_mock.reset_mock()
    exchange._lev_prep("XYZ-TSLA/USDC:USDC", 10.0, "sell")
    assert api_mock.set_margin_mode.call_count == 1
    api_mock.set_margin_mode.assert_called_with("isolated", "XYZ-TSLA/USDC:USDC", {"leverage": 10})

    # HIP-3 VNTL market
    api_mock.reset_mock()
    exchange._lev_prep("VNTL-SPACEX/USDH:USDH", 2.5, "buy")
    assert api_mock.set_margin_mode.call_count == 1
    api_mock.set_margin_mode.assert_called_with(
        "isolated", "VNTL-SPACEX/USDH:USDH", {"leverage": 2}
    )

    api_mock.reset_mock()
    exchange._lev_prep("VNTL-ANTHROPIC/USDH:USDH", 3.0, "sell")
    assert api_mock.set_margin_mode.call_count == 1
    api_mock.set_margin_mode.assert_called_with(
        "isolated", "VNTL-ANTHROPIC/USDH:USDH", {"leverage": 3}
    )


def test_hyperliquid_fetch_order(default_conf_usdt, mocker, markets_hip3):
    default_conf_usdt["dry_run"] = False
    default_conf_usdt["trading_mode"] = "futures"
    default_conf_usdt["margin_mode"] = "isolated"
    default_conf_usdt["exchange"]["hip3_dexes"] = ["xyz", "vntl"]

    api_mock = MagicMock()

    # Test with normal market
    api_mock.fetch_order = MagicMock(
        return_value={
            "id": "12345",
            "symbol": "ETH/USDC:USDC",
            "status": "closed",
            "filled": 0.1,
            "average": None,
            "timestamp": 1630000000,
        }
    )

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    gtfo_mock = mocker.patch(
        f"{EXMS}.get_trades_for_order",
        return_value=[
            {
                "order_id": "12345",
                "price": 1000,
                "amount": 3,
                "filled": 3,
                "remaining": 0,
            },
            {
                "order_id": "12345",
                "price": 3000,
                "amount": 1,
                "filled": 1,
                "remaining": 0,
            },
        ],
    )
    mocker.patch("freqtrade.exchange.hyperliquid.Hyperliquid.validate_config")
    exchange = get_patched_exchange(
        mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=markets_hip3
    )
    o = exchange.fetch_order("12345", "ETH/USDC:USDC")
    # Uses weighted average
    assert o["average"] == 1500
    assert gtfo_mock.call_count == 1

    # Test with HIP-3 XYZ market
    api_mock.fetch_order = MagicMock(
        return_value={
            "id": "67890",
            "symbol": "XYZ-TSLA/USDC:USDC",
            "status": "closed",
            "filled": 2.5,
            "average": None,
            "timestamp": 1630000100,
        }
    )
    gtfo_mock.reset_mock()
    gtfo_mock.return_value = [
        {
            "order_id": "67890",
            "price": 250,
            "amount": 1.5,
            "filled": 1.5,
            "remaining": 0,
        },
        {
            "order_id": "67890",
            "price": 260,
            "amount": 1.0,
            "filled": 1.0,
            "remaining": 0,
        },
    ]

    o = exchange.fetch_order("67890", "XYZ-TSLA/USDC:USDC")
    # Weighted average: (250*1.5 + 260*1.0) / 2.5 = 254
    assert o["average"] == 254
    assert gtfo_mock.call_count == 1

    # Test with HIP-3 VNTL market
    api_mock.fetch_order = MagicMock(
        return_value={
            "id": "11111",
            "symbol": "VNTL-SPACEX/USDH:USDH",
            "status": "closed",
            "filled": 5.0,
            "average": None,
            "timestamp": 1630000200,
        }
    )
    gtfo_mock.reset_mock()
    gtfo_mock.return_value = [
        {
            "order_id": "11111",
            "price": 100,
            "amount": 3.0,
            "filled": 3.0,
            "remaining": 0,
        },
        {
            "order_id": "11111",
            "price": 105,
            "amount": 2.0,
            "filled": 2.0,
            "remaining": 0,
        },
    ]

    o = exchange.fetch_order("11111", "VNTL-SPACEX/USDH:USDH")
    assert o["average"] == 102
    assert gtfo_mock.call_count == 1


def test_hyperliquid_hip3_config_validation(default_conf_usdt, mocker, markets_hip3):
    """Test HIP-3 DEX configuration validation."""

    api_mock = MagicMock()
    default_conf_usdt["stake_currency"] = "USDC"

    # Futures mode, no dex configured
    default_conf_copy = deepcopy(default_conf_usdt)
    default_conf_copy["trading_mode"] = "futures"
    default_conf_copy["margin_mode"] = "isolated"
    exchange = get_patched_exchange(
        mocker, default_conf_copy, api_mock, exchange="hyperliquid", mock_markets=markets_hip3
    )
    exchange.validate_config(default_conf_copy)

    # Not in futures mode - no dex configured - no error
    get_patched_exchange(
        mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=markets_hip3
    )
    # Not in futures mode
    default_conf_usdt["exchange"]["hip3_dexes"] = ["xyz"]
    with pytest.raises(
        ConfigurationError, match=r"HIP-3 DEXes are only supported in FUTURES trading mode\."
    ):
        get_patched_exchange(
            mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=markets_hip3
        )
    # Valid single DEX
    default_conf_usdt["trading_mode"] = "futures"
    default_conf_usdt["margin_mode"] = "isolated"
    default_conf_usdt["exchange"]["hip3_dexes"] = ["xyz"]
    exchange = get_patched_exchange(
        mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=markets_hip3
    )
    assert exchange._get_configured_hip3_dexes() == ["xyz"]

    # Invalid DEX
    default_conf_usdt["exchange"]["hip3_dexes"] = ["invalid_dex"]
    with pytest.raises(ConfigurationError, match="Invalid HIP-3 DEXes configured"):
        exchange = get_patched_exchange(
            mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=markets_hip3
        )
        exchange.validate_config(default_conf_usdt)

    # Mix of valid and invalid DEX
    default_conf_usdt["exchange"]["hip3_dexes"] = ["xyz", "invalid_dex"]
    with pytest.raises(ConfigurationError, match="Invalid HIP-3 DEXes configured"):
        exchange = get_patched_exchange(
            mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=markets_hip3
        )
        exchange.validate_config(default_conf_usdt)

    default_conf_usdt["margin_mode"] = "cross"
    with pytest.raises(ConfigurationError, match="HIP-3 DEXes require 'isolated' margin mode"):
        exchange = get_patched_exchange(
            mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=markets_hip3
        )
        exchange.validate_config(default_conf_usdt)


def test_hyperliquid_get_balances_hip3(default_conf, mocker, caplog, markets_hip3):
    """Test balance fetching from HIP-3 DEXes."""
    api_mock = MagicMock()

    api_mock.load_markets = get_mock_coro()

    # Mock balance responses
    default_balance = {"USDC": {"free": 1000, "used": 0, "total": 1000}}
    xyz_balance = {"USDC": {"free": 0, "used": 600, "total": 600}}
    vntl_balance = {"USDH": {"free": 0, "used": 300, "total": 300}}

    def fetch_balance_side_effect(params=None):
        if params and params.get("dex") == "xyz":
            return xyz_balance
        elif params and params.get("dex") == "vntl":
            return vntl_balance
        elif params and params.get("dex") == "flx":
            raise Exception("FLX DEX error")
        return default_balance

    api_mock.fetch_balance = MagicMock(side_effect=fetch_balance_side_effect)

    # Test with two HIP-3 DEXes
    default_conf["exchange"]["hip3_dexes"] = ["xyz", "vntl", "flx"]
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    mocker.patch("freqtrade.exchange.hyperliquid.Hyperliquid.validate_config")
    exchange = get_patched_exchange(
        mocker, default_conf, api_mock, exchange="hyperliquid", mock_markets=markets_hip3
    )

    balances = exchange.get_balances()

    # Should have combined balances
    assert balances["USDC"]["free"] == 1000
    assert balances["USDC"]["used"] == 600
    assert balances["USDC"]["total"] == 1600
    assert balances["USDH"]["free"] == 0
    assert balances["USDH"]["used"] == 300
    assert balances["USDH"]["total"] == 300

    assert api_mock.fetch_balance.call_count == 4
    assert log_has_re("Could not fetch balance for HIP-3 DEX.*", caplog)


def test_hyperliquid_fetch_positions_hip3(default_conf, mocker, caplog, markets_hip3):
    """Test position fetching from HIP-3 DEXes."""
    api_mock = MagicMock()

    # Mock position responses
    default_positions = [{"symbol": "BTC/USDC:USDC", "contracts": 0.5}]
    xyz_positions = [{"symbol": "XYZ-AAPL/USDC:USDC", "contracts": 10}]
    vntl_positions = [{"symbol": "VNTL-SPACEX/USDH:USDH", "contracts": 5}]

    def fetch_positions_side_effect(symbols=None, params=None):
        if params and params.get("dex") == "xyz":
            return xyz_positions
        elif params and params.get("dex") == "vntl":
            return vntl_positions
        elif params and params.get("dex") == "flx":
            raise Exception("FLX DEX error")
        return default_positions

    positions_mock = MagicMock(side_effect=fetch_positions_side_effect)

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["exchange"]["hip3_dexes"] = ["xyz", "vntl", "flx"]

    mocker.patch("freqtrade.exchange.hyperliquid.Hyperliquid.validate_config")
    exchange = get_patched_exchange(
        mocker, default_conf, api_mock, exchange="hyperliquid", mock_markets=markets_hip3
    )

    # Mock super().fetch_positions() to return default positions
    mocker.patch(f"{EXMS}.fetch_positions", positions_mock)

    positions = exchange.fetch_positions()

    assert log_has_re("Could not fetch positions from HIP-3 .*", caplog)

    # Should have all positions combined (default + HIP-3)
    assert len(positions) == 3
    assert any(p["symbol"] == "BTC/USDC:USDC" for p in positions)
    assert any(p["symbol"] == "XYZ-AAPL/USDC:USDC" for p in positions)
    assert any(p["symbol"] == "VNTL-SPACEX/USDH:USDH" for p in positions)

    # Verify API calls (xyz + vntl, default is mocked separately)
    assert positions_mock.call_count == 4


def test_hyperliquid_market_is_tradable(default_conf_usdt, mocker, markets_hip3):
    """Test market_is_tradable filters HIP-3 markets correctly."""
    default_conf_usdt["stake_currency"] = "USDC"
    default_conf_usdt["trading_mode"] = "futures"
    default_conf_usdt["margin_mode"] = "isolated"
    api_mock = MagicMock()
    api_mock.load_markets = get_mock_coro(return_value=markets_hip3)
    api_mock.markets = markets_hip3
    # Mock parent call - we only want to test hyperliquid specifics here.
    mocker.patch(f"{EXMS}.market_is_tradable", return_value=True)

    # Test 1: No HIP-3 DEXes configured - only default markets tradable
    default_conf_usdt["exchange"]["hip3_dexes"] = []
    exchange = get_patched_exchange(
        mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=False
    )

    assert exchange.market_is_tradable(markets_hip3["BTC/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["ETH/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["XYZ-AAPL/USDC:USDC"]) is False
    assert exchange.market_is_tradable(markets_hip3["XYZ-TSLA/USDC:USDC"]) is False
    assert exchange.market_is_tradable(markets_hip3["VNTL-SPACEX/USDH:USDH"]) is False
    assert exchange.market_is_tradable(markets_hip3["FLX-TOKEN/USDC:USDC"]) is False

    # Test 2: Only 'xyz' configured - default + xyz markets tradable
    default_conf_usdt["exchange"]["hip3_dexes"] = ["xyz"]
    exchange = get_patched_exchange(
        mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=False
    )

    assert exchange.market_is_tradable(markets_hip3["BTC/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["ETH/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["XYZ-AAPL/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["XYZ-TSLA/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["VNTL-SPACEX/USDH:USDH"]) is False
    assert exchange.market_is_tradable(markets_hip3["FLX-TOKEN/USDC:USDC"]) is False

    # Test 3: 'xyz' and 'vntl' configured - default + xyz + vntl markets tradable
    default_conf_usdt["exchange"]["hip3_dexes"] = ["xyz", "flx"]
    exchange = get_patched_exchange(
        mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=False
    )

    assert exchange.market_is_tradable(markets_hip3["BTC/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["ETH/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["XYZ-AAPL/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["XYZ-TSLA/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["VNTL-SPACEX/USDH:USDH"]) is False
    assert exchange.market_is_tradable(markets_hip3["FLX-TOKEN/USDC:USDC"]) is True

    # Use USDH stake currency to enable VNTL markets
    default_conf_usdt["exchange"]["hip3_dexes"] = ["vntl"]
    default_conf_usdt["stake_currency"] = "USDH"
    exchange = get_patched_exchange(
        mocker, default_conf_usdt, api_mock, exchange="hyperliquid", mock_markets=False
    )
    assert exchange.market_is_tradable(markets_hip3["BTC/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["ETH/USDC:USDC"]) is True
    assert exchange.market_is_tradable(markets_hip3["XYZ-AAPL/USDC:USDC"]) is False
    assert exchange.market_is_tradable(markets_hip3["XYZ-TSLA/USDC:USDC"]) is False
    assert exchange.market_is_tradable(markets_hip3["VNTL-SPACEX/USDH:USDH"]) is True
    assert exchange.market_is_tradable(markets_hip3["FLX-TOKEN/USDC:USDC"]) is False
