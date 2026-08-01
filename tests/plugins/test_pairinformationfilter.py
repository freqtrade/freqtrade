from copy import deepcopy

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import get_markets, get_patched_exchange


@pytest.fixture(scope="function")
def pif_config(default_conf_usdt):

    default_conf_usdt["exchange"]["pair_whitelist"] = [
        "ETH/USDT",
        "XRP/USDT",
        "BTC/USDT",
    ]
    default_conf_usdt["exchange"]["pair_blacklist"] = ["BLK/USDT"]

    return default_conf_usdt


@pytest.mark.parametrize(
    "missing_key,error_msg",
    [
        ("info_key", "`info_key` not specified"),
        ("info_compare_value", "`info_compare_value` not specified"),
        ("selection_mode", "`selection_mode` not configured correctly"),
    ],
)
def test_PairInformationFilter_validation(mocker, pif_config, missing_key, error_msg):

    pif_config["pairlists"] = [
        {
            "method": "PairInformationFilter",
            "selection_mode": "whitelist",
            "info_key": "info.contractType",
            "info_compare_value": "TRADIFI_PERPETUAL",
            "refresh_period": 1800,
        }
    ]
    exchange = get_patched_exchange(mocker, pif_config)

    with pytest.raises(OperationalException, match=error_msg):
        if missing_key == "selection_mode":
            pif_config["pairlists"][0]["selection_mode"] = "invalid_mode"
        else:
            pif_config["pairlists"][0].pop(missing_key)
        PairListManager(exchange, pif_config)


def test_PairInformationFilter_filter_float(mocker, pif_config):

    pif_config["pairlists"] = [
        {
            "method": "StaticPairList",
        },
        {
            "method": "PairInformationFilter",
            "selection_mode": "whitelist",
            "info_key": "limits.amount.min",
            "info_compare_value": 0.01,
            "refresh_period": 1800,
        },
    ]
    exchange = get_patched_exchange(mocker, pif_config)
    pairlist_manager = PairListManager(exchange, pif_config)

    pairlist_manager.refresh_pairlist()
    pairlist = pairlist_manager.whitelist
    assert set(pairlist) == {"XRP/USDT"}


def _get_pairinformation_test_markets() -> dict:
    markets = get_markets()
    custom_markets = {
        pair: deepcopy(markets[pair]) for pair in ["ETH/USDT", "XRP/USDT", "BTC/USDT"]
    }
    custom_markets["ETH/USDT"]["info"] = {"contractType": "TRADIFI_PERPETUAL"}
    custom_markets["XRP/USDT"]["info"] = {}
    custom_markets["BTC/USDT"]["info"] = {"contractType": "CURRENT_QUARTER"}
    return custom_markets


def test_PairInformationFilter_filter_nested_info_string_whitelist(mocker, pif_config):
    markets = _get_pairinformation_test_markets()

    pif_config["pairlists"] = [
        {
            "method": "StaticPairList",
        },
        {
            "method": "PairInformationFilter",
            "selection_mode": "whitelist",
            "info_key": "info.contractType",
            "info_compare_value": "TRADIFI_PERPETUAL",
            "refresh_period": 1800,
        },
    ]
    exchange = get_patched_exchange(mocker, pif_config, mock_markets=markets)
    pairlist_manager = PairListManager(exchange, pif_config)

    pairlist_manager.refresh_pairlist()
    pairlist = pairlist_manager.whitelist
    assert pairlist == ["ETH/USDT"]


def test_PairInformationFilter_filter_nested_info_string_blacklist(mocker, pif_config):
    markets = _get_pairinformation_test_markets()

    pif_config["pairlists"] = [
        {
            "method": "StaticPairList",
        },
        {
            "method": "PairInformationFilter",
            "selection_mode": "blacklist",
            "info_key": "info.contractType",
            "info_compare_value": "TRADIFI_PERPETUAL",
            "refresh_period": 1800,
        },
    ]
    exchange = get_patched_exchange(mocker, pif_config, mock_markets=markets)
    pairlist_manager = PairListManager(exchange, pif_config)

    pairlist_manager.refresh_pairlist()
    pairlist = pairlist_manager.whitelist
    assert pairlist == ["XRP/USDT", "BTC/USDT"]


def test_PairInformationFilter_filter_nested_info_combi(mocker, pif_config):
    markets = _get_pairinformation_test_markets()

    pif_config["pairlists"] = [
        {
            "method": "StaticPairList",
        },
        {
            "method": "PairInformationFilter",
            "selection_mode": "blacklist",
            "info_key": "info.contractType",
            "info_compare_value": "TRADIFI_PERPETUAL",
            "refresh_period": 1800,
        },
        {
            "method": "PairInformationFilter",
            "selection_mode": "whitelist",
            "info_key": "info.contractType",
            "info_compare_value": "CURRENT_QUARTER",
            "refresh_period": 1800,
        },
    ]
    exchange = get_patched_exchange(mocker, pif_config, mock_markets=markets)
    pairlist_manager = PairListManager(exchange, pif_config)

    pairlist_manager.refresh_pairlist()
    pairlist = pairlist_manager.whitelist
    assert pairlist == ["BTC/USDT"]

    desc = pairlist_manager._pairlist_handlers[1].description()
    assert desc == "Filter pairs based upon any information in their market data."
    short_desc = pairlist_manager._pairlist_handlers[1].short_desc()
    assert short_desc == (
        "PairInformationFilter - Returns blacklist pairs by comparing "
        "info.contractType matches TRADIFI_PERPETUAL."
    )
    short_desc2 = pairlist_manager._pairlist_handlers[2].short_desc()
    assert short_desc2 == (
        "PairInformationFilter - Returns whitelist pairs by comparing "
        "info.contractType matches CURRENT_QUARTER."
    )
