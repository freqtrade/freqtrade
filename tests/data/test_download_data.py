from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade.configuration.config_setup import setup_utils_configuration
from freqtrade.data.history.history_utils import download_data_main
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from tests.conftest import EXMS, log_has, patch_exchange


def test_download_data_main_no_markets(mocker, caplog):
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data",
        MagicMock(return_value=["ETH/BTC", "XRP/BTC"]),
    )
    patch_exchange(mocker, exchange="binance")
    mocker.patch(f"{EXMS}.get_markets", return_value={})
    config = setup_utils_configuration({"exchange": "binance"}, RunMode.UTIL_EXCHANGE)
    config.update({"days": 20, "pairs": ["ETH/BTC", "XRP/BTC"], "timeframes": ["5m", "1h"]})
    download_data_main(config)
    assert dl_mock.call_args[1]["timerange"].starttype == "date"
    assert log_has("Pairs [ETH/BTC,XRP/BTC] not available on exchange Binance.", caplog)


def test_download_data_main_all_pairs(mocker, markets):
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_ohlcv_data",
        MagicMock(return_value=["ETH/BTC", "XRP/BTC"]),
    )
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))

    config = setup_utils_configuration({"exchange": "binance"}, RunMode.UTIL_EXCHANGE)
    config.update({"pairs": [".*/USDT"], "timeframes": ["5m", "1h"]})
    download_data_main(config)
    expected = set(["BTC/USDT", "ETH/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"])
    assert set(dl_mock.call_args_list[0][1]["pairs"]) == expected
    assert dl_mock.call_count == 1

    dl_mock.reset_mock()

    config.update({"pairs": [".*/USDT"], "timeframes": ["5m", "1h"], "include_inactive": True})
    download_data_main(config)
    expected = set(["BTC/USDT", "ETH/USDT", "LTC/USDT", "XRP/USDT", "NEO/USDT", "TKN/USDT"])
    assert set(dl_mock.call_args_list[0][1]["pairs"]) == expected


def test_download_data_main_trades(mocker):
    dl_mock = mocker.patch(
        "freqtrade.data.history.history_utils.refresh_backtest_trades_data",
        MagicMock(return_value=[]),
    )
    convert_mock = mocker.patch(
        "freqtrade.data.history.history_utils.convert_trades_to_ohlcv", MagicMock(return_value=[])
    )
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_markets", return_value={})
    config = setup_utils_configuration({"exchange": "binance"}, RunMode.UTIL_EXCHANGE)
    config.update(
        {
            "days": 20,
            "pairs": ["ETH/BTC", "XRP/BTC"],
            "timeframes": ["5m", "1h"],
            "download_trades": True,
        }
    )
    download_data_main(config)

    assert dl_mock.call_args[1]["timerange"].starttype == "date"
    assert dl_mock.call_count == 1
    assert convert_mock.call_count == 0
    dl_mock.reset_mock()

    config.update(
        {
            "convert_trades": True,
        }
    )
    download_data_main(config)

    assert dl_mock.call_args[1]["timerange"].starttype == "date"
    assert dl_mock.call_count == 1
    assert convert_mock.call_count == 1

    # Exchange that doesn't support historic downloads
    config["exchange"]["name"] = "bybit"
    with pytest.raises(OperationalException, match=r"Trade history not available for .*"):
        download_data_main(config)


def test_download_data_main_data_invalid(mocker):
    patch_exchange(mocker, exchange="kraken")
    mocker.patch(f"{EXMS}.get_markets", return_value={})
    config = setup_utils_configuration({"exchange": "kraken"}, RunMode.UTIL_EXCHANGE)
    config.update(
        {
            "days": 20,
            "pairs": ["ETH/BTC", "XRP/BTC"],
            "timeframes": ["5m", "1h"],
        }
    )
    with pytest.raises(OperationalException, match=r"Historic klines not available for .*"):
        download_data_main(config)
