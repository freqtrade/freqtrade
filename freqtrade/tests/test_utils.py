import re
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

from freqtrade.state import RunMode
from freqtrade.tests.conftest import get_args, log_has, patch_exchange
from freqtrade.utils import (setup_utils_configuration, start_download_data,
                             start_list_exchanges)


def test_setup_utils_configuration():
    args = [
        '--config', 'config.json.example',
    ]

    config = setup_utils_configuration(get_args(args), RunMode.OTHER)
    assert "exchange" in config
    assert config['exchange']['dry_run'] is True
    assert config['exchange']['key'] == ''
    assert config['exchange']['secret'] == ''


def test_list_exchanges(capsys):

    args = [
        "list-exchanges",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.match(r"Exchanges supported by ccxt and available.*", captured.out)
    assert re.match(r".*binance,.*", captured.out)
    assert re.match(r".*bittrex,.*", captured.out)

    # Test with --one-column
    args = [
        "list-exchanges",
        "--one-column",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert not re.match(r"Exchanges supported by ccxt and available.*", captured.out)
    assert re.search(r"^binance$", captured.out, re.MULTILINE)
    assert re.search(r"^bittrex$", captured.out, re.MULTILINE)


def test_download_data(mocker, markets, caplog):
    dl_mock = mocker.patch('freqtrade.utils.download_pair_history', MagicMock())
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets)
    )
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch.object(Path, "unlink", MagicMock())

    args = [
        "download-data",
        "--exchange", "binance",
        "--pairs", "ETH/BTC", "XRP/BTC",
        "--erase"
    ]
    start_download_data(get_args(args))

    assert dl_mock.call_count == 4
    assert log_has("Deleting existing data for pair ETH/BTC, interval 1m.", caplog)
    assert log_has("Downloading pair ETH/BTC, interval 1m.", caplog)


def test_download_data_no_markets(mocker, caplog):
    dl_mock = mocker.patch('freqtrade.utils.download_pair_history', MagicMock())
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    args = [
        "download-data",
        "--exchange", "binance",
        "--pairs", "ETH/BTC", "XRP/BTC"
    ]
    start_download_data(get_args(args))

    assert dl_mock.call_count == 0
    assert log_has("Skipping pair ETH/BTC...", caplog)
    assert log_has("Pairs [ETH/BTC,XRP/BTC] not available on exchange binance.", caplog)
