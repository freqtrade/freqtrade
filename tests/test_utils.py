import re
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from freqtrade import OperationalException
from freqtrade.state import RunMode
from freqtrade.utils import (setup_utils_configuration, start_create_userdir,
                             start_download_data, start_list_exchanges,
                             start_list_markets, start_list_timeframes)
from tests.conftest import get_args, log_has, patch_exchange


def test_setup_utils_configuration():
    args = [
        'list-exchanges', '--config', 'config.json.example',
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
    assert re.match(r"Exchanges available for Freqtrade.*", captured.out)
    assert re.match(r".*binance,.*", captured.out)
    assert re.match(r".*bittrex,.*", captured.out)

    # Test with --one-column
    args = [
        "list-exchanges",
        "--one-column",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^binance$", captured.out, re.MULTILINE)
    assert re.search(r"^bittrex$", captured.out, re.MULTILINE)

    # Test with --all
    args = [
        "list-exchanges",
        "--all",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.match(r"All exchanges supported by the ccxt library.*", captured.out)
    assert re.match(r".*binance,.*", captured.out)
    assert re.match(r".*bittrex,.*", captured.out)
    assert re.match(r".*bitmex,.*", captured.out)

    # Test with --one-column --all
    args = [
        "list-exchanges",
        "--one-column",
        "--all",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^binance$", captured.out, re.MULTILINE)
    assert re.search(r"^bittrex$", captured.out, re.MULTILINE)
    assert re.search(r"^bitmex$", captured.out, re.MULTILINE)


def test_list_timeframes(mocker, capsys):

    api_mock = MagicMock()
    api_mock.timeframes = {'1m': 'oneMin',
                           '5m': 'fiveMin',
                           '30m': 'thirtyMin',
                           '1h': 'hour',
                           '1d': 'day',
                           }
    patch_exchange(mocker, api_mock=api_mock)
    args = [
        "list-timeframes",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"This command requires a configured exchange.*"):
        start_list_timeframes(pargs)

    # Test with --config config.json.example
    args = [
        "list-timeframes",
        '--config', 'config.json.example',
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match("Timeframes available for the exchange `Bittrex`: "
                    "1m, 5m, 30m, 1h, 1d",
                    captured.out)

    # Test with --exchange bittrex
    args = [
        "list-timeframes",
        "--exchange", "bittrex",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match("Timeframes available for the exchange `Bittrex`: "
                    "1m, 5m, 30m, 1h, 1d",
                    captured.out)

    api_mock.timeframes = {'1m': '1m',
                           '5m': '5m',
                           '15m': '15m',
                           '30m': '30m',
                           '1h': '1h',
                           '6h': '6h',
                           '12h': '12h',
                           '1d': '1d',
                           '3d': '3d',
                           }
    patch_exchange(mocker, api_mock=api_mock, id='binance')
    # Test with --exchange binance
    args = [
        "list-timeframes",
        "--exchange", "binance",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.match("Timeframes available for the exchange `Binance`: "
                    "1m, 5m, 15m, 30m, 1h, 6h, 12h, 1d, 3d",
                    captured.out)

    # Test with --one-column
    args = [
        "list-timeframes",
        '--config', 'config.json.example',
        "--one-column",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^1m$", captured.out, re.MULTILINE)
    assert re.search(r"^5m$", captured.out, re.MULTILINE)
    assert re.search(r"^1h$", captured.out, re.MULTILINE)
    assert re.search(r"^1d$", captured.out, re.MULTILINE)

    # Test with --exchange binance --one-column
    args = [
        "list-timeframes",
        "--exchange", "binance",
        "--one-column",
    ]
    start_list_timeframes(get_args(args))
    captured = capsys.readouterr()
    assert re.search(r"^1m$", captured.out, re.MULTILINE)
    assert re.search(r"^5m$", captured.out, re.MULTILINE)
    assert re.search(r"^1h$", captured.out, re.MULTILINE)
    assert re.search(r"^1d$", captured.out, re.MULTILINE)


def test_list_markets(mocker, markets, capsys):

    api_mock = MagicMock()
    api_mock.markets = markets
    patch_exchange(mocker, api_mock=api_mock)

    # Test with no --config
    args = [
        "list-markets",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"This command requires a configured exchange.*"):
        start_list_markets(pargs, False)

    # Test with --config config.json.example
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 9 active markets: "
            "BLK/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/USD, NEO/BTC, TKN/BTC, XLTCUSDT, XRP/BTC.\n"
            in captured.out)

    patch_exchange(mocker, api_mock=api_mock, id="binance")
    # Test with --exchange
    args = [
        "list-markets",
        "--exchange", "binance"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_list_markets(pargs, False)
    captured = capsys.readouterr()
    assert re.match("\nExchange Binance has 9 active markets:\n",
                    captured.out)

    patch_exchange(mocker, api_mock=api_mock, id="bittrex")
    # Test with --all: all markets
    args = [
        "list-markets", "--all",
        '--config', 'config.json.example',
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 11 markets: "
            "BLK/BTC, BTT/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/USD, LTC/USDT, NEO/BTC, "
            "TKN/BTC, XLTCUSDT, XRP/BTC.\n"
            in captured.out)

    # Test list-pairs subcommand: active pairs
    args = [
        "list-pairs",
        '--config', 'config.json.example',
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 8 active pairs: "
            "BLK/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/USD, NEO/BTC, TKN/BTC, XRP/BTC.\n"
            in captured.out)

    # Test list-pairs subcommand with --all: all pairs
    args = [
        "list-pairs", "--all",
        '--config', 'config.json.example',
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 10 pairs: "
            "BLK/BTC, BTT/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/USD, LTC/USDT, NEO/BTC, "
            "TKN/BTC, XRP/BTC.\n"
            in captured.out)

    # active markets, base=ETH, LTC
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "ETH", "LTC",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 5 active markets with ETH, LTC as base currencies: "
            "ETH/BTC, ETH/USDT, LTC/BTC, LTC/USD, XLTCUSDT.\n"
            in captured.out)

    # active markets, base=LTC
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "LTC",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 3 active markets with LTC as base currency: "
            "LTC/BTC, LTC/USD, XLTCUSDT.\n"
            in captured.out)

    # active markets, quote=USDT, USD
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--quote", "USDT", "USD",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 3 active markets with USDT, USD as quote currencies: "
            "ETH/USDT, LTC/USD, XLTCUSDT.\n"
            in captured.out)

    # active markets, quote=USDT
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--quote", "USDT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 2 active markets with USDT as quote currency: "
            "ETH/USDT, XLTCUSDT.\n"
            in captured.out)

    # active markets, base=LTC, quote=USDT
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "LTC", "--quote", "USDT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 1 active market with LTC as base currency and "
            "with USDT as quote currency: XLTCUSDT.\n"
            in captured.out)

    # active pairs, base=LTC, quote=USDT
    args = [
        "list-pairs",
        '--config', 'config.json.example',
        "--base", "LTC", "--quote", "USD",
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 1 active pair with LTC as base currency and "
            "with USD as quote currency: LTC/USD.\n"
            in captured.out)

    # active markets, base=LTC, quote=USDT, NONEXISTENT
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "LTC", "--quote", "USDT", "NONEXISTENT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 1 active market with LTC as base currency and "
            "with USDT, NONEXISTENT as quote currencies: XLTCUSDT.\n"
            in captured.out)

    # active markets, base=LTC, quote=NONEXISTENT
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "LTC", "--quote", "NONEXISTENT",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 0 active markets with LTC as base currency and "
            "with NONEXISTENT as quote currency.\n"
            in captured.out)

    # Test tabular output
    args = [
        "list-markets",
        '--config', 'config.json.example',
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 9 active markets:\n"
            in captured.out)

    # Test tabular output, no markets found
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--base", "LTC", "--quote", "NONEXISTENT",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 0 active markets with LTC as base currency and "
            "with NONEXISTENT as quote currency.\n"
            in captured.out)

    # Test --print-json
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--print-json"
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ('["BLK/BTC","ETH/BTC","ETH/USDT","LTC/BTC","LTC/USD","NEO/BTC",'
            '"TKN/BTC","XLTCUSDT","XRP/BTC"]'
            in captured.out)

    # Test --print-csv
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--print-csv"
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Id,Symbol,Base,Quote,Active,Is pair" in captured.out)
    assert ("blkbtc,BLK/BTC,BLK,BTC,True,True" in captured.out)
    assert ("USD-LTC,LTC/USD,LTC,USD,True,True" in captured.out)

    # Test --one-column
    args = [
        "list-markets",
        '--config', 'config.json.example',
        "--one-column"
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert re.search(r"^BLK/BTC$", captured.out, re.MULTILINE)
    assert re.search(r"^LTC/USD$", captured.out, re.MULTILINE)


def test_create_datadir_failed(caplog):

    args = [
        "create-userdir",
    ]
    with pytest.raises(SystemExit):
        start_create_userdir(get_args(args))
    assert log_has("`create-userdir` requires --userdir to be set.", caplog)


def test_create_datadir(caplog, mocker):
    cud = mocker.patch("freqtrade.utils.create_userdata_dir", MagicMock())
    args = [
        "create-userdir",
        "--userdir",
        "/temp/freqtrade/test"
    ]
    start_create_userdir(get_args(args))

    assert cud.call_count == 1
    assert len(caplog.record_tuples) == 0


def test_download_data_keyboardInterrupt(mocker, caplog, markets):
    dl_mock = mocker.patch('freqtrade.utils.refresh_backtest_ohlcv_data',
                           MagicMock(side_effect=KeyboardInterrupt))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets)
    )
    args = [
        "download-data",
        "--exchange", "binance",
        "--pairs", "ETH/BTC", "XRP/BTC",
    ]
    with pytest.raises(SystemExit):
        start_download_data(get_args(args))

    assert dl_mock.call_count == 1


def test_download_data_no_markets(mocker, caplog):
    dl_mock = mocker.patch('freqtrade.utils.refresh_backtest_ohlcv_data',
                           MagicMock(return_value=["ETH/BTC", "XRP/BTC"]))
    patch_exchange(mocker, id='binance')
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    args = [
        "download-data",
        "--exchange", "binance",
        "--pairs", "ETH/BTC", "XRP/BTC",
        "--days", "20"
    ]
    start_download_data(get_args(args))
    assert dl_mock.call_args[1]['timerange'].starttype == "date"
    assert log_has("Pairs [ETH/BTC,XRP/BTC] not available on exchange Binance.", caplog)


def test_download_data_no_exchange(mocker, caplog):
    mocker.patch('freqtrade.utils.refresh_backtest_ohlcv_data',
                 MagicMock(return_value=["ETH/BTC", "XRP/BTC"]))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    args = [
        "download-data",
        ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"This command requires a configured exchange.*"):
        start_download_data(pargs)


def test_download_data_no_pairs(mocker, caplog):

    mocker.patch.object(Path, "exists", MagicMock(return_value=False))

    mocker.patch('freqtrade.utils.refresh_backtest_ohlcv_data',
                 MagicMock(return_value=["ETH/BTC", "XRP/BTC"]))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    args = [
        "download-data",
        "--exchange",
        "binance",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"Downloading data requires a list of pairs\..*"):
        start_download_data(pargs)


def test_download_data_trades(mocker, caplog):
    dl_mock = mocker.patch('freqtrade.utils.refresh_backtest_trades_data',
                           MagicMock(return_value=[]))
    convert_mock = mocker.patch('freqtrade.utils.convert_trades_to_ohlcv',
                                MagicMock(return_value=[]))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    args = [
        "download-data",
        "--exchange", "kraken",
        "--pairs", "ETH/BTC", "XRP/BTC",
        "--days", "20",
        "--dl-trades"
    ]
    start_download_data(get_args(args))
    assert dl_mock.call_args[1]['timerange'].starttype == "date"
    assert dl_mock.call_count == 1
    assert convert_mock.call_count == 1
