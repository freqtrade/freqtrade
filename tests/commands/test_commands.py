import json
import re
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock
from zipfile import ZipFile

import arrow
import pytest

from freqtrade.commands import (start_backtesting_show, start_convert_data, start_convert_trades,
                                start_create_userdir, start_download_data, start_hyperopt_list,
                                start_hyperopt_show, start_install_ui, start_list_data,
                                start_list_exchanges, start_list_markets, start_list_strategies,
                                start_list_timeframes, start_new_strategy, start_show_trades,
                                start_test_pairlist, start_trading, start_webserver)
from freqtrade.commands.deploy_commands import (clean_ui_subdir, download_and_install_ui,
                                                get_ui_download_url, read_ui_version)
from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from tests.conftest import (create_mock_trades, get_args, log_has, log_has_re, patch_exchange,
                            patched_configuration_load_config_file)
from tests.conftest_trades import MOCK_TRADE_COUNT


def test_setup_utils_configuration():
    args = [
        'list-exchanges', '--config', 'config_examples/config_bittrex.example.json',
    ]

    config = setup_utils_configuration(get_args(args), RunMode.OTHER)
    assert "exchange" in config
    assert config['dry_run'] is True


def test_start_trading_fail(mocker, caplog):

    mocker.patch("freqtrade.worker.Worker.run", MagicMock(side_effect=OperationalException))

    mocker.patch("freqtrade.worker.Worker.__init__", MagicMock(return_value=None))

    exitmock = mocker.patch("freqtrade.worker.Worker.exit", MagicMock())
    args = [
        'trade',
        '-c', 'config_examples/config_bittrex.example.json'
    ]
    start_trading(get_args(args))
    assert exitmock.call_count == 1

    exitmock.reset_mock()
    caplog.clear()
    mocker.patch("freqtrade.worker.Worker.__init__", MagicMock(side_effect=OperationalException))
    start_trading(get_args(args))
    assert exitmock.call_count == 0
    assert log_has('Fatal exception!', caplog)


def test_start_webserver(mocker, caplog):

    api_server_mock = mocker.patch("freqtrade.rpc.api_server.ApiServer", )

    args = [
        'webserver',
        '-c', 'config_examples/config_bittrex.example.json'
    ]
    start_webserver(get_args(args))
    assert api_server_mock.call_count == 1


def test_list_exchanges(capsys):

    args = [
        "list-exchanges",
    ]

    start_list_exchanges(get_args(args))
    captured = capsys.readouterr()
    assert re.match(r"Exchanges available for Freqtrade.*", captured.out)
    assert re.search(r".*binance.*", captured.out)
    assert re.search(r".*bittrex.*", captured.out)

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
    assert re.search(r".*binance.*", captured.out)
    assert re.search(r".*bittrex.*", captured.out)
    assert re.search(r".*bitmex.*", captured.out)

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
    patch_exchange(mocker, api_mock=api_mock, id='bittrex')
    args = [
        "list-timeframes",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"This command requires a configured exchange.*"):
        start_list_timeframes(pargs)

    # Test with --config config_examples/config_bittrex.example.json
    args = [
        "list-timeframes",
        '--config', 'config_examples/config_bittrex.example.json',
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
        '--config', 'config_examples/config_bittrex.example.json',
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


def test_list_markets(mocker, markets_static, capsys):

    api_mock = MagicMock()
    patch_exchange(mocker, api_mock=api_mock, id='bittrex', mock_markets=markets_static)

    # Test with no --config
    args = [
        "list-markets",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match=r"This command requires a configured exchange.*"):
        start_list_markets(pargs, False)

    # Test with --config config_examples/config_bittrex.example.json
    args = [
        "list-markets",
        '--config', 'config_examples/config_bittrex.example.json',
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 10 active markets: "
            "BLK/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/ETH, LTC/USD, NEO/BTC, "
            "TKN/BTC, XLTCUSDT, XRP/BTC.\n"
            in captured.out)

    patch_exchange(mocker, api_mock=api_mock, id="binance", mock_markets=markets_static)
    # Test with --exchange
    args = [
        "list-markets",
        "--exchange", "binance"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_list_markets(pargs, False)
    captured = capsys.readouterr()
    assert re.match("\nExchange Binance has 10 active markets:\n",
                    captured.out)

    patch_exchange(mocker, api_mock=api_mock, id="bittrex", mock_markets=markets_static)
    # Test with --all: all markets
    args = [
        "list-markets", "--all",
        '--config', 'config_examples/config_bittrex.example.json',
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 12 markets: "
            "BLK/BTC, BTT/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/ETH, LTC/USD, LTC/USDT, NEO/BTC, "
            "TKN/BTC, XLTCUSDT, XRP/BTC.\n"
            in captured.out)

    # Test list-pairs subcommand: active pairs
    args = [
        "list-pairs",
        '--config', 'config_examples/config_bittrex.example.json',
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 9 active pairs: "
            "BLK/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/ETH, LTC/USD, NEO/BTC, TKN/BTC, XRP/BTC.\n"
            in captured.out)

    # Test list-pairs subcommand with --all: all pairs
    args = [
        "list-pairs", "--all",
        '--config', 'config_examples/config_bittrex.example.json',
        "--print-list",
    ]
    start_list_markets(get_args(args), True)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 11 pairs: "
            "BLK/BTC, BTT/BTC, ETH/BTC, ETH/USDT, LTC/BTC, LTC/ETH, LTC/USD, LTC/USDT, NEO/BTC, "
            "TKN/BTC, XRP/BTC.\n"
            in captured.out)

    # active markets, base=ETH, LTC
    args = [
        "list-markets",
        '--config', 'config_examples/config_bittrex.example.json',
        "--base", "ETH", "LTC",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 6 active markets with ETH, LTC as base currencies: "
            "ETH/BTC, ETH/USDT, LTC/BTC, LTC/ETH, LTC/USD, XLTCUSDT.\n"
            in captured.out)

    # active markets, base=LTC
    args = [
        "list-markets",
        '--config', 'config_examples/config_bittrex.example.json',
        "--base", "LTC",
        "--print-list",
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 4 active markets with LTC as base currency: "
            "LTC/BTC, LTC/ETH, LTC/USD, XLTCUSDT.\n"
            in captured.out)

    # active markets, quote=USDT, USD
    args = [
        "list-markets",
        '--config', 'config_examples/config_bittrex.example.json',
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
        '--config', 'config_examples/config_bittrex.example.json',
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
        '--config', 'config_examples/config_bittrex.example.json',
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
        '--config', 'config_examples/config_bittrex.example.json',
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
        '--config', 'config_examples/config_bittrex.example.json',
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
        '--config', 'config_examples/config_bittrex.example.json',
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
        '--config', 'config_examples/config_bittrex.example.json',
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ("Exchange Bittrex has 10 active markets:\n"
            in captured.out)

    # Test tabular output, no markets found
    args = [
        "list-markets",
        '--config', 'config_examples/config_bittrex.example.json',
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
        '--config', 'config_examples/config_bittrex.example.json',
        "--print-json"
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert ('["BLK/BTC","ETH/BTC","ETH/USDT","LTC/BTC","LTC/ETH","LTC/USD","NEO/BTC",'
            '"TKN/BTC","XLTCUSDT","XRP/BTC"]'
            in captured.out)

    # Test --print-csv
    args = [
        "list-markets",
        '--config', 'config_examples/config_bittrex.example.json',
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
        '--config', 'config_examples/config_bittrex.example.json',
        "--one-column"
    ]
    start_list_markets(get_args(args), False)
    captured = capsys.readouterr()
    assert re.search(r"^BLK/BTC$", captured.out, re.MULTILINE)
    assert re.search(r"^LTC/USD$", captured.out, re.MULTILINE)

    mocker.patch('freqtrade.exchange.Exchange.markets', PropertyMock(side_effect=ValueError))
    # Test --one-column
    args = [
        "list-markets",
        '--config', 'config_examples/config_bittrex.example.json',
        "--one-column"
    ]
    with pytest.raises(OperationalException, match=r"Cannot get markets.*"):
        start_list_markets(get_args(args), False)


def test_create_datadir_failed(caplog):

    args = [
        "create-userdir",
    ]
    with pytest.raises(SystemExit):
        start_create_userdir(get_args(args))
    assert log_has("`create-userdir` requires --userdir to be set.", caplog)


def test_create_datadir(caplog, mocker):

    cud = mocker.patch("freqtrade.commands.deploy_commands.create_userdata_dir", MagicMock())
    csf = mocker.patch("freqtrade.commands.deploy_commands.copy_sample_files", MagicMock())
    args = [
        "create-userdir",
        "--userdir",
        "/temp/freqtrade/test"
    ]
    start_create_userdir(get_args(args))

    assert cud.call_count == 1
    assert csf.call_count == 1


def test_start_new_strategy(mocker, caplog):
    wt_mock = mocker.patch.object(Path, "write_text", MagicMock())
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))

    args = [
        "new-strategy",
        "--strategy",
        "CoolNewStrategy"
    ]
    start_new_strategy(get_args(args))

    assert wt_mock.call_count == 1
    assert "CoolNewStrategy" in wt_mock.call_args_list[0][0][0]
    assert log_has_re("Writing strategy to .*", caplog)

    mocker.patch('freqtrade.commands.deploy_commands.setup_utils_configuration')
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    with pytest.raises(OperationalException,
                       match=r".* already exists. Please choose another Strategy Name\."):
        start_new_strategy(get_args(args))


def test_start_new_strategy_no_arg(mocker, caplog):
    args = [
        "new-strategy",
    ]
    with pytest.raises(OperationalException,
                       match="`new-strategy` requires --strategy to be set."):
        start_new_strategy(get_args(args))


def test_start_install_ui(mocker):
    clean_mock = mocker.patch('freqtrade.commands.deploy_commands.clean_ui_subdir')
    get_url_mock = mocker.patch('freqtrade.commands.deploy_commands.get_ui_download_url',
                                return_value=('https://example.com/whatever', '0.0.1'))
    download_mock = mocker.patch('freqtrade.commands.deploy_commands.download_and_install_ui')
    mocker.patch('freqtrade.commands.deploy_commands.read_ui_version', return_value=None)
    args = [
        "install-ui",
    ]
    start_install_ui(get_args(args))
    assert clean_mock.call_count == 1
    assert get_url_mock.call_count == 1
    assert download_mock.call_count == 1

    clean_mock.reset_mock()
    get_url_mock.reset_mock()
    download_mock.reset_mock()

    args = [
        "install-ui",
        "--erase",
    ]
    start_install_ui(get_args(args))
    assert clean_mock.call_count == 1
    assert get_url_mock.call_count == 1
    assert download_mock.call_count == 0


def test_clean_ui_subdir(mocker, tmpdir, caplog):
    mocker.patch("freqtrade.commands.deploy_commands.Path.is_dir",
                 side_effect=[True, True])
    mocker.patch("freqtrade.commands.deploy_commands.Path.is_file",
                 side_effect=[False, True])
    rd_mock = mocker.patch("freqtrade.commands.deploy_commands.Path.rmdir")
    ul_mock = mocker.patch("freqtrade.commands.deploy_commands.Path.unlink")

    mocker.patch("freqtrade.commands.deploy_commands.Path.glob",
                 return_value=[Path('test1'), Path('test2'), Path('.gitkeep')])
    folder = Path(tmpdir) / "uitests"
    clean_ui_subdir(folder)
    assert log_has("Removing UI directory content.", caplog)
    assert rd_mock.call_count == 1
    assert ul_mock.call_count == 1


def test_download_and_install_ui(mocker, tmpdir):
    # Create zipfile
    requests_mock = MagicMock()
    file_like_object = BytesIO()
    with ZipFile(file_like_object, mode='w') as zipfile:
        for file in ('test1.txt', 'hello/', 'test2.txt'):
            zipfile.writestr(file, file)
    file_like_object.seek(0)
    requests_mock.content = file_like_object.read()

    mocker.patch("freqtrade.commands.deploy_commands.requests.get", return_value=requests_mock)

    mocker.patch("freqtrade.commands.deploy_commands.Path.is_dir",
                 side_effect=[True, False])
    wb_mock = mocker.patch("freqtrade.commands.deploy_commands.Path.write_bytes")

    folder = Path(tmpdir) / "uitests_dl"
    folder.mkdir(exist_ok=True)

    assert read_ui_version(folder) is None

    download_and_install_ui(folder, 'http://whatever.xxx/download/file.zip', '22')

    assert wb_mock.call_count == 2

    assert read_ui_version(folder) == '22'


def test_get_ui_download_url(mocker):
    response = MagicMock()
    response.json = MagicMock(
        side_effect=[[{'assets_url': 'http://whatever.json', 'name': '0.0.1'}],
                     [{'browser_download_url': 'http://download.zip'}]])
    get_mock = mocker.patch("freqtrade.commands.deploy_commands.requests.get",
                            return_value=response)
    x, last_version = get_ui_download_url()
    assert get_mock.call_count == 2
    assert last_version == '0.0.1'
    assert x == 'http://download.zip'


def test_get_ui_download_url_direct(mocker):
    response = MagicMock()
    response.json = MagicMock(
        return_value=[
            {
                'assets_url': 'http://whatever.json',
                'name': '0.0.2',
                'assets': [{'browser_download_url': 'http://download22.zip'}]
            },
            {
                'assets_url': 'http://whatever.json',
                'name': '0.0.1',
                'assets': [{'browser_download_url': 'http://download1.zip'}]
            },
        ])
    get_mock = mocker.patch("freqtrade.commands.deploy_commands.requests.get",
                            return_value=response)
    x, last_version = get_ui_download_url()
    assert get_mock.call_count == 1
    assert last_version == '0.0.2'
    assert x == 'http://download22.zip'
    get_mock.reset_mock()
    response.json.reset_mock()

    x, last_version = get_ui_download_url('0.0.1')
    assert last_version == '0.0.1'
    assert x == 'http://download1.zip'

    with pytest.raises(ValueError, match="UI-Version not found."):
        x, last_version = get_ui_download_url('0.0.3')


def test_download_data_keyboardInterrupt(mocker, caplog, markets):
    dl_mock = mocker.patch('freqtrade.commands.data_commands.refresh_backtest_ohlcv_data',
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


def test_download_data_timerange(mocker, caplog, markets):
    dl_mock = mocker.patch('freqtrade.commands.data_commands.refresh_backtest_ohlcv_data',
                           MagicMock(return_value=["ETH/BTC", "XRP/BTC"]))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets)
    )
    args = [
        "download-data",
        "--exchange", "binance",
        "--pairs", "ETH/BTC", "XRP/BTC",
        "--days", "20",
        "--timerange", "20200101-"
    ]
    with pytest.raises(OperationalException,
                       match=r"--days and --timerange are mutually.*"):
        start_download_data(get_args(args))
    assert dl_mock.call_count == 0

    args = [
        "download-data",
        "--exchange", "binance",
        "--pairs", "ETH/BTC", "XRP/BTC",
        "--days", "20",
    ]
    start_download_data(get_args(args))
    assert dl_mock.call_count == 1
    # 20days ago
    days_ago = arrow.get(arrow.now().shift(days=-20).date()).int_timestamp
    assert dl_mock.call_args_list[0][1]['timerange'].startts == days_ago

    dl_mock.reset_mock()
    args = [
        "download-data",
        "--exchange", "binance",
        "--pairs", "ETH/BTC", "XRP/BTC",
        "--timerange", "20200101-"
    ]
    start_download_data(get_args(args))
    assert dl_mock.call_count == 1

    assert dl_mock.call_args_list[0][1]['timerange'].startts == arrow.Arrow(
        2020, 1, 1).int_timestamp


def test_download_data_no_markets(mocker, caplog):
    dl_mock = mocker.patch('freqtrade.commands.data_commands.refresh_backtest_ohlcv_data',
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
    mocker.patch('freqtrade.commands.data_commands.refresh_backtest_ohlcv_data',
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

    mocker.patch('freqtrade.commands.data_commands.refresh_backtest_ohlcv_data',
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


def test_download_data_all_pairs(mocker, markets):

    mocker.patch.object(Path, "exists", MagicMock(return_value=False))

    dl_mock = mocker.patch('freqtrade.commands.data_commands.refresh_backtest_ohlcv_data',
                           MagicMock(return_value=["ETH/BTC", "XRP/BTC"]))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets)
    )
    args = [
        "download-data",
        "--exchange",
        "binance",
        "--pairs",
        ".*/USDT"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_download_data(pargs)
    expected = set(['ETH/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT'])
    assert set(dl_mock.call_args_list[0][1]['pairs']) == expected
    assert dl_mock.call_count == 1

    dl_mock.reset_mock()
    args = [
        "download-data",
        "--exchange",
        "binance",
        "--pairs",
        ".*/USDT",
        "--include-inactive-pairs",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_download_data(pargs)
    expected = set(['ETH/USDT', 'LTC/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT'])
    assert set(dl_mock.call_args_list[0][1]['pairs']) == expected


def test_download_data_trades(mocker, caplog):
    dl_mock = mocker.patch('freqtrade.commands.data_commands.refresh_backtest_trades_data',
                           MagicMock(return_value=[]))
    convert_mock = mocker.patch('freqtrade.commands.data_commands.convert_trades_to_ohlcv',
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


def test_start_convert_trades(mocker, caplog):
    convert_mock = mocker.patch('freqtrade.commands.data_commands.convert_trades_to_ohlcv',
                                MagicMock(return_value=[]))
    patch_exchange(mocker)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    args = [
        "trades-to-ohlcv",
        "--exchange", "kraken",
        "--pairs", "ETH/BTC", "XRP/BTC",
    ]
    start_convert_trades(get_args(args))
    assert convert_mock.call_count == 1


def test_start_list_strategies(mocker, caplog, capsys):

    args = [
        "list-strategies",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy" / "strats"),
        "-1"
    ]
    pargs = get_args(args)
    # pargs['config'] = None
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert "TestStrategyLegacyV1" in captured.out
    assert "legacy_strategy_v1.py" not in captured.out
    assert "StrategyTestV2" in captured.out

    # Test regular output
    args = [
        "list-strategies",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy" / "strats"),
        '--no-color',
    ]
    pargs = get_args(args)
    # pargs['config'] = None
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert "TestStrategyLegacyV1" in captured.out
    assert "legacy_strategy_v1.py" in captured.out
    assert "StrategyTestV2" in captured.out

    # Test color output
    args = [
        "list-strategies",
        "--strategy-path",
        str(Path(__file__).parent.parent / "strategy" / "strats"),
    ]
    pargs = get_args(args)
    # pargs['config'] = None
    start_list_strategies(pargs)
    captured = capsys.readouterr()
    assert "TestStrategyLegacyV1" in captured.out
    assert "legacy_strategy_v1.py" in captured.out
    assert "StrategyTestV2" in captured.out
    assert "LOAD FAILED" in captured.out


def test_start_test_pairlist(mocker, caplog, tickers, default_conf, capsys):
    patch_exchange(mocker, mock_markets=True)
    mocker.patch.multiple('freqtrade.exchange.Exchange',
                          exchange_has=MagicMock(return_value=True),
                          get_tickers=tickers,
                          )

    default_conf['pairlists'] = [
        {
            "method": "VolumePairList",
            "number_assets": 5,
            "sort_key": "quoteVolume",
        },
        {"method": "PrecisionFilter"},
        {"method": "PriceFilter", "low_price_ratio": 0.02},
    ]

    patched_configuration_load_config_file(mocker, default_conf)
    args = [
        'test-pairlist',
        '-c', 'config_examples/config_bittrex.example.json'
    ]

    start_test_pairlist(get_args(args))

    assert log_has_re(r"^Using resolved pairlist VolumePairList.*", caplog)
    assert log_has_re(r"^Using resolved pairlist PrecisionFilter.*", caplog)
    assert log_has_re(r"^Using resolved pairlist PriceFilter.*", caplog)
    captured = capsys.readouterr()
    assert re.match(r"Pairs for .*", captured.out)
    assert re.match("['ETH/BTC', 'TKN/BTC', 'BLK/BTC', 'LTC/BTC', 'XRP/BTC']", captured.out)

    args = [
        'test-pairlist',
        '-c', 'config_examples/config_bittrex.example.json',
        '--one-column',
    ]
    start_test_pairlist(get_args(args))
    captured = capsys.readouterr()
    assert re.match(r"ETH/BTC\nTKN/BTC\nBLK/BTC\nLTC/BTC\nXRP/BTC\n", captured.out)

    args = [
        'test-pairlist',
        '-c', 'config_examples/config_bittrex.example.json',
        '--print-json',
    ]
    start_test_pairlist(get_args(args))
    captured = capsys.readouterr()
    try:
        json_pairs = json.loads(captured.out)
        assert 'ETH/BTC' in json_pairs
        assert 'TKN/BTC' in json_pairs
        assert 'BLK/BTC' in json_pairs
        assert 'LTC/BTC' in json_pairs
        assert 'XRP/BTC' in json_pairs
    except json.decoder.JSONDecodeError:
        pytest.fail(f'Expected well formed JSON, but failed to parse: {captured.out}')


def test_hyperopt_list(mocker, capsys, caplog, saved_hyperopt_results, tmpdir):
    csv_file = Path(tmpdir) / "test.csv"
    mocker.patch(
        'freqtrade.optimize.hyperopt_tools.HyperoptTools._test_hyperopt_results_exist',
        return_value=True
        )

    def fake_iterator(*args, **kwargs):
        yield from [saved_hyperopt_results]

    mocker.patch(
        'freqtrade.optimize.hyperopt_tools.HyperoptTools._read_results',
        side_effect=fake_iterator
    )

    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 1/12", " 2/12", " 3/12", " 4/12", " 5/12",
                         " 6/12", " 7/12", " 8/12", " 9/12", " 10/12",
                         " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--best",
        "--no-details",
        "--no-color",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 1/12", " 5/12", " 10/12"])
    assert all(x not in captured.out
               for x in [" 2/12", " 3/12", " 4/12", " 6/12", " 7/12", " 8/12", " 9/12",
                         " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-details",
        "--no-color",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 2/12", " 10/12"])
    assert all(x not in captured.out
               for x in [" 1/12", " 3/12", " 4/12", " 5/12", " 6/12", " 7/12", " 8/12", " 9/12",
                         " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-color",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 2/12", " 10/12", "Best result:", "Buy hyperspace params",
                         "Sell hyperspace params", "ROI table", "Stoploss"])
    assert all(x not in captured.out
               for x in [" 1/12", " 3/12", " 4/12", " 5/12", " 6/12", " 7/12", " 8/12", " 9/12",
                         " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--min-trades", "20",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 3/12", " 6/12", " 7/12", " 9/12", " 11/12"])
    assert all(x not in captured.out
               for x in [" 1/12", " 2/12", " 4/12", " 5/12", " 8/12", " 10/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-details",
        "--no-color",
        "--max-trades", "20",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 2/12", " 10/12"])
    assert all(x not in captured.out
               for x in [" 1/12", " 3/12", " 4/12", " 5/12", " 6/12", " 7/12", " 8/12", " 9/12",
                         " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-details",
        "--no-color",
        "--min-avg-profit", "0.11",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 2/12"])
    assert all(x not in captured.out
               for x in [" 1/12", " 3/12", " 4/12", " 5/12", " 6/12", " 7/12", " 8/12", " 9/12",
                         " 10/12", " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--max-avg-profit", "0.10",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 1/12", " 3/12", " 5/12", " 6/12", " 7/12", " 8/12", " 9/12",
                         " 11/12"])
    assert all(x not in captured.out
               for x in [" 2/12", " 4/12", " 10/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--min-total-profit", "0.4",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 10/12"])
    assert all(x not in captured.out
               for x in [" 1/12", " 2/12", " 3/12", " 4/12", " 5/12", " 6/12", " 7/12", " 8/12",
                         " 9/12", " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--max-total-profit", "0.4",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 1/12", " 2/12", " 3/12", " 5/12", " 6/12", " 7/12", " 8/12",
                         " 9/12", " 11/12"])
    assert all(x not in captured.out
               for x in [" 4/12", " 10/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--min-objective", "0.1",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 10/12"])
    assert all(x not in captured.out
               for x in [" 1/12", " 2/12", " 3/12", " 4/12", " 5/12", " 6/12", " 7/12", " 8/12",
                         " 9/12", " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--no-details",
        "--max-objective", "0.1",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 1/12", " 2/12", " 3/12", " 5/12", " 6/12", " 7/12", " 8/12",
                         " 9/12", " 11/12"])
    assert all(x not in captured.out
               for x in [" 4/12", " 10/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--profitable",
        "--no-details",
        "--no-color",
        "--min-avg-time", "2000",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 10/12"])
    assert all(x not in captured.out
               for x in [" 1/12", " 2/12", " 3/12", " 4/12", " 5/12", " 6/12", " 7/12",
                         " 8/12", " 9/12", " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--max-avg-time", "1500",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    assert all(x in captured.out
               for x in [" 2/12", " 6/12"])
    assert all(x not in captured.out
               for x in [" 1/12", " 3/12", " 4/12", " 5/12", " 7/12", " 8/12"
                         " 9/12", " 10/12", " 11/12", " 12/12"])
    args = [
        "hyperopt-list",
        "--no-details",
        "--no-color",
        "--export-csv",
        str(csv_file),
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_list(pargs)
    captured = capsys.readouterr()
    log_has("CSV file created: test_file.csv", caplog)
    assert csv_file.is_file()
    line = csv_file.read_text()
    assert ('Best,1,2,-1.25%,-1.2222,-0.00125625,,-2.51,"3,930.0 m",0.43662' in line
            or "Best,1,2,-1.25%,-1.2222,-0.00125625,,-2.51,2 days 17:30:00,0.43662" in line)
    csv_file.unlink()


def test_hyperopt_show(mocker, capsys, saved_hyperopt_results):
    mocker.patch(
        'freqtrade.optimize.hyperopt_tools.HyperoptTools._test_hyperopt_results_exist',
        return_value=True
    )

    def fake_iterator(*args, **kwargs):
        yield from [saved_hyperopt_results]

    mocker.patch(
        'freqtrade.optimize.hyperopt_tools.HyperoptTools._read_results',
        side_effect=fake_iterator
    )
    mocker.patch('freqtrade.commands.hyperopt_commands.show_backtest_result')

    args = [
        "hyperopt-show",
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert " 12/12" in captured.out

    args = [
        "hyperopt-show",
        "--best"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert " 10/12" in captured.out

    args = [
        "hyperopt-show",
        "-n", "1"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert " 1/12" in captured.out

    args = [
        "hyperopt-show",
        "--best",
        "-n", "2"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert " 5/12" in captured.out

    args = [
        "hyperopt-show",
        "--best",
        "-n", "-1"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_hyperopt_show(pargs)
    captured = capsys.readouterr()
    assert " 10/12" in captured.out

    args = [
        "hyperopt-show",
        "--best",
        "-n", "-4"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match="The index of the epoch to show should be greater than -4."):
        start_hyperopt_show(pargs)

    args = [
        "hyperopt-show",
        "--best",
        "-n", "4"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    with pytest.raises(OperationalException,
                       match="The index of the epoch to show should be less than 4."):
        start_hyperopt_show(pargs)


def test_convert_data(mocker, testdatadir):
    ohlcv_mock = mocker.patch("freqtrade.commands.data_commands.convert_ohlcv_format")
    trades_mock = mocker.patch("freqtrade.commands.data_commands.convert_trades_format")
    args = [
        "convert-data",
        "--format-from",
        "json",
        "--format-to",
        "jsongz",
        "--datadir",
        str(testdatadir),
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_convert_data(pargs, True)
    assert trades_mock.call_count == 0
    assert ohlcv_mock.call_count == 1
    assert ohlcv_mock.call_args[1]['convert_from'] == 'json'
    assert ohlcv_mock.call_args[1]['convert_to'] == 'jsongz'
    assert ohlcv_mock.call_args[1]['erase'] is False


def test_convert_data_trades(mocker, testdatadir):
    ohlcv_mock = mocker.patch("freqtrade.commands.data_commands.convert_ohlcv_format")
    trades_mock = mocker.patch("freqtrade.commands.data_commands.convert_trades_format")
    args = [
        "convert-trade-data",
        "--format-from",
        "jsongz",
        "--format-to",
        "json",
        "--datadir",
        str(testdatadir),
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_convert_data(pargs, False)
    assert ohlcv_mock.call_count == 0
    assert trades_mock.call_count == 1
    assert trades_mock.call_args[1]['convert_from'] == 'jsongz'
    assert trades_mock.call_args[1]['convert_to'] == 'json'
    assert trades_mock.call_args[1]['erase'] is False


def test_start_list_data(testdatadir, capsys):
    args = [
        "list-data",
        "--data-format-ohlcv",
        "json",
        "--datadir",
        str(testdatadir),
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert "Found 17 pair / timeframe combinations." in captured.out
    assert "\n|         Pair |       Timeframe |\n" in captured.out
    assert "\n| UNITTEST/BTC | 1m, 5m, 8m, 30m |\n" in captured.out

    args = [
        "list-data",
        "--data-format-ohlcv",
        "json",
        "--pairs", "XRP/ETH",
        "--datadir",
        str(testdatadir),
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_list_data(pargs)
    captured = capsys.readouterr()
    assert "Found 2 pair / timeframe combinations." in captured.out
    assert "\n|    Pair |   Timeframe |\n" in captured.out
    assert "UNITTEST/BTC" not in captured.out
    assert "\n| XRP/ETH |      1m, 5m |\n" in captured.out


@pytest.mark.usefixtures("init_persistence")
def test_show_trades(mocker, fee, capsys, caplog):
    mocker.patch("freqtrade.persistence.init_db")
    create_mock_trades(fee)
    args = [
        "show-trades",
        "--db-url",
        "sqlite:///"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_show_trades(pargs)
    assert log_has(f"Printing {MOCK_TRADE_COUNT} Trades: ", caplog)
    captured = capsys.readouterr()
    assert "Trade(id=1" in captured.out
    assert "Trade(id=2" in captured.out
    assert "Trade(id=3" in captured.out
    args = [
        "show-trades",
        "--db-url",
        "sqlite:///",
        "--print-json",
        "--trade-ids", "1", "2"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_show_trades(pargs)

    captured = capsys.readouterr()
    assert log_has("Printing 2 Trades: ", caplog)
    assert '"trade_id": 1' in captured.out
    assert '"trade_id": 2' in captured.out
    assert '"trade_id": 3' not in captured.out
    args = [
        "show-trades",
    ]
    pargs = get_args(args)
    pargs['config'] = None

    with pytest.raises(OperationalException, match=r"--db-url is required for this command."):
        start_show_trades(pargs)


def test_backtesting_show(mocker, testdatadir, capsys):
    sbr = mocker.patch('freqtrade.optimize.optimize_reports.show_backtest_results')
    args = [
        "backtesting-show",
        "--export-filename",
        f"{testdatadir / 'backtest-result_new.json'}",
        "--show-pair-list"
    ]
    pargs = get_args(args)
    pargs['config'] = None
    start_backtesting_show(pargs)
    assert sbr.call_count == 1
    out, err = capsys.readouterr()
    assert "Pairs for Strategy" in out
