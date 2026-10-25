from pathlib import Path
from unittest.mock import MagicMock

import pytest
import rapidjson

from freqtrade.commands.build_config_commands import start_new_config
from freqtrade.configuration.deploy_config import (
    ask_user_config,
    ask_user_overwrite,
    deploy_new_config,
    exchange_requires_password,
    validate_is_float,
    validate_is_int,
)
from freqtrade.exceptions import OperationalException
from tests.conftest import get_args, log_has_re


def test_validate_is_float():
    assert validate_is_float("2.0")
    assert validate_is_float("2.1")
    assert validate_is_float("0.1")
    assert validate_is_float("-0.5")
    assert not validate_is_float("-0.5e")


def test_validate_is_int():
    assert validate_is_int("2")
    assert validate_is_int("6")
    assert validate_is_int("-1")
    assert validate_is_int("500")
    assert not validate_is_int("2.0")
    assert not validate_is_int("2.1")
    assert not validate_is_int("-2.1")
    assert not validate_is_int("-ee")


@pytest.mark.parametrize(
    "exchange",
    ["bybit", "binance", "gate", "gateio", "htx", "kraken", "kucoin", "okx"],
)
@pytest.mark.parametrize(
    "credential",
    ["sampleKey", 'quoted"secret', r"backslash\new", "line\nbreak", "secret\u79d8\u5bc6"],
)
def test_start_new_config(mocker, caplog, tmp_path, exchange, credential):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"dry_run": false}')
    mocker.patch("freqtrade.configuration.deploy_config.ask_user_overwrite", return_value=True)

    sample_selections = {
        "max_open_trades": 3,
        "stake_currency": "USDT",
        "stake_amount": 100,
        "fiat_display_currency": "EUR",
        "timeframe": "15m",
        "dry_run": True,
        "trading_mode": "spot",
        "margin_mode": "",
        "exchange_name": exchange,
        "exchange_api_key": credential,
        "exchange_secret": credential,
        "exchange_api_key_password": credential,
        "telegram": False,
        "telegram_token": credential,
        "telegram_chat_id": credential,
        "api_server": False,
        "api_server_listen_addr": "127.0.0.1",
        "api_server_username": credential,
        "api_server_password": credential,
        "api_server_jwt_key": credential,
        "api_server_ws_token": credential,
    }
    mocker.patch(
        "freqtrade.configuration.deploy_config.ask_user_config", return_value=sample_selections
    )
    args = ["new-config", "--config", str(config_path)]
    start_new_config(get_args(args))

    assert log_has_re("Writing config to .*", caplog)
    result = rapidjson.loads(
        config_path.read_text(encoding="utf-8"),
        parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS,
    )
    assert result["exchange"]["name"] == exchange
    assert result["exchange"]["api_key"] == credential
    assert result["exchange"]["secret"] == credential
    if exchange in ("kucoin", "okx"):
        assert result["exchange"]["password"] == credential
    if exchange in ("gate", "gateio"):
        assert result["exchange"]["unknown_fee_rate"] == 1
    if exchange == "htx":
        assert result["exchange"]["pair_blacklist"] == ["HT/.*"]
    assert result["telegram"]["token"] == credential
    assert result["telegram"]["chat_id"] == credential
    assert result["api_server"]["username"] == credential
    assert result["api_server"]["password"] == credential
    assert result["api_server"]["jwt_secret_key"] == credential
    assert result["api_server"]["ws_token"] == credential
    assert result["timeframe"] == "15m"


@pytest.mark.parametrize(
    "exchange,has_password",
    [
        ("bybit", False),
        ("binance", False),
        ("bitget", True),
        ("gate", False),
        ("gateio", False),
        ("htx", False),
        ("kraken", False),
        ("kucoin", True),
        ("myokx", True),
        ("okx", True),
    ],
)
def test_deploy_new_config_optional_credentials(tmp_path, exchange, has_password):
    config_path = tmp_path / "config.json"
    selections = {
        "max_open_trades": 3,
        "stake_currency": "USDT",
        "stake_amount": "unlimited",
        "dry_run": True,
        "trading_mode": "spot",
        "margin_mode": "",
        "exchange_name": exchange,
        "telegram": False,
        "api_server": False,
        "api_server_jwt_key": "test-jwt-key",
        "api_server_ws_token": "test-ws-token",
    }

    deploy_new_config(config_path, selections)
    result = rapidjson.loads(config_path.read_text(encoding="utf-8"))

    assert result["dry_run"] is True
    assert result["stake_currency"] == "USDT"
    assert result["stake_amount"] == "unlimited"
    assert result["max_open_trades"] == 3
    assert result["exchange"]["api_key"] == ""
    assert result["exchange"]["secret"] == ""
    if has_password:
        assert result["exchange"]["password"] == ""
    else:
        assert "password" not in result["exchange"]
    assert result["telegram"]["token"] == ""
    assert result["telegram"]["chat_id"] == ""
    assert result["api_server"]["username"] == ""
    assert result["api_server"]["password"] == ""
    assert result["api_server"]["listen_ip_address"] == "127.0.0.1"
    assert "timeframe" not in result
    assert "fiat_display_currency" not in result


@pytest.mark.parametrize(
    "exchange,expected",
    [("okx", True), ("okxus", True), ("bitget", True), ("binance", False), ("doesnotexist", False)],
)
def test_exchange_requires_password(exchange, expected):
    assert exchange_requires_password(exchange) is expected


def test_start_new_config_exists(mocker, caplog):
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch("freqtrade.configuration.deploy_config.ask_user_overwrite", return_value=False)
    args = ["new-config", "--config", "coolconfig.json"]
    with pytest.raises(OperationalException, match=r"Configuration .* already exists\."):
        start_new_config(get_args(args))


def test_start_new_config_cancelled_overwrite(mocker, tmp_path):
    config_path = tmp_path / "config.json"
    original_config = '{"dry_run": true}'
    config_path.write_text(original_config)
    mocker.patch("freqtrade.configuration.deploy_config.ask_user_overwrite", return_value=True)
    mocker.patch(
        "freqtrade.configuration.deploy_config.ask_user_config",
        side_effect=OperationalException("User interrupted interactive questions."),
    )

    with pytest.raises(OperationalException, match="User interrupted interactive questions"):
        start_new_config(get_args(["new-config", "--config", str(config_path)]))

    assert config_path.read_text() == original_config


def test_ask_user_overwrite(mocker):
    """
    Once https://github.com/tmbo/questionary/issues/35 is implemented, improve this test.
    """
    prompt_mock = mocker.patch(
        "freqtrade.configuration.deploy_config.prompt", return_value={"overwrite": False}
    )
    assert not ask_user_overwrite(Path("test.json"))
    assert prompt_mock.call_count == 1

    prompt_mock.reset_mock()
    prompt_mock = mocker.patch(
        "freqtrade.configuration.deploy_config.prompt", return_value={"overwrite": True}
    )
    assert ask_user_overwrite(Path("test.json"))
    assert prompt_mock.call_count == 1


def test_ask_user_config(mocker):
    """
    Once https://github.com/tmbo/questionary/issues/35 is implemented, improve this test.
    """
    prompt_mock = mocker.patch(
        "freqtrade.configuration.deploy_config.prompt", return_value={"overwrite": False}
    )
    answers = ask_user_config()
    assert isinstance(answers, dict)
    assert prompt_mock.call_count == 1

    questions = {q["name"]: q for q in prompt_mock.call_args[0][0] if "name" in q}
    assert questions["stake_amount"]["filter"]("unlimited") == "unlimited"
    assert questions["stake_amount"]["filter"]("122") == 122
    assert isinstance(questions["stake_amount"]["filter"]("122"), int)
    assert questions["stake_amount"]["filter"]("0.5") == 0.5
    assert questions["max_open_trades"]["filter"]("-1") == -1

    prompt_mock = mocker.patch("freqtrade.configuration.deploy_config.prompt", return_value={})

    with pytest.raises(OperationalException, match=r"User interrupted interactive questions\."):
        ask_user_config()
