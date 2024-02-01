import logging
import secrets
from pathlib import Path
from typing import Any, Dict, List

from questionary import Separator, prompt

from freqtrade.configuration.detect_environment import running_in_docker
from freqtrade.configuration.directory_operations import chown_user_directory
from freqtrade.constants import UNLIMITED_STAKE_AMOUNT
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import MAP_EXCHANGE_CHILDCLASS, available_exchanges
from freqtrade.util import render_template


logger = logging.getLogger(__name__)


def validate_is_int(val):
    try:
        _ = int(val)
        return True
    except Exception:
        return False


def validate_is_float(val):
    try:
        _ = float(val)
        return True
    except Exception:
        return False


def ask_user_overwrite(config_path: Path) -> bool:
    questions = [
        {
            "type": "confirm",
            "name": "overwrite",
            "message": f"File {config_path} already exists. Overwrite?",
            "default": False,
        },
    ]
    answers = prompt(questions)
    return answers['overwrite']


def ask_user_config() -> Dict[str, Any]:
    """
    Ask user a few questions to build the configuration.
    Interactive questions built using https://github.com/tmbo/questionary
    :returns: Dict with keys to put into template
    """
    questions: List[Dict[str, Any]] = [
        {
            "type": "confirm",
            "name": "dry_run",
            "message": "Do you want to enable Dry-run (simulated trades)?",
            "default": True,
        },
        {
            "type": "text",
            "name": "stake_currency",
            "message": "Please insert your stake currency:",
            "default": 'USDT',
        },
        {
            "type": "text",
            "name": "stake_amount",
            "message": f"Please insert your stake amount (Number or '{UNLIMITED_STAKE_AMOUNT}'):",
            "default": "unlimited",
            "validate": lambda val: val == UNLIMITED_STAKE_AMOUNT or validate_is_float(val),
            "filter": lambda val: '"' + UNLIMITED_STAKE_AMOUNT + '"'
            if val == UNLIMITED_STAKE_AMOUNT
            else val
        },
        {
            "type": "text",
            "name": "max_open_trades",
            "message": "Please insert max_open_trades (Integer or -1 for unlimited open trades):",
            "default": "3",
            "validate": lambda val: validate_is_int(val)
        },
        {
            "type": "select",
            "name": "timeframe_in_config",
            "message": "Time",
            "choices": ["Have the strategy define timeframe.", "Override in configuration."]
        },
        {
            "type": "text",
            "name": "timeframe",
            "message": "Please insert your desired timeframe (e.g. 5m):",
            "default": "5m",
            "when": lambda x: x["timeframe_in_config"] == 'Override in configuration.'

        },
        {
            "type": "text",
            "name": "fiat_display_currency",
            "message": "Please insert your display Currency (for reporting):",
            "default": 'USD',
        },
        {
            "type": "select",
            "name": "exchange_name",
            "message": "Select exchange",
            "choices": [
                "binance",
                "binanceus",
                "gate",
                "htx",
                "kraken",
                "kucoin",
                "okx",
                Separator("------------------"),
                "other",
            ],
        },
        {
            "type": "confirm",
            "name": "trading_mode",
            "message": "Do you want to trade Perpetual Swaps (perpetual futures)?",
            "default": False,
            "filter": lambda val: 'futures' if val else 'spot',
            "when": lambda x: x["exchange_name"] in ['binance', 'gate', 'okx'],
        },
        {
            "type": "autocomplete",
            "name": "exchange_name",
            "message": "Type your exchange name (Must be supported by ccxt)",
            "choices": available_exchanges(),
            "when": lambda x: x["exchange_name"] == 'other'
        },
        {
            "type": "password",
            "name": "exchange_key",
            "message": "Insert Exchange Key",
            "when": lambda x: not x['dry_run']
        },
        {
            "type": "password",
            "name": "exchange_secret",
            "message": "Insert Exchange Secret",
            "when": lambda x: not x['dry_run']
        },
        {
            "type": "password",
            "name": "exchange_key_password",
            "message": "Insert Exchange API Key password",
            "when": lambda x: not x['dry_run'] and x['exchange_name'] in ('kucoin', 'okx')
        },
        {
            "type": "confirm",
            "name": "telegram",
            "message": "Do you want to enable Telegram?",
            "default": False,
        },
        {
            "type": "password",
            "name": "telegram_token",
            "message": "Insert Telegram token",
            "when": lambda x: x['telegram']
        },
        {
            "type": "password",
            "name": "telegram_chat_id",
            "message": "Insert Telegram chat id",
            "when": lambda x: x['telegram']
        },
        {
            "type": "confirm",
            "name": "api_server",
            "message": "Do you want to enable the Rest API (includes FreqUI)?",
            "default": False,
        },
        {
            "type": "text",
            "name": "api_server_listen_addr",
            "message": ("Insert Api server Listen Address (0.0.0.0 for docker, "
                        "otherwise best left untouched)"),
            "default": "127.0.0.1" if not running_in_docker() else "0.0.0.0",
            "when": lambda x: x['api_server']
        },
        {
            "type": "text",
            "name": "api_server_username",
            "message": "Insert api-server username",
            "default": "freqtrader",
            "when": lambda x: x['api_server']
        },
        {
            "type": "password",
            "name": "api_server_password",
            "message": "Insert api-server password",
            "when": lambda x: x['api_server']
        },
    ]
    answers = prompt(questions)

    if not answers:
        # Interrupted questionary sessions return an empty dict.
        raise OperationalException("User interrupted interactive questions.")
    # Ensure default is set for non-futures exchanges
    answers['trading_mode'] = answers.get('trading_mode', "spot")
    answers['margin_mode'] = (
        'isolated'
        if answers.get('trading_mode') == 'futures'
        else ''
    )
    # Force JWT token to be a random string
    answers['api_server_jwt_key'] = secrets.token_hex()
    answers['api_server_ws_token'] = secrets.token_urlsafe(25)

    return answers


def deploy_new_config(config_path: Path, selections: Dict[str, Any]) -> None:
    """
    Applies selections to the template and writes the result to config_path
    :param config_path: Path object for new config file. Should not exist yet
    :param selections: Dict containing selections taken by the user.
    """
    from jinja2.exceptions import TemplateNotFound
    try:
        exchange_template = MAP_EXCHANGE_CHILDCLASS.get(
            selections['exchange_name'], selections['exchange_name'])

        selections['exchange'] = render_template(
            templatefile=f"subtemplates/exchange_{exchange_template}.j2",
            arguments=selections
        )
    except TemplateNotFound:
        selections['exchange'] = render_template(
            templatefile="subtemplates/exchange_generic.j2",
            arguments=selections
        )

    config_text = render_template(templatefile='base_config.json.j2',
                                  arguments=selections)

    logger.info(f"Writing config to `{config_path}`.")
    logger.info(
        "Please make sure to check the configuration contents and adjust settings to your needs.")

    config_path.write_text(config_text)


def start_new_config(args: Dict[str, Any]) -> None:
    """
    Create a new strategy from a template
    Asking the user questions to fill out the template accordingly.
    """

    config_path = Path(args['config'][0])
    chown_user_directory(config_path.parent)
    if config_path.exists():
        overwrite = ask_user_overwrite(config_path)
        if overwrite:
            config_path.unlink()
        else:
            raise OperationalException(
                f"Configuration file `{config_path}` already exists. "
                "Please delete it or use a different configuration file name.")
    selections = ask_user_config()
    deploy_new_config(config_path, selections)
