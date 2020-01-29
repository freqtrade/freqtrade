import logging
from pathlib import Path
from typing import Any, Dict

from questionary import Separator, prompt

from freqtrade.exchange import available_exchanges
from freqtrade.misc import render_template

logger = logging.getLogger(__name__)


def ask_user_config() -> Dict[str, Any]:
    """
    Ask user a few questions to build the configuration.
    :returns: Dict with keys to put into template
    """
    questions = [
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
            "default": 'BTC',
        },
        {
            "type": "text",
            "name": "stake_amount",
            "message": "Please insert your stake amount:",
            "default": "0.01",
        },
        {
            "type": "text",
            "name": "max_open_trades",
            "message": "Please insert max_open_trades:",
            "default": "3",
        },
        {
            "type": "text",
            "name": "ticker_interval",
            "message": "Please insert your ticker interval:",
            "default": "5m",
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
                "bittrex",
                "binance",
                "binanceje",
                "binanceus",
                "kraken",
                Separator(),
                "other",
            ],
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
            "type": "text",
            "name": "telegram_chat_id",
            "message": "Insert Telegram chat id",
            "when": lambda x: x['telegram']
        },
    ]
    answers = prompt(questions)

    print(answers)

    sample_selections = {
        'max_open_trades': 3,
        'stake_currency': 'USDT',
        'stake_amount': 100,
        'fiat_display_currency': 'EUR',
        'ticker_interval': '15m',
        'dry_run': True,
        'exchange_name': 'binance',
        'exchange_key': 'sampleKey',
        'exchange_secret': 'Samplesecret',
        'telegram': False,
        'telegram_token': 'asdf1244',
        'telegram_chat_id': '1144444',
    }
    return sample_selections


def deploy_new_config(config_path: Path, selections: Dict[str, Any]) -> None:
    """
    Applies selections to the template and writes the result to config_path
    :param config_path: Path object for new config file. Should not exist yet
    :param selecions: Dict containing selections taken by the user.
    """
    from jinja2.exceptions import TemplateNotFound
    try:
        selections['exchange'] = render_template(
            templatefile=f"subtemplates/exchange_{selections['exchange_name']}.j2",
            arguments=selections
            )
    except TemplateNotFound:
        selections['exchange'] = render_template(
            templatefile=f"subtemplates/exchange_generic.j2",
            arguments=selections
        )

    config_text = render_template(templatefile='base_config.json.j2',
                                  arguments=selections)

    logger.info(f"Writing config to `{config_path}`.")
    config_path.write_text(config_text)


def start_new_config(args: Dict[str, Any]) -> None:
    """
    Create a new strategy from a template
    Asking the user questions to fill out the templateaccordingly.
    """
    selections = ask_user_config()
    config_path = Path(args['config'][0])
    deploy_new_config(config_path, selections)
