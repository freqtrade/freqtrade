import argparse
import enum
from typing import Optional

from wrapt import synchronized


class State(enum.Enum):
    RUNNING = 0
    STOPPED = 1
    BACKTESTING = 2


# Command-line arguments
_ARGS: argparse.Namespace = None

# Current application state
_STATE = State.STOPPED


@synchronized
def update_state(state: State) -> None:
    """
    Updates the application state
    :param state: new state
    :return: None
    """
    global _STATE
    _STATE = state


@synchronized
def get_state() -> State:
    """
    Gets the current application state
    :return:
    """
    return _STATE


def parse_args(argv: Optional[list] = None) -> None:
    """
    Parses and stores command-line arguments.
    :param argv: (optional) command-line arguments
    :return: None
    """
    global _ARGS

    parser = argparse.ArgumentParser(
        description='Simple High Frequency Trading Bot for crypto currencies')
    parser.add_argument('-b', '--backtesting', action='store_true',
                        help='test bot performance on offline data and print results')
    _ARGS = parser.parse_args(argv[1:] if argv else [])


def get_args():
    """
    Gets the current command-line arguments.
    :return: arguments
    """
    return _ARGS


# Required json-schema for user specified config
CONF_SCHEMA = {
    'type': 'object',
    'properties': {
        'max_open_trades': {'type': 'integer', 'minimum': 1},
        'stake_currency': {'type': 'string', 'enum': ['BTC', 'ETH', 'USDT']},
        'stake_amount': {'type': 'number', 'minimum': 0.0005},
        'dry_run': {'type': 'boolean'},
        'minimal_roi': {
            'type': 'object',
            'patternProperties': {
                '^[0-9.]+$': {'type': 'number'}
            },
            'minProperties': 1
        },
        'stoploss': {'type': 'number', 'maximum': 0, 'exclusiveMaximum': True},
        'bid_strategy': {
            'type': 'object',
            'properties': {
                'ask_last_balance': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 1,
                    'exclusiveMaximum': False
                },
            },
            'required': ['ask_last_balance']
        },
        'exchange': {'$ref': '#/definitions/exchange'},
        'telegram': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'boolean'},
                'token': {'type': 'string'},
                'chat_id': {'type': 'string'},
            },
            'required': ['enabled', 'token', 'chat_id']
        },
        'initial_state': {'type': 'string', 'enum': ['running', 'stopped']},
    },
    'definitions': {
        'exchange': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'key': {'type': 'string'},
                'secret': {'type': 'string'},
                'pair_whitelist': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'uniqueItems': True
                }
            },
            'required': ['name', 'key', 'secret', 'pair_whitelist']
        }
    },
    'anyOf': [
        {'required': ['exchange']}
    ],
    'required': [
        'max_open_trades',
        'stake_currency',
        'stake_amount',
        'dry_run',
        'minimal_roi',
        'bid_strategy',
        'telegram'
    ]
}
