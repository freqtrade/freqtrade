import argparse
import enum
import logging
import os
import time
from typing import Any, Callable

from wrapt import synchronized

from freqtrade import __version__

logger = logging.getLogger(__name__)


class FreqtradeException(BaseException):
    pass


class State(enum.Enum):
    RUNNING = 0
    STOPPED = 1


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


def throttle(func: Callable[..., Any], min_secs: float, *args, **kwargs) -> Any:
    """
    Throttles the given callable that it
    takes at least `min_secs` to finish execution.
    :param func: Any callable
    :param min_secs: minimum execution time in seconds
    :return: Any
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    duration = max(min_secs - (end - start), 0.0)
    logger.debug('Throttling %s for %.2f seconds', func.__name__, duration)
    time.sleep(duration)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """ Builds and returns an ArgumentParser instance """
    parser = argparse.ArgumentParser(
        description='Simple High Frequency Trading Bot for crypto currencies'
    )
    parser.add_argument(
        '-c', '--config',
        help='specify configuration file (default: config.json)',
        dest='config',
        default='config.json',
        type=str,
        metavar='PATH',
    )
    parser.add_argument(
        '-v', '--verbose',
        help='be verbose',
        action='store_const',
        dest='loglevel',
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {}'.format(__version__),
    )
    parser.add_argument(
        '--dynamic-whitelist',
        help='dynamically generate and update whitelist based on 24h BaseVolume',
        action='store_true',
    )
    build_subcommands(parser)
    return parser


def build_subcommands(parser: argparse.ArgumentParser) -> None:
    """ Builds and attaches all subcommands """
    subparsers = parser.add_subparsers(dest='subparser')
    backtest = subparsers.add_parser('backtesting', help='backtesting module')
    backtest.set_defaults(func=start_backtesting)
    backtest.add_argument(
        '-l', '--live',
        action='store_true',
        dest='live',
        help='using live data',
    )
    backtest.add_argument(
        '-i', '--ticker-interval',
        help='specify ticker interval in minutes (default: 5)',
        dest='ticker_interval',
        default=5,
        type=int,
        metavar='INT',
    )


def start_backtesting(args) -> None:
    """
    Exports all args as environment variables and starts backtesting via pytest.
    :param args: arguments namespace
    :return:
    """
    import pytest

    os.environ.update({
        'BACKTEST': 'true',
        'BACKTEST_LIVE': 'true' if args.live else '',
        'BACKTEST_CONFIG': args.config,
        'BACKTEST_TICKER_INTERVAL': str(args.ticker_interval),
    })
    path = os.path.join(os.path.dirname(__file__), 'tests', 'test_backtesting.py')
    pytest.main(['-s', path])


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
        'internals': {
            'type': 'object',
            'properties': {
                'process_throttle_secs': {'type': 'number'}
            }
        }
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
