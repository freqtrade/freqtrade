import argparse
import enum
import json
import logging
import time
import os
from typing import Any, Callable, Dict, List

from jsonschema import Draft4Validator, validate
from jsonschema.exceptions import ValidationError, best_match
from wrapt import synchronized

from freqtrade import __version__

logger = logging.getLogger(__name__)


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


def load_config(path: str) -> Dict:
    """
    Loads a config file from the given path
    :param path: path as str
    :return: configuration as dictionary
    """
    with open(path) as file:
        conf = json.load(file)
    if 'internals' not in conf:
        conf['internals'] = {}
    logger.info('Validating configuration ...')
    try:
        validate(conf, CONF_SCHEMA)
        return conf
    except ValidationError as exception:
        logger.fatal('Invalid configuration. See config.json.example. Reason: %s', exception)
        raise ValidationError(
            best_match(Draft4Validator(CONF_SCHEMA).iter_errors(conf)).message
        )


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


def common_args_parser(description: str):
    """
    Parses given common arguments and returns them as a parsed object.
    """
    parser = argparse.ArgumentParser(
        description=description
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
        '-c', '--config',
        help='specify configuration file (default: config.json)',
        dest='config',
        default='config.json',
        type=str,
        metavar='PATH',
    )
    return parser


def parse_args(args: List[str], description: str):
    """
    Parses given arguments and returns an argparse Namespace instance.
    Returns None if a sub command has been selected and executed.
    """
    parser = common_args_parser(description)
    parser.add_argument(
        '--dry-run-db',
        help='Force dry run to use a local DB "tradesv3.dry_run.sqlite" \
             instead of memory DB. Work only if dry_run is enabled.',
        action='store_true',
        dest='dry_run_db',
    )
    parser.add_argument(
        '-dd', '--datadir',
        help='path to backtest data (default freqdata/tests/testdata',
        dest='datadir',
        default=os.path.join('freqtrade', 'tests', 'testdata'),
        type=str,
        metavar='PATH',
    )
    parser.add_argument(
        '--dynamic-whitelist',
        help='dynamically generate and update whitelist \
             based on 24h BaseVolume (Default 20 currencies)',  # noqa
        dest='dynamic_whitelist',
        const=20,
        type=int,
        metavar='INT',
        nargs='?',
    )

    build_subcommands(parser)
    return parser.parse_args(args)


def build_subcommands(parser: argparse.ArgumentParser) -> None:
    """ Builds and attaches all subcommands """
    from freqtrade.optimize import backtesting, hyperopt

    subparsers = parser.add_subparsers(dest='subparser')

    # Add backtesting subcommand
    backtesting_cmd = subparsers.add_parser('backtesting', help='backtesting module')
    backtesting_cmd.set_defaults(func=backtesting.start)
    backtesting_cmd.add_argument(
        '-l', '--live',
        action='store_true',
        dest='live',
        help='using live data',
    )
    backtesting_cmd.add_argument(
        '-i', '--ticker-interval',
        help='specify ticker interval in minutes (default: 5)',
        dest='ticker_interval',
        default=5,
        type=int,
        metavar='INT',
    )
    backtesting_cmd.add_argument(
        '--realistic-simulation',
        help='uses max_open_trades from config to simulate real world limitations',
        action='store_true',
        dest='realistic_simulation',
    )
    backtesting_cmd.add_argument(
        '-r', '--refresh-pairs-cached',
        help='refresh the pairs files in tests/testdata with the latest data from Bittrex. \
              Use it if you want to run your backtesting with up-to-date data.',
        action='store_true',
        dest='refresh_pairs',
    )

    # Add hyperopt subcommand
    hyperopt_cmd = subparsers.add_parser('hyperopt', help='hyperopt module')
    hyperopt_cmd.set_defaults(func=hyperopt.start)
    hyperopt_cmd.add_argument(
        '-e', '--epochs',
        help='specify number of epochs (default: 100)',
        dest='epochs',
        default=100,
        type=int,
        metavar='INT',
    )
    hyperopt_cmd.add_argument(
        '--use-mongodb',
        help='parallelize evaluations with mongodb (requires mongod in PATH)',
        dest='mongodb',
        action='store_true',
    )
    hyperopt_cmd.add_argument(
        '-i', '--ticker-interval',
        help='specify ticker interval in minutes (default: 5)',
        dest='ticker_interval',
        default=5,
        type=int,
        metavar='INT',
    )


# Required json-schema for user specified config
CONF_SCHEMA = {
    'type': 'object',
    'properties': {
        'max_open_trades': {'type': 'integer', 'minimum': 1},
        'ticker_interval': {'type': 'integer', 'minimum':1, 'maximum':1440},
        'stake_currency': {'type': 'string', 'enum': ['BTC', 'ETH', 'USDT']},
        'stake_amount': {'type': 'number', 'minimum': 0.0005},
        'fiat_display_currency': {'type': 'string', 'enum': ['AUD', 'BRL', 'CAD', 'CHF',
                                                             'CLP', 'CNY', 'CZK', 'DKK',
                                                             'EUR', 'GBP', 'HKD', 'HUF',
                                                             'IDR', 'ILS', 'INR', 'JPY',
                                                             'KRW', 'MXN', 'MYR', 'NOK',
                                                             'NZD', 'PHP', 'PKR', 'PLN',
                                                             'RUB', 'SEK', 'SGD', 'THB',
                                                             'TRY', 'TWD', 'ZAR', 'USD']},
        'dry_run': {'type': 'boolean'},
        'minimal_roi': {
            'type': 'object',
            'patternProperties': {
                '^[0-9.]+$': {'type': 'number'}
            },
            'minProperties': 1
        },
        'stoploss': {'type': 'number', 'maximum': 0, 'exclusiveMaximum': True},
        'unfilledtimeout': {'type': 'integer', 'minimum': 0},
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
        'experimental': {
            'type': 'object',
            'properties': {
                'use_sell_signal': {'type': 'boolean'},
                'sell_profit_only': {'type': 'boolean'}
            }
        },
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
                'process_throttle_secs': {'type': 'number'},
                'interval': {'type': 'integer'}
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
                    'items': {
                        'type': 'string',
                        'pattern': '^[0-9A-Z]+_[0-9A-Z]+$'
                    },
                    'uniqueItems': True
                },
                'pair_blacklist': {
                    'type': 'array',
                    'items': {
                        'type': 'string',
                        'pattern': '^[0-9A-Z]+_[0-9A-Z]+$'
                    },
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
        'ticker_interval',
        'stake_currency',
        'stake_amount',
        'fiat_display_currency',
        'dry_run',
        'minimal_roi',
        'bid_strategy',
        'telegram'
    ]
}
