# pragma pylint: disable=too-few-public-methods

"""
List bot constants
"""


class Constants(object):
    """
    Static class that contain all bot constants
    """
    DYNAMIC_WHITELIST = 20  # pairs
    PROCESS_THROTTLE_SECS = 5  # sec
    TICKER_INTERVAL = 5  # min
    HYPEROPT_EPOCH = 100  # epochs
    RETRY_TIMEOUT = 30  # sec
    DEFAULT_STRATEGY = 'default_strategy'

    # Required json-schema for user specified config
    CONF_SCHEMA = {
        'type': 'object',
        'properties': {
            'max_open_trades': {'type': 'integer', 'minimum': 1},
            'ticker_interval': {'type': 'integer', 'enum': [1, 5, 30, 60, 1440]},
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
            'stake_currency',
            'stake_amount',
            'fiat_display_currency',
            'dry_run',
            'bid_strategy',
            'telegram'
        ]
    }
