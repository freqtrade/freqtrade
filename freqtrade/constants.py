# pragma pylint: disable=too-few-public-methods

"""
bot constants
"""
DEFAULT_CONFIG = 'config.json'
DYNAMIC_WHITELIST = 20  # pairs
PROCESS_THROTTLE_SECS = 5  # sec
TICKER_INTERVAL = 5  # min
HYPEROPT_EPOCH = 100  # epochs
RETRY_TIMEOUT = 30  # sec
DEFAULT_STRATEGY = 'DefaultStrategy'
DEFAULT_HYPEROPT = 'DefaultHyperOpts'
DEFAULT_DB_PROD_URL = 'sqlite:///tradesv3.sqlite'
DEFAULT_DB_DRYRUN_URL = 'sqlite://'
UNLIMITED_STAKE_AMOUNT = 'unlimited'
DEFAULT_AMOUNT_RESERVE_PERCENT = 0.05
REQUIRED_ORDERTIF = ['buy', 'sell']
REQUIRED_ORDERTYPES = ['buy', 'sell', 'stoploss', 'stoploss_on_exchange']
ORDERTYPE_POSSIBILITIES = ['limit', 'market']
ORDERTIF_POSSIBILITIES = ['gtc', 'fok', 'ioc']
AVAILABLE_PAIRLISTS = ['StaticPairList', 'VolumePairList']

TICKER_INTERVAL_MINUTES = {
    '1m': 1,
    '3m': 3,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '4h': 240,
    '6h': 360,
    '8h': 480,
    '12h': 720,
    '1d': 1440,
    '3d': 4320,
    '1w': 10080,
}

SUPPORTED_FIAT = [
    "AUD", "BRL", "CAD", "CHF", "CLP", "CNY", "CZK", "DKK",
    "EUR", "GBP", "HKD", "HUF", "IDR", "ILS", "INR", "JPY",
    "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PKR", "PLN",
    "RUB", "SEK", "SGD", "THB", "TRY", "TWD", "ZAR", "USD",
    "BTC", "XBT", "ETH", "XRP", "LTC", "BCH", "USDT"
]

# Required json-schema for user specified config
CONF_SCHEMA = {
    'type': 'object',
    'properties': {
        'max_open_trades': {'type': 'integer', 'minimum': -1},
        'ticker_interval': {'type': 'string', 'enum': list(TICKER_INTERVAL_MINUTES.keys())},
        'stake_currency': {'type': 'string', 'enum': ['BTC', 'XBT', 'ETH', 'USDT', 'EUR', 'USD']},
        'stake_amount': {
            "type": ["number", "string"],
            "minimum": 0.0005,
            "pattern": UNLIMITED_STAKE_AMOUNT
        },
        'fiat_display_currency': {'type': 'string', 'enum': SUPPORTED_FIAT},
        'dry_run': {'type': 'boolean'},
        'process_only_new_candles': {'type': 'boolean'},
        'minimal_roi': {
            'type': 'object',
            'patternProperties': {
                '^[0-9.]+$': {'type': 'number'}
            },
            'minProperties': 1
        },
        'amount_reserve_percent': {'type': 'number', 'minimum': 0.0, 'maximum': 0.5},
        'stoploss': {'type': 'number', 'maximum': 0, 'exclusiveMaximum': True},
        'trailing_stop': {'type': 'boolean'},
        'trailing_stop_positive': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'trailing_stop_positive_offset': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'unfilledtimeout': {
            'type': 'object',
            'properties': {
                'buy': {'type': 'number', 'minimum': 3},
                'sell': {'type': 'number', 'minimum': 10}
            }
        },
        'bid_strategy': {
            'type': 'object',
            'properties': {
                'ask_last_balance': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 1,
                    'exclusiveMaximum': False,
                    'use_order_book': {'type': 'boolean'},
                    'order_book_top': {'type': 'number', 'maximum': 20, 'minimum': 1},
                    'check_depth_of_market': {
                        'type': 'object',
                        'properties': {
                            'enabled': {'type': 'boolean'},
                            'bids_to_ask_delta': {'type': 'number', 'minimum': 0},
                        }
                    },
                },
            },
            'required': ['ask_last_balance']
        },
        'ask_strategy': {
            'type': 'object',
            'properties': {
                'use_order_book': {'type': 'boolean'},
                'order_book_min': {'type': 'number', 'minimum': 1},
                'order_book_max': {'type': 'number', 'minimum': 1, 'maximum': 50}
            }
        },
        'order_types': {
            'type': 'object',
            'properties': {
                'buy': {'type': 'string', 'enum': ORDERTYPE_POSSIBILITIES},
                'sell': {'type': 'string', 'enum': ORDERTYPE_POSSIBILITIES},
                'stoploss': {'type': 'string', 'enum': ORDERTYPE_POSSIBILITIES},
                'stoploss_on_exchange': {'type': 'boolean'},
                'stoploss_on_exchange_interval': {'type': 'number'}
            },
            'required': ['buy', 'sell', 'stoploss', 'stoploss_on_exchange']
        },
        'order_time_in_force': {
            'type': 'object',
            'properties': {
                'buy': {'type': 'string', 'enum': ORDERTIF_POSSIBILITIES},
                'sell': {'type': 'string', 'enum': ORDERTIF_POSSIBILITIES}
            },
            'required': ['buy', 'sell']
        },
        'exchange': {'$ref': '#/definitions/exchange'},
        'edge': {'$ref': '#/definitions/edge'},
        'experimental': {
            'type': 'object',
            'properties': {
                'use_sell_signal': {'type': 'boolean'},
                'sell_profit_only': {'type': 'boolean'},
                'ignore_roi_if_buy_signal_true': {'type': 'boolean'}
            }
        },
        'pairlist': {
            'type': 'object',
            'properties': {
                'method': {'type': 'string', 'enum': AVAILABLE_PAIRLISTS},
                'config': {'type': 'object'}
            },
            'required': ['method']
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
        'webhook': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'boolean'},
                'webhookbuy': {'type': 'object'},
                'webhooksell': {'type': 'object'},
                'webhookstatus': {'type': 'object'},
            },
        },
        'db_url': {'type': 'string'},
        'initial_state': {'type': 'string', 'enum': ['running', 'stopped']},
        'forcebuy_enable': {'type': 'boolean'},
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
                'sandbox': {'type': 'boolean'},
                'key': {'type': 'string'},
                'secret': {'type': 'string'},
                'password': {'type': 'string'},
                'uid': {'type': 'string'},
                'pair_whitelist': {
                    'type': 'array',
                    'items': {
                        'type': 'string',
                        'pattern': '^[0-9A-Z]+/[0-9A-Z]+$'
                    },
                    'uniqueItems': True
                },
                'pair_blacklist': {
                    'type': 'array',
                    'items': {
                        'type': 'string',
                        'pattern': '^[0-9A-Z]+/[0-9A-Z]+$'
                    },
                    'uniqueItems': True
                },
                'outdated_offset': {'type': 'integer', 'minimum': 1},
                'markets_refresh_interval': {'type': 'integer'},
                'ccxt_config': {'type': 'object'},
                'ccxt_async_config': {'type': 'object'}
            },
            'required': ['name', 'key', 'secret', 'pair_whitelist']
        },
        'edge': {
            'type': 'object',
            'properties': {
                "enabled": {'type': 'boolean'},
                "process_throttle_secs": {'type': 'integer', 'minimum': 600},
                "calculate_since_number_of_days": {'type': 'integer'},
                "allowed_risk": {'type': 'number'},
                "capital_available_percentage": {'type': 'number'},
                "stoploss_range_min": {'type': 'number'},
                "stoploss_range_max": {'type': 'number'},
                "stoploss_range_step": {'type': 'number'},
                "minimum_winrate": {'type': 'number'},
                "minimum_expectancy": {'type': 'number'},
                "min_trade_number": {'type': 'number'},
                "max_trade_duration_minute": {'type': 'integer'},
                "remove_pumps": {'type': 'boolean'}
            },
            'required': ['process_throttle_secs', 'allowed_risk', 'capital_available_percentage']
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
        'bid_strategy',
        'telegram'
    ]
}
