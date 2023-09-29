# pragma pylint: disable=too-few-public-methods

"""
bot constants
"""
from typing import Any, Dict, List, Literal, Tuple

from freqtrade.enums import CandleType, PriceType, RPCMessageType


DOCS_LINK = "https://www.freqtrade.io/en/stable"
DEFAULT_CONFIG = 'config.json'
PROCESS_THROTTLE_SECS = 5  # sec
HYPEROPT_EPOCH = 100  # epochs
RETRY_TIMEOUT = 30  # sec
TIMEOUT_UNITS = ['minutes', 'seconds']
EXPORT_OPTIONS = ['none', 'trades', 'signals']
DEFAULT_DB_PROD_URL = 'sqlite:///tradesv3.sqlite'
DEFAULT_DB_DRYRUN_URL = 'sqlite:///tradesv3.dryrun.sqlite'
UNLIMITED_STAKE_AMOUNT = 'unlimited'
DEFAULT_AMOUNT_RESERVE_PERCENT = 0.05
REQUIRED_ORDERTIF = ['entry', 'exit']
REQUIRED_ORDERTYPES = ['entry', 'exit', 'stoploss', 'stoploss_on_exchange']
PRICING_SIDES = ['ask', 'bid', 'same', 'other']
ORDERTYPE_POSSIBILITIES = ['limit', 'market']
_ORDERTIF_POSSIBILITIES = ['GTC', 'FOK', 'IOC', 'PO']
ORDERTIF_POSSIBILITIES = _ORDERTIF_POSSIBILITIES + [t.lower() for t in _ORDERTIF_POSSIBILITIES]
STOPLOSS_PRICE_TYPES = [p for p in PriceType]
HYPEROPT_LOSS_BUILTIN = ['ShortTradeDurHyperOptLoss', 'OnlyProfitHyperOptLoss',
                         'SharpeHyperOptLoss', 'SharpeHyperOptLossDaily',
                         'SortinoHyperOptLoss', 'SortinoHyperOptLossDaily',
                         'CalmarHyperOptLoss',
                         'MaxDrawDownHyperOptLoss', 'MaxDrawDownRelativeHyperOptLoss',
                         'ProfitDrawDownHyperOptLoss']
AVAILABLE_PAIRLISTS = ['StaticPairList', 'VolumePairList', 'ProducerPairList', 'RemotePairList',
                       'AgeFilter', "FullTradesFilter", 'OffsetFilter', 'PerformanceFilter',
                       'PrecisionFilter', 'PriceFilter', 'RangeStabilityFilter',
                       'ShuffleFilter', 'SpreadFilter', 'VolatilityFilter']
AVAILABLE_PROTECTIONS = ['CooldownPeriod',
                         'LowProfitPairs', 'MaxDrawdown', 'StoplossGuard']
AVAILABLE_DATAHANDLERS = ['json', 'jsongz', 'hdf5', 'feather', 'parquet']
BACKTEST_BREAKDOWNS = ['day', 'week', 'month']
BACKTEST_CACHE_AGE = ['none', 'day', 'week', 'month']
BACKTEST_CACHE_DEFAULT = 'day'
DRY_RUN_WALLET = 1000
DATETIME_PRINT_FORMAT = '%Y-%m-%d %H:%M:%S'
MATH_CLOSE_PREC = 1e-14  # Precision used for float comparisons
DEFAULT_DATAFRAME_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']
# Don't modify sequence of DEFAULT_TRADES_COLUMNS
# it has wide consequences for stored trades files
DEFAULT_TRADES_COLUMNS = ['timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost']
TRADES_DTYPES = {
    'timestamp': 'int64',
    'id': 'str',
    'type': 'str',
    'side': 'str',
    'price': 'float64',
    'amount': 'float64',
    'cost': 'float64',
}
TRADING_MODES = ['spot', 'margin', 'futures']
MARGIN_MODES = ['cross', 'isolated', '']

LAST_BT_RESULT_FN = '.last_result.json'
FTHYPT_FILEVERSION = 'fthypt_fileversion'

USERPATH_HYPEROPTS = 'hyperopts'
USERPATH_STRATEGIES = 'strategies'
USERPATH_NOTEBOOKS = 'notebooks'
USERPATH_FREQAIMODELS = 'freqaimodels'

TELEGRAM_SETTING_OPTIONS = ['on', 'off', 'silent']
WEBHOOK_FORMAT_OPTIONS = ['form', 'json', 'raw']
FULL_DATAFRAME_THRESHOLD = 100
CUSTOM_TAG_MAX_LENGTH = 255
DL_DATA_TIMEFRAMES = ['1m', '5m']

ENV_VAR_PREFIX = 'FREQTRADE__'

CANCELED_EXCHANGE_STATES = ('cancelled', 'canceled', 'expired')
NON_OPEN_EXCHANGE_STATES = CANCELED_EXCHANGE_STATES + ('closed',)

# Define decimals per coin for outputs
# Only used for outputs.
DECIMAL_PER_COIN_FALLBACK = 3  # Should be low to avoid listing all possible FIAT's
DECIMALS_PER_COIN = {
    'BTC': 8,
    'ETH': 5,
}

DUST_PER_COIN = {
    'BTC': 0.0001,
    'ETH': 0.01
}

# Source files with destination directories within user-directory
USER_DATA_FILES = {
    'sample_strategy.py': USERPATH_STRATEGIES,
    'sample_hyperopt_loss.py': USERPATH_HYPEROPTS,
    'strategy_analysis_example.ipynb': USERPATH_NOTEBOOKS,
}

SUPPORTED_FIAT = [
    "AUD", "BRL", "CAD", "CHF", "CLP", "CNY", "CZK", "DKK",
    "EUR", "GBP", "HKD", "HUF", "IDR", "ILS", "INR", "JPY",
    "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PKR", "PLN",
    "RUB", "UAH", "SEK", "SGD", "THB", "TRY", "TWD", "ZAR",
    "USD", "BTC", "ETH", "XRP", "LTC", "BCH"
]

MINIMAL_CONFIG = {
    "stake_currency": "",
    "dry_run": True,
    "exchange": {
        "name": "",
        "key": "",
        "secret": "",
        "pair_whitelist": [],
        "ccxt_async_config": {
        }
    }
}

__MESSAGE_TYPE_DICT: Dict[str, Dict[str, str]] = {x: {'type': 'object'} for x in RPCMessageType}

# Required json-schema for user specified config
CONF_SCHEMA = {
    'type': 'object',
    'properties': {
        'max_open_trades': {'type': ['integer', 'number'], 'minimum': -1},
        'new_pairs_days': {'type': 'integer', 'default': 30},
        'timeframe': {'type': 'string'},
        'stake_currency': {'type': 'string'},
        'stake_amount': {
            'type': ['number', 'string'],
            'minimum': 0.0001,
            'pattern': UNLIMITED_STAKE_AMOUNT
        },
        'tradable_balance_ratio': {
            'type': 'number',
            'minimum': 0.0,
            'maximum': 1,
            'default': 0.99
        },
        'available_capital': {
            'type': 'number',
            'minimum': 0,
        },
        'amend_last_stake_amount': {'type': 'boolean', 'default': False},
        'last_stake_amount_min_ratio': {
            'type': 'number', 'minimum': 0.0, 'maximum': 1.0, 'default': 0.5
        },
        'fiat_display_currency': {'type': 'string', 'enum': SUPPORTED_FIAT},
        'dry_run': {'type': 'boolean'},
        'dry_run_wallet': {'type': 'number', 'default': DRY_RUN_WALLET},
        'cancel_open_orders_on_exit': {'type': 'boolean', 'default': False},
        'process_only_new_candles': {'type': 'boolean'},
        'minimal_roi': {
            'type': 'object',
            'patternProperties': {
                '^[0-9.]+$': {'type': 'number'}
            },
        },
        'amount_reserve_percent': {'type': 'number', 'minimum': 0.0, 'maximum': 0.5},
        'stoploss': {'type': 'number', 'maximum': 0, 'exclusiveMaximum': True},
        'trailing_stop': {'type': 'boolean'},
        'trailing_stop_positive': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'trailing_stop_positive_offset': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'trailing_only_offset_is_reached': {'type': 'boolean'},
        'use_exit_signal': {'type': 'boolean'},
        'exit_profit_only': {'type': 'boolean'},
        'exit_profit_offset': {'type': 'number'},
        'ignore_roi_if_entry_signal': {'type': 'boolean'},
        'ignore_buying_expired_candle_after': {'type': 'number'},
        'trading_mode': {'type': 'string', 'enum': TRADING_MODES},
        'margin_mode': {'type': 'string', 'enum': MARGIN_MODES},
        'reduce_df_footprint': {'type': 'boolean', 'default': False},
        'minimum_trade_amount': {'type': 'number', 'default': 10},
        'targeted_trade_amount': {'type': 'number', 'default': 20},
        'lookahead_analysis_exportfilename': {'type': 'string'},
        'startup_candle': {
            'type': 'array',
            'uniqueItems': True,
            'default': [199, 399, 499, 999, 1999],
        },
        'liquidation_buffer': {'type': 'number', 'minimum': 0.0, 'maximum': 0.99},
        'backtest_breakdown': {
            'type': 'array',
            'items': {'type': 'string', 'enum': BACKTEST_BREAKDOWNS}
        },
        'bot_name': {'type': 'string'},
        'unfilledtimeout': {
            'type': 'object',
            'properties': {
                'entry': {'type': 'number', 'minimum': 1},
                'exit': {'type': 'number', 'minimum': 1},
                'exit_timeout_count': {'type': 'number', 'minimum': 0, 'default': 0},
                'unit': {'type': 'string', 'enum': TIMEOUT_UNITS, 'default': 'minutes'}
            }
        },
        'entry_pricing': {
            'type': 'object',
            'properties': {
                'price_last_balance': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 1,
                    'exclusiveMaximum': False,
                },
                'price_side': {'type': 'string', 'enum': PRICING_SIDES, 'default': 'same'},
                'use_order_book': {'type': 'boolean'},
                'order_book_top': {'type': 'integer', 'minimum': 1, 'maximum': 50, },
                'check_depth_of_market': {
                    'type': 'object',
                    'properties': {
                        'enabled': {'type': 'boolean'},
                        'bids_to_ask_delta': {'type': 'number', 'minimum': 0},
                    }
                },
            },
            'required': ['price_side']
        },
        'exit_pricing': {
            'type': 'object',
            'properties': {
                'price_side': {'type': 'string', 'enum': PRICING_SIDES, 'default': 'same'},
                'price_last_balance': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 1,
                    'exclusiveMaximum': False,
                },
                'use_order_book': {'type': 'boolean'},
                'order_book_top': {'type': 'integer', 'minimum': 1, 'maximum': 50, },
            },
            'required': ['price_side']
        },
        'custom_price_max_distance_ratio': {
            'type': 'number', 'minimum': 0.0
        },
        'order_types': {
            'type': 'object',
            'properties': {
                'entry': {'type': 'string', 'enum': ORDERTYPE_POSSIBILITIES},
                'exit': {'type': 'string', 'enum': ORDERTYPE_POSSIBILITIES},
                'force_exit': {'type': 'string', 'enum': ORDERTYPE_POSSIBILITIES},
                'force_entry': {'type': 'string', 'enum': ORDERTYPE_POSSIBILITIES},
                'emergency_exit': {
                    'type': 'string',
                    'enum': ORDERTYPE_POSSIBILITIES,
                    'default': 'market'},
                'stoploss': {'type': 'string', 'enum': ORDERTYPE_POSSIBILITIES},
                'stoploss_on_exchange': {'type': 'boolean'},
                'stoploss_price_type': {'type': 'string', 'enum': STOPLOSS_PRICE_TYPES},
                'stoploss_on_exchange_interval': {'type': 'number'},
                'stoploss_on_exchange_limit_ratio': {'type': 'number', 'minimum': 0.0,
                                                     'maximum': 1.0}
            },
            'required': ['entry', 'exit', 'stoploss', 'stoploss_on_exchange']
        },
        'order_time_in_force': {
            'type': 'object',
            'properties': {
                'entry': {'type': 'string', 'enum': ORDERTIF_POSSIBILITIES},
                'exit': {'type': 'string', 'enum': ORDERTIF_POSSIBILITIES}
            },
            'required': REQUIRED_ORDERTIF
        },
        'exchange': {'$ref': '#/definitions/exchange'},
        'edge': {'$ref': '#/definitions/edge'},
        'freqai': {'$ref': '#/definitions/freqai'},
        'external_message_consumer': {'$ref': '#/definitions/external_message_consumer'},
        'experimental': {
            'type': 'object',
            'properties': {
                'block_bad_exchanges': {'type': 'boolean'}
            }
        },
        'pairlists': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'method': {'type': 'string', 'enum': AVAILABLE_PAIRLISTS},
                },
                'required': ['method'],
            }
        },
        'protections': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'method': {'type': 'string', 'enum': AVAILABLE_PROTECTIONS},
                    'stop_duration': {'type': 'number', 'minimum': 0.0},
                    'stop_duration_candles': {'type': 'number', 'minimum': 0},
                    'trade_limit': {'type': 'number', 'minimum': 1},
                    'lookback_period': {'type': 'number', 'minimum': 1},
                    'lookback_period_candles': {'type': 'number', 'minimum': 1},
                },
                'required': ['method'],
            }
        },
        'telegram': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'boolean'},
                'token': {'type': 'string'},
                'chat_id': {'type': 'string'},
                'allow_custom_messages': {'type': 'boolean', 'default': True},
                'balance_dust_level': {'type': 'number', 'minimum': 0.0},
                'notification_settings': {
                    'type': 'object',
                    'default': {},
                    'properties': {
                        'status': {'type': 'string', 'enum': TELEGRAM_SETTING_OPTIONS},
                        'warning': {'type': 'string', 'enum': TELEGRAM_SETTING_OPTIONS},
                        'startup': {'type': 'string', 'enum': TELEGRAM_SETTING_OPTIONS},
                        'entry': {'type': 'string', 'enum': TELEGRAM_SETTING_OPTIONS},
                        'entry_fill': {
                            'type': 'string',
                            'enum': TELEGRAM_SETTING_OPTIONS,
                            'default': 'off'
                        },
                        'entry_cancel': {'type': 'string', 'enum': TELEGRAM_SETTING_OPTIONS, },
                        'exit': {
                            'type': ['string', 'object'],
                            'additionalProperties': {
                                'type': 'string',
                                'enum': TELEGRAM_SETTING_OPTIONS
                            }
                        },
                        'exit_fill': {
                            'type': 'string',
                            'enum': TELEGRAM_SETTING_OPTIONS,
                            'default': 'on'
                        },
                        'exit_cancel': {'type': 'string', 'enum': TELEGRAM_SETTING_OPTIONS},
                        'protection_trigger': {
                            'type': 'string',
                            'enum': TELEGRAM_SETTING_OPTIONS,
                            'default': 'on'
                        },
                        'protection_trigger_global': {
                            'type': 'string',
                            'enum': TELEGRAM_SETTING_OPTIONS,
                            'default': 'on'
                        },
                        'show_candle': {
                            'type': 'string',
                            'enum': ['off', 'ohlc'],
                            'default': 'off'
                        },
                        'strategy_msg': {
                            'type': 'string',
                            'enum': TELEGRAM_SETTING_OPTIONS,
                            'default': 'on'
                        },
                    }
                },
                'reload': {'type': 'boolean'},
            },
            'required': ['enabled', 'token', 'chat_id'],
        },
        'webhook': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'boolean'},
                'url': {'type': 'string'},
                'format': {'type': 'string', 'enum': WEBHOOK_FORMAT_OPTIONS, 'default': 'form'},
                'retries': {'type': 'integer', 'minimum': 0},
                'retry_delay': {'type': 'number', 'minimum': 0},
                **__MESSAGE_TYPE_DICT,
                # **{x: {'type': 'object'} for x in RPCMessageType},
                # Below -> Deprecated
                'webhookentry': {'type': 'object'},
                'webhookentrycancel': {'type': 'object'},
                'webhookentryfill': {'type': 'object'},
                'webhookexit': {'type': 'object'},
                'webhookexitcancel': {'type': 'object'},
                'webhookexitfill': {'type': 'object'},
                'webhookstatus': {'type': 'object'},
            },
        },
        'discord': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'boolean'},
                'webhook_url': {'type': 'string'},
                "exit_fill": {
                    'type': 'array', 'items': {'type': 'object'},
                    'default': [
                        {"Trade ID": "{trade_id}"},
                        {"Exchange": "{exchange}"},
                        {"Pair": "{pair}"},
                        {"Direction": "{direction}"},
                        {"Open rate": "{open_rate}"},
                        {"Close rate": "{close_rate}"},
                        {"Amount": "{amount}"},
                        {"Open date": "{open_date:%Y-%m-%d %H:%M:%S}"},
                        {"Close date": "{close_date:%Y-%m-%d %H:%M:%S}"},
                        {"Profit": "{profit_amount} {stake_currency}"},
                        {"Profitability": "{profit_ratio:.2%}"},
                        {"Enter tag": "{enter_tag}"},
                        {"Exit Reason": "{exit_reason}"},
                        {"Strategy": "{strategy}"},
                        {"Timeframe": "{timeframe}"},
                    ]
                },
                "entry_fill": {
                    'type': 'array', 'items': {'type': 'object'},
                    'default': [
                        {"Trade ID": "{trade_id}"},
                        {"Exchange": "{exchange}"},
                        {"Pair": "{pair}"},
                        {"Direction": "{direction}"},
                        {"Open rate": "{open_rate}"},
                        {"Amount": "{amount}"},
                        {"Open date": "{open_date:%Y-%m-%d %H:%M:%S}"},
                        {"Enter tag": "{enter_tag}"},
                        {"Strategy": "{strategy} {timeframe}"},
                    ]
                },
            }
        },
        'api_server': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'boolean'},
                'listen_ip_address': {'format': 'ipv4'},
                'listen_port': {
                    'type': 'integer',
                    'minimum': 1024,
                    'maximum': 65535
                },
                'username': {'type': 'string'},
                'password': {'type': 'string'},
                'ws_token': {'type': ['string', 'array'], 'items': {'type': 'string'}},
                'jwt_secret_key': {'type': 'string'},
                'CORS_origins': {'type': 'array', 'items': {'type': 'string'}},
                'verbosity': {'type': 'string', 'enum': ['error', 'info']},
            },
            'required': ['enabled', 'listen_ip_address', 'listen_port', 'username', 'password']
        },
        'db_url': {'type': 'string'},
        'export': {'type': 'string', 'enum': EXPORT_OPTIONS, 'default': 'trades'},
        'disableparamexport': {'type': 'boolean'},
        'initial_state': {'type': 'string', 'enum': ['running', 'stopped']},
        'force_entry_enable': {'type': 'boolean'},
        'disable_dataframe_checks': {'type': 'boolean'},
        'internals': {
            'type': 'object',
            'default': {},
            'properties': {
                'process_throttle_secs': {'type': 'integer'},
                'interval': {'type': 'integer'},
                'sd_notify': {'type': 'boolean'},
            }
        },
        'dataformat_ohlcv': {
            'type': 'string',
            'enum': AVAILABLE_DATAHANDLERS,
            'default': 'feather'
        },
        'dataformat_trades': {
            'type': 'string',
            'enum': AVAILABLE_DATAHANDLERS,
            'default': 'feather'
        },
        'position_adjustment_enable': {'type': 'boolean'},
        'max_entry_position_adjustment': {'type': ['integer', 'number'], 'minimum': -1},
    },
    'definitions': {
        'exchange': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'key': {'type': 'string', 'default': ''},
                'secret': {'type': 'string', 'default': ''},
                'password': {'type': 'string', 'default': ''},
                'uid': {'type': 'string'},
                'pair_whitelist': {
                    'type': 'array',
                    'items': {
                        'type': 'string',
                    },
                    'uniqueItems': True
                },
                'pair_blacklist': {
                    'type': 'array',
                    'items': {
                        'type': 'string',
                    },
                    'uniqueItems': True
                },
                'unknown_fee_rate': {'type': 'number'},
                'outdated_offset': {'type': 'integer', 'minimum': 1},
                'markets_refresh_interval': {'type': 'integer'},
                'ccxt_config': {'type': 'object'},
                'ccxt_async_config': {'type': 'object'}
            },
            'required': ['name']
        },
        'edge': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'boolean'},
                'process_throttle_secs': {'type': 'integer', 'minimum': 600},
                'calculate_since_number_of_days': {'type': 'integer'},
                'allowed_risk': {'type': 'number'},
                'stoploss_range_min': {'type': 'number'},
                'stoploss_range_max': {'type': 'number'},
                'stoploss_range_step': {'type': 'number'},
                'minimum_winrate': {'type': 'number'},
                'minimum_expectancy': {'type': 'number'},
                'min_trade_number': {'type': 'number'},
                'max_trade_duration_minute': {'type': 'integer'},
                'remove_pumps': {'type': 'boolean'}
            },
            'required': ['process_throttle_secs', 'allowed_risk']
        },
        'external_message_consumer': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'boolean', 'default': False},
                'producers': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'name': {'type': 'string'},
                            'host': {'type': 'string'},
                            'port': {
                                'type': 'integer',
                                'default': 8080,
                                'minimum': 0,
                                'maximum': 65535
                            },
                            'secure': {'type': 'boolean', 'default': False},
                            'ws_token': {'type': 'string'},
                        },
                        'required': ['name', 'host', 'ws_token']
                    }
                },
                'wait_timeout': {'type': 'integer', 'minimum': 0},
                'sleep_time': {'type': 'integer', 'minimum': 0},
                'ping_timeout': {'type': 'integer', 'minimum': 0},
                'remove_entry_exit_signals': {'type': 'boolean', 'default': False},
                'initial_candle_limit': {
                    'type': 'integer',
                    'minimum': 0,
                    'maximum': 1500,
                    'default': 1500
                },
                'message_size_limit': {  # In megabytes
                    'type': 'integer',
                    'minimum': 1,
                    'maxmium': 20,
                    'default': 8,
                }
            },
            'required': ['producers']
        },
        "freqai": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "default": False},
                "keras": {"type": "boolean", "default": False},
                "write_metrics_to_disk": {"type": "boolean", "default": False},
                "purge_old_models": {"type": ["boolean", "number"], "default": 2},
                "conv_width": {"type": "integer", "default": 1},
                "train_period_days": {"type": "integer", "default": 0},
                "backtest_period_days": {"type": "number", "default": 7},
                "identifier": {"type": "string", "default": "example"},
                "feature_parameters": {
                    "type": "object",
                    "properties": {
                        "include_corr_pairlist": {"type": "array"},
                        "include_timeframes": {"type": "array"},
                        "label_period_candles": {"type": "integer"},
                        "include_shifted_candles": {"type": "integer", "default": 0},
                        "DI_threshold": {"type": "number", "default": 0},
                        "weight_factor": {"type": "number", "default": 0},
                        "principal_component_analysis": {"type": "boolean", "default": False},
                        "use_SVM_to_remove_outliers": {"type": "boolean", "default": False},
                        "plot_feature_importances": {"type": "integer", "default": 0},
                        "svm_params": {"type": "object",
                                       "properties": {
                                           "shuffle": {"type": "boolean", "default": False},
                                           "nu": {"type": "number", "default": 0.1}
                                           },
                                       },
                        "shuffle_after_split": {"type": "boolean", "default": False},
                        "buffer_train_data_candles": {"type": "integer", "default": 0}
                    },
                    "required": ["include_timeframes", "include_corr_pairlist", ]
                },
                "data_split_parameters": {
                    "type": "object",
                    "properties": {
                        "test_size": {"type": "number"},
                        "random_state": {"type": "integer"},
                        "shuffle": {"type": "boolean", "default": False}
                    },
                },
                "model_training_parameters": {
                    "type": "object"
                },
                "rl_config": {
                    "type": "object",
                    "properties": {
                        "drop_ohlc_from_features": {"type": "boolean", "default": False},
                        "train_cycles": {"type": "integer"},
                        "max_trade_duration_candles": {"type": "integer"},
                        "add_state_info": {"type": "boolean", "default": False},
                        "max_training_drawdown_pct": {"type": "number", "default": 0.02},
                        "cpu_count": {"type": "integer", "default": 1},
                        "model_type": {"type": "string", "default": "PPO"},
                        "policy_type": {"type": "string", "default": "MlpPolicy"},
                        "net_arch": {"type": "array", "default": [128, 128]},
                        "randomize_starting_position": {"type": "boolean", "default": False},
                        "progress_bar": {"type": "boolean", "default": True},
                        "model_reward_parameters": {
                            "type": "object",
                            "properties": {
                                "rr": {"type": "number", "default": 1},
                                "profit_aim": {"type": "number", "default": 0.025}
                            }
                        }
                    },
                },
            },
            "required": [
                "enabled",
                "train_period_days",
                "backtest_period_days",
                "identifier",
                "feature_parameters",
                "data_split_parameters"
            ]
        },
    },
}

SCHEMA_TRADE_REQUIRED = [
    'exchange',
    'timeframe',
    'max_open_trades',
    'stake_currency',
    'stake_amount',
    'tradable_balance_ratio',
    'last_stake_amount_min_ratio',
    'dry_run',
    'dry_run_wallet',
    'exit_pricing',
    'entry_pricing',
    'stoploss',
    'minimal_roi',
    'internals',
    'dataformat_ohlcv',
    'dataformat_trades',
]

SCHEMA_BACKTEST_REQUIRED = [
    'exchange',
    'stake_currency',
    'stake_amount',
    'dry_run_wallet',
    'dataformat_ohlcv',
    'dataformat_trades',
]
SCHEMA_BACKTEST_REQUIRED_FINAL = SCHEMA_BACKTEST_REQUIRED + [
    'stoploss',
    'minimal_roi',
    'max_open_trades'
]

SCHEMA_MINIMAL_REQUIRED = [
    'exchange',
    'dry_run',
    'dataformat_ohlcv',
    'dataformat_trades',
]
SCHEMA_MINIMAL_WEBSERVER = SCHEMA_MINIMAL_REQUIRED + [
    'api_server',
]

CANCEL_REASON = {
    "TIMEOUT": "cancelled due to timeout",
    "PARTIALLY_FILLED_KEEP_OPEN": "partially filled - keeping order open",
    "PARTIALLY_FILLED": "partially filled",
    "FULLY_CANCELLED": "fully cancelled",
    "ALL_CANCELLED": "cancelled (all unfilled and partially filled open orders cancelled)",
    "CANCELLED_ON_EXCHANGE": "cancelled on exchange",
    "FORCE_EXIT": "forcesold",
    "REPLACE": "cancelled to be replaced by new limit order",
    "REPLACE_FAILED": "failed to replace order, deleting Trade",
    "USER_CANCEL": "user requested order cancel"
}

# List of pairs with their timeframes
PairWithTimeframe = Tuple[str, str, CandleType]
ListPairsWithTimeframes = List[PairWithTimeframe]

# Type for trades list
TradeList = List[List]

LongShort = Literal['long', 'short']
EntryExit = Literal['entry', 'exit']
BuySell = Literal['buy', 'sell']
MakerTaker = Literal['maker', 'taker']
BidAsk = Literal['bid', 'ask']
OBLiteral = Literal['asks', 'bids']

Config = Dict[str, Any]
# Exchange part of the configuration.
ExchangeConfig = Dict[str, Any]
IntOrInf = float


EntryExecuteMode = Literal['initial', 'pos_adjust', 'replace']
