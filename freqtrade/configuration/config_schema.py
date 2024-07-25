# Required json-schema for user specified config
from typing import Dict

from freqtrade.constants import (
    AVAILABLE_DATAHANDLERS,
    AVAILABLE_PAIRLISTS,
    AVAILABLE_PROTECTIONS,
    BACKTEST_BREAKDOWNS,
    DRY_RUN_WALLET,
    EXPORT_OPTIONS,
    MARGIN_MODES,
    ORDERTIF_POSSIBILITIES,
    ORDERTYPE_POSSIBILITIES,
    PRICING_SIDES,
    REQUIRED_ORDERTIF,
    STOPLOSS_PRICE_TYPES,
    SUPPORTED_FIAT,
    TELEGRAM_SETTING_OPTIONS,
    TIMEOUT_UNITS,
    TRADING_MODES,
    UNLIMITED_STAKE_AMOUNT,
    WEBHOOK_FORMAT_OPTIONS,
)
from freqtrade.enums import RPCMessageType


__MESSAGE_TYPE_DICT: Dict[str, Dict[str, str]] = {x: {"type": "object"} for x in RPCMessageType}

CONF_SCHEMA = {
    "type": "object",
    "properties": {
        "max_open_trades": {
            "description": "Maximum number of open trades. -1 for unlimited.",
            "type": ["integer", "number"],
            "minimum": -1,
        },
        "new_pairs_days": {
            "description": "Download data of new pairs for given number of days",
            "type": "integer",
            "default": 30,
        },
        "timeframe": {
            "description": (
                "The timeframe to use (e.g `1m`, `5m`, `15m`, `30m`, `1h` ...)."
                "Usually missing in configuration and specified in the strategy."
            ),
            "type": "string",
        },
        "stake_currency": {
            "description": "Currency used for staking.",
            "type": "string",
        },
        "stake_amount": {
            "description": "Amount to stake per trade.",
            "type": ["number", "string"],
            "minimum": 0.0001,
            "pattern": UNLIMITED_STAKE_AMOUNT,
        },
        "tradable_balance_ratio": {
            "description": "Ratio of balance that is tradable.",
            "type": "number",
            "minimum": 0.0,
            "maximum": 1,
            "default": 0.99,
        },
        "available_capital": {
            "description": "Total capital available for trading.",
            "type": "number",
            "minimum": 0,
        },
        "amend_last_stake_amount": {
            "description": "Whether to amend the last stake amount.",
            "type": "boolean",
            "default": False,
        },
        "last_stake_amount_min_ratio": {
            "description": "Minimum ratio for the last stake amount.",
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.5,
        },
        "fiat_display_currency": {
            "description": "Fiat currency for display purposes.",
            "type": "string",
            "enum": SUPPORTED_FIAT,
        },
        "dry_run": {
            "description": "Enable or disable dry run mode.",
            "type": "boolean",
        },
        "dry_run_wallet": {
            "description": "Initial wallet balance for dry run mode.",
            "type": "number",
            "default": DRY_RUN_WALLET,
        },
        "cancel_open_orders_on_exit": {
            "description": "Cancel open orders when exiting.",
            "type": "boolean",
            "default": False,
        },
        "process_only_new_candles": {
            "description": "Process only new candles.",
            "type": "boolean",
        },
        "minimal_roi": {
            "description": "Minimum return on investment.",
            "type": "object",
            "patternProperties": {"^[0-9.]+$": {"type": "number"}},
        },
        "amount_reserve_percent": {
            "description": "Percentage of amount to reserve.",
            "type": "number",
            "minimum": 0.0,
            "maximum": 0.5,
        },
        "stoploss": {
            "description": "Value (as ratio) to use as Stoploss value.",
            "type": "number",
            "maximum": 0,
            "exclusiveMaximum": True,
        },
        "trailing_stop": {
            "description": "Enable or disable trailing stop.",
            "type": "boolean",
        },
        "trailing_stop_positive": {
            "description": "Positive offset for trailing stop.",
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "trailing_stop_positive_offset": {
            "description": "Offset for trailing stop to activate.",
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "trailing_only_offset_is_reached": {
            "description": "Use trailing stop only when offset is reached.",
            "type": "boolean",
        },
        "use_exit_signal": {
            "description": "Use exit signal for trades.",
            "type": "boolean",
        },
        "exit_profit_only": {
            "description": (
                "Exit only when in profit. Exit signals are ignored as "
                "long as profit is < exit_profit_offset."
            ),
            "type": "boolean",
        },
        "exit_profit_offset": {
            "description": "Offset for profit exit.",
            "type": "number",
        },
        "fee": {
            "description": "Trading fee percentage. Can help to simulate slippage in backtesting",
            "type": "number",
            "minimum": 0,
            "maximum": 0.1,
        },
        "ignore_roi_if_entry_signal": {
            "description": "Ignore ROI if entry signal is present.",
            "type": "boolean",
        },
        "ignore_buying_expired_candle_after": {
            "description": "Ignore buying after candle expiration time.",
            "type": "number",
        },
        "trading_mode": {
            "description": "Mode of trading (e.g., spot, margin).",
            "type": "string",
            "enum": TRADING_MODES,
        },
        "margin_mode": {
            "description": "Margin mode for trading.",
            "type": "string",
            "enum": MARGIN_MODES,
        },
        "reduce_df_footprint": {
            "description": "Reduce DataFrame footprint by casting columns to float32/int32.",
            "type": "boolean",
            "default": False,
        },
        "minimum_trade_amount": {
            "description": "Minimum amount for a trade - only used for lookahead-analysis",
            "type": "number",
            "default": 10,
        },
        "targeted_trade_amount": {
            "description": "Targeted trade amount for lookahead analysis.",
            "type": "number",
            "default": 20,
        },
        "lookahead_analysis_exportfilename": {
            "description": "csv Filename for lookahead analysis export.",
            "type": "string",
        },
        "startup_candle": {
            "description": "Startup candle configuration.",
            "type": "array",
            "uniqueItems": True,
            "default": [199, 399, 499, 999, 1999],
        },
        "liquidation_buffer": {
            "description": "Buffer ratio for liquidation.",
            "type": "number",
            "minimum": 0.0,
            "maximum": 0.99,
        },
        "backtest_breakdown": {
            "description": "Breakdown configuration for backtesting.",
            "type": "array",
            "items": {"type": "string", "enum": BACKTEST_BREAKDOWNS},
        },
        "bot_name": {
            "description": "Name of the trading bot. Passed via API to a client.",
            "type": "string",
        },
        "unfilledtimeout": {
            "description": "Timeout configuration for unfilled orders.",
            "type": "object",
            "properties": {
                "entry": {
                    "description": "Timeout for entry orders in unit.",
                    "type": "number",
                    "minimum": 1,
                },
                "exit": {
                    "description": "Timeout for exit orders in unit.",
                    "type": "number",
                    "minimum": 1,
                },
                "exit_timeout_count": {
                    "description": "Number of times to retry exit orders before giving up.",
                    "type": "number",
                    "minimum": 0,
                    "default": 0,
                },
                "unit": {
                    "description": "Unit of time for the timeout (e.g., seconds, minutes).",
                    "type": "string",
                    "enum": TIMEOUT_UNITS,
                    "default": "minutes",
                },
            },
        },
        "entry_pricing": {
            "description": "Configuration for entry pricing.",
            "type": "object",
            "properties": {
                "price_last_balance": {
                    "description": "Balance ratio for the last price.",
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "exclusiveMaximum": False,
                },
                "price_side": {
                    "description": "Side of the price to use (e.g., bid, ask, same).",
                    "type": "string",
                    "enum": PRICING_SIDES,
                    "default": "same",
                },
                "use_order_book": {
                    "description": "Whether to use the order book for pricing.",
                    "type": "boolean",
                },
                "order_book_top": {
                    "description": "Top N levels of the order book to consider.",
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                },
                "check_depth_of_market": {
                    "description": "Configuration for checking the depth of the market.",
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "description": "Enable or disable depth of market check.",
                            "type": "boolean",
                        },
                        "bids_to_ask_delta": {
                            "description": "Delta between bids and asks to consider.",
                            "type": "number",
                            "minimum": 0,
                        },
                    },
                },
            },
            "required": ["price_side"],
        },
        "exit_pricing": {
            "description": "Configuration for exit pricing.",
            "type": "object",
            "properties": {
                "price_side": {
                    "description": "Side of the price to use (e.g., bid, ask, same).",
                    "type": "string",
                    "enum": PRICING_SIDES,
                    "default": "same",
                },
                "price_last_balance": {
                    "description": "Balance ratio for the last price.",
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "exclusiveMaximum": False,
                },
                "use_order_book": {
                    "description": "Whether to use the order book for pricing.",
                    "type": "boolean",
                },
                "order_book_top": {
                    "description": "Top N levels of the order book to consider.",
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["price_side"],
        },
        "custom_price_max_distance_ratio": {
            "description": "Maximum distance ratio between current and custom entry or exit price.",
            "type": "number",
            "minimum": 0.0,
            "maximum": 1,
            "default": 0.02,
        },
        "order_types": {
            "description": "Configuration of order types.",
            "type": "object",
            "properties": {
                "entry": {
                    "description": "Order type for entry (e.g., limit, market).",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                },
                "exit": {
                    "description": "Order type for exit (e.g., limit, market).",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                },
                "force_exit": {
                    "description": "Order type for forced exit (e.g., limit, market).",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                },
                "force_entry": {
                    "description": "Order type for forced entry (e.g., limit, market).",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                },
                "emergency_exit": {
                    "description": "Order type for emergency exit (e.g., limit, market).",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                    "default": "market",
                },
                "stoploss": {
                    "description": "Order type for stop loss (e.g., limit, market).",
                    "type": "string",
                    "enum": ORDERTYPE_POSSIBILITIES,
                },
                "stoploss_on_exchange": {
                    "description": "Whether to place stop loss on the exchange.",
                    "type": "boolean",
                },
                "stoploss_price_type": {
                    "description": "Price type for stop loss (e.g., last, mark, index).",
                    "type": "string",
                    "enum": STOPLOSS_PRICE_TYPES,
                },
                "stoploss_on_exchange_interval": {
                    "description": "Interval for stop loss on exchange in seconds.",
                    "type": "number",
                },
                "stoploss_on_exchange_limit_ratio": {
                    "description": "Limit ratio for stop loss on exchange.",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["entry", "exit", "stoploss", "stoploss_on_exchange"],
        },
        "order_time_in_force": {
            "description": "Time in force configuration for orders.",
            "type": "object",
            "properties": {
                "entry": {
                    "description": "Time in force for entry orders.",
                    "type": "string",
                    "enum": ORDERTIF_POSSIBILITIES,
                },
                "exit": {
                    "description": "Time in force for exit orders.",
                    "type": "string",
                    "enum": ORDERTIF_POSSIBILITIES,
                },
            },
            "required": REQUIRED_ORDERTIF,
        },
        "coingecko": {
            "description": "Configuration for CoinGecko API.",
            "type": "object",
            "properties": {
                "is_demo": {
                    "description": "Whether to use CoinGecko in demo mode.",
                    "type": "boolean",
                    "default": True,
                },
                "api_key": {"description": "API key for accessing CoinGecko.", "type": "string"},
            },
            "required": ["is_demo", "api_key"],
        },
        "exchange": {
            "description": "Exchange configuration.",
            "$ref": "#/definitions/exchange",
        },
        "edge": {
            "description": "Edge configuration.",
            "$ref": "#/definitions/edge",
        },
        "freqai": {
            "description": "FreqAI configuration.",
            "$ref": "#/definitions/freqai",
        },
        "external_message_consumer": {
            "description": "Configuration for external message consumer.",
            "$ref": "#/definitions/external_message_consumer",
        },
        "experimental": {
            "description": "Experimental configuration.",
            "type": "object",
            "properties": {"block_bad_exchanges": {"type": "boolean"}},
        },
        "pairlists": {
            "description": "Configuration for pairlists.",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "method": {
                        "description": "Method used for generating the pairlist.",
                        "type": "string",
                        "enum": AVAILABLE_PAIRLISTS,
                    },
                },
                "required": ["method"],
            },
        },
        "protections": {
            "description": "Configuration for various protections.",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "method": {
                        "description": "Method used for the protection.",
                        "type": "string",
                        "enum": AVAILABLE_PROTECTIONS,
                    },
                    "stop_duration": {
                        "description": (
                            "Duration to lock the pair after a protection is triggered, "
                            "in minutes."
                        ),
                        "type": "number",
                        "minimum": 0.0,
                    },
                    "stop_duration_candles": {
                        "description": (
                            "Duration to lock the pair after a protection is triggered, in "
                            "number of candles."
                        ),
                        "type": "number",
                        "minimum": 0,
                    },
                    "trade_limit": {
                        "description": "Minimum number of trades required during lookback period.",
                        "type": "number",
                        "minimum": 1,
                    },
                    "lookback_period": {
                        "description": "Period to look back for protection checks, in minutes.",
                        "type": "number",
                        "minimum": 1,
                    },
                    "lookback_period_candles": {
                        "description": (
                            "Period to look back for protection checks, in number " "of candles."
                        ),
                        "type": "number",
                        "minimum": 1,
                    },
                },
                "required": ["method"],
            },
        },
        "telegram": {
            "type": "object",
            "properties": {
                "enabled": {
                    "description": "Enable Telegram notifications.",
                    "type": "boolean",
                },
                "token": {"description": "Telegram bot token.", "type": "string"},
                "chat_id": {
                    "description": "Telegram chat ID",
                    "type": "string",
                },
                "allow_custom_messages": {
                    "description": "Allow sending custom messages from the Strategy.",
                    "type": "boolean",
                    "default": True,
                },
                "balance_dust_level": {
                    "description": "Minimum balance level to consider as dust.",
                    "type": "number",
                    "minimum": 0.0,
                },
                "notification_settings": {
                    "description": "Settings for different types of notifications.",
                    "type": "object",
                    "default": {},
                    "properties": {
                        "status": {
                            "description": "Telegram setting for status updates.",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "warning": {
                            "description": "Telegram setting for warnings.",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "startup": {
                            "description": "Telegram setting for startup messages.",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "entry": {
                            "description": "Telegram setting for entry signals.",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "entry_fill": {
                            "description": "Telegram setting for entry fill signals.",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                            "default": "off",
                        },
                        "entry_cancel": {
                            "description": "Telegram setting for entry cancel signals.",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "exit": {
                            "description": "Telegram setting for exit signals.",
                            "type": ["string", "object"],
                            "additionalProperties": {
                                "type": "string",
                                "enum": TELEGRAM_SETTING_OPTIONS,
                            },
                        },
                        "exit_fill": {
                            "description": "Telegram setting for exit fill signals.",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                            "default": "on",
                        },
                        "exit_cancel": {
                            "description": "Telegram setting for exit cancel signals.",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                        },
                        "protection_trigger": {
                            "description": "Telegram setting for protection triggers.",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                            "default": "on",
                        },
                        "protection_trigger_global": {
                            "description": "Telegram setting for global protection triggers.",
                            "type": "string",
                            "enum": TELEGRAM_SETTING_OPTIONS,
                            "default": "on",
                        },
                    },
                },
                "reload": {
                    "description": "Add Reload button to certain messages.",
                    "type": "boolean",
                },
            },
            "required": ["enabled", "token", "chat_id"],
        },
        "webhook": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "url": {"type": "string"},
                "format": {"type": "string", "enum": WEBHOOK_FORMAT_OPTIONS, "default": "form"},
                "retries": {"type": "integer", "minimum": 0},
                "retry_delay": {"type": "number", "minimum": 0},
                **__MESSAGE_TYPE_DICT,
            },
        },
        "discord": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "webhook_url": {"type": "string"},
                "exit_fill": {
                    "type": "array",
                    "items": {"type": "object"},
                    "default": [
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
                    ],
                },
                "entry_fill": {
                    "type": "array",
                    "items": {"type": "object"},
                    "default": [
                        {"Trade ID": "{trade_id}"},
                        {"Exchange": "{exchange}"},
                        {"Pair": "{pair}"},
                        {"Direction": "{direction}"},
                        {"Open rate": "{open_rate}"},
                        {"Amount": "{amount}"},
                        {"Open date": "{open_date:%Y-%m-%d %H:%M:%S}"},
                        {"Enter tag": "{enter_tag}"},
                        {"Strategy": "{strategy} {timeframe}"},
                    ],
                },
            },
        },
        "api_server": {
            "description": "API server settings.",
            "type": "object",
            "properties": {
                "enabled": {"description": "Whether the API server is enabled.", "type": "boolean"},
                "listen_ip_address": {
                    "description": "IP address the API server listens on.",
                    "format": "ipv4",
                },
                "listen_port": {
                    "description": "Port the API server listens on.",
                    "type": "integer",
                    "minimum": 1024,
                    "maximum": 65535,
                },
                "username": {
                    "description": "Username for API server authentication.",
                    "type": "string",
                },
                "password": {
                    "description": "Password for API server authentication.",
                    "type": "string",
                },
                "ws_token": {
                    "description": "WebSocket token for API server.",
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                },
                "jwt_secret_key": {
                    "description": "Secret key for JWT authentication.",
                    "type": "string",
                },
                "CORS_origins": {
                    "description": "List of allowed CORS origins.",
                    "type": "array",
                    "items": {"type": "string"},
                },
                "x": {
                    "description": "Logging verbosity level.",
                    "type": "string",
                    "enum": ["error", "info"],
                },
            },
            "required": ["enabled", "listen_ip_address", "listen_port", "username", "password"],
        },
        "db_url": {
            "description": "Database connection URL.",
            "type": "string",
        },
        "export": {
            "description": "Type of data to export.",
            "type": "string",
            "enum": EXPORT_OPTIONS,
            "default": "trades",
        },
        "disableparamexport": {
            "description": "Disable parameter export.",
            "type": "boolean",
        },
        "initial_state": {
            "description": "Initial state of the system.",
            "type": "string",
            "enum": ["running", "stopped"],
        },
        "force_entry_enable": {
            "description": "Force enable entry.",
            "type": "boolean",
        },
        "disable_dataframe_checks": {
            "description": "Disable checks on dataframes.",
            "type": "boolean",
        },
        "internals": {
            "description": "Internal settings.",
            "type": "object",
            "default": {},
            "properties": {
                "process_throttle_secs": {
                    "description": "Throttle time in seconds for processing.",
                    "type": "integer",
                },
                "interval": {
                    "description": "Interval time in seconds.",
                    "type": "integer",
                },
                "sd_notify": {
                    "description": "Enable systemd notify.",
                    "type": "boolean",
                },
            },
        },
        "dataformat_ohlcv": {
            "description": "Data format for OHLCV data.",
            "type": "string",
            "enum": AVAILABLE_DATAHANDLERS,
            "default": "feather",
        },
        "dataformat_trades": {
            "description": "Data format for trade data.",
            "type": "string",
            "enum": AVAILABLE_DATAHANDLERS,
            "default": "feather",
        },
        "position_adjustment_enable": {"type": "boolean"},
        "max_entry_position_adjustment": {"type": ["integer", "number"], "minimum": -1},
        "orderflow": {
            "type": "object",
            "properties": {
                "cache_size": {"type": "number", "minimum": 1, "default": 1500},
                "max_candles": {"type": "number", "minimum": 1, "default": 1500},
                "scale": {"type": "number", "minimum": 0.0},
                "stacked_imbalance_range": {"type": "number", "minimum": 0},
                "imbalance_volume": {"type": "number", "minimum": 0},
                "imbalance_ratio": {"type": "number", "minimum": 0.0},
            },
            "required": [
                "max_candles",
                "scale",
                "stacked_imbalance_range",
                "imbalance_volume",
                "imbalance_ratio",
            ],
        },
    },
    "definitions": {
        "exchange": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "enable_ws": {"type": "boolean", "default": True},
                "key": {"type": "string", "default": ""},
                "secret": {"type": "string", "default": ""},
                "password": {"type": "string", "default": ""},
                "uid": {"type": "string"},
                "pair_whitelist": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "uniqueItems": True,
                },
                "pair_blacklist": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "uniqueItems": True,
                },
                "unknown_fee_rate": {"type": "number"},
                "outdated_offset": {"type": "integer", "minimum": 1},
                "markets_refresh_interval": {"type": "integer"},
                "ccxt_config": {"type": "object"},
                "ccxt_async_config": {"type": "object"},
            },
            "required": ["name"],
        },
        "edge": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "process_throttle_secs": {"type": "integer", "minimum": 600},
                "calculate_since_number_of_days": {"type": "integer"},
                "allowed_risk": {"type": "number"},
                "stoploss_range_min": {"type": "number"},
                "stoploss_range_max": {"type": "number"},
                "stoploss_range_step": {"type": "number"},
                "minimum_winrate": {"type": "number"},
                "minimum_expectancy": {"type": "number"},
                "min_trade_number": {"type": "number"},
                "max_trade_duration_minute": {"type": "integer"},
                "remove_pumps": {"type": "boolean"},
            },
            "required": ["process_throttle_secs", "allowed_risk"],
        },
        "external_message_consumer": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "default": False},
                "producers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "host": {"type": "string"},
                            "port": {
                                "type": "integer",
                                "default": 8080,
                                "minimum": 0,
                                "maximum": 65535,
                            },
                            "secure": {"type": "boolean", "default": False},
                            "ws_token": {"type": "string"},
                        },
                        "required": ["name", "host", "ws_token"],
                    },
                },
                "wait_timeout": {"type": "integer", "minimum": 0},
                "sleep_time": {"type": "integer", "minimum": 0},
                "ping_timeout": {"type": "integer", "minimum": 0},
                "remove_entry_exit_signals": {"type": "boolean", "default": False},
                "initial_candle_limit": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 1500,
                    "default": 1500,
                },
                "message_size_limit": {  # In megabytes
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 8,
                },
            },
            "required": ["producers"],
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
                        "svm_params": {
                            "type": "object",
                            "properties": {
                                "shuffle": {"type": "boolean", "default": False},
                                "nu": {"type": "number", "default": 0.1},
                            },
                        },
                        "shuffle_after_split": {"type": "boolean", "default": False},
                        "buffer_train_data_candles": {"type": "integer", "default": 0},
                    },
                    "required": [
                        "include_timeframes",
                        "include_corr_pairlist",
                    ],
                },
                "data_split_parameters": {
                    "type": "object",
                    "properties": {
                        "test_size": {"type": "number"},
                        "random_state": {"type": "integer"},
                        "shuffle": {"type": "boolean", "default": False},
                    },
                },
                "model_training_parameters": {"type": "object"},
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
                                "profit_aim": {"type": "number", "default": 0.025},
                            },
                        },
                    },
                },
            },
            "required": [
                "enabled",
                "train_period_days",
                "backtest_period_days",
                "identifier",
                "feature_parameters",
                "data_split_parameters",
            ],
        },
    },
}

SCHEMA_TRADE_REQUIRED = [
    "exchange",
    "timeframe",
    "max_open_trades",
    "stake_currency",
    "stake_amount",
    "tradable_balance_ratio",
    "last_stake_amount_min_ratio",
    "dry_run",
    "dry_run_wallet",
    "exit_pricing",
    "entry_pricing",
    "stoploss",
    "minimal_roi",
    "internals",
    "dataformat_ohlcv",
    "dataformat_trades",
]

SCHEMA_BACKTEST_REQUIRED = [
    "exchange",
    "stake_currency",
    "stake_amount",
    "dry_run_wallet",
    "dataformat_ohlcv",
    "dataformat_trades",
]
SCHEMA_BACKTEST_REQUIRED_FINAL = SCHEMA_BACKTEST_REQUIRED + [
    "stoploss",
    "minimal_roi",
    "max_open_trades",
]

SCHEMA_MINIMAL_REQUIRED = [
    "exchange",
    "dry_run",
    "dataformat_ohlcv",
    "dataformat_trades",
]
SCHEMA_MINIMAL_WEBSERVER = SCHEMA_MINIMAL_REQUIRED + [
    "api_server",
]
