# Required json-schema for user specified config


from freqtrade.constants import (
    AVAILABLE_DATAHANDLERS,
    AVAILABLE_PAIRLISTS,
    BACKTEST_BREAKDOWNS,
    BACKTEST_CACHE_AGE,
    DRY_RUN_WALLET,
    EXPORT_OPTIONS,
    HYPEROPT_LOSS_BUILTIN,
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


__MESSAGE_TYPE_DICT: dict[str, dict[str, str]] = {x: {"type": "object"} for x in RPCMessageType}

__IN_STRATEGY = "\nUsually specified in the strategy and missing in the configuration."

__VIA_ENV = "Recommended to be set via environment variable"

CONF_SCHEMA = {
    "type": "object",
    "properties": {
        "max_open_trades": {
            "description": "Maximum number of open trades. -1 for unlimited.",
            "type": ["integer", "number"],
            "minimum": -1,
        },
        "timeframe": {
            "description": (
                f"The timeframe to use (e.g `1m`, `5m`, `15m`, `30m`, `1h` ...). {__IN_STRATEGY}"
            ),
            "type": "string",
        },
        "proxy_coin": {
            "description": "Proxy coin - must be used for specific futures modes (e.g. BNFCR)",
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
            "type": ["number", "object"],
            "default": DRY_RUN_WALLET,
            "patternProperties": {r"^[a-zA-Z0-9]+$": {"type": "number"}},
            "additionalProperties": False,
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
            "description": f"Minimum return on investment. {__IN_STRATEGY}",
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
            "description": f"Value (as ratio) to use as Stoploss value. {__IN_STRATEGY}",
            "type": "number",
            "maximum": 0,
            "exclusiveMaximum": True,
        },
        "trailing_stop": {
            "description": f"Enable or disable trailing stop. {__IN_STRATEGY}",
            "type": "boolean",
        },
        "trailing_stop_positive": {
            "description": f"Positive offset for trailing stop. {__IN_STRATEGY}",
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "trailing_stop_positive_offset": {
            "description": f"Offset for trailing stop to activate. {__IN_STRATEGY}",
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "trailing_only_offset_is_reached": {
            "description": f"Use trailing stop only when offset is reached. {__IN_STRATEGY}",
            "type": "boolean",
        },
        "use_exit_signal": {
            "description": f"Use exit signal for trades. {__IN_STRATEGY}",
            "type": "boolean",
        },
        "exit_profit_only": {
            "description": (
                "Exit only when in profit. Exit signals are ignored as "
                f"long as profit is < exit_profit_offset. {__IN_STRATEGY}"
            ),
            "type": "boolean",
        },
        "exit_profit_offset": {
            "description": f"Offset for profit exit. {__IN_STRATEGY}",
            "type": "number",
        },
        "recursive_strategy_search": {
            "description": "Enable recursive strategy search.",
            "type": "boolean",
        },
        "user_data_dir": {
            "description": "Path to the user data directory.",
        },
        "datadir": {
            "description": "Path to the data directory.",
        },
        "fee": {
            "description": "Trading fee percentage. Can help to simulate slippage in backtesting",
            "type": "number",
            "minimum": 0,
            "maximum": 0.1,
        },
        "ignore_roi_if_entry_signal": {
            "description": f"Ignore ROI if entry signal is present. {__IN_STRATEGY}",
            "type": "boolean",
        },
        "ignore_buying_expired_candle_after": {
            "description": f"Ignore buying after candle expiration time. {__IN_STRATEGY}",
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
        # Lookahead analysis section
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
        "backtest_cache": {
            "description": "Load a cached backtest result no older than specified age.",
            "type": "string",
            "enum": BACKTEST_CACHE_AGE,
        },
        # Hyperopt
        "hyperopt_path": {
            "description": "Specify additional lookup path for Hyperopt Loss functions.",
            "type": "string",
        },
        "epochs": {
            "description": "Number of training epochs for Hyperopt.",
            "type": "integer",
            "minimum": 1,
        },
        "early_stop": {
            "description": (
                "Early stop hyperopt if no improvement after <epochs>. Set to 0 to disable."
            ),
            "type": "integer",
            "minimum": 0,
        },
        "spaces": {
            "description": (
                "Hyperopt parameter spaces to optimize. Default is the default set and"
                "includes all spaces except for 'trailing', 'protection', and 'trades'."
            ),
            "type": "array",
            "items": {"type": "string"},
            "default": ["default"],
        },
        "analyze_per_epoch": {
            "description": "Perform analysis after each epoch in Hyperopt.",
            "type": "boolean",
        },
        "print_all": {
            "description": "Print all hyperopt trials, not just the best ones.",
            "type": "boolean",
            "default": False,
        },
        "hyperopt_jobs": {
            "description": (
                "The number of concurrently running jobs for hyperoptimization "
                "(hyperopt worker processes). "
                "If -1 (default), all CPUs are used, for -2, all CPUs but one are used, etc. "
                "If 1 is given, no parallel computing is used."
            ),
            "type": "integer",
            "default": -1,
        },
        "hyperopt_random_state": {
            "description": "Random state for hyperopt trials.",
            "type": "integer",
            "minimum": 0,
        },
        "hyperopt_min_trades": {
            "description": "Minimum number of trades per epoch for hyperopt.",
            "type": "integer",
            "minimum": 0,
        },
        "hyperopt_loss": {
            "description": (
                "The class name of the hyperopt loss function class (IHyperOptLoss). "
                "Different functions can generate completely different results, "
                "since the target for optimization is different. "
                f"Built-in Hyperopt-loss-functions are: {', '.join(HYPEROPT_LOSS_BUILTIN)}"
            ),
            "type": "string",
        },
        # end hyperopt
        "bot_name": {
            "description": "Name of the trading bot. Passed via API to a client.",
            "type": "string",
        },
        "unfilledtimeout": {
            "description": f"Timeout configuration for unfilled orders. {__IN_STRATEGY}",
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
            "description": f"Configuration of order types. {__IN_STRATEGY}",
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
            "description": f"Time in force configuration for orders. {__IN_STRATEGY}",
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
        "log_config": {
            "description": "Logging configuration.",
            "$ref": "#/definitions/logging",
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
            "minItems": 1,
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
        # RPC section
        "telegram": {
            "description": "Telegram settings.",
            "type": "object",
            "properties": {
                "enabled": {
                    "description": "Enable Telegram notifications.",
                    "type": "boolean",
                },
                "token": {"description": "Telegram bot token.", "type": "string"},
                "chat_id": {
                    "description": (
                        f"Telegram chat or group ID. {__VIA_ENV} FREQTRADE__TELEGRAM__CHAT_ID"
                    ),
                    "type": "string",
                },
                "topic_id": {
                    "description": (
                        "Telegram topic ID - only applicable for group chats. "
                        f"{__VIA_ENV} FREQTRADE__TELEGRAM__TOPIC_ID"
                    ),
                    "type": "string",
                },
                "authorized_users": {
                    "description": "Authorized users for the bot.",
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
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
                            "type": ["string", "object"],
                            "additionalProperties": {
                                "type": "string",
                                "enum": TELEGRAM_SETTING_OPTIONS,
                            },
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
            "description": "Webhook settings.",
            "type": "object",
            "properties": {
                "enabled": {"description": "Enable webhook notifications.", "type": "boolean"},
                "url": {
                    "description": f"Webhook URL. {__VIA_ENV} FREQTRADE__WEBHOOK__URL",
                    "type": "string",
                },
                "format": {"type": "string", "enum": WEBHOOK_FORMAT_OPTIONS, "default": "form"},
                "retries": {"type": "integer", "minimum": 0},
                "retry_delay": {"type": "number", "minimum": 0},
                **__MESSAGE_TYPE_DICT,
            },
        },
        "discord": {
            "description": "Discord settings.",
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "webhook_url": {
                    "description": (
                        f"Discord webhook URL. {__VIA_ENV} FREQTRADE__DISCORD__WEBHOOK_URL"
                    ),
                    "type": "string",
                },
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
                "verbosity": {
                    "description": "Logging verbosity level.",
                    "type": "string",
                    "enum": ["error", "info"],
                },
            },
            "required": ["enabled", "listen_ip_address", "listen_port", "username", "password"],
        },
        # end of RPC section
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
            "enum": ["running", "paused", "stopped"],
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
                    "description": "Minimum loop duration for one bot iteration in seconds.",
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
        "position_adjustment_enable": {
            "description": f"Enable position adjustment. {__IN_STRATEGY}",
            "type": "boolean",
        },
        # Download data section
        "new_pairs_days": {
            "description": "Download data of new pairs for given number of days",
            "type": "integer",
            "default": 30,
        },
        "download_trades": {
            "description": "Download trades data by default (instead of ohlcv data).",
            "type": "boolean",
        },
        "max_entry_position_adjustment": {
            "description": f"Maximum entry position adjustment allowed. {__IN_STRATEGY}",
            "type": ["integer", "number"],
            "minimum": -1,
        },
        "add_config_files": {
            "description": "Additional configuration files to load.",
            "type": "array",
            "items": {"type": "string"},
        },
        "orderflow": {
            "description": "Settings related to order flow.",
            "type": "object",
            "properties": {
                "cache_size": {
                    "description": "Size of the cache for order flow data.",
                    "type": "number",
                    "minimum": 1,
                    "default": 1500,
                },
                "max_candles": {
                    "description": "Maximum number of candles to consider.",
                    "type": "number",
                    "minimum": 1,
                    "default": 1500,
                },
                "scale": {
                    "description": "Scale factor for order flow data.",
                    "type": "number",
                    "minimum": 0.0,
                },
                "stacked_imbalance_range": {
                    "description": "Range for stacked imbalance.",
                    "type": "number",
                    "minimum": 0,
                },
                "imbalance_volume": {
                    "description": "Volume threshold for imbalance.",
                    "type": "number",
                    "minimum": 0,
                },
                "imbalance_ratio": {
                    "description": "Ratio threshold for imbalance.",
                    "type": "number",
                    "minimum": 0.0,
                },
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
            "description": "Exchange configuration settings.",
            "type": "object",
            "properties": {
                "name": {"description": "Name of the exchange.", "type": "string"},
                "key": {
                    "description": (
                        f"API key for the exchange. {__VIA_ENV} FREQTRADE__EXCHANGE__KEY"
                    ),
                    "type": "string",
                    "default": "",
                },
                "secret": {
                    "description": (
                        f"API secret for the exchange. {__VIA_ENV} FREQTRADE__EXCHANGE__SECRET"
                    ),
                    "type": "string",
                    "default": "",
                },
                "password": {
                    "description": (
                        "Password for the exchange, if required. "
                        f"{__VIA_ENV} FREQTRADE__EXCHANGE__PASSWORD"
                    ),
                    "type": "string",
                    "default": "",
                },
                "uid": {
                    "description": (
                        "User ID for the exchange, if required. "
                        f"{__VIA_ENV} FREQTRADE__EXCHANGE__UID"
                    ),
                    "type": "string",
                },
                "account_id": {
                    "description": (
                        "Account ID for the exchange, if required. "
                        f"{__VIA_ENV} FREQTRADE__EXCHANGE__ACCOUNT_ID"
                    ),
                    "type": "string",
                },
                "wallet_address": {
                    "description": (
                        "Wallet address for the exchange, if required. "
                        "Usually used by DEX exchanges. "
                        f"{__VIA_ENV} FREQTRADE__EXCHANGE__WALLET_ADDRESS"
                    ),
                    "type": "string",
                },
                "private_key": {
                    "description": (
                        "Private key for the exchange, if required. Usually used by DEX exchanges. "
                        f"{__VIA_ENV} FREQTRADE__EXCHANGE__PRIVATE_KEY"
                    ),
                    "type": "string",
                },
                "pair_whitelist": {
                    "description": "List of whitelisted trading pairs.",
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "pair_blacklist": {
                    "description": "List of blacklisted trading pairs.",
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
                "log_responses": {
                    "description": (
                        "Log responses from the exchange."
                        "Useful/required to debug issues with order processing."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                "enable_ws": {
                    "description": "Enable WebSocket connections to the exchange.",
                    "type": "boolean",
                    "default": True,
                },
                "unknown_fee_rate": {
                    "description": "Fee rate for unknown markets.",
                    "type": "number",
                },
                "outdated_offset": {
                    "description": "Offset for outdated data in minutes.",
                    "type": "integer",
                    "minimum": 1,
                },
                "markets_refresh_interval": {
                    "description": "Interval for refreshing market data in minutes.",
                    "type": "integer",
                    "default": 60,
                },
                "ccxt_config": {"description": "CCXT configuration settings.", "type": "object"},
                "ccxt_async_config": {
                    "description": (
                        "CCXT asynchronous configuration settings."
                        "Usually ccxt_config should be used instead."
                    ),
                    "type": "object",
                },
                "ccxt_sync_config": {
                    "description": (
                        "CCXT synchronous configuration settings. "
                        "Usually ccxt_config should be used instead."
                    ),
                    "type": "object",
                },
            },
            "required": ["name"],
        },
        "logging": {
            "type": "object",
            "properties": {
                "version": {"type": "number", "const": 1},
                "formatters": {
                    "type": "object",
                    # In theory the below, but can be more flexible
                    # based on logging.config documentation
                    # "additionalProperties": {
                    #     "type": "object",
                    #     "properties": {
                    #         "format": {"type": "string"},
                    #         "datefmt": {"type": "string"},
                    #     },
                    #     "required": ["format"],
                    # },
                },
                "handlers": {"type": "object"},
                "root": {"type": "object"},
            },
            "required": ["version", "formatters", "handlers", "root"],
        },
        "external_message_consumer": {
            "description": "Configuration for external message consumer.",
            "type": "object",
            "properties": {
                "enabled": {
                    "description": "Whether the external message consumer is enabled.",
                    "type": "boolean",
                    "default": False,
                },
                "producers": {
                    "description": "List of producers for the external message consumer.",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "description": "Name of the producer.",
                                "type": "string",
                            },
                            "host": {
                                "description": "Host of the producer.",
                                "type": "string",
                            },
                            "port": {
                                "description": "Port of the producer.",
                                "type": "integer",
                                "default": 8080,
                                "minimum": 0,
                                "maximum": 65535,
                            },
                            "secure": {
                                "description": "Whether to use SSL to connect to the producer.",
                                "type": "boolean",
                                "default": False,
                            },
                            "ws_token": {
                                "description": "WebSocket token for the producer.",
                                "type": "string",
                            },
                        },
                        "required": ["name", "host", "ws_token"],
                    },
                },
                "wait_timeout": {
                    "description": "Wait timeout in seconds.",
                    "type": "integer",
                    "minimum": 0,
                },
                "sleep_time": {
                    "description": "Sleep time in seconds before retrying to connect.",
                    "type": "integer",
                    "minimum": 0,
                },
                "ping_timeout": {
                    "description": "Ping timeout in seconds.",
                    "type": "integer",
                    "minimum": 0,
                },
                "remove_entry_exit_signals": {
                    "description": "Remove signal columns from the dataframe (set them to 0)",
                    "type": "boolean",
                    "default": False,
                },
                "initial_candle_limit": {
                    "description": "Initial candle limit.",
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 1500,
                    "default": 1500,
                },
                "message_size_limit": {
                    "description": "Message size limit in megabytes.",
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
                "enabled": {
                    "description": "Whether freqAI is enabled.",
                    "type": "boolean",
                    "default": False,
                },
                "identifier": {
                    "description": (
                        "A unique ID for the current model. "
                        "Must be changed when modifying features."
                    ),
                    "type": "string",
                    "default": "example",
                },
                "write_metrics_to_disk": {
                    "description": "Write metrics to disk?",
                    "type": "boolean",
                    "default": False,
                },
                "purge_old_models": {
                    "description": "Number of models to keep on disk.",
                    "type": ["boolean", "number"],
                    "default": 2,
                },
                "conv_width": {
                    "description": "The width of a neural network input tensor.",
                    "type": "integer",
                    "default": 1,
                },
                "train_period_days": {
                    "description": (
                        "Number of days to use for the training data (width of the sliding window)"
                    ),
                    "type": "integer",
                    "default": 0,
                },
                "backtest_period_days": {
                    "description": (
                        "Number of days to inference from the trained model before sliding the "
                        "`train_period_days` window "
                    ),
                    "type": "number",
                    "default": 7,
                },
                "live_retrain_hours": {
                    "description": "Frequency of retraining during dry/live runs.",
                    "type": "number",
                    "default": 0,
                },
                "expiration_hours": {
                    "description": (
                        "Avoid making predictions if a model is more than `expiration_hours` "
                        "old. Defaults to 0 (no expiration)."
                    ),
                    "type": "number",
                    "default": 0,
                },
                "save_backtest_models": {
                    "description": "Save models to disk when running backtesting.",
                    "type": "boolean",
                    "default": False,
                },
                "fit_live_predictions_candles": {
                    "description": (
                        "Number of historical candles to use for computing target (label) "
                        "statistics from prediction data, instead of from the training dataset."
                    ),
                    "type": "integer",
                },
                "data_kitchen_thread_count": {
                    "description": (
                        "Designate the number of threads you want to use for data processing "
                        "(outlier methods, normalization, etc.)."
                    ),
                    "type": "integer",
                },
                "activate_tensorboard": {
                    "description": "Indicate whether or not to activate tensorboard",
                    "type": "boolean",
                    "default": True,
                },
                "wait_for_training_iteration_on_reload": {
                    "description": (
                        "Wait for the next training iteration to complete after /reload or ctrl+c."
                    ),
                    "type": "boolean",
                    "default": True,
                },
                "continual_learning": {
                    "description": (
                        "Use the final state of the most recently trained model "
                        "as starting point for the new model, allowing for "
                        "incremental learning."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                "keras": {
                    "description": "Use Keras for model training.",
                    "type": "boolean",
                    "default": False,
                },
                "override_exchange_check": {
                    "description": (
                        "Override the exchange check to force FreqAI to use exchanges "
                        "that may not have enough historic data. Turn this to True if "
                        "you know your FreqAI model and strategy do not require historical data."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                "feature_parameters": {
                    "description": "The parameters used to engineer the feature set",
                    "type": "object",
                    "properties": {
                        "include_corr_pairlist": {
                            "description": "List of correlated pairs to include in the features.",
                            "type": "array",
                        },
                        "include_timeframes": {
                            "description": (
                                "A list of timeframes that all indicators in "
                                "`feature_engineering_expand_*()` will be created for."
                            ),
                            "type": "array",
                        },
                        "label_period_candles": {
                            "description": (
                                "Number of candles into the future to use for labeling the period."
                                "This can be used in `set_freqai_targets()`."
                            ),
                            "type": "integer",
                        },
                        "include_shifted_candles": {
                            "description": (
                                "Add features from previous candles to subsequent candles with "
                                "the intent of adding historical information."
                            ),
                            "type": "integer",
                            "default": 0,
                        },
                        "DI_threshold": {
                            "description": (
                                "Activates the use of the Dissimilarity Index for "
                                "outlier detection when set to > 0."
                            ),
                            "type": "number",
                            "default": 0,
                        },
                        "weight_factor": {
                            "description": (
                                "Weight training data points according to their recency."
                            ),
                            "type": "number",
                            "default": 0,
                        },
                        "principal_component_analysis": {
                            "description": (
                                "Automatically reduce the dimensionality of the data set using "
                                "Principal Component Analysis"
                            ),
                            "type": "boolean",
                            "default": False,
                        },
                        "indicator_periods_candles": {
                            "description": (
                                "Time periods to calculate indicators for. "
                                "The indicators are added to the base indicator dataset."
                            ),
                            "type": "array",
                            "items": {"type": "number", "minimum": 1},
                        },
                        "use_SVM_to_remove_outliers": {
                            "description": "Use SVM to remove outliers from the features.",
                            "type": "boolean",
                            "default": False,
                        },
                        "plot_feature_importances": {
                            "description": "Create feature importance plots for each model.",
                            "type": "integer",
                            "default": 0,
                        },
                        "svm_params": {
                            "description": (
                                "All parameters available in Sklearn's `SGDOneClassSVM()`."
                            ),
                            "type": "object",
                            "properties": {
                                "shuffle": {
                                    "description": "Whether to shuffle data before applying SVM.",
                                    "type": "boolean",
                                    "default": False,
                                },
                                "nu": {
                                    "type": "number",
                                    "default": 0.1,
                                },
                            },
                        },
                        "shuffle_after_split": {
                            "description": (
                                "Split the data into train and test sets, and then shuffle "
                                "both sets individually."
                            ),
                            "type": "boolean",
                            "default": False,
                        },
                        "buffer_train_data_candles": {
                            "description": (
                                "Cut `buffer_train_data_candles` off the beginning and end of the "
                                "training data *after* the indicators were populated."
                            ),
                            "type": "integer",
                            "default": 0,
                        },
                    },
                    "required": [
                        "include_timeframes",
                        "include_corr_pairlist",
                    ],
                },
                "data_split_parameters": {
                    "descriptions": (
                        "Additional parameters for scikit-learn's test_train_split() function."
                    ),
                    "type": "object",
                    "properties": {
                        "test_size": {"type": "number"},
                        "random_state": {"type": "integer"},
                        "shuffle": {"type": "boolean", "default": False},
                    },
                },
                "model_training_parameters": {
                    "description": (
                        "Flexible dictionary that includes all parameters available by "
                        "the selected model library. "
                    ),
                    "type": "object",
                },
                "rl_config": {
                    "type": "object",
                    "properties": {
                        "drop_ohlc_from_features": {
                            "description": (
                                "Do not include the normalized ohlc data in the feature set."
                            ),
                            "type": "boolean",
                            "default": False,
                        },
                        "train_cycles": {
                            "description": "Number of training cycles to perform.",
                            "type": "integer",
                        },
                        "max_trade_duration_candles": {
                            "description": (
                                "Guides the agent training to keep trades below desired length."
                            ),
                            "type": "integer",
                        },
                        "add_state_info": {
                            "description": (
                                "Include state information in the feature set for "
                                "training and inference."
                            ),
                            "type": "boolean",
                            "default": False,
                        },
                        "max_training_drawdown_pct": {
                            "description": "Maximum allowed drawdown percentage during training.",
                            "type": "number",
                            "default": 0.02,
                        },
                        "cpu_count": {
                            "description": "Number of threads/CPU's to use for training.",
                            "type": "integer",
                            "default": 1,
                        },
                        "model_type": {
                            "description": "Model string from stable_baselines3 or SBcontrib.",
                            "type": "string",
                            "default": "PPO",
                        },
                        "policy_type": {
                            "description": (
                                "One of the available policy types from stable_baselines3."
                            ),
                            "type": "string",
                            "default": "MlpPolicy",
                        },
                        "net_arch": {
                            "description": "Architecture of the neural network.",
                            "type": "array",
                            "default": [128, 128],
                        },
                        "randomize_starting_position": {
                            "description": (
                                "Randomize the starting point of each episode to avoid overfitting."
                            ),
                            "type": "boolean",
                            "default": False,
                        },
                        "progress_bar": {
                            "description": "Display a progress bar with the current progress.",
                            "type": "boolean",
                            "default": True,
                        },
                        "model_reward_parameters": {
                            "description": "Parameters for configuring the reward model.",
                            "type": "object",
                            "properties": {
                                "rr": {
                                    "type": "number",
                                    "default": 1,
                                    "description": "Reward ratio parameter.",
                                },
                                "profit_aim": {
                                    "type": "number",
                                    "default": 0.025,
                                    "description": "Profit aim parameter.",
                                },
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
    "pairlists",
    "internals",
    "dataformat_ohlcv",
    "dataformat_trades",
]

SCHEMA_BACKTEST_REQUIRED = [
    "exchange",
    "stake_currency",
    "stake_amount",
    "pairlists",
    "dry_run_wallet",
    "dataformat_ohlcv",
    "dataformat_trades",
]
SCHEMA_BACKTEST_REQUIRED_FINAL = [
    *SCHEMA_BACKTEST_REQUIRED,
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
SCHEMA_MINIMAL_WEBSERVER = [*SCHEMA_MINIMAL_REQUIRED, "api_server"]
