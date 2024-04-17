# Configure the bot

Freqtrade has many configurable features and possibilities.
By default, these settings are configured via the configuration file (see below).

## The Freqtrade configuration file

The bot uses a set of configuration parameters during its operation that all together conform to the bot configuration. It normally reads its configuration from a file (Freqtrade configuration file).

Per default, the bot loads the configuration from the `config.json` file, located in the current working directory.

You can specify a different configuration file used by the bot with the `-c/--config` command-line option.

If you used the [Quick start](docker_quickstart.md#docker-quick-start) method for installing
the bot, the installation script should have already created the default configuration file (`config.json`) for you.

If the default configuration file is not created we recommend to use `freqtrade new-config --config user_data/config.json` to generate a basic configuration file.

The Freqtrade configuration file is to be written in JSON format.

Additionally to the standard JSON syntax, you may use one-line `// ...` and multi-line `/* ... */` comments in your configuration files and trailing commas in the lists of parameters.

Do not worry if you are not familiar with JSON format -- simply open the configuration file with an editor of your choice, make some changes to the parameters you need, save your changes and, finally, restart the bot or, if it was previously stopped, run it again with the changes you made to the configuration. The bot validates the syntax of the configuration file at startup and will warn you if you made any errors editing it, pointing out problematic lines.

### Environment variables

Set options in the Freqtrade configuration via environment variables.
This takes priority over the corresponding value in configuration or strategy.

Environment variables must be prefixed with  `FREQTRADE__` to be loaded to the freqtrade configuration.

`__` serves as level separator, so the format used should correspond to `FREQTRADE__{section}__{key}`.
As such - an environment variable defined as  `export FREQTRADE__STAKE_AMOUNT=200` would result in `{stake_amount: 200}`.

A more complex example might be `export FREQTRADE__EXCHANGE__KEY=<yourExchangeKey>` to keep your exchange key secret. This will move the value to the `exchange.key` section of the configuration.
Using this scheme, all configuration settings will also be available as environment variables.

Please note that Environment variables will overwrite corresponding settings in your configuration, but command line Arguments will always win.

Common example:

```
FREQTRADE__TELEGRAM__CHAT_ID=<telegramchatid>
FREQTRADE__TELEGRAM__TOKEN=<telegramToken>
FREQTRADE__EXCHANGE__KEY=<yourExchangeKey>
FREQTRADE__EXCHANGE__SECRET=<yourExchangeSecret>
```

!!! Note
    Environment variables detected are logged at startup - so if you can't find why a value is not what you think it should be based on the configuration, make sure it's not loaded from an environment variable.

!!! Tip "Validate combined result"
    You can use the [show-config subcommand](utils.md#show-config) to see the final, combined configuration.

??? Warning "Loading sequence"
    Environment variables are loaded after the initial configuration. As such, you cannot provide the path to the configuration through environment variables. Please use `--config path/to/config.json` for that.
    This also applies to user_dir to some degree. while the user directory can be set through environment variables - the configuration will **not** be loaded from that location.

### Multiple configuration files

Multiple configuration files can be specified and used by the bot or the bot can read its configuration parameters from the process standard input stream.

You can specify additional configuration files in `add_config_files`. Files specified in this parameter will be loaded and merged with the initial config file. The files are resolved relative to the initial configuration file.
This is similar to using multiple `--config` parameters, but simpler in usage as you don't have to specify all files for all commands.

!!! Tip "Validate combined result"
    You can use the [show-config subcommand](utils.md#show-config) to see the final, combined configuration.

!!! Tip "Use multiple configuration files to keep secrets secret"
    You can use a 2nd configuration file containing your secrets. That way you can share your "primary" configuration file, while still keeping your API keys for yourself.
    The 2nd file should only specify what you intend to override.
    If a key is in more than one of the configurations, then the "last specified configuration" wins (in the above example, `config-private.json`).

    For one-off commands, you can also use the below syntax by specifying multiple "--config" parameters.

    ``` bash
    freqtrade trade --config user_data/config1.json --config user_data/config-private.json <...>
    ```

    The below is equivalent to the example above - but having 2 configuration files in the configuration, for easier reuse.

    ``` json title="user_data/config.json"
    "add_config_files": [
        "config1.json",
        "config-private.json"
    ]
    ```

    ``` bash
    freqtrade trade --config user_data/config.json <...>
    ```

??? Note "config collision handling"
    If the same configuration setting takes place in both `config.json` and `config-import.json`, then the parent configuration wins.
    In the below case, `max_open_trades` would be 3 after the merging - as the reusable "import" configuration has this key overwritten.

    ``` json title="user_data/config.json"
    {
        "max_open_trades": 3,
        "stake_currency": "USDT",
        "add_config_files": [
            "config-import.json"
        ]
    }
    ```

    ``` json title="user_data/config-import.json"
    {
        "max_open_trades": 10,
        "stake_amount": "unlimited",
    }
    ```

    Resulting combined configuration:

    ``` json title="Result"
    {
        "max_open_trades": 3,
        "stake_currency": "USDT",
        "stake_amount": "unlimited"
    }
    ```

    If multiple files are in the `add_config_files` section, then they will be assumed to be at identical levels, having the last occurrence override the earlier config (unless a parent already defined such a key).

## Configuration parameters

The table below will list all configuration parameters available.

Freqtrade can also load many options via command line (CLI) arguments (check out the commands `--help` output for details).

### Configuration option prevalence

The prevalence for all Options is as follows:

- CLI arguments override any other option
- [Environment Variables](#environment-variables)
- Configuration files are used in sequence (the last file wins) and override Strategy configurations.
- Strategy configurations are only used if they are not set via configuration or command-line arguments. These options are marked with [Strategy Override](#parameters-in-the-strategy) in the below table.

### Parameters table

Mandatory parameters are marked as **Required**, which means that they are required to be set in one of the possible ways.

|  Parameter | Description |
|------------|-------------|
| `max_open_trades` | **Required.** Number of open trades your bot is allowed to have. Only one open trade per pair is possible, so the length of your pairlist is another limitation that can apply. If -1 then it is ignored (i.e. potentially unlimited open trades, limited by the pairlist). [More information below](#configuring-amount-per-trade). [Strategy Override](#parameters-in-the-strategy).<br> **Datatype:** Positive integer or -1.
| `stake_currency` | **Required.** Crypto-currency used for trading. <br> **Datatype:** String
| `stake_amount` | **Required.** Amount of crypto-currency your bot will use for each trade. Set it to `"unlimited"` to allow the bot to use all available balance. [More information below](#configuring-amount-per-trade). <br> **Datatype:** Positive float or `"unlimited"`.
| `tradable_balance_ratio` | Ratio of the total account balance the bot is allowed to trade. [More information below](#configuring-amount-per-trade). <br>*Defaults to `0.99` 99%).*<br> **Datatype:** Positive float between `0.1` and `1.0`.
| `available_capital` | Available starting capital for the bot. Useful when running multiple bots on the same exchange account. [More information below](#configuring-amount-per-trade). <br> **Datatype:** Positive float.
| `amend_last_stake_amount` | Use reduced last stake amount if necessary. [More information below](#configuring-amount-per-trade). <br>*Defaults to `false`.* <br> **Datatype:** Boolean
| `last_stake_amount_min_ratio` | Defines minimum stake amount that has to be left and executed. Applies only to the last stake amount when it's amended to a reduced value (i.e. if `amend_last_stake_amount` is set to `true`). [More information below](#configuring-amount-per-trade). <br>*Defaults to `0.5`.* <br> **Datatype:** Float (as ratio)
| `amount_reserve_percent` | Reserve some amount in min pair stake amount. The bot will reserve `amount_reserve_percent` + stoploss value when calculating min pair stake amount in order to avoid possible trade refusals. <br>*Defaults to `0.05` (5%).* <br> **Datatype:** Positive Float as ratio.
| `timeframe` | The timeframe to use (e.g `1m`, `5m`, `15m`, `30m`, `1h` ...). Usually missing in configuration, and specified in the strategy. [Strategy Override](#parameters-in-the-strategy). <br> **Datatype:** String
| `fiat_display_currency` | Fiat currency used to show your profits. [More information below](#what-values-can-be-used-for-fiat_display_currency). <br> **Datatype:** String
| `dry_run` | **Required.** Define if the bot must be in Dry Run or production mode. <br>*Defaults to `true`.* <br> **Datatype:** Boolean
| `dry_run_wallet` | Define the starting amount in stake currency for the simulated wallet used by the bot running in Dry Run mode.<br>*Defaults to `1000`.* <br> **Datatype:** Float
| `cancel_open_orders_on_exit` | Cancel open orders when the `/stop` RPC command is issued, `Ctrl+C` is pressed or the bot dies unexpectedly. When set to `true`, this allows you to use `/stop` to cancel unfilled and partially filled orders in the event of a market crash. It does not impact open positions. <br>*Defaults to `false`.* <br> **Datatype:** Boolean
| `process_only_new_candles` | Enable processing of indicators only when new candles arrive. If false each loop populates the indicators, this will mean the same candle is processed many times creating system load but can be useful of your strategy depends on tick data not only candle. [Strategy Override](#parameters-in-the-strategy). <br>*Defaults to `true`.*  <br> **Datatype:** Boolean
| `minimal_roi` | **Required.** Set the threshold as ratio the bot will use to exit a trade. [More information below](#understand-minimal_roi). [Strategy Override](#parameters-in-the-strategy). <br> **Datatype:** Dict
| `stoploss` |  **Required.** Value as ratio of the stoploss used by the bot. More details in the [stoploss documentation](stoploss.md). [Strategy Override](#parameters-in-the-strategy).  <br> **Datatype:** Float (as ratio)
| `trailing_stop` | Enables trailing stoploss (based on `stoploss` in either configuration or strategy file). More details in the [stoploss documentation](stoploss.md#trailing-stop-loss). [Strategy Override](#parameters-in-the-strategy). <br> **Datatype:** Boolean
| `trailing_stop_positive` | Changes stoploss once profit has been reached. More details in the [stoploss documentation](stoploss.md#trailing-stop-loss-custom-positive-loss). [Strategy Override](#parameters-in-the-strategy). <br> **Datatype:** Float
| `trailing_stop_positive_offset` | Offset on when to apply `trailing_stop_positive`. Percentage value which should be positive. More details in the [stoploss documentation](stoploss.md#trailing-stop-loss-only-once-the-trade-has-reached-a-certain-offset). [Strategy Override](#parameters-in-the-strategy). <br>*Defaults to `0.0` (no offset).* <br> **Datatype:** Float
| `trailing_only_offset_is_reached` | Only apply trailing stoploss when the offset is reached. [stoploss documentation](stoploss.md). [Strategy Override](#parameters-in-the-strategy). <br>*Defaults to `false`.*  <br> **Datatype:** Boolean
| `fee` | Fee used during backtesting / dry-runs. Should normally not be configured, which has freqtrade fall back to the exchange default fee. Set as ratio (e.g. 0.001 = 0.1%). Fee is applied twice for each trade, once when buying, once when selling. <br> **Datatype:** Float (as ratio)
| `futures_funding_rate` | User-specified funding rate to be used when historical funding rates are not available from the exchange. This does not overwrite real historical rates. It is recommended that this be set to 0 unless you are testing a specific coin and you understand how the funding rate will affect freqtrade's profit calculations. [More information here](leverage.md#unavailable-funding-rates) <br>*Defaults to `None`.*<br> **Datatype:** Float
| `trading_mode` | Specifies if you want to trade regularly, trade with leverage, or trade contracts whose prices are derived from matching cryptocurrency prices. [leverage documentation](leverage.md). <br>*Defaults to `"spot"`.*  <br> **Datatype:** String
| `margin_mode` | When trading with leverage, this determines if the collateral owned by the trader will be shared or isolated to each trading pair [leverage documentation](leverage.md). <br> **Datatype:** String
| `liquidation_buffer` | A ratio specifying how large of a safety net to place between the liquidation price and the stoploss to prevent a position from reaching the liquidation price [leverage documentation](leverage.md). <br>*Defaults to `0.05`.*  <br> **Datatype:** Float
| | **Unfilled timeout**
| `unfilledtimeout.entry` | **Required.** How long (in minutes or seconds) the bot will wait for an unfilled entry order to complete, after which the order will be cancelled and repeated at current (new) price, as long as there is a signal. [Strategy Override](#parameters-in-the-strategy).<br> **Datatype:** Integer
| `unfilledtimeout.exit` | **Required.** How long (in minutes or seconds) the bot will wait for an unfilled exit order to complete, after which the order will be cancelled and repeated at current (new) price, as long as there is a signal. [Strategy Override](#parameters-in-the-strategy).<br> **Datatype:** Integer
| `unfilledtimeout.unit` | Unit to use in unfilledtimeout setting. Note: If you set unfilledtimeout.unit to "seconds", "internals.process_throttle_secs" must be inferior or equal to timeout [Strategy Override](#parameters-in-the-strategy). <br> *Defaults to `"minutes"`.* <br> **Datatype:** String
| `unfilledtimeout.exit_timeout_count` | How many times can exit orders time out. Once this number of timeouts is reached, an emergency exit is triggered. 0 to disable and allow unlimited order cancels. [Strategy Override](#parameters-in-the-strategy).<br>*Defaults to `0`.* <br> **Datatype:** Integer
| | **Pricing**
| `entry_pricing.price_side` | Select the side of the spread the bot should look at to get the entry rate. [More information below](#entry-price).<br> *Defaults to `"same"`.* <br> **Datatype:** String (either `ask`, `bid`, `same` or `other`).
| `entry_pricing.price_last_balance` | **Required.** Interpolate the bidding price. More information [below](#entry-price-without-orderbook-enabled).
| `entry_pricing.use_order_book` | Enable entering using the rates in [Order Book Entry](#entry-price-with-orderbook-enabled). <br> *Defaults to `true`.*<br> **Datatype:** Boolean
| `entry_pricing.order_book_top` | Bot will use the top N rate in Order Book "price_side" to enter a trade. I.e. a value of 2 will allow the bot to pick the 2nd entry in [Order Book Entry](#entry-price-with-orderbook-enabled). <br>*Defaults to `1`.*  <br> **Datatype:** Positive Integer
| `entry_pricing. check_depth_of_market.enabled` | Do not enter if the difference of buy orders and sell orders is met in Order Book. [Check market depth](#check-depth-of-market). <br>*Defaults to `false`.* <br> **Datatype:** Boolean
| `entry_pricing. check_depth_of_market.bids_to_ask_delta` | The difference ratio of buy orders and sell orders found in Order Book. A value below 1 means sell order size is greater, while value greater than 1 means buy order size is higher. [Check market depth](#check-depth-of-market) <br> *Defaults to `0`.*  <br> **Datatype:** Float (as ratio)
| `exit_pricing.price_side` | Select the side of the spread the bot should look at to get the exit rate. [More information below](#exit-price-side).<br> *Defaults to `"same"`.* <br> **Datatype:** String (either `ask`, `bid`, `same` or `other`).
| `exit_pricing.price_last_balance` | Interpolate the exiting price. More information [below](#exit-price-without-orderbook-enabled).
| `exit_pricing.use_order_book` | Enable exiting of open trades using [Order Book Exit](#exit-price-with-orderbook-enabled). <br> *Defaults to `true`.*<br> **Datatype:** Boolean
| `exit_pricing.order_book_top` | Bot will use the top N rate in Order Book "price_side" to exit. I.e. a value of 2 will allow the bot to pick the 2nd ask rate in [Order Book Exit](#exit-price-with-orderbook-enabled)<br>*Defaults to `1`.* <br> **Datatype:** Positive Integer
| `custom_price_max_distance_ratio` | Configure maximum distance ratio between current and custom entry or exit price. <br>*Defaults to `0.02` 2%).*<br> **Datatype:** Positive float
| | **TODO**
| `use_exit_signal` | Use exit signals produced by the strategy in addition to the `minimal_roi`. <br>Setting this to false disables the usage of `"exit_long"` and `"exit_short"` columns. Has no influence on other exit methods (Stoploss, ROI, callbacks). [Strategy Override](#parameters-in-the-strategy). <br>*Defaults to `true`.* <br> **Datatype:** Boolean
| `exit_profit_only` | Wait until the bot reaches `exit_profit_offset` before taking an exit decision. [Strategy Override](#parameters-in-the-strategy). <br>*Defaults to `false`.* <br> **Datatype:** Boolean
| `exit_profit_offset` | Exit-signal is only active above this value. Only active in combination with `exit_profit_only=True`. [Strategy Override](#parameters-in-the-strategy). <br>*Defaults to `0.0`.* <br> **Datatype:** Float (as ratio)
| `ignore_roi_if_entry_signal` | Do not exit if the entry signal is still active. This setting takes preference over `minimal_roi` and `use_exit_signal`. [Strategy Override](#parameters-in-the-strategy). <br>*Defaults to `false`.* <br> **Datatype:** Boolean
| `ignore_buying_expired_candle_after` | Specifies the number of seconds until a buy signal is no longer used. <br> **Datatype:** Integer
| `order_types` | Configure order-types depending on the action (`"entry"`, `"exit"`, `"stoploss"`, `"stoploss_on_exchange"`). [More information below](#understand-order_types). [Strategy Override](#parameters-in-the-strategy).<br> **Datatype:** Dict
| `order_time_in_force` | Configure time in force for entry and exit orders. [More information below](#understand-order_time_in_force). [Strategy Override](#parameters-in-the-strategy). <br> **Datatype:** Dict
| `position_adjustment_enable` | Enables the strategy to use position adjustments (additional buys or sells). [More information here](strategy-callbacks.md#adjust-trade-position). <br> [Strategy Override](#parameters-in-the-strategy). <br>*Defaults to `false`.*<br> **Datatype:** Boolean
| `max_entry_position_adjustment` | Maximum additional order(s) for each open trade on top of the first entry Order. Set it to `-1` for unlimited additional orders. [More information here](strategy-callbacks.md#adjust-trade-position). <br> [Strategy Override](#parameters-in-the-strategy). <br>*Defaults to `-1`.*<br> **Datatype:** Positive Integer or -1
| | **Exchange**
| `exchange.name` | **Required.** Name of the exchange class to use. [List below](#user-content-what-values-for-exchangename). <br> **Datatype:** String
| `exchange.key` | API key to use for the exchange. Only required when you are in production mode.<br>**Keep it in secret, do not disclose publicly.** <br> **Datatype:** String
| `exchange.secret` | API secret to use for the exchange. Only required when you are in production mode.<br>**Keep it in secret, do not disclose publicly.** <br> **Datatype:** String
| `exchange.password` | API password to use for the exchange. Only required when you are in production mode and for exchanges that use password for API requests.<br>**Keep it in secret, do not disclose publicly.** <br> **Datatype:** String
| `exchange.uid` | API uid to use for the exchange. Only required when you are in production mode and for exchanges that use uid for API requests.<br>**Keep it in secret, do not disclose publicly.** <br> **Datatype:** String
| `exchange.pair_whitelist` | List of pairs to use by the bot for trading and to check for potential trades during backtesting. Supports regex pairs as `.*/BTC`. Not used by VolumePairList. [More information](plugins.md#pairlists-and-pairlist-handlers). <br> **Datatype:** List
| `exchange.pair_blacklist` | List of pairs the bot must absolutely avoid for trading and backtesting. [More information](plugins.md#pairlists-and-pairlist-handlers). <br> **Datatype:** List
| `exchange.ccxt_config` | Additional CCXT parameters passed to both ccxt instances (sync and async). This is usually the correct place for additional ccxt configurations. Parameters may differ from exchange to exchange and are documented in the [ccxt documentation](https://ccxt.readthedocs.io/en/latest/manual.html#instantiation). Please avoid adding exchange secrets here (use the dedicated fields instead), as they may be contained in logs. <br> **Datatype:** Dict
| `exchange.ccxt_sync_config` | Additional CCXT parameters passed to the regular (sync) ccxt instance. Parameters may differ from exchange to exchange and are documented in the [ccxt documentation](https://ccxt.readthedocs.io/en/latest/manual.html#instantiation) <br> **Datatype:** Dict
| `exchange.ccxt_async_config` | Additional CCXT parameters passed to the async ccxt instance. Parameters may differ from exchange to exchange  and are documented in the [ccxt documentation](https://ccxt.readthedocs.io/en/latest/manual.html#instantiation) <br> **Datatype:** Dict
| `exchange.markets_refresh_interval` | The interval in minutes in which markets are reloaded. <br>*Defaults to `60` minutes.* <br> **Datatype:** Positive Integer
| `exchange.skip_pair_validation` | Skip pairlist validation on startup.<br>*Defaults to `false`*<br> **Datatype:** Boolean
| `exchange.skip_open_order_update` | Skips open order updates on startup should the exchange cause problems. Only relevant in live conditions.<br>*Defaults to `false`*<br> **Datatype:** Boolean
| `exchange.unknown_fee_rate` | Fallback value to use when calculating trading fees. This can be useful for exchanges which have fees in non-tradable currencies. The value provided here will be multiplied with the "fee cost".<br>*Defaults to `None`<br> **Datatype:** float
| `exchange.log_responses` | Log relevant exchange responses. For debug mode only - use with care.<br>*Defaults to `false`*<br> **Datatype:** Boolean
| `experimental.block_bad_exchanges` | Block exchanges known to not work with freqtrade. Leave on default unless you want to test if that exchange works now. <br>*Defaults to `true`.* <br> **Datatype:** Boolean
| | **Plugins**
| `edge.*` | Please refer to [edge configuration document](edge.md) for detailed explanation of all possible configuration options.
| `pairlists` | Define one or more pairlists to be used. [More information](plugins.md#pairlists-and-pairlist-handlers). <br>*Defaults to `StaticPairList`.*  <br> **Datatype:** List of Dicts
| `protections` | Define one or more protections to be used. [More information](plugins.md#protections). <br> **Datatype:** List of Dicts
| | **Telegram**
| `telegram.enabled` | Enable the usage of Telegram. <br> **Datatype:** Boolean
| `telegram.token` | Your Telegram bot token. Only required if `telegram.enabled` is `true`. <br>**Keep it in secret, do not disclose publicly.** <br> **Datatype:** String
| `telegram.chat_id` | Your personal Telegram account id. Only required if `telegram.enabled` is `true`. <br>**Keep it in secret, do not disclose publicly.** <br> **Datatype:** String
| `telegram.balance_dust_level` | Dust-level (in stake currency) - currencies with a balance below this will not be shown by `/balance`. <br> **Datatype:** float
| `telegram.reload` | Allow "reload" buttons on telegram messages. <br>*Defaults to `true`.<br> **Datatype:** boolean
| `telegram.notification_settings.*` | Detailed notification settings. Refer to the [telegram documentation](telegram-usage.md) for details.<br> **Datatype:** dictionary
| `telegram.allow_custom_messages` | Enable the sending of Telegram messages from strategies via the dataprovider.send_msg() function. <br> **Datatype:** Boolean
| | **Webhook**
| `webhook.enabled` | Enable usage of Webhook notifications <br> **Datatype:** Boolean
| `webhook.url` | URL for the webhook. Only required if `webhook.enabled` is `true`. See the [webhook documentation](webhook-config.md) for more details. <br> **Datatype:** String
| `webhook.entry` | Payload to send on entry. Only required if `webhook.enabled` is `true`. See the [webhook documentation](webhook-config.md) for more details. <br> **Datatype:** String
| `webhook.entry_cancel` | Payload to send on entry order cancel. Only required if `webhook.enabled` is `true`. See the [webhook documentation](webhook-config.md) for more details. <br> **Datatype:** String
| `webhook.entry_fill` | Payload to send on entry order filled. Only required if `webhook.enabled` is `true`. See the [webhook documentation](webhook-config.md) for more details. <br> **Datatype:** String
| `webhook.exit` | Payload to send on exit. Only required if `webhook.enabled` is `true`. See the [webhook documentation](webhook-config.md) for more details. <br> **Datatype:** String
| `webhook.exit_cancel` | Payload to send on exit order cancel. Only required if `webhook.enabled` is `true`. See the [webhook documentation](webhook-config.md) for more details. <br> **Datatype:** String
| `webhook.exit_fill` | Payload to send on exit order filled. Only required if `webhook.enabled` is `true`. See the [webhook documentation](webhook-config.md) for more details. <br> **Datatype:** String
| `webhook.status` | Payload to send on status calls. Only required if `webhook.enabled` is `true`. See the [webhook documentation](webhook-config.md) for more details. <br> **Datatype:** String
| `webhook.allow_custom_messages` | Enable the sending of Webhook messages from strategies via the dataprovider.send_msg() function. <br> **Datatype:** Boolean
| | **Rest API / FreqUI / Producer-Consumer**
| `api_server.enabled` | Enable usage of API Server. See the [API Server documentation](rest-api.md) for more details. <br> **Datatype:** Boolean
| `api_server.listen_ip_address` | Bind IP address. See the [API Server documentation](rest-api.md) for more details. <br> **Datatype:** IPv4
| `api_server.listen_port` | Bind Port. See the [API Server documentation](rest-api.md) for more details. <br>**Datatype:** Integer between 1024 and 65535
| `api_server.verbosity` | Logging verbosity. `info` will print all RPC Calls, while "error" will only display errors. <br>**Datatype:** Enum, either `info` or `error`. Defaults to `info`.
| `api_server.username` | Username for API server. See the [API Server documentation](rest-api.md) for more details. <br>**Keep it in secret, do not disclose publicly.**<br> **Datatype:** String
| `api_server.password` | Password for API server. See the [API Server documentation](rest-api.md) for more details. <br>**Keep it in secret, do not disclose publicly.**<br> **Datatype:** String
| `api_server.ws_token` | API token for the Message WebSocket. See the [API Server documentation](rest-api.md) for more details.  <br>**Keep it in secret, do not disclose publicly.** <br> **Datatype:** String
| `bot_name` | Name of the bot. Passed via API to a client - can be shown to distinguish / name bots.<br> *Defaults to `freqtrade`*<br> **Datatype:** String
| `external_message_consumer` | Enable [Producer/Consumer mode](producer-consumer.md) for more details. <br> **Datatype:** Dict
| | **Other**
| `initial_state` | Defines the initial application state. If set to stopped, then the bot has to be explicitly started via `/start` RPC command. <br>*Defaults to `stopped`.* <br> **Datatype:** Enum, either `stopped` or `running`
| `force_entry_enable` | Enables the RPC Commands to force a Trade entry. More information below. <br> **Datatype:** Boolean
| `disable_dataframe_checks` | Disable checking the OHLCV dataframe returned from the strategy methods for correctness. Only use when intentionally changing the dataframe and understand what you are doing. [Strategy Override](#parameters-in-the-strategy).<br> *Defaults to `False`*. <br> **Datatype:** Boolean
| `internals.process_throttle_secs` | Set the process throttle, or minimum loop duration for one bot iteration loop. Value in second. <br>*Defaults to `5` seconds.* <br> **Datatype:** Positive Integer
| `internals.heartbeat_interval` | Print heartbeat message every N seconds. Set to 0 to disable heartbeat messages. <br>*Defaults to `60` seconds.* <br> **Datatype:** Positive Integer or 0
| `internals.sd_notify` | Enables use of the sd_notify protocol to tell systemd service manager about changes in the bot state and issue keep-alive pings. See [here](installation.md#7-optional-configure-freqtrade-as-a-systemd-service) for more details. <br> **Datatype:** Boolean
| `strategy` | **Required** Defines Strategy class to use. Recommended to be set via `--strategy NAME`. <br> **Datatype:** ClassName
| `strategy_path` | Adds an additional strategy lookup path (must be a directory). <br> **Datatype:** String
| `recursive_strategy_search` | Set to `true` to recursively search sub-directories inside `user_data/strategies` for a strategy. <br> **Datatype:** Boolean
| `user_data_dir` | Directory containing user data. <br> *Defaults to `./user_data/`*. <br> **Datatype:** String
| `db_url` | Declares database URL to use. NOTE: This defaults to `sqlite:///tradesv3.dryrun.sqlite` if `dry_run` is `true`, and to `sqlite:///tradesv3.sqlite` for production instances. <br> **Datatype:** String, SQLAlchemy connect string
| `logfile` | Specifies logfile name. Uses a rolling strategy for log file rotation for 10 files with the 1MB limit per file. <br> **Datatype:** String
| `add_config_files` | Additional config files. These files will be loaded and merged with the current config file. The files are resolved relative to the initial file.<br> *Defaults to `[]`*. <br> **Datatype:** List of strings
| `dataformat_ohlcv` | Data format to use to store historical candle (OHLCV) data. <br> *Defaults to `feather`*. <br> **Datatype:** String
| `dataformat_trades` | Data format to use to store historical trades data. <br> *Defaults to `feather`*. <br> **Datatype:** String
| `reduce_df_footprint` | Recast all numeric columns to float32/int32, with the objective of reducing ram/disk usage (and decreasing train/inference timing in FreqAI). (Currently only affects FreqAI use-cases) <br> **Datatype:** Boolean. <br> Default: `False`.

### Parameters in the strategy

The following parameters can be set in the configuration file or strategy.
Values set in the configuration file always overwrite values set in the strategy.

* `minimal_roi`
* `timeframe`
* `stoploss`
* `max_open_trades`
* `trailing_stop`
* `trailing_stop_positive`
* `trailing_stop_positive_offset`
* `trailing_only_offset_is_reached`
* `use_custom_stoploss`
* `process_only_new_candles`
* `order_types`
* `order_time_in_force`
* `unfilledtimeout`
* `disable_dataframe_checks`
- `use_exit_signal`
* `exit_profit_only`
- `exit_profit_offset`
- `ignore_roi_if_entry_signal`
* `ignore_buying_expired_candle_after`
* `position_adjustment_enable`
* `max_entry_position_adjustment`

### Configuring amount per trade

There are several methods to configure how much of the stake currency the bot will use to enter a trade. All methods respect the [available balance configuration](#tradable-balance) as explained below.

#### Minimum trade stake

The minimum stake amount will depend on exchange and pair and is usually listed in the exchange support pages.

Assuming the minimum tradable amount for XRP/USD is 20 XRP (given by the exchange), and the price is 0.6$, the minimum stake amount to buy this pair is `20 * 0.6 ~= 12`.
This exchange has also a limit on USD - where all orders must be > 10$ - which however does not apply in this case.

To guarantee safe execution, freqtrade will not allow buying with a stake-amount of 10.1$, instead, it'll make sure that there's enough space to place a stoploss below the pair (+ an offset, defined by `amount_reserve_percent`, which defaults to 5%).

With a reserve of 5%, the minimum stake amount would be ~12.6$ (`12 * (1 + 0.05)`). If we take into account a stoploss of 10% on top of that - we'd end up with a value of ~14$ (`12.6 / (1 - 0.1)`).

To limit this calculation in case of large stoploss values, the calculated minimum stake-limit will never be more than 50% above the real limit.

!!! Warning
    Since the limits on exchanges are usually stable and are not updated often, some pairs can show pretty high minimum limits, simply because the price increased a lot since the last limit adjustment by the exchange. Freqtrade adjusts the stake-amount to this value, unless it's > 30% more than the calculated/desired stake-amount - in which case the trade is rejected.

#### Tradable balance

By default, the bot assumes that the `complete amount - 1%` is at it's disposal, and when using [dynamic stake amount](#dynamic-stake-amount), it will split the complete balance into `max_open_trades` buckets per trade.
Freqtrade will reserve 1% for eventual fees when entering a trade and will therefore not touch that by default.

You can configure the "untouched" amount by using the `tradable_balance_ratio` setting.

For example, if you have 10 ETH available in your wallet on the exchange and `tradable_balance_ratio=0.5` (which is 50%), then the bot will use a maximum amount of 5 ETH for trading and considers this as an available balance. The rest of the wallet is untouched by the trades.

!!! Danger
    This setting should **not** be used when running multiple bots on the same account. Please look at [Available Capital to the bot](#assign-available-capital) instead.

!!! Warning
    The `tradable_balance_ratio` setting applies to the current balance (free balance + tied up in trades). Therefore, assuming the starting balance of 1000, a configuration with `tradable_balance_ratio=0.99` will not guarantee that 10 currency units will always remain available on the exchange. For example, the free amount may reduce to 5 units if the total balance is reduced to 500 (either by a losing streak or by withdrawing balance).

#### Assign available Capital

To fully utilize compounding profits when using multiple bots on the same exchange account, you'll want to limit each bot to a certain starting balance.
This can be accomplished by setting `available_capital` to the desired starting balance.

Assuming your account has 10000 USDT and you want to run 2 different strategies on this exchange.
You'd set `available_capital=5000` - granting each bot an initial capital of 5000 USDT.
The bot will then split this starting balance equally into `max_open_trades` buckets.
Profitable trades will result in increased stake-sizes for this bot - without affecting the stake-sizes of the other bot.

Adjusting `available_capital` requires reloading the configuration to take effect. Adjusting the `available_capital` adds the difference between the previous `available_capital` and the new `available_capital`. Decreasing the available capital when trades are open doesn't exit the trades. The difference is returned to the wallet when the trades conclude. The outcome of this differs depending on the price movement between the adjustment and exiting the trades.

!!! Warning "Incompatible with `tradable_balance_ratio`"
    Setting this option will replace any configuration of `tradable_balance_ratio`.

#### Amend last stake amount

Assuming we have the tradable balance of 1000 USDT, `stake_amount=400`, and `max_open_trades=3`.
The bot would open 2 trades and will be unable to fill the last trading slot, since the requested 400 USDT are no longer available since 800 USDT are already tied in other trades.

To overcome this, the option `amend_last_stake_amount` can be set to `True`, which will enable the bot to reduce stake_amount to the available balance to fill the last trade slot.

In the example above this would mean:

- Trade1: 400 USDT
- Trade2: 400 USDT
- Trade3: 200 USDT

!!! Note
    This option only applies with [Static stake amount](#static-stake-amount) - since [Dynamic stake amount](#dynamic-stake-amount) divides the balances evenly.

!!! Note
    The minimum last stake amount can be configured using `last_stake_amount_min_ratio` - which defaults to 0.5 (50%). This means that the minimum stake amount that's ever used is `stake_amount * 0.5`. This avoids very low stake amounts, that are close to the minimum tradable amount for the pair and can be refused by the exchange.

#### Static stake amount

The `stake_amount` configuration statically configures the amount of stake-currency your bot will use for each trade.

The minimal configuration value is 0.0001, however, please check your exchange's trading minimums for the stake currency you're using to avoid problems.

This setting works in combination with `max_open_trades`. The maximum capital engaged in trades is `stake_amount * max_open_trades`.
For example, the bot will at most use (0.05 BTC x 3) = 0.15 BTC, assuming a configuration of `max_open_trades=3` and `stake_amount=0.05`.

!!! Note
    This setting respects the [available balance configuration](#available-balance).

#### Dynamic stake amount

Alternatively, you can use a dynamic stake amount, which will use the available balance on the exchange, and divide that equally by the number of allowed trades (`max_open_trades`).

To configure this, set `stake_amount="unlimited"`. We also recommend to set `tradable_balance_ratio=0.99` (99%) - to keep a minimum balance for eventual fees.

In this case a trade amount is calculated as:

```python
currency_balance / (max_open_trades - current_open_trades)
```

To allow the bot to trade all the available `stake_currency` in your account (minus `tradable_balance_ratio`) set

```json
"stake_amount" : "unlimited",
"tradable_balance_ratio": 0.99,
```

!!! Tip "Compounding profits"
    This configuration will allow increasing/decreasing stakes depending on the performance of the bot (lower stake if the bot is losing, higher stakes if the bot has a winning record since higher balances are available), and will result in profit compounding.

!!! Note "When using Dry-Run Mode"
    When using `"stake_amount" : "unlimited",` in combination with Dry-Run, Backtesting or Hyperopt, the balance will be simulated starting with a stake of `dry_run_wallet` which will evolve.
    It is therefore important to set `dry_run_wallet` to a sensible value (like 0.05 or 0.01 for BTC and 1000 or 100 for USDT, for example), otherwise, it may simulate trades with 100 BTC (or more) or 0.05 USDT (or less) at once - which may not correspond to your real available balance or is less than the exchange minimal limit for the order amount for the stake currency.

#### Dynamic stake amount with position adjustment

When you want to use position adjustment with unlimited stakes, you must also implement `custom_stake_amount` to a return a value depending on your strategy.
Typical value would be in the range of 25% - 50% of the proposed stakes, but depends highly on your strategy and how much you wish to leave into the wallet as position adjustment buffer.

For example if your position adjustment assumes it can do 2 additional buys with the same stake amounts then your buffer should be 66.6667% of the initially proposed unlimited stake amount.

Or another example if your position adjustment assumes it can do 1 additional buy with 3x the original stake amount then `custom_stake_amount` should return 25% of proposed stake amount and leave 75% for possible later position adjustments.

--8<-- "includes/pricing.md"

### Understand minimal_roi

The `minimal_roi` configuration parameter is a JSON object where the key is a duration
in minutes and the value is the minimum ROI as a ratio.
See the example below:

```json
"minimal_roi": {
    "40": 0.0,    # Exit after 40 minutes if the profit is not negative
    "30": 0.01,   # Exit after 30 minutes if there is at least 1% profit
    "20": 0.02,   # Exit after 20 minutes if there is at least 2% profit
    "0":  0.04    # Exit immediately if there is at least 4% profit
},
```

Most of the strategy files already include the optimal `minimal_roi` value.
This parameter can be set in either Strategy or Configuration file. If you use it in the configuration file, it will override the
`minimal_roi` value from the strategy file.
If it is not set in either Strategy or Configuration, a default of 1000% `{"0": 10}` is used, and minimal ROI is disabled unless your trade generates 1000% profit.

!!! Note "Special case to forceexit after a specific time"
    A special case presents using `"<N>": -1` as ROI. This forces the bot to exit a trade after N Minutes, no matter if it's positive or negative, so represents a time-limited force-exit.

### Understand force_entry_enable

The `force_entry_enable` configuration parameter enables the usage of force-enter (`/forcelong`, `/forceshort`) commands via Telegram and REST API.
For security reasons, it's disabled by default, and freqtrade will show a warning message on startup if enabled.
For example, you can send `/forceenter ETH/BTC` to the bot, which will result in freqtrade buying the pair and holds it until a regular exit-signal (ROI, stoploss, /forceexit) appears.

This can be dangerous with some strategies, so use with care.

See [the telegram documentation](telegram-usage.md) for details on usage.

### Ignoring expired candles

When working with larger timeframes (for example 1h or more) and using a low `max_open_trades` value, the last candle can be processed as soon as a trade slot becomes available. When processing the last candle, this can lead to a situation where it may not be desirable to use the buy signal on that candle. For example, when using a condition in your strategy where you use a cross-over, that point may have passed too long ago for you to start a trade on it.

In these situations, you can enable the functionality to ignore candles that are beyond a specified period by setting `ignore_buying_expired_candle_after` to a positive number, indicating the number of seconds after which the buy signal becomes expired.

For example, if your strategy is using a 1h timeframe, and you only want to buy within the first 5 minutes when a new candle comes in, you can add the following configuration to your strategy:

``` json
  {
    //...
    "ignore_buying_expired_candle_after": 300,
    // ...
  }
```

!!! Note
    This setting resets with each new candle, so it will not prevent sticking-signals from executing on the 2nd or 3rd candle they're active. Best use a "trigger" selector for buy signals, which are only active for one candle.

### Understand order_types

The `order_types` configuration parameter maps actions (`entry`, `exit`, `stoploss`, `emergency_exit`, `force_exit`, `force_entry`) to order-types (`market`, `limit`, ...) as well as configures stoploss to be on the exchange and defines stoploss on exchange update interval in seconds.

This allows to enter using limit orders, exit using limit-orders, and create stoplosses using market orders.
It also allows to set the
stoploss "on exchange" which means stoploss order would be placed immediately once the buy order is fulfilled.

`order_types` set in the configuration file overwrites values set in the strategy as a whole, so you need to configure the whole `order_types` dictionary in one place.

If this is configured, the following 4 values (`entry`, `exit`, `stoploss` and `stoploss_on_exchange`) need to be present, otherwise, the bot will fail to start.

For information on (`emergency_exit`,`force_exit`, `force_entry`, `stoploss_on_exchange`,`stoploss_on_exchange_interval`,`stoploss_on_exchange_limit_ratio`) please see stop loss documentation [stop loss on exchange](stoploss.md)

Syntax for Strategy:

```python
order_types = {
    "entry": "limit",
    "exit": "limit",
    "emergency_exit": "market",
    "force_entry": "market",
    "force_exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": False,
    "stoploss_on_exchange_interval": 60,
    "stoploss_on_exchange_limit_ratio": 0.99,
}
```

Configuration:

```json
"order_types": {
    "entry": "limit",
    "exit": "limit",
    "emergency_exit": "market",
    "force_entry": "market",
    "force_exit": "market",
    "stoploss": "market",
    "stoploss_on_exchange": false,
    "stoploss_on_exchange_interval": 60
}
```

!!! Note "Market order support"
    Not all exchanges support "market" orders.
    The following message will be shown if your exchange does not support market orders:
    `"Exchange <yourexchange> does not support market orders."` and the bot will refuse to start.

!!! Warning "Using market orders"
    Please carefully read the section [Market order pricing](#market-order-pricing) section when using market orders.

!!! Note "Stoploss on exchange"
    `order_types.stoploss_on_exchange_interval` is not mandatory. Do not change its value if you are
    unsure of what you are doing. For more information about how stoploss works please
    refer to [the stoploss documentation](stoploss.md).

    If `order_types.stoploss_on_exchange` is enabled and the stoploss is cancelled manually on the exchange, then the bot will create a new stoploss order.

!!! Warning "Warning: order_types.stoploss_on_exchange failures"
    If stoploss on exchange creation fails for some reason, then an "emergency exit" is initiated. By default, this will exit the trade using a market order. The order-type for the emergency-exit can be changed by setting the `emergency_exit` value in the `order_types` dictionary - however, this is not advised.

### Understand order_time_in_force

The `order_time_in_force` configuration parameter defines the policy by which the order
is executed on the exchange. Three commonly used time in force are:

**GTC (Good Till Canceled):**

This is most of the time the default time in force. It means the order will remain
on exchange till it is cancelled by the user. It can be fully or partially fulfilled.
If partially fulfilled, the remaining will stay on the exchange till cancelled.

**FOK (Fill Or Kill):**

It means if the order is not executed immediately AND fully then it is cancelled by the exchange.

**IOC (Immediate Or Canceled):**

It is the same as FOK (above) except it can be partially fulfilled. The remaining part
is automatically cancelled by the exchange.

**PO (Post only):**

Post only order. The order is either placed as a maker order, or it is canceled.
This means the order must be placed on orderbook for at at least time in an unfilled state.

#### time_in_force config

The `order_time_in_force` parameter contains a dict with entry and exit time in force policy values.
This can be set in the configuration file or in the strategy.
Values set in the configuration file overwrites values set in the strategy.

The possible values are: `GTC` (default), `FOK` or `IOC`.

``` python
"order_time_in_force": {
    "entry": "GTC",
    "exit": "GTC"
},
```

!!! Warning
    This is ongoing work. For now, it is supported only for binance, gate and kucoin.
    Please don't change the default value unless you know what you are doing and have researched the impact of using different values for your particular exchange.

### What values can be used for fiat_display_currency?

The `fiat_display_currency` configuration parameter sets the base currency to use for the
conversion from coin to fiat in the bot Telegram reports.

The valid values are:

```json
"AUD", "BRL", "CAD", "CHF", "CLP", "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HUF", "IDR", "ILS", "INR", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PKR", "PLN", "RUB", "SEK", "SGD", "THB", "TRY", "TWD", "ZAR", "USD"
```

In addition to fiat currencies, a range of crypto currencies is supported.

The valid values are:

```json
"BTC", "ETH", "XRP", "LTC", "BCH", "BNB"
```

Removing `fiat_display_currency` completely from the configuration will skip initializing coingecko, and will not show any FIAT currency conversion. This has no importance for the correct functioning of the bot.

## Using Dry-run mode

We recommend starting the bot in the Dry-run mode to see how your bot will
behave and what is the performance of your strategy. In the Dry-run mode, the
bot does not engage your money. It only runs a live simulation without
creating trades on the exchange.

1. Edit your `config.json` configuration file.
2. Switch `dry-run` to `true` and specify `db_url` for a persistence database.

```json
"dry_run": true,
"db_url": "sqlite:///tradesv3.dryrun.sqlite",
```

3. Remove your Exchange API key and secret (change them by empty values or fake credentials):

```json
"exchange": {
    "name": "binance",
    "key": "key",
    "secret": "secret",
    ...
}
```

Once you will be happy with your bot performance running in the Dry-run mode, you can switch it to production mode.

!!! Note
    A simulated wallet is available during dry-run mode and will assume a starting capital of `dry_run_wallet` (defaults to 1000).

### Considerations for dry-run

* API-keys may or may not be provided. Only Read-Only operations (i.e. operations that do not alter account state) on the exchange are performed in dry-run mode.
* Wallets (`/balance`) are simulated based on `dry_run_wallet`.
* Orders are simulated, and will not be posted to the exchange.
* Market orders fill based on orderbook volume the moment the order is placed.
* Limit orders fill once the price reaches the defined level - or time out based on `unfilledtimeout` settings.
* Limit orders will be converted to market orders if they cross the price by more than 1%.
* In combination with `stoploss_on_exchange`, the stop_loss price is assumed to be filled.
* Open orders (not trades, which are stored in the database) are kept open after bot restarts, with the assumption that they were not filled while being offline.

## Switch to production mode

In production mode, the bot will engage your money. Be careful, since a wrong strategy can lose all your money.
Be aware of what you are doing when you run it in production mode.

When switching to Production mode, please make sure to use a different / fresh database to avoid dry-run trades messing with your exchange money and eventually tainting your statistics.

### Setup your exchange account

You will need to create API Keys (usually you get `key` and `secret`, some exchanges require an additional `password`) from the Exchange website and you'll need to insert this into the appropriate fields in the configuration or when asked by the `freqtrade new-config` command.
API Keys are usually only required for live trading (trading for real money, bot running in "production mode", executing real orders on the exchange) and are not required for the bot running in dry-run (trade simulation) mode. When you set up the bot in dry-run mode, you may fill these fields with empty values.

### To switch your bot in production mode

**Edit your `config.json`  file.**

**Switch dry-run to false and don't forget to adapt your database URL if set:**

```json
"dry_run": false,
```

**Insert your Exchange API key (change them by fake API keys):**

```json
{
    "exchange": {
        "name": "binance",
        "key": "af8ddd35195e9dc500b9a6f799f6f5c93d89193b",
        "secret": "08a9dc6db3d7b53e1acebd9275677f4b0a04f1a5",
        //"password": "", // Optional, not needed by all exchanges)
        // ...
    }
    //...
}
```

You should also make sure to read the [Exchanges](exchanges.md) section of the documentation to be aware of potential configuration details specific to your exchange.

!!! Hint "Keep your secrets secret"
    To keep your secrets secret, we recommend using a 2nd configuration for your API keys.
    Simply use the above snippet in a new configuration file (e.g. `config-private.json`) and keep your settings in this file.
    You can then start the bot with `freqtrade trade --config user_data/config.json --config user_data/config-private.json <...>` to have your keys loaded.

    **NEVER** share your private configuration file or your exchange keys with anyone!

### Using proxy with Freqtrade

To use a proxy with freqtrade, export your proxy settings using the variables `"HTTP_PROXY"` and `"HTTPS_PROXY"` set to the appropriate values.
This will have the proxy settings applied to everything (telegram, coingecko, ...) **except** for exchange requests.

``` bash
export HTTP_PROXY="http://addr:port"
export HTTPS_PROXY="http://addr:port"
freqtrade
```

#### Proxy exchange requests

To use a proxy for exchange connections - you will have to define the proxies as part of the ccxt configuration.

``` json
{ 
  "exchange": {
    "ccxt_config": {
      "httpsProxy": "http://addr:port",
    }
  }
}
```

For more information on available proxy types, please consult the [ccxt proxy documentation](https://docs.ccxt.com/#/README?id=proxy).

## Next step

Now you have configured your config.json, the next step is to [start your bot](bot-usage.md).
