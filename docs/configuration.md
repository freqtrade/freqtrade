# Configure the bot

This page explains how to configure your `config.json` file.

## Setup config.json

We recommend to copy and use the `config.json.example` as a template
for your bot configuration.

The table below will list all configuration parameters.

Mandatory Parameters are marked as **Required**.

|  Command | Default | Description |
|----------|---------|-------------|
| `max_open_trades` | 3 | **Required.** Number of trades open your bot will have. If -1 then it is ignored (i.e. potentially unlimited open trades)
| `stake_currency` | BTC | **Required.** Crypto-currency used for trading. [Strategy Override](#parameters-in-the-strategy).
| `stake_amount` | 0.05 | **Required.** Amount of crypto-currency your bot will use for each trade. Per default, the bot will use (0.05 BTC x 3) = 0.15 BTC in total will be always engaged. Set it to `"unlimited"` to allow the bot to use all available balance. [Strategy Override](#parameters-in-the-strategy).
| `amount_reserve_percent` | 0.05 | Reserve some amount in min pair stake amount. Default is 5%. The bot will reserve `amount_reserve_percent` + stop-loss value when calculating min pair stake amount in order to avoid possible trade refusals.
| `ticker_interval` | [1m, 5m, 15m, 30m, 1h, 1d, ...] | The ticker interval to use (1min, 5 min, 15 min, 30 min, 1 hour or 1 day). Default is 5 minutes. [Strategy Override](#parameters-in-the-strategy).
| `fiat_display_currency` | USD | **Required.** Fiat currency used to show your profits. More information below.
| `dry_run` | true | **Required.** Define if the bot must be in Dry-run or production mode.
| `dry_run_wallet` | 999.9 | Overrides the default amount of 999.9 stake currency units in the wallet used by the bot running in the Dry Run mode if you need it for any reason.
| `process_only_new_candles` | false | If set to true indicators are processed only once a new candle arrives. If false each loop populates the indicators, this will mean the same candle is processed many times creating system load but can be useful of your strategy depends on tick data not only candle. [Strategy Override](#parameters-in-the-strategy).
| `minimal_roi` | See below | Set the threshold in percent the bot will use to sell a trade. More information below. [Strategy Override](#parameters-in-the-strategy).
| `stoploss` | -0.10 | Value of the stoploss in percent used by the bot. More information below. More details in the [stoploss documentation](stoploss.md). [Strategy Override](#parameters-in-the-strategy).
| `trailing_stop` | false | Enables trailing stop-loss (based on `stoploss` in either configuration or strategy file). More details in the [stoploss documentation](stoploss.md). [Strategy Override](#parameters-in-the-strategy).
| `trailing_stop_positive` | 0 | Changes stop-loss once profit has been reached. More details in the [stoploss documentation](stoploss.md). [Strategy Override](#parameters-in-the-strategy).
| `trailing_stop_positive_offset` | 0 | Offset on when to apply `trailing_stop_positive`. Percentage value which should be positive. More details in the [stoploss documentation](stoploss.md). [Strategy Override](#parameters-in-the-strategy).
| `trailing_only_offset_is_reached` | false | Only apply trailing stoploss when the offset is reached. [stoploss documentation](stoploss.md). [Strategy Override](#parameters-in-the-strategy).
| `unfilledtimeout.buy` | 10 | **Required.** How long (in minutes) the bot will wait for an unfilled buy order to complete, after which the order will be cancelled.
| `unfilledtimeout.sell` | 10 | **Required.** How long (in minutes) the bot will wait for an unfilled sell order to complete, after which the order will be cancelled.
| `bid_strategy.ask_last_balance` | 0.0 | **Required.** Set the bidding price. More information [below](#understand-ask_last_balance).
| `bid_strategy.use_order_book` | false | Allows buying of pair using the rates in Order Book Bids.
| `bid_strategy.order_book_top` | 0 | Bot will use the top N rate in Order Book Bids. Ie. a value of 2 will allow the bot to pick the 2nd bid rate in Order Book Bids.
| `bid_strategy. check_depth_of_market.enabled` | false | Does not buy if the % difference of buy orders and sell orders is met in Order Book.
| `bid_strategy. check_depth_of_market.bids_to_ask_delta` | 0 | The % difference of buy orders and sell orders found in Order Book. A value lesser than 1 means sell orders is greater, while value greater than 1 means buy orders is higher.
| `ask_strategy.use_order_book` | false | Allows selling of open traded pair using the rates in Order Book Asks.
| `ask_strategy.order_book_min` | 0 | Bot will scan from the top min to max Order Book Asks searching for a profitable rate.
| `ask_strategy.order_book_max` | 0 | Bot will scan from the top min to max Order Book Asks searching for a profitable rate.
| `order_types` | None | Configure order-types depending on the action (`"buy"`, `"sell"`, `"stoploss"`, `"stoploss_on_exchange"`). [More information below](#understand-order_types). [Strategy Override](#parameters-in-the-strategy).
| `order_time_in_force` | None | Configure time in force for buy and sell orders. [More information below](#understand-order_time_in_force). [Strategy Override](#parameters-in-the-strategy).
| `exchange.name` |  | **Required.** Name of the exchange class to use. [List below](#user-content-what-values-for-exchangename).
| `exchange.sandbox` | false | Use the 'sandbox' version of the exchange, where the exchange provides a sandbox for risk-free integration. See [here](sandbox-testing.md) in more details.
| `exchange.key` | '' | API key to use for the exchange. Only required when you are in production mode.
| `exchange.secret` | '' | API secret to use for the exchange. Only required when you are in production mode.
| `exchange.pair_whitelist` | [] | List of currency to use by the bot. Can be overrided with `--dynamic-whitelist` param.
| `exchange.pair_blacklist` | [] | List of currency the bot must avoid. Useful when using `--dynamic-whitelist` param.
| `exchange.ccxt_config` | None | Additional CCXT parameters passed to the regular ccxt instance. Parameters may differ from exchange to exchange and are documented in the [ccxt documentation](https://ccxt.readthedocs.io/en/latest/manual.html#instantiation)
| `exchange.ccxt_async_config` | None | Additional CCXT parameters passed to the async ccxt instance. Parameters may differ from exchange to exchange  and are documented in the [ccxt documentation](https://ccxt.readthedocs.io/en/latest/manual.html#instantiation)
| `exchange.markets_refresh_interval` | 60 | The interval in minutes in which markets are reloaded.
| `edge` | false | Please refer to [edge configuration document](edge.md) for detailed explanation.
| `experimental.use_sell_signal` | false | Use your sell strategy in addition of the `minimal_roi`. [Strategy Override](#parameters-in-the-strategy).
| `experimental.sell_profit_only` | false | Waits until you have made a positive profit before taking a sell decision. [Strategy Override](#parameters-in-the-strategy).
| `experimental.ignore_roi_if_buy_signal` | false | Does not sell if the buy-signal is still active. Takes preference over `minimal_roi` and `use_sell_signal`. [Strategy Override](#parameters-in-the-strategy).
| `pairlist.method` | StaticPairList | Use Static whitelist. [More information below](#dynamic-pairlists).
| `pairlist.config` | None | Additional configuration for dynamic pairlists. [More information below](#dynamic-pairlists).
| `telegram.enabled` | true | **Required.** Enable or not the usage of Telegram.
| `telegram.token` | token | Your Telegram bot token. Only required if `telegram.enabled` is `true`.
| `telegram.chat_id` | chat_id | Your personal Telegram account id. Only required if `telegram.enabled` is `true`.
| `webhook.enabled` | false | Enable usage of Webhook notifications
| `webhook.url` | false | URL for the webhook. Only required if `webhook.enabled` is `true`. See the [webhook documentation](webhook-config.md) for more details.
| `webhook.webhookbuy` | false | Payload to send on buy. Only required if `webhook.enabled` is `true`. See the [webhook documentationV](webhook-config.md) for more details.
| `webhook.webhooksell` | false | Payload to send on sell. Only required if `webhook.enabled` is `true`. See the [webhook documentationV](webhook-config.md) for more details.
| `webhook.webhookstatus` | false | Payload to send on status calls. Only required if `webhook.enabled` is `true`. See the [webhook documentationV](webhook-config.md) for more details.
| `db_url` | `sqlite:///tradesv3.sqlite`| Declares database URL to use. NOTE: This defaults to `sqlite://` if `dry_run` is `True`.
| `initial_state` | running | Defines the initial application state. More information below.
| `forcebuy_enable` | false | Enables the RPC Commands to force a buy. More information below.
| `strategy` | DefaultStrategy | Defines Strategy class to use.
| `strategy_path` | null | Adds an additional strategy lookup path (must be a folder).
| `internals.process_throttle_secs` | 5 | **Required.** Set the process throttle. Value in second.
| `internals.sd_notify` | false | Enables use of the sd_notify protocol to tell systemd service manager about changes in the bot state and issue keep-alive pings. See [here](installation.md#7-optional-configure-freqtrade-as-a-systemd-service) for more details.
| `logfile` | | Specify Logfile. Uses a rolling strategy of 10 files, with 1Mb per file.

### Parameters in the strategy

The following parameters can be set in either configuration file or strategy.
Values set in the configuration file always overwrite values set in the strategy.

* `stake_currency`
* `stake_amount`
* `ticker_interval`
* `minimal_roi`
* `stoploss`
* `trailing_stop`
* `trailing_stop_positive`
* `trailing_stop_positive_offset`
* `process_only_new_candles`
* `order_types`
* `order_time_in_force`
* `use_sell_signal` (experimental)
* `sell_profit_only` (experimental)
* `ignore_roi_if_buy_signal` (experimental)

### Understand stake_amount

The `stake_amount` configuration parameter is an amount of crypto-currency your bot will use for each trade.
The minimal value is 0.0005. If there is not enough crypto-currency in
the account an exception is generated.
To allow the bot to trade all the available `stake_currency` in your account set

```json
"stake_amount" : "unlimited",
```

In this case a trade amount is calclulated as:

```python
currency_balanse / (max_open_trades - current_open_trades)
```

### Understand minimal_roi

The `minimal_roi` configuration parameter is a JSON object where the key is a duration
in minutes and the value is the minimum ROI in percent.
See the example below:

```json
"minimal_roi": {
    "40": 0.0,    # Sell after 40 minutes if the profit is not negative
    "30": 0.01,   # Sell after 30 minutes if there is at least 1% profit
    "20": 0.02,   # Sell after 20 minutes if there is at least 2% profit
    "0":  0.04    # Sell immediately if there is at least 4% profit
},
```

Most of the strategy files already include the optimal `minimal_roi` value.
This parameter can be set in either Strategy or Configuration file. If you use it in the configuration file, it will override the
`minimal_roi` value from the strategy file.
If it is not set in either Strategy or Configuration, a default of 1000% `{"0": 10}` is used, and minimal roi is disabled unless your trade generates 1000% profit.

### Understand stoploss

The `stoploss` configuration parameter is loss in percentage that should trigger a sale.
For example, value `-0.10` will cause immediate sell if the
profit dips below -10% for a given trade. This parameter is optional.

Most of the strategy files already include the optimal `stoploss`
value. This parameter is optional. If you use it in the configuration file, it will take over the
`stoploss` value from the strategy file.

### Understand trailing stoploss

Go to the [trailing stoploss Documentation](stoploss.md) for details on trailing stoploss.

### Understand initial_state

The `initial_state` configuration parameter is an optional field that defines the initial application state.
Possible values are `running` or `stopped`. (default=`running`)
If the value is `stopped` the bot has to be started with `/start` first.

### Understand forcebuy_enable

The `forcebuy_enable` configuration parameter enables the usage of forcebuy commands via Telegram.
This is disabled for security reasons by default, and will show a warning message on startup if enabled.
For example, you can send `/forcebuy ETH/BTC` Telegram command when this feature if enabled to the bot,
who then buys the pair and holds it until a regular sell-signal (ROI, stoploss, /forcesell) appears.

This can be dangerous with some strategies, so use with care.

See [the telegram documentation](telegram-usage.md) for details on usage.

### Understand process_throttle_secs

The `process_throttle_secs` configuration parameter is an optional field that defines in seconds how long the bot should wait
before asking the strategy if we should buy or a sell an asset. After each wait period, the strategy is asked again for
every opened trade wether or not we should sell, and for all the remaining pairs (either the dynamic list of pairs or
the static list of pairs) if we should buy.

### Understand ask_last_balance

The `ask_last_balance` configuration parameter sets the bidding price. Value `0.0` will use `ask` price, `1.0` will
use the `last` price and values between those interpolate between ask and last
price. Using `ask` price will guarantee quick success in bid, but bot will also
end up paying more then would probably have been necessary.

### Understand order_types

The `order_types` configuration parameter contains a dict mapping order-types to
market-types as well as stoploss on or off exchange type and stoploss on exchange
update interval in seconds. This allows to buy using limit orders, sell using
limit-orders, and create stoploss orders using market. It also allows to set the
stoploss "on exchange" which means stoploss order would be placed immediately once
the buy order is fulfilled. In case stoploss on exchange and `trailing_stop` are
both set, then the bot will use `stoploss_on_exchange_interval` to check it periodically
and update it if necessary (e.x. in case of trailing stoploss).
This can be set in the configuration file or in the strategy.
Values set in the configuration file  overwrites values set in the strategy.

If this is configured, all 4 values (`buy`, `sell`, `stoploss` and
`stoploss_on_exchange`) need to be present, otherwise the bot will warn about it and fail to start.
The below is the default which is used if this is not configured in either strategy or configuration file.

Syntax for Strategy:

```python
order_types = {
    "buy": "limit",
    "sell": "limit",
    "stoploss": "market",
    "stoploss_on_exchange": False,
    "stoploss_on_exchange_interval": 60
}
```

Configuration:

```json
"order_types": {
    "buy": "limit",
    "sell": "limit",
    "stoploss": "market",
    "stoploss_on_exchange": false,
    "stoploss_on_exchange_interval": 60
}
```

!!! Note
    Not all exchanges support "market" orders.
    The following message will be shown if your exchange does not support market orders:
    `"Exchange <yourexchange>  does not support market orders."`

!!! Note
    Stoploss on exchange interval is not mandatory. Do not change its value if you are
    unsure of what you are doing. For more information about how stoploss works please
    read [the stoploss documentation](stoploss.md).

!!! Note
    In case of stoploss on exchange if the stoploss is cancelled manually then
    the bot would recreate one.

### Understand order_time_in_force

The `order_time_in_force` configuration parameter defines the policy by which the order
is executed on the exchange. Three commonly used time in force are:

**GTC (Good Till Canceled):**

This is most of the time the default time in force. It means the order will remain
on exchange till it is canceled by user. It can be fully or partially fulfilled.
If partially fulfilled, the remaining will stay on the exchange till cancelled.

**FOK (Full Or Kill):**

It means if the order is not executed immediately AND fully then it is canceled by the exchange.

**IOC (Immediate Or Canceled):**

It is the same as FOK (above) except it can be partially fulfilled. The remaining part
is automatically cancelled by the exchange.

The `order_time_in_force` parameter contains a dict with buy and sell time in force policy values.
This can be set in the configuration file or in the strategy.
Values set in the configuration file overwrites values set in the strategy.

The possible values are: `gtc` (default), `fok` or `ioc`.

``` python
"order_time_in_force": {
    "buy": "gtc",
    "sell": "gtc"
},
```

!!! Warning
    This is an ongoing work. For now it is supported only for binance and only for buy orders.
    Please don't change the default value unless you know what you are doing.

### Exchange configuration

Freqtrade is based on [CCXT library](https://github.com/ccxt/ccxt) that supports over 100 cryptocurrency
exchange markets and trading APIs. The complete up-to-date list can be found in the
[CCXT repo homepage](https://github.com/ccxt/ccxt/tree/master/python). However, the bot was tested
with only Bittrex and Binance.

The bot was tested with the following exchanges:

- [Bittrex](https://bittrex.com/): "bittrex"
- [Binance](https://www.binance.com/): "binance"

Feel free to test other exchanges and submit your PR to improve the bot.

#### Sample exchange configuration

A exchange configuration for "binance" would look as follows:

```json
"exchange": {
    "name": "binance",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "ccxt_config": {"enableRateLimit": true},
    "ccxt_async_config": {
        "enableRateLimit": true,
        "rateLimit": 200
    },
```

This configuration enables binance, as well as rate limiting to avoid bans from the exchange.
`"rateLimit": 200` defines a wait-event of 0.2s between each call. This can also be completely disabled by setting `"enableRateLimit"` to false.

!!! Note
    Optimal settings for rate limiting depend on the exchange and the size of the whitelist, so an ideal parameter will vary on many other settings.
    We try to provide sensible defaults per exchange where possible, if you encounter bans please make sure that `"enableRateLimit"` is enabled and increase the `"rateLimit"` parameter step by step. 


### What values can be used for fiat_display_currency?

The `fiat_display_currency` configuration parameter sets the base currency to use for the
conversion from coin to fiat in the bot Telegram reports.

The valid values are:

```json
"AUD", "BRL", "CAD", "CHF", "CLP", "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HUF", "IDR", "ILS", "INR", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PKR", "PLN", "RUB", "SEK", "SGD", "THB", "TRY", "TWD", "ZAR", "USD"
```

In addition to fiat currencies, a range of cryto currencies are supported.

The valid values are:

```json
"BTC", "ETH", "XRP", "LTC", "BCH", "USDT"
```

## Switch to Dry-run mode

We recommend starting the bot in the Dry-run mode to see how your bot will
behave and what is the performance of your strategy. In the Dry-run mode the
bot does not engage your money. It only runs a live simulation without
creating trades on the exchange.

1. Edit your `config.json` configuration file.
2. Switch `dry-run` to `true` and specify `db_url` for a persistence database.

```json
"dry_run": true,
"db_url": "sqlite:///tradesv3.dryrun.sqlite",
```

3. Remove your Exchange API key and secrete (change them by empty values or fake credentials):

```json
"exchange": {
        "name": "bittrex",
        "key": "key",
        "secret": "secret",
        ...
}
```

Once you will be happy with your bot performance running in the Dry-run mode,
you can switch it to production mode.

### Dynamic Pairlists

Dynamic pairlists select pairs for you based on the logic configured.
The bot runs against all pairs (with that stake) on the exchange, and a number of assets
(`number_assets`) is selected based on the selected criteria.

By default, the `StaticPairList` method is used.
The Pairlist method is configured as `pair_whitelist` parameter under the `exchange`
section of the configuration.

**Available Pairlist methods:**

* `StaticPairList`
  * It uses configuration from `exchange.pair_whitelist` and `exchange.pair_blacklist`.
* `VolumePairList`
  * Formerly available as `--dynamic-whitelist [<number_assets>]`. This command line
option is deprecated and should no longer be used.
  * It selects `number_assets` top pairs based on `sort_key`, which can be one of
`askVolume`, `bidVolume` and `quoteVolume`, defaults to `quoteVolume`.
  * There is a possibility to filter low-value coins that would not allow setting a stop loss
(set `precision_filter` parameter to `true` for this).

Example:

```json
"pairlist": {
        "method": "VolumePairList",
        "config": {
            "number_assets": 20,
            "sort_key": "quoteVolume",
            "precision_filter": false
        }
    },
```

## Switch to production mode

In production mode, the bot will engage your money. Be careful, since a wrong
strategy can lose all your money. Be aware of what you are doing when
you run it in production mode.

### To switch your bot in production mode

**Edit your `config.json`  file.**

**Switch dry-run to false and don't forget to adapt your database URL if set:**

```json
"dry_run": false,
```

**Insert your Exchange API key (change them by fake api keys):**

```json
"exchange": {
        "name": "bittrex",
        "key": "af8ddd35195e9dc500b9a6f799f6f5c93d89193b",
        "secret": "08a9dc6db3d7b53e1acebd9275677f4b0a04f1a5",
        ...
}

```
!!! Note
    If you have an exchange API key yet, [see our tutorial](/pre-requisite).

### Using proxy with FreqTrade

To use a proxy with freqtrade, add the kwarg `"aiohttp_trust_env"=true` to the `"ccxt_async_kwargs"` dict in the exchange section of the configuration.

An example for this can be found in `config_full.json.example`

``` json
"ccxt_async_config": {
    "aiohttp_trust_env": true
}
```

Then, export your proxy settings using the variables `"HTTP_PROXY"` and `"HTTPS_PROXY"` set to the appropriate values

``` bash
export HTTP_PROXY="http://addr:port"
export HTTPS_PROXY="http://addr:port"
freqtrade
```


### Embedding Strategies

FreqTrade provides you with with an easy way to embed the strategy into your configuration file.
This is done by utilizing BASE64 encoding and providing this string at the strategy configuration field,
in your chosen config file.

#### Encoding a string as BASE64

This is a quick example, how to generate the BASE64 string in python

```python
from base64 import urlsafe_b64encode

with open(file, 'r') as f:
    content = f.read()
content = urlsafe_b64encode(content.encode('utf-8'))
```

The variable 'content', will contain the strategy file in a BASE64 encoded form. Which can now be set in your configurations file as following

```json
"strategy": "NameOfStrategy:BASE64String"
```

Please ensure that 'NameOfStrategy' is identical to the strategy name!

## Next step

Now you have configured your config.json, the next step is to [start your bot](bot-usage.md).
