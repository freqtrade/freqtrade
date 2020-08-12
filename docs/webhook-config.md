# Webhook usage

## Configuration

Enable webhooks by adding a webhook-section to your configuration file, and setting `webhook.enabled` to `true`.

Sample configuration (tested using IFTTT).

```json
  "webhook": {
        "enabled": true,
        "url": "https://maker.ifttt.com/trigger/<YOUREVENT>/with/key/<YOURKEY>/",
        "webhookbuy": {
            "value1": "Buying {pair}",
            "value2": "limit {limit:8f}",
            "value3": "{stake_amount:8f} {stake_currency}"
        },
        "webhookbuycancel": {
            "value1": "Cancelling Open Buy Order for {pair}",
            "value2": "limit {limit:8f}",
            "value3": "{stake_amount:8f} {stake_currency}"
        },
        "webhooksell": {
            "value1": "Selling {pair}",
            "value2": "limit {limit:8f}",
            "value3": "profit: {profit_amount:8f} {stake_currency} ({profit_ratio})"
        },
        "webhooksellcancel": {
            "value1": "Cancelling Open Sell Order for {pair}",
            "value2": "limit {limit:8f}",
            "value3": "profit: {profit_amount:8f} {stake_currency} ({profit_ratio})"
        },
        "webhookstatus": {
            "value1": "Status: {status}",
            "value2": "",
            "value3": ""
        }
    },
```

The url in `webhook.url` should point to the correct url for your webhook. If you're using [IFTTT](https://ifttt.com) (as shown in the sample above) please insert our event and key to the url.

Different payloads can be configured for different events. Not all fields are necessary, but you should configure at least one of the dicts, otherwise the webhook will never be called.

### Webhookbuy

The fields in `webhook.webhookbuy` are filled when the bot executes a buy. Parameters are filled using string.format.
Possible parameters are:

* `trade_id`
* `exchange`
* `pair`
* `limit`
* `amount`
* `open_date`
* `stake_amount`
* `stake_currency`
* `fiat_currency`
* `order_type`
* `current_rate`

### Webhookbuycancel

The fields in `webhook.webhookbuycancel` are filled when the bot cancels a buy order. Parameters are filled using string.format.
Possible parameters are:

* `trade_id`
* `exchange`
* `pair`
* `limit`
* `amount`
* `open_date`
* `stake_amount`
* `stake_currency`
* `fiat_currency`
* `order_type`
* `current_rate`

### Webhooksell

The fields in `webhook.webhooksell` are filled when the bot sells a trade. Parameters are filled using string.format.
Possible parameters are:

* `trade_id`
* `exchange`
* `pair`
* `gain`
* `limit`
* `amount`
* `open_rate`
* `current_rate`
* `profit_amount`
* `profit_ratio`
* `stake_currency`
* `fiat_currency`
* `sell_reason`
* `order_type`
* `open_date`
* `close_date`

### Webhooksellcancel

The fields in `webhook.webhooksellcancel` are filled when the bot cancels a sell order. Parameters are filled using string.format.
Possible parameters are:

* `trade_id`
* `exchange`
* `pair`
* `gain`
* `limit`
* `amount`
* `open_rate`
* `current_rate`
* `profit_amount`
* `profit_ratio`
* `stake_currency`
* `fiat_currency`
* `sell_reason`
* `order_type`
* `open_date`
* `close_date`

### Webhookstatus

The fields in `webhook.webhookstatus` are used for regular status messages (Started / Stopped / ...). Parameters are filled using string.format.

The only possible value here is `{status}`.
