## Pairlists and Pairlist Handlers

Pairlist Handlers define the list of pairs (pairlist) that the bot should trade. They are configured in the `pairlists` section of the configuration settings.

In your configuration, you can use Static Pairlist (defined by the [`StaticPairList`](#static-pair-list) Pairlist Handler) and Dynamic Pairlist (defined by the [`VolumePairList`](#volume-pair-list) Pairlist Handler).

Additionally, [`AgeFilter`](#agefilter), [`PrecisionFilter`](#precisionfilter), [`PriceFilter`](#pricefilter), [`ShuffleFilter`](#shufflefilter), [`SpreadFilter`](#spreadfilter) and [`VolatilityFilter`](#volatilityfilter) act as Pairlist Filters, removing certain pairs and/or moving their positions in the pairlist.

If multiple Pairlist Handlers are used, they are chained and a combination of all Pairlist Handlers forms the resulting pairlist the bot uses for trading and backtesting. Pairlist Handlers are executed in the sequence they are configured. You should always configure either `StaticPairList` or `VolumePairList` as the starting Pairlist Handler.

Inactive markets are always removed from the resulting pairlist. Explicitly blacklisted pairs (those in the `pair_blacklist` configuration setting) are also always removed from the resulting pairlist.

### Pair blacklist

The pair blacklist (configured via `exchange.pair_blacklist` in the configuration) disallows certain pairs from trading.
This can be as simple as excluding `DOGE/BTC` - which will remove exactly this pair.

The pair-blacklist does also support wildcards (in regex-style) - so `BNB/.*` will exclude ALL pairs that start with BNB.
You may also use something like `.*DOWN/BTC` or `.*UP/BTC` to exclude leveraged tokens (check Pair naming conventions for your exchange!)

### Available Pairlist Handlers

* [`StaticPairList`](#static-pair-list) (default, if not configured differently)
* [`VolumePairList`](#volume-pair-list)
* [`ProducerPairList`](#producerpairlist)
* [`RemotePairList`](#remotepairlist)
* [`AgeFilter`](#agefilter)
* [`FullTradesFilter`](#fulltradesfilter)
* [`OffsetFilter`](#offsetfilter)
* [`PerformanceFilter`](#performancefilter)
* [`PrecisionFilter`](#precisionfilter)
* [`PriceFilter`](#pricefilter)
* [`ShuffleFilter`](#shufflefilter)
* [`SpreadFilter`](#spreadfilter)
* [`RangeStabilityFilter`](#rangestabilityfilter)
* [`VolatilityFilter`](#volatilityfilter)

!!! Tip "Testing pairlists"
    Pairlist configurations can be quite tricky to get right. Best use the [`test-pairlist`](utils.md#test-pairlist) utility sub-command to test your configuration quickly.

#### Static Pair List

By default, the `StaticPairList` method is used, which uses a statically defined pair whitelist from the configuration. The pairlist also supports wildcards (in regex-style) - so `.*/BTC` will include all pairs with BTC as a stake.

It uses configuration from `exchange.pair_whitelist` and `exchange.pair_blacklist`.

```json
"pairlists": [
    {"method": "StaticPairList"}
],
```

By default, only currently enabled pairs are allowed.
To skip pair validation against active markets, set `"allow_inactive": true` within the `StaticPairList` configuration.
This can be useful for backtesting expired pairs (like quarterly spot-markets).
This option must be configured along with `exchange.skip_pair_validation` in the exchange configuration.

When used in a "follow-up" position (e.g. after VolumePairlist), all pairs in `'pair_whitelist'` will be added to the end of the pairlist.

#### Volume Pair List

`VolumePairList` employs sorting/filtering of pairs by their trading volume. It selects `number_assets` top pairs with sorting based on the `sort_key` (which can only be `quoteVolume`).

When used in the chain of Pairlist Handlers in a non-leading position (after StaticPairList and other Pairlist Filters), `VolumePairList` considers outputs of previous Pairlist Handlers, adding its sorting/selection of the pairs by the trading volume.

When used in the leading position of the chain of Pairlist Handlers, the `pair_whitelist` configuration setting is ignored. Instead, `VolumePairList` selects the top assets from all available markets with matching stake-currency on the exchange.

The `refresh_period` setting allows to define the period (in seconds), at which the pairlist will be refreshed. Defaults to 1800s (30 minutes).
The pairlist cache (`refresh_period`) on `VolumePairList` is only applicable to generating pairlists.
Filtering instances (not the first position in the list) will not apply any cache and will always use up-to-date data.

`VolumePairList` is per default based on the ticker data from exchange, as reported by the ccxt library:

* The `quoteVolume` is the amount of quote (stake) currency traded (bought or sold) in last 24 hours.

```json
"pairlists": [
    {
        "method": "VolumePairList",
        "number_assets": 20,
        "sort_key": "quoteVolume",
        "min_value": 0,
        "refresh_period": 1800
    }
],
```

You can define a minimum volume with `min_value` - which will filter out pairs with a volume lower than the specified value in the specified timerange.

##### VolumePairList Advanced mode

`VolumePairList` can also operate in an advanced mode to build volume over a given timerange of specified candle size. It utilizes exchange historical candle data, builds a typical price (calculated by (open+high+low)/3) and multiplies the typical price with every candle's volume. The sum is the `quoteVolume` over the given range. This allows different scenarios, for a  more smoothened volume, when using longer ranges with larger candle sizes, or the opposite when using a short range with small candles.

For convenience `lookback_days` can be specified, which will imply that 1d candles will be used for the lookback. In the example below the pairlist would be created based on the last 7 days:

```json
"pairlists": [
    {
        "method": "VolumePairList",
        "number_assets": 20,
        "sort_key": "quoteVolume",
        "min_value": 0,
        "refresh_period": 86400,
        "lookback_days": 7
    }
],
```

!!! Warning "Range look back and refresh period"
    When used in conjunction with `lookback_days` and `lookback_timeframe` the `refresh_period` can not be smaller than the candle size in seconds. As this will result in unnecessary requests to the exchanges API.

!!! Warning "Performance implications when using lookback range"
    If used in first position in combination with lookback, the computation of the range based volume can be time and resource consuming, as it downloads candles for all tradable pairs. Hence it's highly advised to use the standard approach with `VolumeFilter` to narrow the pairlist down for further range volume calculation.

??? Tip "Unsupported exchanges"
    On some exchanges (like Gemini), regular VolumePairList does not work as the api does not natively provide 24h volume. This can be worked around by using candle data to build the volume.
    To roughly simulate 24h volume, you can use the following configuration.
    Please note that These pairlists will only refresh once per day.

    ```json
    "pairlists": [
        {
            "method": "VolumePairList",
            "number_assets": 20,
            "sort_key": "quoteVolume",
            "min_value": 0,
            "refresh_period": 86400,
            "lookback_days": 1
        }
    ],
    ```

More sophisticated approach can be used, by using `lookback_timeframe` for candle size and `lookback_period` which specifies the amount of candles. This example will build the volume pairs based on a rolling period of 3 days of 1h candles:

```json
"pairlists": [
    {
        "method": "VolumePairList",
        "number_assets": 20,
        "sort_key": "quoteVolume",
        "min_value": 0,
        "refresh_period": 3600,
        "lookback_timeframe": "1h",
        "lookback_period": 72
    }
],
```

!!! Note
    `VolumePairList` does not support backtesting mode.

#### ProducerPairList

With `ProducerPairList`, you can reuse the pairlist from a [Producer](producer-consumer.md) without explicitly defining the pairlist on each consumer.

[Consumer mode](producer-consumer.md) is required for this pairlist to work.

The pairlist will perform a check on active pairs against the current exchange configuration to avoid attempting to trade on invalid markets.

You can limit the length of the pairlist with the optional parameter `number_assets`. Using `"number_assets"=0` or omitting this key will result in the reuse of all producer pairs valid for the current setup.

```json
"pairlists": [
    {
        "method": "ProducerPairList",
        "number_assets": 5,
        "producer_name": "default",
    }
],
```


!!! Tip "Combining pairlists"
    This pairlist can be combined with all other pairlists and filters for further pairlist reduction, and can also act as an "additional" pairlist, on top of already defined pairs.
    `ProducerPairList` can also be used multiple times in sequence, combining the pairs from multiple producers.
    Obviously in complex such configurations, the Producer may not provide data for all pairs, so the strategy must be fit for this.

#### RemotePairList

It allows the user to fetch a pairlist from a remote server or a locally stored json file within the freqtrade directory, enabling dynamic updates and customization of the trading pairlist.

The RemotePairList is defined in the pairlists section of the configuration settings. It uses the following configuration options:

```json
"pairlists": [
    {
        "method": "RemotePairList",
        "mode": "whitelist",
        "processing_mode": "filter",
        "pairlist_url": "https://example.com/pairlist",
        "number_assets": 10,
        "refresh_period": 1800,
        "keep_pairlist_on_failure": true,
        "read_timeout": 60,
        "bearer_token": "my-bearer-token"
    }
]
```

The optional `mode` option specifies if the pairlist should be used as a `blacklist` or as a `whitelist`. The default value is "whitelist".

The optional `processing_mode` option in the RemotePairList configuration determines how the retrieved pairlist is processed. It can have two values: "filter" or "append".

In "filter" mode, the retrieved pairlist is used as a filter. Only the pairs present in both the original pairlist and the retrieved pairlist are included in the final pairlist. Other pairs are filtered out.

In "append" mode, the retrieved pairlist is added to the original pairlist. All pairs from both lists are included in the final pairlist without any filtering.

The `pairlist_url` option specifies the URL of the remote server where the pairlist is located, or the path to a local file (if file:/// is prepended). This allows the user to use either a remote server or a local file as the source for the pairlist.

The user is responsible for providing a server or local file that returns a JSON object with the following structure:

```json
{
    "pairs": ["XRP/USDT", "ETH/USDT", "LTC/USDT"],
    "refresh_period": 1800
}
```

The `pairs` property should contain a list of strings with the trading pairs to be used by the bot. The `refresh_period` property is optional and specifies the number of seconds that the pairlist should be cached before being refreshed.

The optional `keep_pairlist_on_failure` specifies whether the previous received pairlist should be used if the remote server is not reachable or returns an error. The default value is true.

The optional `read_timeout` specifies the maximum amount of time (in seconds) to wait for a response from the remote source, The default value is 60.

The optional `bearer_token` will be included in the requests Authorization Header.

!!! Note
    In case of a server error the last received pairlist will be kept if `keep_pairlist_on_failure` is set to true, when set to false a empty pairlist is returned.

#### AgeFilter

Removes pairs that have been listed on the exchange for less than `min_days_listed` days (defaults to `10`) or more than `max_days_listed` days (defaults `None` mean infinity).

When pairs are first listed on an exchange they can suffer huge price drops and volatility
in the first few days while the pair goes through its price-discovery period. Bots can often
be caught out buying before the pair has finished dropping in price.

This filter allows freqtrade to ignore pairs until they have been listed for at least `min_days_listed` days and listed before `max_days_listed`.

#### FullTradesFilter

Shrink whitelist to consist only in-trade pairs when the trade slots are full (when `max_open_trades` isn't being set to `-1` in the config).

When the trade slots are full, there is no need to calculate indicators of the rest of the pairs (except informative pairs) since no new trade can be opened. By shrinking the whitelist to just the in-trade pairs, you can improve calculation speeds and reduce CPU usage. When a trade slot is free (either a trade is closed or `max_open_trades` value in config is increased), then the whitelist will return to normal state.

When multiple pairlist filters are being used, it's recommended to put this filter at second position directly below the primary pairlist, so when the trade slots are full, the bot doesn't have to download data for the rest of the filters.

!!! Warning "Backtesting"
    `FullTradesFilter` does not support backtesting mode.

#### OffsetFilter

Offsets an incoming pairlist by a given `offset` value.

As an example it can be used in conjunction with `VolumeFilter` to remove the top X volume pairs. Or to split a larger pairlist on two bot instances.

Example to remove the first 10 pairs from the pairlist, and takes the next 20 (taking items 10-30 of the initial list):

```json
"pairlists": [
    // ...
    {
        "method": "OffsetFilter",
        "offset": 10,
        "number_assets": 20
    }
],
```

!!! Warning
    When `OffsetFilter` is used to split a larger pairlist among multiple bots in combination with `VolumeFilter` 
    it can not be guaranteed that pairs won't overlap due to slightly different refresh intervals for the
    `VolumeFilter`.

!!! Note
    An offset larger than the total length of the incoming pairlist will result in an empty pairlist.

#### PerformanceFilter

Sorts pairs by past trade performance, as follows:

1. Positive performance.
2. No closed trades yet.
3. Negative performance.

Trade count is used as a tie breaker.

You can use the `minutes` parameter to only consider performance of the past X minutes (rolling window).
Not defining this parameter (or setting it to 0) will use all-time performance.

The optional `min_profit` (as ratio -> a setting of `0.01` corresponds to 1%) parameter defines the minimum profit a pair must have to be considered.
Pairs below this level will be filtered out.
Using this parameter without `minutes` is highly discouraged, as it can lead to an empty pairlist without a way to recover.

```json
"pairlists": [
    // ...
    {
        "method": "PerformanceFilter",
        "minutes": 1440,  // rolling 24h
        "min_profit": 0.01  // minimal profit 1%
    }
],
```

As this Filter uses past performance of the bot, it'll have some startup-period - and should only be used after the bot has a few 100 trades in the database.

!!! Warning "Backtesting"
    `PerformanceFilter` does not support backtesting mode.

#### PrecisionFilter

Filters low-value coins which would not allow setting stoplosses.

!!! Warning "Backtesting"
    `PrecisionFilter` does not support backtesting mode using multiple strategies.

#### PriceFilter

The `PriceFilter` allows filtering of pairs by price. Currently the following price filters are supported:

* `min_price`
* `max_price`
* `max_value`
* `low_price_ratio`

The `min_price` setting removes pairs where the price is below the specified price. This is useful if you wish to avoid trading very low-priced pairs.
This option is disabled by default, and will only apply if set to > 0.

The `max_price` setting removes pairs where the price is above the specified price. This is useful if you wish to trade only low-priced pairs.
This option is disabled by default, and will only apply if set to > 0.

The `max_value` setting removes pairs where the minimum value change is above a specified value.
This is useful when an exchange has unbalanced limits. For example, if step-size = 1 (so you can only buy 1, or 2, or 3, but not 1.1 Coins) - and the price is pretty high (like 20\$) as the coin has risen sharply since the last limit adaption.
As a result of the above, you can only buy for 20\$, or 40\$ - but not for 25\$.
On exchanges that deduct fees from the receiving currency (e.g. binance) - this can result in high value coins / amounts that are unsellable as the amount is slightly below the limit.

The `low_price_ratio` setting removes pairs where a raise of 1 price unit (pip) is above the `low_price_ratio` ratio.
This option is disabled by default, and will only apply if set to > 0.

For `PriceFilter` at least one of its `min_price`, `max_price` or `low_price_ratio` settings must be applied.

Calculation example:

Min price precision for SHITCOIN/BTC is 8 decimals. If its price is 0.00000011 - one price step above would be 0.00000012, which is ~9% higher than the previous price value. You may filter out this pair by using PriceFilter with `low_price_ratio` set to 0.09 (9%) or with `min_price` set to 0.00000011, correspondingly.

!!! Warning "Low priced pairs"
    Low priced pairs with high "1 pip movements" are dangerous since they are often illiquid and it may also be impossible to place the desired stoploss, which can often result in high losses since price needs to be rounded to the next tradable price - so instead of having a stoploss of -5%, you could end up with a stoploss of -9% simply due to price rounding.

#### ShuffleFilter

Shuffles (randomizes) pairs in the pairlist. It can be used for preventing the bot from trading some of the pairs more frequently then others when you want all pairs be treated with the same priority.

By default, ShuffleFilter will shuffle pairs once per candle.
To shuffle on every iteration, set `"shuffle_frequency"` to `"iteration"` instead of  the default of `"candle"`.

``` json
    {
        "method": "ShuffleFilter", 
        "shuffle_frequency": "candle",
        "seed": 42
    }

```

!!! Tip
    You may set the `seed` value for this Pairlist to obtain reproducible results, which can be useful for repeated backtesting sessions. If `seed` is not set, the pairs are shuffled in the non-repeatable random order. ShuffleFilter will automatically detect runmodes and apply the `seed` only for backtesting modes - if a `seed` value is set.

#### SpreadFilter

Removes pairs that have a difference between asks and bids above the specified ratio, `max_spread_ratio` (defaults to `0.005`).

Example:

If `DOGE/BTC` maximum bid is 0.00000026 and minimum ask is 0.00000027, the ratio is calculated as: `1 - bid/ask ~= 0.037` which is `> 0.005` and this pair will be filtered out.

#### RangeStabilityFilter

Removes pairs where the difference between lowest low and highest high over `lookback_days` days is below `min_rate_of_change` or above `max_rate_of_change`. Since this is a filter that requires additional data, the results are cached for `refresh_period`.

In the below example:
If the trading range over the last 10 days is <1% or >99%, remove the pair from the whitelist.

```json
"pairlists": [
    {
        "method": "RangeStabilityFilter",
        "lookback_days": 10,
        "min_rate_of_change": 0.01,
        "max_rate_of_change": 0.99,
        "refresh_period": 86400
    }
]
```

!!! Tip
    This Filter can be used to automatically remove stable coin pairs, which have a very low trading range, and are therefore extremely difficult to trade with profit.
    Additionally, it can also be used to automatically remove pairs with extreme high/low variance over a given amount of time.

#### VolatilityFilter

Volatility is the degree of historical variation of a pairs over time, it is measured by the standard deviation of logarithmic daily returns. Returns are assumed to be normally distributed, although actual distribution might be different. In a normal distribution, 68% of observations fall within one standard deviation and 95% of observations fall within two standard deviations. Assuming a volatility of 0.05 means that the expected returns for 20 out of 30 days is expected to be less than 5% (one standard deviation). Volatility is a positive ratio of the expected deviation of return and can be greater than 1.00. Please refer to the wikipedia definition of [`volatility`](https://en.wikipedia.org/wiki/Volatility_(finance)).

This filter removes pairs if the average volatility over a `lookback_days` days is below `min_volatility` or above `max_volatility`. Since this is a filter that requires additional data, the results are cached for `refresh_period`.

This filter can be used to narrow down your pairs to a certain volatility or avoid very volatile pairs. 

In the below example:
If the volatility over the last 10 days is not in the range of 0.05-0.50, remove the pair from the whitelist. The filter is applied every 24h.

```json
"pairlists": [
    {
        "method": "VolatilityFilter",
        "lookback_days": 10,
        "min_volatility": 0.05,
        "max_volatility": 0.50,
        "refresh_period": 86400
    }
]
```

### Full example of Pairlist Handlers

The below example blacklists `BNB/BTC`, uses `VolumePairList` with `20` assets, sorting pairs by `quoteVolume` and applies [`PrecisionFilter`](#precisionfilter) and [`PriceFilter`](#pricefilter), filtering all assets where 1 price unit is > 1%. Then the [`SpreadFilter`](#spreadfilter) and [`VolatilityFilter`](#volatilityfilter) is applied and pairs are finally shuffled with the random seed set to some predefined value.

```json
"exchange": {
    "pair_whitelist": [],
    "pair_blacklist": ["BNB/BTC"]
},
"pairlists": [
    {
        "method": "VolumePairList",
        "number_assets": 20,
        "sort_key": "quoteVolume"
    },
    {"method": "AgeFilter", "min_days_listed": 10},
    {"method": "PrecisionFilter"},
    {"method": "PriceFilter", "low_price_ratio": 0.01},
    {"method": "SpreadFilter", "max_spread_ratio": 0.005},
    {
        "method": "RangeStabilityFilter",
        "lookback_days": 10,
        "min_rate_of_change": 0.01,
        "refresh_period": 86400
    },
    {
        "method": "VolatilityFilter",
        "lookback_days": 10,
        "min_volatility": 0.05,
        "max_volatility": 0.50,
        "refresh_period": 86400
    },
    {"method": "ShuffleFilter", "seed": 42}
],
```
