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
* [`AgeFilter`](#agefilter)
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

#### Volume Pair List

`VolumePairList` employs sorting/filtering of pairs by their trading volume. It selects `number_assets` top pairs with sorting based on the `sort_key` (which can only be `quoteVolume`).

When used in the chain of Pairlist Handlers in a non-leading position (after StaticPairList and other Pairlist Filters), `VolumePairList` considers outputs of previous Pairlist Handlers, adding its sorting/selection of the pairs by the trading volume.

When used on the leading position of the chain of Pairlist Handlers, it does not consider `pair_whitelist` configuration setting, but selects the top assets from all available markets (with matching stake-currency) on the exchange.

The `refresh_period` setting allows to define the period (in seconds), at which the pairlist will be refreshed. Defaults to 1800s (30 minutes).
The pairlist cache (`refresh_period`) on `VolumePairList` is only applicable to generating pairlists.
Filtering instances (not the first position in the list) will not apply any cache and will always use up-to-date data.

`VolumePairList` is based on the ticker data from exchange, as reported by the ccxt library:

* The `quoteVolume` is the amount of quote (stake) currency traded (bought or sold) in last 24 hours.

```json
"pairlists": [{
        "method": "VolumePairList",
        "number_assets": 20,
        "sort_key": "quoteVolume",
        "refresh_period": 1800
}],
```

!!! Note
    `VolumePairList` does not support backtesting mode.

#### AgeFilter

Removes pairs that have been listed on the exchange for less than `min_days_listed` days (defaults to `10`).

When pairs are first listed on an exchange they can suffer huge price drops and volatility
in the first few days while the pair goes through its price-discovery period. Bots can often
be caught out buying before the pair has finished dropping in price.

This filter allows freqtrade to ignore pairs until they have been listed for at least `min_days_listed` days.

#### PerformanceFilter

Sorts pairs by past trade performance, as follows:

1. Positive performance.
2. No closed trades yet.
3. Negative performance.

Trade count is used as a tie breaker.

!!! Note
    `PerformanceFilter` does not support backtesting mode.

#### PrecisionFilter

Filters low-value coins which would not allow setting stoplosses.

#### PriceFilter

The `PriceFilter` allows filtering of pairs by price. Currently the following price filters are supported:

* `min_price`
* `max_price`
* `low_price_ratio`

The `min_price` setting removes pairs where the price is below the specified price. This is useful if you wish to avoid trading very low-priced pairs.
This option is disabled by default, and will only apply if set to > 0.

The `max_price` setting removes pairs where the price is above the specified price. This is useful if you wish to trade only low-priced pairs.
This option is disabled by default, and will only apply if set to > 0.

The `low_price_ratio` setting removes pairs where a raise of 1 price unit (pip) is above the `low_price_ratio` ratio.
This option is disabled by default, and will only apply if set to > 0.

For `PriceFiler` at least one of its `min_price`, `max_price` or `low_price_ratio` settings must be applied.

Calculation example:

Min price precision for SHITCOIN/BTC is 8 decimals. If its price is 0.00000011 - one price step above would be 0.00000012, which is ~9% higher than the previous price value. You may filter out this pair by using PriceFilter with `low_price_ratio` set to 0.09 (9%) or with `min_price` set to 0.00000011, correspondingly.

!!! Warning "Low priced pairs"
    Low priced pairs with high "1 pip movements" are dangerous since they are often illiquid and it may also be impossible to place the desired stoploss, which can often result in high losses since price needs to be rounded to the next tradable price - so instead of having a stoploss of -5%, you could end up with a stoploss of -9% simply due to price rounding.

#### ShuffleFilter

Shuffles (randomizes) pairs in the pairlist. It can be used for preventing the bot from trading some of the pairs more frequently then others when you want all pairs be treated with the same priority.

!!! Tip
    You may set the `seed` value for this Pairlist to obtain reproducible results, which can be useful for repeated backtesting sessions. If `seed` is not set, the pairs are shuffled in the non-repeatable random order.

#### SpreadFilter

Removes pairs that have a difference between asks and bids above the specified ratio, `max_spread_ratio` (defaults to `0.005`).

Example:

If `DOGE/BTC` maximum bid is 0.00000026 and minimum ask is 0.00000027, the ratio is calculated as: `1 - bid/ask ~= 0.037` which is `> 0.005` and this pair will be filtered out.

#### RangeStabilityFilter

Removes pairs where the difference between lowest low and highest high over `lookback_days` days is below `min_rate_of_change`. Since this is a filter that requires additional data, the results are cached for `refresh_period`.

In the below example:
If the trading range over the last 10 days is <1%, remove the pair from the whitelist.

```json
"pairlists": [
    {
        "method": "RangeStabilityFilter",
        "lookback_days": 10,
        "min_rate_of_change": 0.01,
        "refresh_period": 1440
    }
]
```

!!! Tip
    This Filter can be used to automatically remove stable coin pairs, which have a very low trading range, and are therefore extremely difficult to trade with profit.

#### VolatilityFilter

Volatility is the degree of historical variation of a pairs over time, is is measured by the standard deviation of logarithmic daily returns. Returns are assumed to be normally distributed, although actual distribution might be different. In a normal distribution, 68% of observations fall within one standard deviation and 95% of observations fall within two standard deviations. Assuming a volatility of 0.05 means that the expected returns for 20 out of 30 days is expected to be less than 5% (one standard deviation). Volatility is a positive ratio of the expected deviation of return and can be greater than 1.00. Please refer to the wikipedia definition of [`volatility`](https://en.wikipedia.org/wiki/Volatility_(finance)).

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

The below example blacklists `BNB/BTC`, uses `VolumePairList` with `20` assets, sorting pairs by `quoteVolume` and applies [`PrecisionFilter`](#precisionfilter) and [`PriceFilter`](#price-filter), filtering all assets where 1 price unit is > 1%. Then the [`SpreadFilter`](#spreadfilter) and [`VolatilityFilter`](#volatilityfilter) is applied and pairs are finally shuffled with the random seed set to some predefined value.

```json
"exchange": {
    "pair_whitelist": [],
    "pair_blacklist": ["BNB/BTC"]
},
"pairlists": [
    {
        "method": "VolumePairList",
        "number_assets": 20,
        "sort_key": "quoteVolume",
    },
    {"method": "AgeFilter", "min_days_listed": 10},
    {"method": "PrecisionFilter"},
    {"method": "PriceFilter", "low_price_ratio": 0.01},
    {"method": "SpreadFilter", "max_spread_ratio": 0.005},
    {
        "method": "RangeStabilityFilter",
        "lookback_days": 10,
        "min_rate_of_change": 0.01,
        "refresh_period": 1440
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
