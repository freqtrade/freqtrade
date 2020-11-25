## Protections

Protections will protect your strategy from unexpected events and market conditions.

!!! Note
    Not all Protections will work for all strategies, and parameters will need to be tuned for your strategy.

!!! Tip
    Each Protection can be configured multiple times with different parameters, to allow different levels of protection (short-term / long-term).

### Available Protection Handlers

* [`StoplossGuard`](#stoploss-guard) Stop trading if a certain amount of stoploss occurred within a certain time window.
* [`LowProfitPairs`](#low-profit-pairs) Lock pairs with low profits
* [`CooldownPeriod`](#cooldown-period) Don't enter a trade right after selling a trade.

#### Stoploss Guard

`StoplossGuard` selects all trades within a `lookback_period` (in minutes), and determines if the amount of trades that resulted in stoploss are above `trade_limit` - in which case trading will stop for `stop_duration`.
This applies across all pairs, unless `only_per_pair` is set to true, which will then only look at one pair at a time.

```json
"protections": [
    {
        "method": "StoplossGuard",
        "lookback_period": 60,
        "trade_limit": 4,
        "stop_duration": 60,
        "only_per_pair": false
    }
],
```

!!! Note
    `StoplossGuard` considers all trades with the results `"stop_loss"` and `"trailing_stop_loss"` if the result was negative.
    `trade_limit` and `lookback_period` will need to be tuned for your strategy.

#### Low Profit Pairs

`LowProfitPairs` uses all trades for a pair within a `lookback_period` (in minutes) to determine the overall profit ratio.
If that ratio is below `required_profit`, that pair will be locked for `stop_duration` (in minutes).

```json
"protections": [
    {
        "method": "LowProfitPairs",
        "lookback_period": 60,
        "trade_limit": 4,
        "stop_duration": 60,
        "required_profit": 0.02
    }
],
```

#### Cooldown Period

`CooldownPeriod` locks a pair for `stop_duration` (in minutes) after selling, avoiding a re-entry for this pair for `stop_duration` minutes.

```json
"protections": [
    {
        "method": "CooldownPeriod",
        "stop_duration": 60
    }
],
```

!!! Note:
    This Protection applies only at pair-level, and will never lock all pairs globally.

### Full example of Protections

All protections can be combined at will, also with different parameters, creating a increasing wall for under-performing pairs.
All protections are evaluated in the sequence they are defined.

The below example:

* stops trading if more than 4 stoploss occur for all pairs within a 1 hour (60 minute) limit (`StoplossGuard`).
* Locks each pair after selling for an additional 10 minutes (`CooldownPeriod`), giving other pairs a chance to get filled.
* Locks all pairs that had 4 Trades within the last 6 hours with a combined profit ratio of below 0.02 (<2%). (`LowProfitPairs`)
* Locks all pairs for 120 minutes that had a profit of below 0.01 (<1%) within the last 24h (`60 * 24 = 1440`), a minimum of 7 trades

```json
"protections": [
    {
        "method": "CooldownPeriod",
        "stop_duration": 10
    },
    {
        "method": "StoplossGuard",
        "lookback_period": 60,
        "trade_limit": 4,
        "stop_duration": 60
    },
    {
        "method": "LowProfitPairs",
        "lookback_period": 360,
        "trade_limit": 4,
        "stop_duration": 60,
        "required_profit": 0.02
    },
        {
        "method": "LowProfitPairs",
        "lookback_period": 1440,
        "trade_limit": 7,
        "stop_duration": 120,
        "required_profit": 0.01
    }
    ],
```
