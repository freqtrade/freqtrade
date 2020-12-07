## Protections

!!! Warning "Beta feature"
    This feature is still in it's testing phase. Should you notice something you think is wrong please let us know via Discord, Slack or via Issue.

Protections will protect your strategy from unexpected events and market conditions by temporarily stop trading for either one pair, or for all pairs.
All protection end times are rounded up to the next candle to avoid sudden, unexpected intra-candle buys.

!!! Note
    Not all Protections will work for all strategies, and parameters will need to be tuned for your strategy.

!!! Tip
    Each Protection can be configured multiple times with different parameters, to allow different levels of protection (short-term / long-term).

### Available Protections

* [`StoplossGuard`](#stoploss-guard) Stop trading if a certain amount of stoploss occurred within a certain time window.
* [`MaxDrawdown`](#maxdrawdown) Stop trading if max-drawdown is reached.
* [`LowProfitPairs`](#low-profit-pairs) Lock pairs with low profits
* [`CooldownPeriod`](#cooldown-period) Don't enter a trade right after selling a trade.

### Common settings to all Protections

* `stop_duration` (minutes) - how long should protections be locked.
* `lookback_period` (minutes) - Only trades that completed after `current_time - lookback_period` will be considered (may be ignored by some Protections).

#### Stoploss Guard

`StoplossGuard` selects all trades within `lookback_period` (in minutes), and determines if the amount of trades that resulted in stoploss are above `trade_limit` - in which case trading will stop for `stop_duration`.
This applies across all pairs, unless `only_per_pair` is set to true, which will then only look at one pair at a time.

The below example stops trading for all pairs for 2 hours (120min) after the last trade if the bot hit stoploss 4 times within the last 24h.

```json
"protections": [
    {
        "method": "StoplossGuard",
        "lookback_period": 1440,
        "trade_limit": 4,
        "stop_duration": 120,
        "only_per_pair": false
    }
],
```

!!! Note
    `StoplossGuard` considers all trades with the results `"stop_loss"` and `"trailing_stop_loss"` if the resulting profit was negative.
    `trade_limit` and `lookback_period` will need to be tuned for your strategy.

#### MaxDrawdown

`MaxDrawdown` uses all trades within `lookback_period` (in minutes) to determine the maximum drawdown. If the drawdown is below `max_allowed_drawdown`, trading will stop for `stop_duration` (in minutes) after the last trade - assuming that the bot needs some time to let markets recover.

The below sample stops trading for 12 hours (720min) if max-drawdown is > 20% considering all trades within the last 2 days (2880min).

```json
"protections": [
      {
        "method": "MaxDrawdown",
        "lookback_period": 2880,
        "trade_limit": 20,
        "stop_duration": 720,
        "max_allowed_drawdown": 0.2
      },
],

```

#### Low Profit Pairs

`LowProfitPairs` uses all trades for a pair within `lookback_period` (in minutes) to determine the overall profit ratio.
If that ratio is below `required_profit`, that pair will be locked for `stop_duration` (in minutes).

The below example will stop trading a pair for 60 minutes if the pair does not have a required profit of 2% (and a minimum of 2 trades) within the last 6 hours (360min).

```json
"protections": [
    {
        "method": "LowProfitPairs",
        "lookback_period": 360,
        "trade_limit": 2,
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

!!! Note
    This Protection applies only at pair-level, and will never lock all pairs globally.
    This Protection does not consider `lookback_period` as it only looks at the latest trade.

### Full example of Protections

All protections can be combined at will, also with different parameters, creating a increasing wall for under-performing pairs.
All protections are evaluated in the sequence they are defined.

The below example:

* Locks each pair after selling for an additional 10 minutes (`CooldownPeriod`), giving other pairs a chance to get filled.
* Stops trading if the last 2 days had 20 trades, which caused a max-drawdown of more than 20%. (`MaxDrawdown`).
* Stops trading if more than 4 stoploss occur for all pairs within a 1 day (1440min) limit (`StoplossGuard`).
* Locks all pairs that had 4 Trades within the last 6 hours (`60 * 6 = 360`) with a combined profit ratio of below 0.02 (<2%) (`LowProfitPairs`).
* Locks all pairs for 120 minutes that had a profit of below 0.01 (<1%) within the last 24h (`60 * 24 = 1440`), a minimum of 4 trades.

```json
"protections": [
    {
        "method": "CooldownPeriod",
        "stop_duration": 10
    },
    {
        "method": "MaxDrawdown",
        "lookback_period": 2880,
        "trade_limit": 20,
        "stop_duration": 720,
        "max_allowed_drawdown": 0.2
    },
    {
        "method": "StoplossGuard",
        "lookback_period": 1440,
        "trade_limit": 4,
        "stop_duration": 120,
        "only_per_pair": false
    },
    {
        "method": "LowProfitPairs",
        "lookback_period": 360,
        "trade_limit": 2,
        "stop_duration": 60,
        "required_profit": 0.02
    },
    {
        "method": "LowProfitPairs",
        "lookback_period": 1440,
        "trade_limit": 4,
        "stop_duration": 120,
        "required_profit": 0.01
    }
    ],
```
