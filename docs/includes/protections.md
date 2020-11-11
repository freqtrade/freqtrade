## Protections

Protections will protect your strategy from unexpected events and market conditions.

### Available Protection Handlers

* [`StoplossGuard`](#stoploss-guard) (default, if not configured differently)

#### Stoploss Guard

`StoplossGuard` selects all trades within a `lookback_period` (in minutes), and determines if the amount of trades that resulted in stoploss are above `trade_limit` - in which case it will stop trading until this condition is no longer true.

```json
"protections": [{
        "method": "StoplossGuard",
        "lookback_period": 60,
        "trade_limit": 4
}],
```

!!! Note
    `StoplossGuard` considers all trades with the results `"stop_loss"` and `"trailing_stop_loss"` if the result was negative.

#### Low Profit Pairs

`LowProfitpairs` uses all trades for a pair within a `lookback_period` (in minutes) to determine the overall profit ratio.
If that ratio is below `required_profit`, that pair will be locked for `stop_duration` (in minutes).

```json
"protections": [{
        "method": "LowProfitpairs",
        "lookback_period": 60,
        "trade_limit": 4,
        "stop_duration": 60,
        "required_profit": 0.02
}],
```

### Full example of Protections

The below example stops trading if more than 4 stoploss occur within a 1 hour (60 minute) limit.

```json
"protections": [
    {
        "method": "StoplossGuard",
        "lookback_period": 60,
        "trade_limit": 4
    }
    ],
```
