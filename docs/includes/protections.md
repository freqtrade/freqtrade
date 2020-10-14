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
