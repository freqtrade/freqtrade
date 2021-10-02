## Protections

!!! Warning "Beta feature"
    This feature is still in it's testing phase. Should you notice something you think is wrong please let us know via Discord or via Github Issue.

Protections will protect your strategy from unexpected events and market conditions by temporarily stop trading for either one pair, or for all pairs.
All protection end times are rounded up to the next candle to avoid sudden, unexpected intra-candle buys.

!!! Note
    Not all Protections will work for all strategies, and parameters will need to be tuned for your strategy to improve performance.  

!!! Tip
    Each Protection can be configured multiple times with different parameters, to allow different levels of protection (short-term / long-term).

!!! Note "Backtesting"
    Protections are supported by backtesting and hyperopt, but must be explicitly enabled by using the `--enable-protections` flag.

!!! Warning "Setting protections from the configuration"
    Setting protections from the configuration via `"protections": [],` key should be considered deprecated and will be removed in a future version.
    It is also no longer guaranteed that your protections apply to the strategy in cases where the strategy defines [protections as property](hyperopt.md#optimizing-protections).

### Available Protections

* [`StoplossGuard`](#stoploss-guard) Stop trading if a certain amount of stoploss occurred within a certain time window.
* [`MaxDrawdown`](#maxdrawdown) Stop trading if max-drawdown is reached.
* [`LowProfitPairs`](#low-profit-pairs) Lock pairs with low profits
* [`CooldownPeriod`](#cooldown-period) Don't enter a trade right after selling a trade.

### Common settings to all Protections

|  Parameter| Description |
|------------|-------------|
| `method` | Protection name to use. <br> **Datatype:** String, selected from [available Protections](#available-protections)
| `stop_duration_candles` | For how many candles should the lock be set? <br> **Datatype:** Positive integer (in candles)
| `stop_duration` | how many minutes should protections be locked. <br>Cannot be used together with `stop_duration_candles`. <br> **Datatype:** Float (in minutes)
| `lookback_period_candles` | Only trades that completed within the last `lookback_period_candles` candles will be considered. This setting may be ignored by some Protections. <br> **Datatype:** Positive integer (in candles).
| `lookback_period` | Only trades that completed after `current_time - lookback_period` will be considered. <br>Cannot be used together with `lookback_period_candles`. <br>This setting may be ignored by some Protections. <br> **Datatype:**  Float (in minutes)
| `trade_limit` | Number of trades required at minimum (not used by all Protections). <br> **Datatype:** Positive integer

!!! Note "Durations"
    Durations (`stop_duration*` and `lookback_period*` can be defined in either minutes or candles).
    For more flexibility when testing different timeframes, all below examples will use the "candle" definition.

#### Stoploss Guard

`StoplossGuard` selects all trades within `lookback_period` in minutes (or in candles when using `lookback_period_candles`).
If `trade_limit` or more trades resulted in stoploss, trading will stop for `stop_duration` in minutes (or in candles when using `stop_duration_candles`).

This applies across all pairs, unless `only_per_pair` is set to true, which will then only look at one pair at a time.

The below example stops trading for all pairs for 4 candles after the last trade if the bot hit stoploss 4 times within the last 24 candles.

``` python
@property
def protections(self):
    return [
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 24,
            "trade_limit": 4,
            "stop_duration_candles": 4,
            "only_per_pair": False
        }
    ]
```

!!! Note
    `StoplossGuard` considers all trades with the results `"stop_loss"`, `"stoploss_on_exchange"` and `"trailing_stop_loss"` if the resulting profit was negative.
    `trade_limit` and `lookback_period` will need to be tuned for your strategy.

#### MaxDrawdown

`MaxDrawdown` uses all trades within `lookback_period` in minutes (or in candles when using `lookback_period_candles`) to determine the maximum drawdown. If the drawdown is below `max_allowed_drawdown`, trading will stop for `stop_duration` in minutes (or in candles when using `stop_duration_candles`) after the last trade - assuming that the bot needs some time to let markets recover.

The below sample stops trading for 12 candles if max-drawdown is > 20% considering all pairs - with a minimum of `trade_limit` trades - within the last 48 candles. If desired, `lookback_period` and/or `stop_duration` can be used.

``` python
@property
def protections(self):
    return  [
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 48,
            "trade_limit": 20,
            "stop_duration_candles": 12,
            "max_allowed_drawdown": 0.2
        },
    ]
```

#### Low Profit Pairs

`LowProfitPairs` uses all trades for a pair within `lookback_period` in minutes (or in candles when using `lookback_period_candles`) to determine the overall profit ratio.
If that ratio is below `required_profit`, that pair will be locked for `stop_duration` in minutes (or in candles when using `stop_duration_candles`).

The below example will stop trading a pair for 60 minutes if the pair does not have a required profit of 2% (and a minimum of 2 trades) within the last 6 candles.

``` python
@property
def protections(self):
    return [
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 6,
            "trade_limit": 2,
            "stop_duration": 60,
            "required_profit": 0.02
        }
    ]
```

#### Cooldown Period

`CooldownPeriod` locks a pair for `stop_duration` in minutes (or in candles when using `stop_duration_candles`) after selling, avoiding a re-entry for this pair for `stop_duration` minutes.

The below example will stop trading a pair for 2 candles after closing a trade, allowing this pair to "cool down".

``` python
@property
def protections(self):
    return  [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2
        }
    ]
```

!!! Note
    This Protection applies only at pair-level, and will never lock all pairs globally.
    This Protection does not consider `lookback_period` as it only looks at the latest trade.

### Full example of Protections

All protections can be combined at will, also with different parameters, creating a increasing wall for under-performing pairs.
All protections are evaluated in the sequence they are defined.

The below example assumes a timeframe of 1 hour:

* Locks each pair after selling for an additional 5 candles (`CooldownPeriod`), giving other pairs a chance to get filled.
* Stops trading for 4 hours (`4 * 1h candles`) if the last 2 days (`48 * 1h candles`) had 20 trades, which caused a max-drawdown of more than 20%. (`MaxDrawdown`).
* Stops trading if more than 4 stoploss occur for all pairs within a 1 day (`24 * 1h candles`) limit (`StoplossGuard`).
* Locks all pairs that had 4 Trades within the last 6 hours (`6 * 1h candles`) with a combined profit ratio of below 0.02 (<2%) (`LowProfitPairs`).
* Locks all pairs for 2 candles that had a profit of below 0.01 (<1%) within the last 24h (`24 * 1h candles`), a minimum of 4 trades.

``` python
from freqtrade.strategy import IStrategy

class AwesomeStrategy(IStrategy)
    timeframe = '1h'
    
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]
    # ...
```
