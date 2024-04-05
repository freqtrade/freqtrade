# Freqtrade basics

This page provides you some basic concepts on how Freqtrade works and operates.

## Freqtrade terminology

* **Strategy**: Your trading strategy, telling the bot what to do.
* **Trade**: Open position.
* **Open Order**: Order which is currently placed on the exchange, and is not yet complete.
* **Pair**: Tradable pair, usually in the format of Base/Quote (e.g. `XRP/USDT` for spot, `XRP/USDT:USDT` for futures).
* **Timeframe**: Candle length to use (e.g. `"5m"`, `"1h"`, ...).
* **Indicators**: Technical indicators (SMA, EMA, RSI, ...).
* **Limit order**: Limit orders which execute at the defined limit price or better.
* **Market order**: Guaranteed to fill, may move price depending on the order size.
* **Current Profit**: Currently pending (unrealized) profit for this trade. This is mainly used throughout the bot and UI.
* **Realized Profit**: Already realized profit. Only relevant in combination with [partial exits](strategy-callbacks.md#adjust-trade-position) - which also explains the calculation logic for this.
* **Total Profit**: Combined realized and unrealized profit. The relative number (%) is calculated against the total investment in this trade.

## Fee handling

All profit calculations of Freqtrade include fees. For Backtesting / Hyperopt / Dry-run modes, the exchange default fee is used (lowest tier on the exchange). For live operations, fees are used as applied by the exchange (this includes BNB rebates etc.).

## Pair naming

Freqtrade follows the [ccxt naming convention](https://docs.ccxt.com/#/README?id=consistency-of-base-and-quote-currencies) for currencies.
Using the wrong naming convention in the wrong market will usually result in the bot not recognizing the pair, usually resulting in errors like "this pair is not available".

### Spot pair naming

For spot pairs, naming will be `base/quote` (e.g. `ETH/USDT`).

### Futures pair naming

For futures pairs, naming will be `base/quote:settle` (e.g. `ETH/USDT:USDT`).

## Bot execution logic

Starting freqtrade in dry-run or live mode (using `freqtrade trade`) will start the bot and start the bot iteration loop.
This will also run the `bot_start()` callback.

By default, the bot loop runs every few seconds (`internals.process_throttle_secs`) and performs the following actions:

* Fetch open trades from persistence.
* Calculate current list of tradable pairs.
* Download OHLCV data for the pairlist including all [informative pairs](strategy-customization.md#get-data-for-non-tradeable-pairs)  
  This step is only executed once per Candle to avoid unnecessary network traffic.
* Call `bot_loop_start()` strategy callback.
* Analyze strategy per pair.
  * Call `populate_indicators()`
  * Call `populate_entry_trend()`
  * Call `populate_exit_trend()`
* Update trades open order state from exchange.
  * Call `order_filled()` strategy callback for filled orders.
  * Check timeouts for open orders.
    * Calls `check_entry_timeout()` strategy callback for open entry orders.
    * Calls `check_exit_timeout()` strategy callback for open exit orders.
    * Calls `adjust_entry_price()` strategy callback for open entry orders.
* Verifies existing positions and eventually places exit orders.
  * Considers stoploss, ROI and exit-signal, `custom_exit()` and `custom_stoploss()`.
  * Determine exit-price based on `exit_pricing` configuration setting or by using the `custom_exit_price()` callback.
  * Before a exit order is placed, `confirm_trade_exit()` strategy callback is called.
* Check position adjustments for open trades if enabled by calling `adjust_trade_position()` and place additional order if required.
* Check if trade-slots are still available (if `max_open_trades` is reached).
* Verifies entry signal trying to enter new positions.
  * Determine entry-price based on `entry_pricing` configuration setting, or by using the `custom_entry_price()` callback.
  * In Margin and Futures mode, `leverage()` strategy callback is called to determine the desired leverage.
  * Determine stake size by calling the `custom_stake_amount()` callback.
  * Before an entry order is placed, `confirm_trade_entry()` strategy callback is called.

This loop will be repeated again and again until the bot is stopped.

## Backtesting / Hyperopt execution logic

[backtesting](backtesting.md) or [hyperopt](hyperopt.md) do only part of the above logic, since most of the trading operations are fully simulated.

* Load historic data for configured pairlist.
* Calls `bot_start()` once.
* Calculate indicators (calls `populate_indicators()` once per pair).
* Calculate entry / exit signals (calls `populate_entry_trend()` and `populate_exit_trend()` once per pair).
* Loops per candle simulating entry and exit points.
  * Calls `bot_loop_start()` strategy callback.
  * Check for Order timeouts, either via the `unfilledtimeout` configuration, or via `check_entry_timeout()` / `check_exit_timeout()` strategy callbacks.
  * Calls `adjust_entry_price()` strategy callback for open entry orders.
  * Check for trade entry signals (`enter_long` / `enter_short` columns).
  * Confirm trade entry / exits (calls `confirm_trade_entry()` and `confirm_trade_exit()` if implemented in the strategy).
  * Call `custom_entry_price()` (if implemented in the strategy) to determine entry price (Prices are moved to be within the opening candle).
  * In Margin and Futures mode, `leverage()` strategy callback is called to determine the desired leverage.
  * Determine stake size by calling the `custom_stake_amount()` callback.
  * Check position adjustments for open trades if enabled and call `adjust_trade_position()` to determine if an additional order is requested.
  * Call `order_filled()` strategy callback for filled entry orders.
  * Call `custom_stoploss()` and `custom_exit()` to find custom exit points.
  * For exits based on exit-signal, custom-exit and partial exits: Call `custom_exit_price()` to determine exit price (Prices are moved to be within the closing candle).
  * Call `order_filled()` strategy callback for filled exit orders.
* Generate backtest report output

!!! Note
    Both Backtesting and Hyperopt include exchange default Fees in the calculation. Custom fees can be passed to backtesting / hyperopt by specifying the `--fee` argument.

!!! Warning "Callback call frequency"
    Backtesting will call each callback at max. once per candle (`--timeframe-detail` modifies this behavior to once per detailed candle).
    Most callbacks will be called once per iteration in live (usually every ~5s) - which can cause backtesting mismatches.
