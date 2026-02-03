## Exit logic comparisons

Freqtrade allows your strategy to implement different exit logic using signal-based or callback-based functions.
This section aims to compare each different function, helping you to choose the one that best fits your needs.

* **`populate_exit_trend()`** - Vectorized signal-based exit logic using indicators in the main dataframe
  ✅ **Use** to define exit signals based on indicators or other data that can be calculated in a vectorized manner.
  🚫 **Don't use** to customize exit conditions for each individual trade, or if trade data is necessary to make an exit decision.
* **`custom_exit()`** - Custom exit logic that will fully exit a trade immediately, called for every open trade at every bot loop iteration until a trade is closed.
  ✅ **Use** to specify exit conditions for each individual trade (including any additional adjusted orders using `adjust_trade_position()`), or if trade data is necessary to make an exit decision, e.g. using profit data to exit.
  🚫 **Don't use** when you want to exit using vectorised indicator-based data (use a `populate_exit_trend()` signal instead), or as a proxy for `custom_stoploss()`, and be aware that rate-based exits in backtesting can be inaccurate.
* **`custom_stoploss()`** - Custom trailing stoploss, called for every open trade every iteration until a trade is closed. The value returned here is also used for [stoploss on exchange](stoploss.md#stop-loss-on-exchangefreqtrade).  
  ✅ **Use** to customize the stoploss logic to set a dynamic stoploss based on trade data or other conditions.
  🚫 **Don't use** to exit a trade immediately based on a specific condition. Use `custom_exit()` for that purpose.
* **`custom_roi()`** - Custom ROI, called for every open trade every iteration until a trade is closed.
  ✅ **Use** to specify a minimum ROI threshold ("take-profit") to exit a trade at this ROI level at some point within the trade duration, based on profit or other conditions.
  🚫 **Don't use** to exit a trade immediately based on a specific condition. Use `custom_exit()`.
  🚫 **Don't use** for static ROI. Use `minimal_roi`.
