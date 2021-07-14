# Margin-db Pull Request Review

## Calculations

### Binance interest formula

    I (interest) = P (borrowed money) * R (daily_interest/24) * ceiling(T) (in hours)

[source](https://www.binance.com/en/support/faq/360030157812)

### Kraken interest formula

    Opening fee  = P (borrowed money) * R (quat_hourly_interest)
    Rollover fee = P (borrowed money) * R (quat_hourly_interest) * ceiling(T/4) (in hours)
    I (interest) = Opening fee + Rollover fee

[source](https://support.kraken.com/hc/en-us/articles/206161568-What-are-the-fees-for-margin-trading-)

### Profit ratio for short trades

`profit_ratio = (1 - (close_trade_value/self.open_trade_value))`

### Profit ratio for leveraged trades

`leveraged_profit_ratio = profit_ratio * leverage`

## Tests

To add shorting and leverage functionality to freqtrade, the majority of changes involved editing existing methods within `freqtrade/persistence/models.py`. `freqtrade/persistence/models` was already fully tested, so to test the new functionality, new tests were created inside `tests/persistence/test_persistence_short.py` and `tests/persistence/test_persistence_leverage.py`, which mirrored the the tests inside `tests/persistence/test_persistence.py`, but accounted for additional factors created by _leverage_ and _short trading_

### Factors added to freqtrade/persistence

#### Short trading

- Exit trade amounts are slightly higher than their enter amounts, due to the extra amount purchased for paying back interest owed for the amount of currency borrowed
- Profit calculation were calculated inversely compared to longs
- Stoplosses moved in the reverse direction
- The entering/starting trade was a "sell" and the exiting/closing trade was a "buy"

#### Leveraged long trading

- Leveraged long trades had the amount of interest owed subtracted from the value of their exiting trade

<hr>

These are all the tests from `test_persistence.py` for which mirror versions were created inside `test_persistence_short.py` and `test_persistence_long.py`. These tests were chosen because the factors listed above affect the output of these methods. Some tests were only mirrored within `test_persistence_short.py` because the methods they test would not be impacted by leverage. `test_persistence_short.py`repeated tests from `test_persistence.py`, but for short trades, to make sure that the methods tested calculated the right values for shorts, and `test_persistence_long.py`repeated tests from `test_persistence.py`, but for leveraged long trades, to make sure that the methods tested calculated the right values when leverage was involved

<table style="width:100%">
  <tr>
    <th>test_persistence</th>
    <th>test_persistence_short</th> 
    <th>test_persistence_leverage</th>
  </tr>
  <tr>
    <td>test_update_with_binance</td>
    <td>test_update_with_binance_short</td>
    <td>test_update_with_binance_lev</td>
  </tr>
  <tr>
    <td>test_update_market_order</td>
    <td>test_update_market_order_short</td>
    <td>test_update_market_order_lev</td>
  </tr>
  <tr>
    <td>test_update_open_order</td>
    <td>test_update_open_order_short</td>
    <td>test_update_open_order_lev</td>
  </tr>
  <tr>
    <td>test_calc_open_trade_value</td>
    <td>test_calc_open_trade_value_short</td>
    <td>test_calc_open_trade_value_lev</td>
  </tr>
  <tr>
    <td>test_calc_open_close_trade_price</td>
    <td>test_calc_open_close_trade_price_short</td>
    <td>test_calc_open_close_trade_price_lev</td>
  </tr>
  <tr>
    <td>test_trade_close</td>
    <td>test_trade_close_short</td>
    <td>test_trade_close_lev</td>
  </tr>
  <tr>
    <td>test_calc_close_trade_price_exception</td>
    <td>test_calc_close_trade_price_exception_short</td>
    <td>test_calc_close_trade_price_exception_lev</td>
  </tr>
  <tr>
    <td>test_calc_close_trade_price</td>
    <td>test_calc_close_trade_price_short</td>
    <td>test_calc_close_trade_price_lev</td>
  </tr>
  <tr>
    <td>test_calc_close_trade_price_exception</td>
    <td>test_calc_close_trade_price_exception_short</td>
    <td>test_calc_close_trade_price_exception_lev</td>
  </tr>
  <tr>
    <td>test_calc_profit & test_calc_profit_ratio</td>
    <td>test_calc_profit_short</td>
    <td>test_calc_profit_lev</td>
  </tr>
  <tr>
    <th colspan="3">
    Tests with no equivelent in test_persistence_lev
    </th>
  </tr>
  <tr>
    <td>test_adjust_stop_loss</td>
    <td>test_adjust_stop_loss_short</td>
  </tr>
  <tr>
    <td>test_get_open</td>
    <td>test_get_open_short</td>
  </tr>
  <tr>
    <td>test_total_open_trades_stakes</td>
    <td>test_total_open_trades_stakes_short</td>
  </tr>
  <tr>
    <td>test_stoploss_reinitialization</td>
    <td>test_stoploss_reinitialization_short</td>
  </tr>
  <tr>
    <td>test_get_best_pair</td>
    <td>test_get_best_pair_short</td>
  </tr>
</table>

### Tests not repeated

These tests did not have an equivelent version created inside `test_persistence_short.py` or `test_persistence_lev.py` because no new situations arise in the methods they test when adding leverage or short trading to `freqtrade/persistence`

- <small>test_init_create_session</small>
- <small>test_init_custom_db_url</small>
- <small>test_init_invalid_db_url</small>
- <small>test_init_prod_db</small>
- <small>test_init_dryrun_db</small>
- <small>test_update_order_from_ccxt</small>
- <small>test_select_order</small>
- <small>test_Trade_object_idem</small>
- <small>test_get_trades_proxy</small>
- <small>test_update_fee</small>
- <small>test_fee_updated</small>
- <small>test_to_json</small>
- <small>test_migrate_old</small>
- <small>test_migrate_new</small>
- <small>test_migrate_mid_state</small>
- <small>test_clean_dry_run_db</small>
- <small>test_update_invalid_order</small>
- <small>test_adjust_min_max_rates</small>
- <small>test_get_trades_backtest</small>
- <small>test_get_overall_performance</small>

<hr>

### Original tests

These methods were added to `LocalTrade` in `freqtrade/persistence/models.py`

- `is_opening_trade`
- `is_closing_trade`
- `set_stop_loss`
- `set_liquidation_price`
- `calculate_interest`
- `borrowed(calculated property)`

These were the tests created to test these methods

<table style="width:100%">
  <tr>
    <th>test_persistence</th>
    <th>test_persistence_short</th> 
    <th>test_persistence_lev</th>
  </tr>
  <tr>
    <td>test_is_opening_closing_trade</td>
    <td>/</td>
    <td>/</td>
  </tr>
  <tr>
    <td>test_set_stop_loss_liquidation_price</td>
    <td>/</td>
    <td>/</td>
  </tr>
  <tr>
    <td>/</td>
    <td>test_interest_kraken_short</td>
    <td>test_interest_binance_short</td>
  </tr>
  <tr>
    <td>/</td>
    <td>test_interest_kraken_lev</td>
    <td>test_interest_binance_lev</td>
  </tr>
</table>

#### test_is_opening_closing_trade

Tested methods from `freqtrade/persistence/models.py`

- `LocalTrade.is_opening_trade`
- `LocalTrade.is_closing_trade`

Tested to check that the correct boolean value is returned according to this truth table

<table style="width:100%">
  <tr>
    <th colspan="1"></th>
    <th colspan="2">is_opening_trade</th>
    <th colspan="2">is_closing_trade</th> 
  </tr>
  <tr>
    <td></td>
    <td>short</td>
    <td>long</td>
    <td>short</td>
    <td>long</td>
  </tr>
  <tr>
    <td>side="buy"</td>
    <td>False</td>
    <td>True</td>
    <td>True</td>
    <td>False</td>
  </tr>
  <tr>
    <td>side="sell"</td>
    <td>True</td>
    <td>False</td>
    <td>False</td>    
    <td>True</td>
  </tr>
</table>

#### test_set_stop_loss_liquidation_price

Tested methods from `freqtrade/persistence/models.py`

- `LocalTrade.set_stop_loss`
- `LocalTrade.set_liquidation_price`

Tested to check for these conditions

- `LocalTrade.stop_loss` is never lower than `LocalTrade.liquidation_price` for longs
- `LocalTrade.stop_loss` is never higher than `LocalTrade.liquidation_price` for shorts
- `LocalTrade.stop_loss` and `LocalTrade.intial_stop_loss` are assigned to the value of `LocalTrade.liquidation_price` when no `LocalTrade.stop_loss` is not assigned
- `LocalTrade.liquidation_price` is not changed when `LocalTrade.stop_loss` gets assigned

#### LocalTrade.calculate_interest

Tests created to test this method

- `test_interest_kraken_short`
- `test_interest_binance_short`
- `test_interest_kraken_lev`
- `test_interest_binance_lev`

Conditions tested

- correct interest calculated for a short trade made on kraken
- correct interest calculated for a long trade made on kraken
- correct interest calculated for a short trade made on binance
- correct interest calculated for a long trade made on binance

#### LocalTrade.borrowed

Tested within

- test_update_with_binance_short
- test_update_with_binance_lev

Conditions tested for

- borrowed was equal to `amount` for short trades
- borrowed was equal to `amount * leverage-1` for long trades (1 is subtracted from leverage because the collateral is in the borrowed currency and is already owned)
