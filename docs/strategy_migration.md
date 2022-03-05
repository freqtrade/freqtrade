# Strategy Migration between V2 and V3

We have put a great effort into keeping compatibility with existing strategies, so if you just want to continue using freqtrade in spot markets, there should be no changes necessary for now.

To support new markets and trade-types (namely short trades / trades with leverage), some things had to change in the interface.
If you intend on using markets other than spot markets, please migrate your strategy to the new format.

## Quick summary / checklist

* Dataframe columns:
  * `buy` -> `enter_long`
  * `sell` -> `exit_long`
  * `buy_tag` -> `enter_tag` (used for both long and short trades)
  * New column `enter_short` and corresponding new column `exit_short`
* trade-object now has the following new properties: `is_short`, `enter_side`, `exit_side` and `trade_direction`.
* New `side` argument to callbacks without trade object
  * `custom_stake_amount`
  * `confirm_trade_entry`
* Renamed `trade.nr_of_successful_buys` to `trade.nr_of_successful_entries`.
* Introduced new `leverage` callback
* `@informative` decorator now takes an optional `candle_type` argument
* helper methods `stoploss_from_open` and `stoploss_from_absolute` now take `is_short` as additional argument.
* `INTERFACE_VERSION` should be set to 3.

## Extensive explanation


