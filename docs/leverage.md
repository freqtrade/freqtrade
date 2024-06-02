# Trading with Leverage

!!! Warning "Beta feature"
    This feature is still in it's testing phase. Should you notice something you think is wrong please let us know via Discord or via Github Issue.

!!! Note "Multiple bots on one account"
    You can't run 2 bots on the same account with leverage. For leveraged / margin trading, freqtrade assumes it's the only user of the account, and all liquidation levels are calculated based on this assumption.

!!! Danger "Trading with leverage is very risky"
    Do not trade with a leverage > 1 using a strategy that hasn't shown positive results in a live run using the spot market. Check the stoploss of your strategy. With a leverage of 2, a stoploss of 0.5 (50%) would be too low, and these trades would be liquidated before reaching that stoploss.
    We do not assume any responsibility for eventual losses that occur from using this software or this mode.

    Please only use advanced trading modes when you know how freqtrade (and your strategy) works.
    Also, never risk more than what you can afford to lose.

If you already have an existing strategy, please read the [strategy migration guide](strategy_migration.md#strategy-migration-between-v2-and-v3) to migrate your strategy from a freqtrade v2 strategy, to strategy of version 3 which can short and trade futures.

## Shorting

Shorting is not possible when trading with [`trading_mode`](#leverage-trading-modes) set to `spot`. To short trade, `trading_mode` must be set to `margin`(currently unavailable) or [`futures`](#futures), with [`margin_mode`](#margin-mode) set to `cross`(currently unavailable) or [`isolated`](#isolated-margin-mode)

For a strategy to short, the strategy class must set the class variable `can_short = True`

Please read [strategy customization](strategy-customization.md#entry-signal-rules) for instructions on how to set signals to enter and exit short trades.

## Understand `trading_mode`

The possible values are: `spot` (default), `margin`(*Currently unavailable*) or `futures`.

### Spot

Regular trading mode (low risk)

- Long trades only (No short trades).
- No leverage.
- No Liquidation.
- Profits gained/lost are equal to the change in value of the assets (minus trading fees).

### Leverage trading modes

With leverage, a trader borrows capital from the exchange. The capital must be re-payed fully to the exchange (potentially with interest), and the trader keeps any profits, or pays any losses, from any trades made using the borrowed capital.

Because the capital must always be re-payed, exchanges will **liquidate** (forcefully sell the traders assets) a trade made using borrowed capital when the total value of assets in the leverage account drops to a certain point (a point where the total value of losses is less than the value of the collateral that the trader actually owns in the leverage account), in order to ensure that the trader has enough capital to pay the borrowed assets back to the exchange. The exchange will also charge a **liquidation fee**, adding to the traders losses.

For this reason, **DO NOT TRADE WITH LEVERAGE IF YOU DON'T KNOW EXACTLY WHAT YOUR DOING. LEVERAGE TRADING IS HIGH RISK, AND CAN RESULT IN THE VALUE OF YOUR ASSETS DROPPING TO 0 VERY QUICKLY, WITH NO CHANCE OF INCREASING IN VALUE AGAIN.**

#### Margin (currently unavailable)

Trading occurs on the spot market, but the exchange lends currency to you in an amount equal to the chosen leverage. You pay the amount lent to you back to the exchange with interest, and your profits/losses are multiplied by the leverage specified.

#### Futures

Perpetual swaps (also known as Perpetual Futures) are contracts traded at a price that is closely tied to the underlying asset they are based off of (ex.). You are not trading the actual asset but instead are trading a derivative contract. Perpetual swap contracts can last indefinitely, in contrast to futures or option contracts.

In addition to the gains/losses from the change in price of the futures contract, traders also exchange _funding fees_, which are gains/losses worth an amount that is derived from the difference in price between the futures contract and the underlying asset. The difference in price between a futures contract and the underlying asset varies between exchanges.

To trade in futures markets, you'll have to set `trading_mode` to "futures".
You will also have to pick a "margin mode" (explanation below) - with freqtrade currently only supporting isolated margin.

``` json
"trading_mode": "futures",
"margin_mode": "isolated"
```

##### Pair namings

Freqtrade follows the [ccxt naming conventions for futures](https://docs.ccxt.com/#/README?id=perpetual-swap-perpetual-future).
A futures pair will therefore have the naming of `base/quote:settle` (e.g. `ETH/USDT:USDT`).

### Margin mode

On top of `trading_mode` - you will also have to configure your `margin_mode`.
While freqtrade currently only supports one margin mode, this will change, and by configuring it now you're all set for future updates.

The possible values are: `isolated`, or `cross`(*currently unavailable*).

#### Isolated margin mode

Each market(trading pair), keeps collateral in a separate account

``` json
"margin_mode": "isolated"
```

#### Cross margin mode (currently unavailable)

One account is used to share collateral between markets (trading pairs). Margin is taken from total account balance to avoid liquidation when needed.

``` json
"margin_mode": "cross"
```

Please read the [exchange specific notes](exchanges.md) for exchanges that support this mode and how they differ.

## Set leverage to use

Different strategies and risk profiles will require different levels of leverage.
While you could configure one static leverage value - freqtrade offers you the flexibility to adjust this via [strategy leverage callback](strategy-callbacks.md#leverage-callback) - which allows you to use different leverages by pair, or based on some other factor benefitting your strategy result.

If not implemented, leverage defaults to 1x (no leverage).

!!! Warning
    Higher leverage also equals higher risk - be sure you fully understand the implications of using leverage!

## Understand `liquidation_buffer`

*Defaults to `0.05`*

A ratio specifying how large of a safety net to place between the liquidation price and the stoploss to prevent a position from reaching the liquidation price.
This artificial liquidation price is calculated as:

`freqtrade_liquidation_price = liquidation_price ± (abs(open_rate - liquidation_price) * liquidation_buffer)`

- `±` = `+` for long trades
- `±` = `-` for short trades

Possible values are any floats between 0.0 and 0.99

**ex:** If a trade is entered at a price of 10 coin/USDT, and the liquidation price of this trade is 8 coin/USDT, then with `liquidation_buffer` set to `0.05` the minimum stoploss for this trade would be $8 + ((10 - 8) * 0.05) = 8 + 0.1 = 8.1$

!!! Danger "A `liquidation_buffer` of 0.0, or a low `liquidation_buffer` is likely to result in liquidations, and liquidation fees"
    Currently Freqtrade is able to calculate liquidation prices, but does not calculate liquidation fees. Setting your `liquidation_buffer` to 0.0, or using a low `liquidation_buffer` could result in your positions being liquidated. Freqtrade does not track liquidation fees, so liquidations will result in inaccurate profit/loss results for your bot. If you use a low `liquidation_buffer`, it is recommended to use `stoploss_on_exchange` if your exchange supports this.

## Unavailable funding rates

For futures data, exchanges commonly provide the futures candles, the marks, and the funding rates. However, it is common that whilst candles and marks might be available, the funding rates are not. This can affect backtesting timeranges, i.e. you may only be able to test recent timeranges and not earlier, experiencing the `No data found. Terminating.` error. To get around this, add the `futures_funding_rate` config option as listed in [configuration.md](configuration.md), and it is recommended that you set this to `0`, unless you know a given specific funding rate for your pair, exchange and timerange. Setting this to anything other than `0` can have drastic effects on your profit calculations within strategy, e.g. within the `custom_exit`, `custom_stoploss`, etc functions.

!!! Warning "This will mean your backtests are inaccurate."
    This will not overwrite funding rates that are available from the exchange, but bear in mind that setting a false funding rate will mean backtesting results will be inaccurate for historical timeranges where funding rates are not available.

### Developer

#### Margin mode

For shorts, the currency which pays the interest fee for the `borrowed` currency is purchased at the same time of the closing trade (This means that the amount purchased in short closing trades is greater than the amount sold in short opening trades).

For longs, the currency which pays the interest fee for the `borrowed` will already be owned by the user and does not need to be purchased. The interest is subtracted from the `close_value` of the trade.

All Fees are included in `current_profit` calculations during the trade.

#### Futures mode

Funding fees are either added or subtracted from the total amount of a trade
