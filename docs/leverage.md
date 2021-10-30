# Leverage

!!! Warning "Beta feature"
    This feature is still in it's testing phase. Should you notice something you think is wrong please let us know via Discord or via Github Issue.

!!! Note "Multiple bots on one account"
    You can't run 2 bots on the same account with leverage. For leveraged / margin trading, freqtrade assumes it's the only user of the account, and all liquidation levels are calculated based on this assumption.

!!! Danger "Trading with leverage is very risky"
    Do not trade with a leverage > 1 using a strategy that hasn't shown positive results in a live run using the spot market. Check the stoploss of your strategy. With a leverage of 2, a stoploss of 0.5 would be too low, and these trades would be liquidated before reaching that stoploss.
    We do not assume any responsibility for eventual losses that occur from using this software or this mode.

    Please only use advanced trading modes when you know how freqtrade (and your strategy) works.
    Also, never risk more than what you can afford to lose.

## Understand `trading_mode`

The possible values are: `spot` (default), `margin`(*coming soon*) or `futures`.

### Spot

Regular trading mode (low risk)

- Long trades only (No short trades).
- No leverage.
- No Liquidation.
- Profits gained/lost are equal to the change in value of the assets (minus trading fees).

### Leverage trading modes

# TODO-lev: include a resource to help calculate stoplosses that are above the liquidation price

With leverage, a trader borrows capital from the exchange. The capital must be repayed fully to the exchange(potentially with interest), and the trader keeps any profits, or pays any losses, from any trades made using the borrowed capital.

Because the capital must always be repayed, exchanges will **liquidate** a trade (forcefully sell the traders assets) made using borrowed capital when the total value of assets in a leverage account drops to a certain point(a point where the total value of losses is less than the value of the collateral that the trader actually owns in the leverage account), in order to ensure that the trader has enough capital to pay back the borrowed assets to the exchange. The exchange will also charge a **liquidation fee**, adding to the traders losses. For this reason, **DO NOT TRADE WITH LEVERAGE IF YOU DON'T KNOW EXACTLY WHAT YOUR DOING. LEVERAGE TRADING IS HIGH RISK, AND CAN RESULT IN THE VALUE OF YOUR ASSETS DROPPING TO 0 VERY QUICKLY, WITH NO CHANCE OF INCREASING IN VALUE AGAIN**

#### MARGIN
*Currently unavailable*
    Trading occurs on the spot market, but the exchange lends currency to you in an amount equal to the chosen leverage. You pay the amount lent to you back to the exchange with interest, and your profits/losses are multiplied by the leverage specified
    
#### FUTURES

Perpetual swaps (also known as Perpetual Futures) are contracts traded at a price that is closely tied to the underlying asset they are based off of(ex. ). You are not trading the actual asset but instead are trading a derivative contract. Perpetual swap contracts can last indefinately, in contrast to futures or option contracts.

In addition to the gains/losses from the change in price of the contract, traders also exchange funding fees, which are gains/losses worth an amount that is derived from the difference in price between the contract and the underlying asset. The difference in price between a contract and the underlying asset varies between exchanges.

In addition to the gains/losses from the change in price of the futures contract, traders also exchange funding fees, which are gains/losses worth an amount that is derived from the difference in price between the futures contract and the underlying asset. The difference in price between a futures contract and the underlying asset varies between exchanges.

``` json
"trading_mode": "futures"
```

### Collateral

The possible values are: `isolated`, or `cross`(*currently unavailable*)

#### ISOLATED

Each market(trading pair), keeps collateral in a separate account

``` json
"collateral": "isolated"
```

#### CROSS
*currently unavailable*
One account is used to share collateral between markets (trading pairs). Margin is taken from total account balance to avoid liquidation when needed.

``` json
"collateral": "cross"
```

### Developer

#### Margin mode

For shorts, the currency which pays the interest fee for the `borrowed` currency is purchased at the same time of the closing trade (This means that the amount purchased in short closing trades is greater than the amount sold in short opening trades).

For longs, the currency which pays the interest fee for the `borrowed` will already be owned by the user and does not need to be purchased. The interest is subtracted from the `close_value` of the trade.

All Fees are included in `current_profit` calculations during the trade.

#### Binance margin trading interest formula

$$
I (interest) = P (borrowed money) * R (daily_interest/24) * ceiling(T) (in hours)
$$

[source](https://www.binance.com/en/support/faq/360030157812)

#### Kraken margin trading interest formula

$$\begin{align*}
& Opening fee = P (borrowed_money) * R (quat_hourly_interest) \\
& Rollover fee = P (borrowed_money) * R (quat_hourly_interest) * ceiling(T/4) (in hours) \\
& I (interest) = Opening_fee + Rollover_fee
\end{align*}$$

[source](https://support.kraken.com/hc/en-us/articles/206161568-What-are-the-fees-for-margin-trading-)

#### FUTURES MODE

Funding fees are either added or subtracted from the total amount of a trade
