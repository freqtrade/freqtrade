# Leverage
*You can't run 2 bots on the same account with leverage*

!!! Warning "Trading with leverage is very risky"
    *(`trading_mode="margin"` or `trading_mode="futures"`)* Do not trade with a leverage > 1 using a strategy that hasn't shown positive results in a live run using the spot market. Check the stoploss of your strategy. With a leverage of 2, a stoploss of 0.5 would be too low, and these trades would be liquidated before reaching that amount

> I've only been using freqtrade for a couple weeks, but I feel like I'm pretty good and could use leverage

**No you're not. Do not use leverage yet.**
### Understand trading_mode

The possible values are: `spot` (default), `margin`(*coming soon*) or `futures`.

**SPOT**
    Regular trading mode. 
        - Shorting is not available
        - There is no liquidation price
        - Profits gained/lost are equal to the change in value of the assets(minus trading fees)

#### Leverage trading modes

# TODO: include a resource to help calculate stoplosses that are above the liquidation price

#TODO: Taken from investopedia, is that ok?
Leverage results from using borrowed capital as a funding source when investing to expand the firm's asset base and generate returns on risk capital. Leverage is an investment strategy of using borrowed money—specifically, the use of various financial instruments or borrowed capital—to increase the potential return of an investment. 


**MARGIN**
*coming soon*
    Trading occurs on the spot market, but the exchange lends currency to you in an amount equal to the chosen leverage. You pay the amount lent to you back to the exchange with interest, and your profits/losses are multiplied by the leverage specified
    
**FUTURES**
*Freqtrade can only trade **perpetual futures***

    Perpetual futures contracts are traded at a price that mirrors the underlying asset they are based off of. You are not trading the actual asset but instead are trading a derivative contract. In contract to regular futures contracts, perpetual futures can last indefinately. 

    In addition to the gains/losses from the change in price of the futures contract, traders also exchange funding fees, which are gains/losses worth an amount that is derived from the difference in price between the futures contract and the underlying asset. The difference in price between a futures contract and the underlying asset varies between exchanges.


``` python
"trading_mode": "futures"
```

### Collateral

The possible values are: `isolated`, or `cross`(*coming soon*)

# TODO: I took this definition from bitmex, is that fine? https://www.bitmex.com/app/isolatedMargin
**ISOLATED** 

Margin assigned to a position is restricted to a certain amount. If the margin falls below the Maintenance Margin level, the position is liquidated.

**CROSS**

Margin is shared between open positions. When needed, a position will draw more margin from the total account balance to avoid liquidation. 

``` python
"collateral": "isolated"
```

### Developer

For shorts, the currency which pays the interest fee for the `borrowed` currency is purchased at the same time of the closing trade (This means that the amount purchased in short closing trades is greater than the amount sold in short opening trades).

For longs, the currency which pays the interest fee for the `borrowed` will already be owned by the user and does not need to be purchased. The interest is subtracted from the close_value of the trade.

#### Binance margin trading interest formula

    I (interest) = P (borrowed money) * R (daily_interest/24) * ceiling(T) (in hours)
    [source](https://www.binance.com/en/support/faq/360030157812)

#### Kraken margin trading interest formula

    Opening fee  = P (borrowed money) * R (quat_hourly_interest)
    Rollover fee = P (borrowed money) * R (quat_hourly_interest) * ceiling(T/4) (in hours)
    I (interest) = Opening fee + Rollover fee
    [source](https://support.kraken.com/hc/en-us/articles/206161568-What-are-the-fees-for-margin-trading-)



