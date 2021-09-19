# Leverage

For shorts, the currency which pays the interest fee for the `borrowed` currency is purchased at the same time of the closing trade (This means that the amount purchased in short closing trades is greater than the amount sold in short opening trades).

For longs, the currency which pays the interest fee for the `borrowed` will already be owned by the user and does not need to be purchased. The interest is subtracted from the close_value of the trade.

## Binance margin trading interest formula

    I (interest) = P (borrowed money) * R (daily_interest/24) * ceiling(T) (in hours)
    [source](https://www.binance.com/en/support/faq/360030157812)

## Kraken margin trading interest formula

    Opening fee  = P (borrowed money) * R (quat_hourly_interest)
    Rollover fee = P (borrowed money) * R (quat_hourly_interest) * ceiling(T/4) (in hours)
    I (interest) = Opening fee + Rollover fee
    [source](https://support.kraken.com/hc/en-us/articles/206161568-What-are-the-fees-for-margin-trading-)

# TODO-lev: Mention that says you can't run 2 bots on the same account with leverage,

#TODO-lev: Create a huge risk disclaimer