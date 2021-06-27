An instance of a `Trade`/`LocalTrade` object is given either a value for `leverage` or a value for `borrowed`, but not both, on instantiation/update with a short/long.

- If given a value for `leverage`, then the `amount` value of the `Trade`/`Local` object is multiplied by the `leverage` value to obtain the new value for `amount`. The borrowed value is also calculated from the `amount` and `leverage` value
- If given a value for `borrowed`, then the `leverage` value is calculated from `borrowed` and `amount`

For shorts, the currency which pays the interest fee for the `borrowed` currency is purchased at the same time of the closing trade (This means that the amount purchased in short closing trades is greater than the amount sold in short opening trades).

For longs, the currency which pays the interest fee for the `borrowed` will already be owned by the user and does not need to be purchased

The interest fee is paid following the closing trade, or simultaneously depending on the exchange
