# Exchange-specific Notes

This page combines common gotchas and informations which are exchange-specific and most likely don't apply to other exchanges.

## Binance

!!! Tip "Stoploss on Exchange"
    Binance supports `stoploss_on_exchange` and uses stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.

### Blacklists

For Binance, please add `"BNB/<STAKE>"` to your blacklist to avoid issues.
Accounts having BNB accounts use this to pay for fees - if your first trade happens to be on `BNB`, further trades will consume this position and make the initial BNB order unsellable as the expected amount is not there anymore.

### Binance sites

Binance has been split into 3, and users must use the correct ccxt exchange ID for their exchange, otherwise API keys are not recognized.

* [binance.com](https://www.binance.com/) - International users. Use exchange id: `binance`.
* [binance.us](https://www.binance.us/) - US based users. Use exchange id: `binanceus`.
* [binance.je](https://www.binance.je/) - Binance Jersey, trading fiat currencies. Use exchange id: `binanceje`.

## Kraken

!!! Tip "Stoploss on Exchange"
    Kraken supports `stoploss_on_exchange` and uses stop-loss-market orders. It provides great advantages, so we recommend to benefit from it.

### Historic Kraken data

The Kraken API does only provide 720 historic candles, which is sufficient for Freqtrade dry-run and live trade modes, but is a problem for backtesting.
To download data for the Kraken exchange, using `--dl-trades` is mandatory, otherwise the bot will download the same 720 candles over and over, and you'll not have enough backtest data.

Due to the heavy rate-limiting applied by Kraken, the following configuration section should be used to download data:

``` json
    "ccxt_async_config": {
        "enableRateLimit": true,
        "rateLimit": 3100
    },
```

## Bittrex

### Order types

Bittrex does not support market orders. If you have a message at the bot startup about this, you should change order type values set in your configuration and/or in the strategy from `"market"` to `"limit"`. See some more details on this [here in the FAQ](faq.md#im-getting-the-exchange-bittrex-does-not-support-market-orders-message-and-cannot-run-my-strategy).

### Restricted markets

Bittrex split its exchange into US and International versions.
The International version has more pairs available, however the API always returns all pairs, so there is currently no automated way to detect if you're affected by the restriction.

If you have restricted pairs in your whitelist, you'll get a warning message in the log on Freqtrade startup for each restricted pair.

The warning message will look similar to the following:

``` output
[...] Message: bittrex {"success":false,"message":"RESTRICTED_MARKET","result":null,"explanation":null}"
```

If you're an "International" customer on the Bittrex exchange, then this warning will probably not impact you.
If you're a US customer, the bot will fail to create orders for these pairs, and you should remove them from your whitelist.

You can get a list of restricted markets by using the following snippet:

``` python
import ccxt
ct = ccxt.bittrex()
_ = ct.load_markets()
res = [ f"{x['MarketCurrency']}/{x['BaseCurrency']}" for x in ct.publicGetMarkets()['result'] if x['IsRestricted']]
print(res)
```

## FTX

!!! Tip "Stoploss on Exchange"
    FTX supports `stoploss_on_exchange` and can use both stop-loss-market and stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.
    You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide.


### Using subaccounts

To use subaccounts with FTX, you need to edit the configuration and add the following:

``` json
"exchange": {
    "ccxt_config": {
        "headers": {
            "FTX-SUBACCOUNT": "name"
        }
    },
}
```

!!! Note
    Older versions of freqtrade may require this key to be added to `"ccxt_async_config"` as well.

## All exchanges

Should you experience constant errors with Nonce (like `InvalidNonce`), it is best to regenerate the API keys. Resetting Nonce is difficult and it's usually easier to regenerate the API keys.


## Random notes for other exchanges

* The Ocean (exchange id: `theocean`) exchange uses Web3 functionality and requires `web3` python package to be installed:
```shell
$ pip3 install web3
```

### Getting latest price / Incomplete candles

Most exchanges return current incomplete candle via their OHLCV/klines API interface.
By default, Freqtrade assumes that incomplete candle is fetched from the exchange and removes the last candle assuming it's the incomplete candle.

Whether your exchange returns incomplete candles or not can be checked using [the helper script](developer.md#Incomplete-candles) from the Contributor documentation.

Due to the danger of repainting, Freqtrade does not allow you to use this incomplete candle.

However, if it is based on the need for the latest price for your strategy - then this requirement can be acquired using the [data provider](strategy-customization.md#possible-options-for-dataprovider) from within the strategy.
