# Exchange-specific Notes

This page combines common gotchas and informations which are exchange-specific and most likely don't apply to other exchanges.

## Binance

!!! Tip "Stoploss on Exchange"
    Binance is currently the only exchange supporting `stoploss_on_exchange`. It provides great advantages, so we recommend to benefit from it.

### Blacklists

For Binance, please add `"BNB/<STAKE>"` to your blacklist to avoid issues.
Accounts having BNB accounts use this to pay for fees - if your first trade happens to be on `BNB`, further trades will consume this position and make the initial BNB order unsellable as the expected amount is not there anymore.

### Binance sites

Binance has been split into 3, and users must use the correct ccxt exchange ID for their exchange, otherwise API keys are not recognized.

* [binance.com](https://www.binance.com/) - International users - ccxt id: `binance`
* [binance.us](https://www.binance.us/) US based users- ccxt id: `binanceus`
* [binance.je](https://www.binance.je/) Trading FIAT currencies - ccxt id: `binanceje`

## Kraken

### Historic Kraken data

The Kraken API does only provide 720 historic candles, which is sufficient for FreqTrade dry-run and live trade modes, but is a problem for backtesting.
To download data for the Kraken exchange, using `--dl-trades` is mandatory, otherwise the bot will download the same 720 candles over and over, and you'll not have enough backtest data.

## Bittrex

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

You can get a list of restricted markets by using the following snipptet:

``` python
import ccxt
ct = ccxt.bittrex()
_ = ct.load_markets()
res = [ f"{x['MarketCurrency']}/{x['BaseCurrency']}" for x in ct.publicGetMarkets()['result'] if x['IsRestricted']]
print(res)

```

## Random notes for other exchanges

* The Ocean (ccxt id: 'theocean') exchange uses Web3 functionality and requires web3 package to be installed:
```shell
$ pip3 install web3
```
