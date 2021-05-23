# Exchange-specific Notes

This page combines common gotchas and informations which are exchange-specific and most likely don't apply to other exchanges.

## Binance

!!! Tip "Stoploss on Exchange"
    Binance supports `stoploss_on_exchange` and uses stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.

### Binance Blacklist

For Binance, please add `"BNB/<STAKE>"` to your blacklist to avoid issues.
Accounts having BNB accounts use this to pay for fees - if your first trade happens to be on `BNB`, further trades will consume this position and make the initial BNB trade unsellable as the expected amount is not there anymore.

### Binance sites

Binance has been split into 3, and users must use the correct ccxt exchange ID for their exchange, otherwise API keys are not recognized.

* [binance.com](https://www.binance.com/) - International users. Use exchange id: `binance`.
* [binance.us](https://www.binance.us/) - US based users. Use exchange id: `binanceus`.
* [binance.je](https://www.binance.je/) - Binance Jersey, trading fiat currencies. Use exchange id: `binanceje`.

## Kraken

!!! Tip "Stoploss on Exchange"
    Kraken supports `stoploss_on_exchange` and can use both stop-loss-market and stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.
    You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type to use.

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

!!! Warning "Downloading data from kraken"
    Downloading kraken data will require significantly more memory (RAM) than any other exchange, as the trades-data needs to be converted into candles on your machine.
    It will also take a long time, as freqtrade will need to download every single trade that happened on the exchange for the pair / timerange combination, therefore please be patient.

!!! Warning "rateLimit tuning"
    Please pay attention that rateLimit configuration entry holds delay in milliseconds between requests, NOT requests\sec rate.
    So, in order to mitigate Kraken API "Rate limit exceeded" exception, this configuration should be increased, NOT decreased.

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
    You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type of stoploss shall be used.

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

## Kucoin

Kucoin requries a passphrase for each api key, you will therefore need to add this key into the configuration so your exchange section looks as follows:

```json
"exchange": {
    "name": "kucoin",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
```

### Kucoin Blacklists

For Kucoin, please add `"KCS/<STAKE>"` to your blacklist to avoid issues.
Accounts having KCS accounts use this to pay for fees - if your first trade happens to be on `KCS`, further trades will consume this position and make the initial KCS trade unsellable as the expected amount is not there anymore.

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

### Advanced Freqtrade Exchange configuration

Advanced options can be configured using the `_ft_has_params` setting, which will override Defaults and exchange-specific behavior.

Available options are listed in the exchange-class as `_ft_has_default`.

For example, to test the order type `FOK` with Kraken, and modify candle limit to 200 (so you only get 200 candles per API call):

```json
"exchange": {
    "name": "kraken",
    "_ft_has_params": {
        "order_time_in_force": ["gtc", "fok"],
        "ohlcv_candle_limit": 200
        }
```

!!! Warning
    Please make sure to fully understand the impacts of these settings before modifying them.
