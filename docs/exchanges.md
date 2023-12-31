# Exchange-specific Notes

This page combines common gotchas and Information which are exchange-specific and most likely don't apply to other exchanges.

## Exchange configuration

Freqtrade is based on [CCXT library](https://github.com/ccxt/ccxt) that supports over 100 cryptocurrency
exchange markets and trading APIs. The complete up-to-date list can be found in the
[CCXT repo homepage](https://github.com/ccxt/ccxt/tree/master/python).
However, the bot was tested by the development team with only a few exchanges.
A current list of these can be found in the "Home" section of this documentation.

Feel free to test other exchanges and submit your feedback or PR to improve the bot or confirm exchanges that work flawlessly..

Some exchanges require special configuration, which can be found below.

### Sample exchange configuration

A exchange configuration for "binance" would look as follows:

```json
"exchange": {
    "name": "binance",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "ccxt_config": {},
    "ccxt_async_config": {},
    // ... 
```

### Setting rate limits

Usually, rate limits set by CCXT are reliable and work well.
In case of problems related to rate-limits (usually DDOS Exceptions in your logs), it's easy to change rateLimit settings to other values.

```json
"exchange": {
    "name": "kraken",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "ccxt_config": {"enableRateLimit": true},
    "ccxt_async_config": {
        "enableRateLimit": true,
        "rateLimit": 3100
    },
```

This configuration enables kraken, as well as rate-limiting to avoid bans from the exchange.
`"rateLimit": 3100` defines a wait-event of 3.1s between each call. This can also be completely disabled by setting `"enableRateLimit"` to false.

!!! Note
    Optimal settings for rate-limiting depend on the exchange and the size of the whitelist, so an ideal parameter will vary on many other settings.
    We try to provide sensible defaults per exchange where possible, if you encounter bans please make sure that `"enableRateLimit"` is enabled and increase the `"rateLimit"` parameter step by step.

## Binance

!!! Warning "Server location and geo-ip restrictions"
    Please be aware that Binance restricts API access regarding the server country. The current and non-exhaustive countries blocked are Canada, Malaysia, Netherlands and United States. Please go to [binance terms > b. Eligibility](https://www.binance.com/en/terms) to find up to date list.

Binance supports [time_in_force](configuration.md#understand-order_time_in_force).

!!! Tip "Stoploss on Exchange"
    Binance supports `stoploss_on_exchange` and uses `stop-loss-limit` orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange.
    On futures, Binance supports both `stop-limit` as well as `stop-market` orders. You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type to use.

### Binance Blacklist recommendation

For Binance, it is suggested to add `"BNB/<STAKE>"` to your blacklist to avoid issues, unless you are willing to maintain enough extra `BNB` on the account or unless you're willing to disable using `BNB` for fees.
Binance accounts may use `BNB` for fees, and if a trade happens to be on `BNB`, further trades may consume this position and make the initial BNB trade unsellable as the expected amount is not there anymore.

### Binance sites

Binance has been split into 2, and users must use the correct ccxt exchange ID for their exchange, otherwise API keys are not recognized.

* [binance.com](https://www.binance.com/) - International users. Use exchange id: `binance`.
* [binance.us](https://www.binance.us/) - US based users. Use exchange id: `binanceus`.

### Binance RSA keys

Freqtrade supports binance RSA API keys.

We recommend to use them as environment variable.

``` bash
export FREQTRADE__EXCHANGE__SECRET="$(cat ./rsa_binance.private)"
```

They can however also be configured via configuration file. Since json doesn't support multi-line strings, you'll have to replace all newlines with `\n` to have a valid json file.

``` json
// ...
 "key": "<someapikey>",
 "secret": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBABACAFQA<...>s8KX8=\n-----END PRIVATE KEY-----"
// ...
```

### Binance Futures

Binance has specific (unfortunately complex) [Futures Trading Quantitative Rules](https://www.binance.com/en/support/faq/4f462ebe6ff445d4a170be7d9e897272) which need to be followed, and which prohibit a too low stake-amount (among others) for too many orders.
Violating these rules will result in a trading restriction.

When trading on Binance Futures market, orderbook must be used because there is no price ticker data for futures.

``` jsonc
  "entry_pricing": {
      "use_order_book": true,
      "order_book_top": 1,
      "check_depth_of_market": {
          "enabled": false,
          "bids_to_ask_delta": 1
      }
  },
  "exit_pricing": {
      "use_order_book": true,
      "order_book_top": 1
  },
```

#### Binance futures settings

Users will also have to have the futures-setting "Position Mode" set to "One-way Mode", and "Asset Mode" set to "Single-Asset Mode".
These settings will be checked on startup, and freqtrade will show an error if this setting is wrong.

![Binance futures settings](assets/binance_futures_settings.png)

Freqtrade will not attempt to change these settings.

## Kraken

Kraken supports [time_in_force](configuration.md#understand-order_time_in_force) with settings "GTC" (good till cancelled), "IOC" (immediate-or-cancel) and "PO" (Post only) settings.

!!! Tip "Stoploss on Exchange"
    Kraken supports `stoploss_on_exchange` and can use both stop-loss-market and stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.
    You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type to use.

### Historic Kraken data

The Kraken API does only provide 720 historic candles, which is sufficient for Freqtrade dry-run and live trade modes, but is a problem for backtesting.
To download data for the Kraken exchange, using `--dl-trades` is mandatory, otherwise the bot will download the same 720 candles over and over, and you'll not have enough backtest data.

To speed up downloading, you can download the [trades zip files](https://support.kraken.com/hc/en-us/articles/360047543791-Downloadable-historical-market-data-time-and-sales-) kraken provides.
These are usually updated once per quarter. Freqtrade expects these files to be placed in `user_data/data/kraken/trades_csv`.

A structure as follows can make sense if using incremental files, with the "full" history in one directory, and incremental files in different directories.
The assumption for this mode is that the data is downloaded and unzipped keeping filenames as they are.
Duplicate content will be ignored (based on timestamp) - though the assumption is that there is no gap in the data.

This means, if your "full" history ends in Q4 2022 - then both incremental updates Q1 2023 and Q2 2023 are available.
Not having this will lead to incomplete data, and therefore invalid results while using the data.

```
└── trades_csv
    ├── Kraken_full_history
    │   ├── BCHEUR.csv
    │   └── XBTEUR.csv
    ├── Kraken_Trading_History_Q1_2023
    │   ├── BCHEUR.csv
    │   └── XBTEUR.csv
    └── Kraken_Trading_History_Q2_2023
        ├── BCHEUR.csv
        └── XBTEUR.csv
```

You can convert these files into freqtrade files:

``` bash
freqtrade convert-trade-data --exchange kraken --format-from kraken_csv --format-to feather
# Convert trade data to different ohlcv timeframes
freqtrade trades-to-ohlcv -p BTC/EUR BCH/EUR --exchange kraken -t 1m 5m 15m 1h
```

The converted data also makes downloading data possible, and will start the download after the latest loaded trade.

``` bash
freqtrade download-data --exchange kraken --dl-trades -p BTC/EUR BCH/EUR 
```

!!! Warning "Downloading data from kraken"
    Downloading kraken data will require significantly more memory (RAM) than any other exchange, as the trades-data needs to be converted into candles on your machine.
    It will also take a long time, as freqtrade will need to download every single trade that happened on the exchange for the pair / timerange combination, therefore please be patient.

!!! Warning "rateLimit tuning"
    Please pay attention that rateLimit configuration entry holds delay in milliseconds between requests, NOT requests\sec rate.
    So, in order to mitigate Kraken API "Rate limit exceeded" exception, this configuration should be increased, NOT decreased.

## Kucoin

Kucoin requires a passphrase for each api key, you will therefore need to add this key into the configuration so your exchange section looks as follows:

```json
"exchange": {
    "name": "kucoin",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
    // ...
}
```

Kucoin supports [time_in_force](configuration.md#understand-order_time_in_force).

!!! Tip "Stoploss on Exchange"
    Kucoin supports `stoploss_on_exchange` and can use both stop-loss-market and stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.
    You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type of stoploss shall be used.

### Kucoin Blacklists

For Kucoin, it is suggested to add `"KCS/<STAKE>"` to your blacklist to avoid issues, unless you are willing to maintain enough extra `KCS` on the account or unless you're willing to disable using `KCS` for fees. 
Kucoin accounts may use `KCS` for fees, and if a trade happens to be on `KCS`, further trades may consume this position and make the initial `KCS` trade unsellable as the expected amount is not there anymore.

## Huobi

!!! Tip "Stoploss on Exchange"
    Huobi supports `stoploss_on_exchange` and uses `stop-limit` orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange.

## OKX (former OKEX)

OKX requires a passphrase for each api key, you will therefore need to add this key into the configuration so your exchange section looks as follows:

```json
"exchange": {
    "name": "okx",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
    // ...
}
```

!!! Warning
    OKX only provides 100 candles per api call. Therefore, the strategy will only have a pretty low amount of data available in backtesting mode.

!!! Warning "Futures"
    OKX Futures has the concept of "position mode" - which can be "Buy/Sell" or long/short (hedge mode).
    Freqtrade supports both modes (we recommend to use Buy/Sell mode) - but changing the mode mid-trading is not supported and will lead to exceptions and failures to place trades.
    OKX also only provides MARK candles for the past ~3 months. Backtesting futures prior to that date will therefore lead to slight deviations, as funding-fees cannot be calculated correctly without this data.

## Gate.io

!!! Tip "Stoploss on Exchange"
    Gate.io supports `stoploss_on_exchange` and uses `stop-loss-limit` orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange..

Gate.io allows the use of `POINT` to pay for fees. As this is not a tradable currency (no regular market available), automatic fee calculations will fail (and default to a fee of 0).
The configuration parameter `exchange.unknown_fee_rate` can be used to specify the exchange rate between Point and the stake currency. Obviously, changing the stake-currency will also require changes to this value.

## Bybit

Futures trading on bybit is currently supported for USDT markets, and will use isolated futures mode.
Users with unified accounts (there's no way back) can create a Sub-account which will start as "non-unified", and can therefore use isolated futures.
On startup, freqtrade will set the position mode to "One-way Mode" for the whole (sub)account. This avoids making this call over and over again (slowing down bot operations), but means that changes to this setting may result in exceptions and errors

As bybit doesn't provide funding rate history, the dry-run calculation is used for live trades as well.

API Keys for live futures trading (Subaccount on non-unified) must have the following permissions:
* Read-write
* Contract - Orders
* Contract - Positions

We do strongly recommend to limit all API keys to the IP you're going to use it from.

!!! Tip "Stoploss on Exchange"
    Bybit (futures only) supports `stoploss_on_exchange` and uses `stop-loss-limit` orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange.
    On futures, Bybit supports both `stop-limit` as well as `stop-market` orders. You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type to use.

## Bitmart

Bitmart requires the API key Memo (the name you give the API key) to go along with the exchange key and secret.
It's therefore required to pass the UID as well.

```json
"exchange": {
    "name": "bitmart",
    "uid": "your_bitmart_api_key_memo",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
    // ...
}
```

!!! Warning "Necessary Verification"
    Bitmart requires Verification Lvl2 to successfully trade on the spot market through the API - even though trading via UI works just fine with just Lvl1 verification.

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
        "order_time_in_force": ["GTC", "FOK"],
        "ohlcv_candle_limit": 200
        }
    //...
}
```

!!! Warning
    Please make sure to fully understand the impacts of these settings before modifying them.
