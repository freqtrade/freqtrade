# Exchange-specific Notes

This page combines common gotchas and Information which are exchange-specific and most likely don't apply to other exchanges.

## Quick overview of supported exchange features

--8<-- "includes/exchange-features.md"

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

If not enough `BNB` is available to cover transaction fees, then fees will not be covered by `BNB` and no fee reduction will occur. Freqtrade will never buy BNB to cover for fees. BNB needs to be bought and monitored manually to this end.

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

#### Binance isolated futures settings

Users will also have to have the futures-setting "Position Mode" set to "One-way Mode", and "Asset Mode" set to "Single-Asset Mode".
These settings will be checked on startup, and freqtrade will show an error if this setting is wrong.

![Binance futures settings](assets/binance_futures_settings.png)

Freqtrade will not attempt to change these settings.

#### Binance BNFCR futures

BNFCR mode are a special type of futures mode on Binance to work around regulatory issues in Europe.  
To use BNFCR futures, you will have to have the following combination of settings:

``` jsonc
{
    // ...
    "trading_mode": "futures",
    "margin_mode": "cross",
    "proxy_coin": "BNFCR",
    "stake_currency": "USDT" // or "USDC"
    // ...
}
```

The `stake_currency` setting defines the markets the bot will be operating in. This choice is really arbitrary.

On the exchange, you'll have to use "Multi-asset Mode" - and "Position Mode set to "One-way Mode".  
Freqtrade will check these settings on startup, but won't attempt to change them.

## Bingx

BingX supports [time_in_force](configuration.md#understand-order_time_in_force) with settings "GTC" (good till cancelled), "IOC" (immediate-or-cancel) and "PO" (Post only) settings.

!!! Tip "Stoploss on Exchange"
    Bingx supports `stoploss_on_exchange` and can use both stop-limit and stop-market orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange.

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
    Please pay attention that rateLimit configuration entry holds delay in milliseconds between requests, NOT requests/sec rate.
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

Kucoin supports [time_in_force](configuration.md#understand-order_time_in_force) with settings "GTC" (good till cancelled), "FOK" (full-or-cancel) and "IOC" (immediate-or-cancel) settings.

!!! Tip "Stoploss on Exchange"
    Kucoin supports `stoploss_on_exchange` and can use both stop-loss-market and stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.
    You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type of stoploss shall be used.

### Kucoin Blacklists

For Kucoin, it is suggested to add `"KCS/<STAKE>"` to your blacklist to avoid issues, unless you are willing to maintain enough extra `KCS` on the account or unless you're willing to disable using `KCS` for fees. 
Kucoin accounts may use `KCS` for fees, and if a trade happens to be on `KCS`, further trades may consume this position and make the initial `KCS` trade unsellable as the expected amount is not there anymore.

## HTX

!!! Tip "Stoploss on Exchange"
    HTX supports `stoploss_on_exchange` and uses `stop-limit` orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange.

## OKX

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

If you've registered with OKX on the host my.okx.com (OKX EAA)- you will need to use `"myokx"` as the exchange name.
Using the wrong exchange will result in the error "OKX Error 50119: API key doesn't exist" - as the 2 are separate entities.

!!! Warning
    OKX only provides 100 candles per api call. Therefore, the strategy will only have a pretty low amount of data available in backtesting mode.

!!! Warning "Futures"
    OKX Futures has the concept of "position mode" - which can be "Buy/Sell" or long/short (hedge mode).
    Freqtrade supports both modes (we recommend to use Buy/Sell mode) - but changing the mode mid-trading is not supported and will lead to exceptions and failures to place trades.
    OKX also only provides MARK candles for the past ~3 months. Backtesting futures prior to that date will therefore lead to slight deviations, as funding-fees cannot be calculated correctly without this data.

## Gate.io

!!! Tip "Stoploss on Exchange"
    Gate.io supports `stoploss_on_exchange` and uses `stop-loss-limit` orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange.

Gate.io supports [time_in_force](configuration.md#understand-order_time_in_force) with settings "GTC" (good till cancelled), and "IOC" (immediate-or-cancel) settings.

Gate.io allows the use of `POINT` to pay for fees. As this is not a tradable currency (no regular market available), automatic fee calculations will fail (and default to a fee of 0).
The configuration parameter `exchange.unknown_fee_rate` can be used to specify the exchange rate between Point and the stake currency. Obviously, changing the stake-currency will also require changes to this value.

Gate API keys require the following permissions on top of the market type you want to trade:

* "Spot Trade" _or_ "Perpetual Futures" (Read and Write) (either select both, or the one matching the market you want to trade)
* "Wallet" (read only)
* "Account" (read only)

Without these permissions, the bot will not start correctly and show errors like "permission missing".

## Bybit

!!! Tip "Stoploss on Exchange"
    Bybit (futures only) supports `stoploss_on_exchange` and uses `stop-loss-limit` orders. It provides great advantages, so we recommend to benefit from it by enabling stoploss on exchange.
    On futures, Bybit supports both `stop-limit` as well as `stop-market` orders. You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type to use.

Bybit supports [time_in_force](configuration.md#understand-order_time_in_force) with settings "GTC" (good till cancelled), "FOK" (full-or-cancel), "IOC" (immediate-or-cancel) and "PO" (Post only) settings.

!!! Warning "Unified accounts"
    Freqtrade assumes accounts to be dedicated to the bot.
    We therefore recommend the usage of one subaccount per bot. This is especially important when using unified accounts.  
    Other configurations (multiple bots on one account, manual non-bot trades on the bot account) are not supported and may lead to unexpected behavior.

### Bybit Futures

Futures trading on bybit is supported for isolated futures mode.

On startup, freqtrade will set the position mode to "One-way Mode" for the whole (sub)account. This avoids making this call over and over again (slowing down bot operations), but means that manual changes to this setting may result in exceptions and errors.

As bybit doesn't provide funding rate history, the dry-run calculation is used for live trades as well.

API Keys for live futures trading must have the following permissions:

* Read-write
* Contract - Orders
* Contract - Positions

We do strongly recommend to limit all API keys to the IP you're going to use it from.


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

## Bitget

Bitget requires a passphrase for each api key, you will therefore need to add this key into the configuration so your exchange section looks as follows:

```json
"exchange": {
    "name": "bitget",
    "key": "your_exchange_key",
    "secret": "your_exchange_secret",
    "password": "your_exchange_api_key_password",
    // ...
}
```

Bitget supports [time_in_force](configuration.md#understand-order_time_in_force) with settings "GTC" (good till cancelled), "FOK" (full-or-cancel), "IOC" (immediate-or-cancel) and "PO" (Post only) settings.

!!! Tip "Stoploss on Exchange"
    Bitget supports `stoploss_on_exchange` and can use both stop-loss-market and stop-loss-limit orders. It provides great advantages, so we recommend to benefit from it.
    You can use either `"limit"` or `"market"` in the `order_types.stoploss` configuration setting to decide which type of stoploss shall be used.

### Bitget Futures

Futures trading on bitget is supported for isolated futures mode.

On startup, freqtrade will set the position mode to "One-way Mode" for the whole (sub)account. This avoids making this call over and over again (slowing down bot operations), but means that manual changes to this setting may result in exceptions and errors.

## Hyperliquid

!!! Tip "Stoploss on Exchange"
    Hyperliquid supports `stoploss_on_exchange` and uses `stop-loss-limit` orders. It provides great advantages, so we recommend to benefit from it.

Hyperliquid is a Decentralized Exchange (DEX). Decentralized exchanges work a bit different compared to normal exchanges. Instead of authenticating private API calls using an API key, private API calls need to be signed with the private key of your wallet (We recommend using an api Wallet for this, generated either on Hyperliquid or in your wallet of choice).
This needs to be configured like this:

```json
"exchange": {
    "name": "hyperliquid",
    "walletAddress": "your_eth_wallet_address",  // This should NOT be your API Wallet Address!
    "privateKey": "your_api_private_key",
    // ...
}
```

* walletAddress in hex format: `0x<40 hex characters>` - Can be easily copied from your wallet - and should be your main wallet address, not your API Wallet Address.
* privateKey in hex format: `0x<64 hex characters>` - Use the key the API Wallet shows on creation.

Hyperliquid handles deposits and withdrawals on the Arbitrum One chain, a Layer 2 scaling solution built on top of Ethereum. Hyperliquid uses USDC as quote / collateral. The process of depositing USDC on Hyperliquid requires a couple of steps, see [how to start trading](https://hyperliquid.gitbook.io/hyperliquid-docs/onboarding/how-to-start-trading) for details on what steps are needed.

!!! Note "Hyperliquid general usage Notes"
    Hyperliquid does not support market orders, however ccxt will simulate market orders by placing limit orders with a maximum slippage of 5%.  
    Unfortunately, hyperliquid only offers 5000 historic candles, so backtesting will either need to build candles historically (by waiting and downloading the data incrementally over time) - or will be limited to the last 5000 candles.

!!! Info "Some general best practices (non exhaustive)"
    * Beware of supply chain attacks, like pip package poisoning etcetera. Whenever you use your private key, make sure your environment is safe.
    * Don't use your actual wallet private key for trading. Use the Hyperliquid [API generator](https://app.hyperliquid.xyz/API) to create a separate API wallet.
    * Don't store your actual wallet private key on the server you use for freqtrade. Use the API wallet private key instead. This key won't allow withdrawals, only trading.
    * Always keep your mnemonic phrase and private key private.
    * Don't use the same mnemonic as the one you had to backup when initializing a hardware wallet, using the same mnemonic basically deletes the security of your hardware wallet.
    * Create a different software wallet, only transfer the funds you want to trade with to that wallet, and use that wallet to trade on Hyperliquid.
    * If you have funds you don't want to use for trading (after making a profit for example), transfer them back to your hardware wallet.

### Hyperliquid Vault / Subaccount

Hyperliquid allows you to create either a vault or a subaccount.  
To use these with Freqtrade, you will need to use the following configuration pattern:

``` json
"exchange": {
    "name": "hyperliquid",
    "walletAddress": "your_master_wallet_address", // Your master wallet address (not the API wallet address and not the vault/subaccount address).
    "privateKey": "your_api_private_key", // API wallet private key (see https://app.hyperliquid.xyz/API). You'll only need the private key.
    "ccxt_config": {
        "options": {
            "vaultAddress": "your_vault_address", // Optional, only if you want to use a vault ...
            "subAccountAddress": "your_subaccount_address" // OR optional, only if you want to use a subaccount
        }
    },
    // ...
}
```

Your balance and trades will now be used from your vault / subaccount - and no longer from your main account.

!!! Note
    You can only use either a vault or a subaccount - not both at the same time.

### Historic Hyperliquid data

The Hyperliquid API does not provide historic data beyond the single call to fetch current data, so downloading data is not possible, as the downloaded data would not constitute proper historic data.

### HIP-3 DEXes

Hyperliquid supports HIP-3 decentralized exchanges (DEXes), which are independent exchanges built on top of the Hyperliquid infrastructure.
These DEXes operate similarly to the main Hyperliquid exchange but are community-created and managed.

To trade on HIP-3 DEXes with Freqtrade, you need to add them to your configuration using the `hip3_dexes` parameter:

```json
"exchange": {
    "name": "hyperliquid",
    "walletAddress": "your_master_wallet_address",
    "privateKey": "your_api_private_key",
    "hip3_dexes": ["dex_name_1", "dex_name_2"]
}
```

Replace `"dex_name_1"` and `"dex_name_2"` with the actual names of the HIP-3 DEXes you want to trade on (e.g. `vntl` and `xyz`).

!!! Warning "Performance and Rate Limit Impact"
    Each HIP-3 DEX you add significantly impacts bot performance and rate limits.

    * **Additional API Calls**: For each HIP-3 DEX configured, Freqtrade needs to make additional API calls.
    * **Rate Limit Pressure**: Additional API calls contribute to Hyperliquid's strict rate limits. With multiple DEXes, you may hit rate limits faster, or rather, slow down bot operations due to enforced delays.

    Please only add HIP-3 DEXes that you actively trade on. Monitor your logs for rate limit warnings or signs of slowed operations, and adjust your configuration accordingly.  
    Different HIP-3 DEXes may also use different quote currencies - so make sure to only add DEXes that are compatible with your stake currency to avoid unnecessary delays.

!!! Note
    HIP-3 DEXes share the same wallet and free amount of collateral as your main Hyperliquid account. Trades on different DEXes will affect your overall account balance and margin.

## Bitvavo

If your account is required to use an operatorId, you can set it in the configuration file as follows:

``` json
"exchange": {
        "name": "bitvavo",
        "key": "",
        "secret": "",
        "ccxt_config": {
            "options": {
                "operatorId": "123567"
            }
        },
   }
```

Bitvavo expects the `operatorId` to be an integer.

## All exchanges

Should you experience constant errors with Nonce (like `InvalidNonce`), it is best to regenerate the API keys. Resetting Nonce is difficult and it's usually easier to regenerate the API keys.

## Random notes for other exchanges

* The Ocean (exchange id: `theocean`) exchange uses Web3 functionality and requires `web3` python package to be installed:

```shell
pip3 install web3
```

### Getting latest price / Incomplete candles

Most exchanges return current incomplete candle via their OHLCV/klines API interface.
By default, Freqtrade assumes that incomplete candle is fetched from the exchange and removes the last candle assuming it's the incomplete candle.

Whether your exchange returns incomplete candles or not can be checked using [the helper script](developer.md#incomplete-candles) from the Contributor documentation.

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
    Using `_ft_has_params` overrides may lead to unexpected behavior, and may even break your bot. 
    We will not be able to provide support for issues caused by custom settings in `_ft_has_params`.
