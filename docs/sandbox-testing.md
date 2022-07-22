# Sandbox API testing

Some exchanges provide sandboxes or testbeds for risk-free testing, while running the bot against a real exchange.
With some configuration, freqtrade (in combination with ccxt) provides access to these.

This document is an overview to configure Freqtrade to be used with sandboxes.
This can be useful to developers and trader alike.

!!! Warning
    Sandboxes usually have very low volume, and either a very wide spread, or no orders available at all.
    Therefore, sandboxes will usually not do a good job of showing you how a strategy would work in real trading.

## Exchanges known to have a sandbox / testnet

* [binance](https://testnet.binance.vision/)
* [coinbasepro](https://public.sandbox.pro.coinbase.com)
* [gemini](https://exchange.sandbox.gemini.com/)
* [huobipro](https://www.testnet.huobi.pro/)
* [kucoin](https://sandbox.kucoin.com/)
* [phemex](https://testnet.phemex.com/) 

!!! Note
    We did not test correct functioning of all of the above testnets. Please report your experiences with each sandbox.

---

## Configure a Sandbox account

When testing your API connectivity, make sure to use the appropriate sandbox / testnet URL.

In general, you should follow these steps to enable an exchange's sandbox:

* Figure out if an exchange has a sandbox (most likely by using google or the exchange's support documents)
* Create a sandbox account (often the sandbox-account requires separate registration)
* [Add some test assets to account](#add-test-funds)
* Create API keys

### Add test funds

Usually, sandbox exchanges allow depositing funds directly via web-interface.
You should make sure to have a realistic amount of funds available to your test-account, so results are representable of your real account funds.

!!! Warning
    Test exchanges will **NEVER** require your real credit card or banking details!

## Configure freqtrade to use a exchange's sandbox

### Sandbox URLs

Freqtrade makes use of CCXT which in turn provides a list of URLs to Freqtrade.
These include `['test']` and `['api']`.

* `[Test]` if available will point to an Exchanges sandbox.
* `[Api]` normally used, and resolves to live API target on the exchange.

To make use of sandbox / test add "sandbox": true, to your config.json

```json
  "exchange": {
        "name": "coinbasepro",
        "sandbox": true,
        "key": "5wowfxemogxeowo;heiohgmd",
        "secret": "/ZMH1P62rCVmwefewrgcewX8nh4gob+lywxfwfxwwfxwfNsH1ySgvWCUR/w==",
        "password": "1bkjfkhfhfu6sr",
        "outdated_offset": 5
        "pair_whitelist": [
            "BTC/USD"
        ]
  },
  "datadir": "user_data/data/coinbasepro_sandbox"
```

Also the following information:

* api-key (created for the sandbox webpage)
* api-secret (noted earlier)
* password (the passphrase - noted earlier)

!!! Tip "Different data directory"
    We also recommend to set `datadir` to something identifying downloaded data as sandbox data, to avoid having sandbox data mixed with data from the real exchange.
    This can be done by adding the `"datadir"` key to the configuration.
    Now, whenever you use this configuration, your data directory will be set to this directory.

---

## You should now be ready to test your sandbox

Ensure Freqtrade logs show the sandbox URL, and trades made are shown in sandbox. Also make sure to select a pair which shows at least some decent value (which very often is BTC/<somestablecoin>).

## Common problems with sandbox exchanges

Sandbox exchange instances often have very low volume, which can cause some problems which usually are not seen on a real exchange instance.

### Old Candles problem

Since Sandboxes often have low volume, candles can be quite old and show no volume.
To disable the error "Outdated history for pair ...", best increase the parameter `"outdated_offset"` to a number that seems realistic for the sandbox you're using.

### Unfilled orders

Sandboxes often have very low volumes - which means that many trades can go unfilled, or can go unfilled for a very long time.

To mitigate this, you can try to match the first order on the opposite orderbook side using the following configuration:

``` jsonc
  "order_types": {
    "entry": "limit",
    "exit": "limit"
    // ...
  },
  "entry_pricing": {
    "price_side": "other",
    // ...
  },
  "exit_pricing":{
    "price_side": "other",
    // ...
  },
  ```

  The configuration is similar to the suggested configuration for market orders - however by using limit-orders you can avoid moving the price too much, and you can set the worst price you might get.
