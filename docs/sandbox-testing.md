# Sandbox API testing
Where an exchange provides a sandbox for risk-free integration, or end-to-end, testing CCXT provides access to these.

This document is a *light overview of configuring Freqtrade and GDAX sandbox.
This can be useful to developers and trader alike as Freqtrade is quite customisable.

When testing your API connectivity, make sure to use the following URLs.
***Website**
https://public.sandbox.gdax.com
***REST API**
https://api-public.sandbox.gdax.com

---
# Configure a Sandbox account on Gdax
Aim of this document section
- An sanbox account
- create 2FA (needed to create an API)
- Add test 50BTC to account 
- Create :
- - API-KEY
- - API-Secret
- - API Password

## Acccount

This link will redirect to the sandbox main page to login / create account dialogues:
https://public.sandbox.pro.coinbase.com/orders/

After registration and Email confimation you wil be redirected into your sanbox account.  It is easy to verify you're in sandbox by checking the URL bar.
> https://public.sandbox.pro.coinbase.com/

## Enable 2Fa (a prerequisite to creating sandbox API Keys)
From within sand box site select your profile, top right.
>Or as a direct link: https://public.sandbox.pro.coinbase.com/profile

From the menu panel to the left of the screen select 
> Security: "*View or Update*"

In the new site select "enable authenticator" as typical google Authenticator. 
- open Google Authenticator on your phone
- scan barcode 
- enter your generated 2fa 

## Enable API Access 
From within sandbox select profile>api>create api-keys
>or as a direct link: https://public.sandbox.pro.coinbase.com/profile/api

Click on "create one" and ensure  **view** and **trade**  are "checked" and sumbit your 2Fa
- **Copy and paste the Passphase** into a notepade this will be needed later
- **Copy and paste the API Secret** popup into a notepad this will needed later
- **Copy and paste the API Key** into a notepad this will needed later

## Add 50 BTC test funds
To add funds, use the web interface deposit and withdraw buttons.


To begin select 'Wallets' from the top menu.
> Or as a direct link: https://public.sandbox.pro.coinbase.com/wallets

- Deposits (bottom left of screen)
- - Deposit Funds Bitcoin 
- - - Coinbase BTC Wallet 
- - - - Max (50 BTC) 
- - - - - Deposit

*This process may be repeated for other currencies, ETH as example*
---
# Configure Freqtrade to use Gax Sandbox 

The aim of this document section
 - Enable sandbox URLs in Freqtrade
 - Configure API 
 - - secret
 - - key
 - - passphrase
 
## Sandbox URLs
Freqtrade makes use of CCXT which in turn provides a list of URLs to Freqtrade. 
These include `['test']` and `['api']`. 
- `[Test]` if available will point to an Exchanges sandbox. 
- `[Api]` normally used, and resolves to live API target on the exchange 

To make use of sandbox / test add "sandbox": true, to your config.json
```
  "exchange": {
        "name": "gdax",
        "sandbox": true,
        "key": "5wowfxemogxeowo;heiohgmd",
        "secret": "/ZMH1P62rCVmwefewrgcewX8nh4gob+lywxfwfxwwfxwfNsH1ySgvWCUR/w==",
        "password": "1bkjfkhfhfu6sr",
        "pair_whitelist": [
            "BTC/USD"
```
Also insert your 
- api-key  (noted earlier)
- api-secret (noted earlier)
- password (the passphrase - noted earlier)

---
## You should now be ready to test your sandbox!
Ensure Freqtrade logs show the sandbox URL, and trades made are shown in sandbox.
** Typically the  BTC/USD has the most activity in sandbox to test against. 

## GDAX - Old Candles problem
It is my experience that GDAX sandbox candles may be 20+- minutes out of date. This can cause trades to fail as one of Freqtrades safety checks 

To disable this check, edit: 
>strategy/interface.py
Look for the following section:
```
      # Check if dataframe is out of date
        signal_date = arrow.get(latest['date'])
        interval_minutes = constants.TICKER_INTERVAL_MINUTES[interval]
        if signal_date < (arrow.utcnow().shift(minutes=-(interval_minutes * 2 + 5))):
            logger.warning(
                'Outdated history for pair %s. Last tick is %s minutes old',
                pair,
                (arrow.utcnow() - signal_date).seconds // 60
            )
            return False, False
```

You could Hash out the entire check as follows:
```
       # # Check if dataframe is out of date
        # signal_date = arrow.get(latest['date'])
        # interval_minutes = constants.TICKER_INTERVAL_MINUTES[interval]
        # if signal_date < (arrow.utcnow().shift(minutes=-(interval_minutes * 2 + 5))):
        #     logger.warning(
        #         'Outdated history for pair %s. Last tick is %s minutes old',
        #         pair,
        #         (arrow.utcnow() - signal_date).seconds // 60
        #     )
        #     return False, False
 ```
 
 Or inrease the timeout to offer a level of protection/alignment of this test to freqtrade in live.
 
 As example, to allow an additional 30 minutes. "(interval_minutes * 2 + 5 + 30)"
 ```
      # Check if dataframe is out of date
        signal_date = arrow.get(latest['date'])
        interval_minutes = constants.TICKER_INTERVAL_MINUTES[interval]
        if signal_date < (arrow.utcnow().shift(minutes=-(interval_minutes * 2 + 5 + 30))):
            logger.warning(
                'Outdated history for pair %s. Last tick is %s minutes old',
                pair,
                (arrow.utcnow() - signal_date).seconds // 60
            )
            return False, False
```