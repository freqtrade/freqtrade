# Telegram usage

This page explains how to command your bot with Telegram.

## Prerequisite
To control your bot with Telegram, you need first to
[set up a Telegram bot](installation.md)
and add your Telegram API keys into your config file.

## Telegram commands
Per default, the Telegram bot shows predefined commands. Some commands
are only available by sending them to the bot. The table below list the
official commands. You can ask at any moment for help with `/help`.

|  Command | Default | Description |
|----------|---------|-------------|
| `/start` | | Starts the trader
| `/stop` | | Stops the trader
| `/stopbuy` | | Stops the trader from opening new trades. Gracefully closes open trades according to their rules.
| `/reload_conf` | | Reloads the configuration file
| `/status` | | Lists all open trades
| `/status table` | | List all open trades in a table format
| `/count` | | Displays number of trades used and available
| `/profit` | | Display a summary of your profit/loss from close trades and some stats about your performance
| `/forcesell <trade_id>` | | Instantly sells the given trade  (Ignoring `minimum_roi`).
| `/forcesell all` | | Instantly sells all open trades (Ignoring `minimum_roi`).
| `/forcebuy <pair> [rate]` | | Instantly buys the given pair. Rate is optional. (`forcebuy_enable` must be set to True)
| `/performance` | | Show performance of each finished trade grouped by pair
| `/balance` | | Show account balance per currency
| `/daily <n>` | 7 | Shows profit or loss per day, over the last n days
| `/whitelist` | | Show the current whitelist
| `/blacklist` | | Show the current blacklist
| `/help` | | Show help message
| `/version` | | Show version

## Telegram commands in action

Below, example of Telegram message you will receive for each command.

### /start

> **Status:** `running`

### /stop

> `Stopping trader ...`
> **Status:** `stopped`

### /stopbuy

> **status:** `Setting max_open_trades to 0. Run /reload_conf to reset.`

Prevents the bot from opening new trades by temporarily setting "max_open_trades" to 0. Open trades will be handled via their regular rules (ROI / Sell-signal, stoploss, ...).

After this, give the bot time to close off open trades (can be checked via `/status table`).
Once all positions are sold, run `/stop` to completely stop the bot.

`/reload_conf` resets "max_open_trades" to the value set in the configuration and resets this command. 

!!! warning
   The stop-buy signal is ONLY active while the bot is running, and is not persisted anyway, so restarting the bot will cause this to reset.

### /status

For each open trade, the bot will send you the following message.

> **Trade ID:** `123`
> **Current Pair:** CVC/BTC
> **Open Since:** `1 days ago`
> **Amount:** `26.64180098`
> **Open Rate:** `0.00007489`
> **Close Rate:** `None`
> **Current Rate:** `0.00007489`
> **Close Profit:** `None`
> **Current Profit:** `12.95%`
> **Open Order:** `None`

### /status table

Return the status of all open trades in a table format.
```
   ID  Pair      Since    Profit
----  --------  -------  --------
  67  SC/BTC    1 d      13.33%
 123  CVC/BTC   1 h      12.95%
```

### /count

Return the number of trades used and available.
```
current    max
---------  -----
     2     10
```

### /profit

Return a summary of your profit/loss and performance.

> **ROI:** Close trades
>   ∙ `0.00485701 BTC (258.45%)`
>   ∙ `62.968 USD`
> **ROI:** All trades
>   ∙ `0.00255280 BTC (143.43%)`
>   ∙ `33.095 EUR`
>
> **Total Trade Count:** `138`
> **First Trade opened:** `3 days ago`
> **Latest Trade opened:** `2 minutes ago`
> **Avg. Duration:** `2:33:45`
> **Best Performing:** `PAY/BTC: 50.23%`

### /forcesell <trade_id>

> **BITTREX:** Selling BTC/LTC with limit `0.01650000 (profit: ~-4.07%, -0.00008168)`

### /forcebuy <pair>

> **BITTREX**: Buying ETH/BTC with limit `0.03400000` (`1.000000 ETH`, `225.290 USD`)

Note that for this to work, `forcebuy_enable` needs to be set to true.

[More details](configuration.md/#understand-forcebuy_enable)

### /performance

Return the performance of each crypto-currency the bot has sold.
> Performance:
> 1. `RCN/BTC 57.77%`
> 2. `PAY/BTC 56.91%`
> 3. `VIB/BTC 47.07%`
> 4. `SALT/BTC 30.24%`
> 5. `STORJ/BTC 27.24%`
> ...

### /balance

Return the balance of all crypto-currency your have on the exchange.

> **Currency:** BTC
> **Available:** 3.05890234
> **Balance:** 3.05890234
> **Pending:** 0.0

> **Currency:** CVC
> **Available:** 86.64180098
> **Balance:** 86.64180098
> **Pending:** 0.0

### /daily <n>

Per default `/daily` will return the 7 last days.
The example below if for `/daily 3`:

> **Daily Profit over the last 3 days:**
```
Day         Profit BTC      Profit USD
----------  --------------  ------------
2018-01-03  0.00224175 BTC  29,142 USD
2018-01-02  0.00033131 BTC   4,307 USD
2018-01-01  0.00269130 BTC  34.986 USD
```

### /whitelist

Shows the current whitelist

> Using whitelist `StaticPairList` with 22 pairs  
> `IOTA/BTC, NEO/BTC, TRX/BTC, VET/BTC, ADA/BTC, ETC/BTC, NCASH/BTC, DASH/BTC, XRP/BTC, XVG/BTC, EOS/BTC, LTC/BTC, OMG/BTC, BTG/BTC, LSK/BTC, ZEC/BTC, HOT/BTC, IOTX/BTC, XMR/BTC, AST/BTC, XLM/BTC, NANO/BTC`


### /blacklist

Shows the current blacklist

> Using blacklist `StaticPairList` with 2 pairs  
>`DODGE/BTC`, `HOT/BTC`.

### /version

> **Version:** `0.14.3`
