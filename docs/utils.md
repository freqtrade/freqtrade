# Utility Subcommands

Besides the Live-Trade and Dry-Run run modes, the `backtesting`, `edge` and `hyperopt` optimization subcommands, and the `download-data` subcommand which prepares historical data, the bot contains a number of utility subcommands. They are described in this section.

## List Exchanges

Use the `list-exchanges` subcommand to see the exchanges available for the bot.

```
usage: freqtrade list-exchanges [-h] [-1] [-a]

optional arguments:
  -h, --help        show this help message and exit
  -1, --one-column  Print output in one column.
  -a, --all         Print all exchanges known to the ccxt library.
```

* Example: see exchanges available for the bot:
```
$ freqtrade list-exchanges
Exchanges available for Freqtrade: _1btcxe, acx, allcoin, bequant, bibox, binance, binanceje, binanceus, bitbank, bitfinex, bitfinex2, bitkk, bitlish, bitmart, bittrex, bitz, bleutrade, btcalpha, btcmarkets, btcturk, buda, cex, cobinhood, coinbaseprime, coinbasepro, coinex, cointiger, coss, crex24, digifinex, dsx, dx, ethfinex, fcoin, fcoinjp, gateio, gdax, gemini, hitbtc2, huobipro, huobiru, idex, kkex, kraken, kucoin, kucoin2, kuna, lbank, mandala, mercado, oceanex, okcoincny, okcoinusd, okex, okex3, poloniex, rightbtc, theocean, tidebit, upbit, zb
```

* Example: see all exchanges supported by the ccxt library (including 'bad' ones, i.e. those that are known to not work with Freqtrade):
```
$ freqtrade list-exchanges -a
All exchanges supported by the ccxt library: _1btcxe, acx, adara, allcoin, anxpro, bcex, bequant, bibox, bigone, binance, binanceje, binanceus, bit2c, bitbank, bitbay, bitfinex, bitfinex2, bitflyer, bitforex, bithumb, bitkk, bitlish, bitmart, bitmex, bitso, bitstamp, bitstamp1, bittrex, bitz, bl3p, bleutrade, braziliex, btcalpha, btcbox, btcchina, btcmarkets, btctradeim, btctradeua, btcturk, buda, bxinth, cex, chilebit, cobinhood, coinbase, coinbaseprime, coinbasepro, coincheck, coinegg, coinex, coinexchange, coinfalcon, coinfloor, coingi, coinmarketcap, coinmate, coinone, coinspot, cointiger, coolcoin, coss, crex24, crypton, deribit, digifinex, dsx, dx, ethfinex, exmo, exx, fcoin, fcoinjp, flowbtc, foxbit, fybse, gateio, gdax, gemini, hitbtc, hitbtc2, huobipro, huobiru, ice3x, idex, independentreserve, indodax, itbit, kkex, kraken, kucoin, kucoin2, kuna, lakebtc, latoken, lbank, liquid, livecoin, luno, lykke, mandala, mercado, mixcoins, negociecoins, nova, oceanex, okcoincny, okcoinusd, okex, okex3, paymium, poloniex, rightbtc, southxchange, stronghold, surbitcoin, theocean, therock, tidebit, tidex, upbit, vaultoro, vbtc, virwox, xbtce, yobit, zaif, zb
```

## List Timeframes

Use the `list-timeframes` subcommand to see the list of ticker intervals (timeframes) available for the exchange.

```
usage: freqtrade list-timeframes [-h] [--exchange EXCHANGE] [-1]

optional arguments:
  -h, --help           show this help message and exit
  --exchange EXCHANGE  Exchange name (default: `bittrex`). Only valid if no
                       config is provided.
  -1, --one-column     Print output in one column.

```

* Example: see the timeframes for the 'binance' exchange, set in the configuration file:

```
$ freqtrade -c config_binance.json list-timeframes
...
Timeframes available for the exchange `binance`: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
```

* Example: enumerate exchanges available for Freqtrade and print timeframes supported by each of them:
```
$ for i in `freqtrade list-exchanges -1`; do freqtrade list-timeframes --exchange $i; done
```
