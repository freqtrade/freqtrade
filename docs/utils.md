# Utility Subcommands

Besides the Live-Trade and Dry-Run run modes, the `backtesting`, `edge` and `hyperopt` optimization subcommands, and the `download-data` subcommand which prepares historical data, the bot contains a number of utility subcommands. They are described in this section.

## Create userdir

Creates the directory structure to hold your files for freqtrade.
Will also create strategy and hyperopt examples for you to get started.
Can be used multiple times - using `--reset` will reset the sample strategy and hyperopt files to their default state. 

```
usage: freqtrade create-userdir [-h] [--userdir PATH] [--reset]

optional arguments:
  -h, --help            show this help message and exit
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
  --reset               Reset sample files to their original state.
```

!!! Warning
    Using `--reset` may result in loss of data, since this will overwrite all sample files without asking again.

```
├── backtest_results
├── data
├── hyperopt_results
├── hyperopts
│   ├── sample_hyperopt_advanced.py
│   ├── sample_hyperopt_loss.py
│   └── sample_hyperopt.py
├── notebooks
│   └── strategy_analysis_example.ipynb
├── plot
└── strategies
    └── sample_strategy.py
```

## Create new strategy

Creates a new strategy from a template similar to SampleStrategy.
The file will be named inline with your class name, and will not overwrite existing files.

Results will be located in `user_data/strategies/<strategyclassname>.py`.

### Sample usage of new-strategy

```bash
freqtrade new-strategy --strategy AwesomeStrategy
```

With custom user directory

```bash
freqtrade new-strategy --userdir ~/.freqtrade/ --strategy AwesomeStrategy
```

### new-strategy complete options

``` output
usage: freqtrade new-strategy [-h] [--userdir PATH] [-s NAME]
                              [--template {full,minimal}]

optional arguments:
  -h, --help            show this help message and exit
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
  -s NAME, --strategy NAME
                        Specify strategy class name which will be used by the
                        bot.
  --template {full,minimal}
                        Use a template which is either `minimal` or `full`
                        (containing multiple sample indicators). Default:
                        `full`.

```

## Create new hyperopt

Creates a new hyperopt from a template similar to SampleHyperopt.
The file will be named inline with your class name, and will not overwrite existing files.

Results will be located in `user_data/hyperopts/<classname>.py`.

### Sample usage of new-hyperopt

```bash
freqtrade new-hyperopt --hyperopt AwesomeHyperopt
```

With custom user directory

```bash
freqtrade new-hyperopt --userdir ~/.freqtrade/ --hyperopt AwesomeHyperopt
```

### new-hyperopt complete options

``` output
usage: freqtrade new-hyperopt [-h] [--userdir PATH] [--hyperopt NAME]
                              [--template {full,minimal}]

optional arguments:
  -h, --help            show this help message and exit
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
  --hyperopt NAME       Specify hyperopt class name which will be used by the
                        bot.
  --template {full,minimal}
                        Use a template which is either `minimal` or `full`
                        (containing multiple sample indicators). Default:
                        `full`.
```

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

## List pairs/list markets

The `list-pairs` and `list-markets` subcommands allow to see the pairs/markets available on exchange.

Pairs are markets with the '/' character between the base currency part and the quote currency part in the market symbol.
For example, in the 'ETH/BTC' pair 'ETH' is the base currency, while 'BTC' is the quote currency.

For pairs traded by Freqtrade the pair quote currency is defined by the value of the `stake_currency` configuration setting.

You can print info about any pair/market with these subcommands - and you can filter output by quote-currency using `--quote BTC`, or by base-currency using `--base ETH` options correspondingly.

These subcommands have same usage and same set of available options:

```
usage: freqtrade list-markets [-h] [--exchange EXCHANGE] [--print-list]
                              [--print-json] [-1] [--print-csv]
                              [--base BASE_CURRENCY [BASE_CURRENCY ...]]
                              [--quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]]
                              [-a]

usage: freqtrade list-pairs [-h] [--exchange EXCHANGE] [--print-list]
                            [--print-json] [-1] [--print-csv]
                            [--base BASE_CURRENCY [BASE_CURRENCY ...]]
                            [--quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]] [-a]

optional arguments:
  -h, --help            show this help message and exit
  --exchange EXCHANGE   Exchange name (default: `bittrex`). Only valid if no
                        config is provided.
  --print-list          Print list of pairs or market symbols. By default data
                        is printed in the tabular format.
  --print-json          Print list of pairs or market symbols in JSON format.
  -1, --one-column      Print output in one column.
  --print-csv           Print exchange pair or market data in the csv format.
  --base BASE_CURRENCY [BASE_CURRENCY ...]
                        Specify base currency(-ies). Space-separated list.
  --quote QUOTE_CURRENCY [QUOTE_CURRENCY ...]
                        Specify quote currency(-ies). Space-separated list.
  -a, --all             Print all pairs or market symbols. By default only
                        active ones are shown.
```

By default, only active pairs/markets are shown. Active pairs/markets are those that can currently be traded
on the exchange. The see the list of all pairs/markets (not only the active ones), use the `-a`/`-all` option.

Pairs/markets are sorted by its symbol string in the printed output.

### Examples

* Print the list of active pairs with quote currency USD on exchange, specified in the default
configuration file (i.e. pairs on the "Bittrex" exchange) in JSON format:

```
$ freqtrade list-pairs --quote USD --print-json
```

* Print the list of all pairs on the exchange, specified in the `config_binance.json` configuration file
(i.e. on the "Binance" exchange) with base currencies BTC or ETH and quote currencies USDT or USD, as the
human-readable list with summary:

```
$ freqtrade -c config_binance.json list-pairs --all --base BTC ETH --quote USDT USD --print-list
```

* Print all markets on exchange "Kraken", in the tabular format:

```
$ freqtrade list-markets --exchange kraken --all
```
