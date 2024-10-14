# Freqtrade Strategies 101: A Quick Start for Strategy Development

For the purposes of this quick start, we are assuming you are familiar with the basics of trading, and have read the 
[Freqtrade basics](bot-basics.md) page.

## Required Knowledge

A strategy in Freqtrade is a Python class that defines the logic for buying and selling cryptocurrency `assets`.

Assets are defined as `pairs`, which represent the `coin` and the `stake`. The coin is the asset you are trading using another currency as the stake.

Data is supplied by the exchange in the form of `candles`, which are made up of a six values: `date`, `open`, `high`, `low`, `close` and `volume`.

`Technical analysis` functions analyse the candle data using various computational and statistical formulae, and produce secondary values called `indicators`.

Indicators are analysed on the asset pair candles to generate `signals`.

Signals are turned into `orders` on a cryptocurrency `exchange`, i.e. `trades`.

We use the terms `entry` and `exit` instead of `buying` and `selling` because Freqtrade supports both `long` and `short` trades.

- **long**: You buy the coin based on a stake, e.g. buying the coin BTC using USDT as your stake, and you make a profit by selling the coin at a higher rate than you paid for. Profits are made in long trades by the coin value going up versus the stake.
- **short**: You borrow capital from from the exchange in the form of the coin, and you pay back the stake value of the coin later. Profits are made in short trades by the coin value going down versus the stake (you pay the loan off at a lower rate).

Whilst Freqtrade supports spot and futures markets for certain exchanges, for simplicity we will focus on spot (long) trades only.

## Structure of a Basic Strategy

### Main dataframe

Freqtrade strategies use a tabular data structure with rows and columns known as a `dataframe` to generate signals to enter and exit trades.

Each pair in your configured pairlist has its own dataframe. Dataframes are indexed by the `date` column, e.g. `2024-06-31 12:00`.

The next 5 columns represent the `open`, `high`, `low`, `close` and `volume` (OHLCV) data.

### Populate indicator values

The `populate_indicators` function adds columns to the dataframe that represent the technical analysis indicator values.

Examples of common indicators include Relative Strength Index, Bollinger Bands, Money Flow Index, Moving Average, and Average True Range.

Columns are added to the dataframe by calling technical analysis functions, e.g. ta-lib's RSI function `ta.RSI()`, and assigning them to a column name, e.g. `rsi`

```python
dataframe['rsi'] = ta.RSI(dataframe)
```

??? Hint "Technical Analysis libraries"
    Different libraries work in different ways to generate indicator values. Please check the documentation of each library to understand
    how to integrate it into your strategy. You can also check the [Freqtrade example strategies](https://github.com/freqtrade/freqtrade-strategies) to give you ideas.

### Populate entry signals

The `populate_entry_trend` function defines conditions for an entry signal.

The dataframe column `enter_long` is added to the dataframe, and when a value of `1` is in this column, Freqtrade sees an entry signal.

??? Hint "Shorting"
    To enter short trades, use the `enter_short` column.

### Populate exit signals

The `populate_exit_trend` function defines conditions for an exit signal.

The dataframe column `exit_long` is added to the dataframe, and when a value of `1` is in this column, Freqtrade sees an exit signal.

??? Hint "Shorting"
    To exit short trades, use the `exit_short` column.

## A simple strategy

Here is a minimal example of a Freqtrade strategy:

```python
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class MyStrategy(IStrategy):

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # generate values for technical analysis indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # generate entry signals based on indicator values
        dataframe.loc[
            (dataframe['rsi'] < 30),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # generate exit signals based on indicator values
        dataframe.loc[
            (dataframe['rsi'] > 70),
            'exit_long'] = 1

        return dataframe
```

## Making trades

When a signal is found (a `1` in an entry or exit column), Freqtrade will attempt to make an order, i.e. a `trade` or `position`.

The number of concurrent trades that can be opened is defined by the `max_open_trades` [configuration](configuration.md) option.

However, there can be a range of scenarios where generating a signal does not always create a trade order. These include:

- not enough remaining stake to buy an asset, or funds in your wallet to sell an asset (including any fees)
- not enough open slots for a new trade to be opened
- there is already an open trade for a pair (Freqtrade cannot stack positions - however it can [adjust existing positions](strategy-callbacks.md#adjust-trade-position))
- if an entry and exit signal is present on the same candle, they cancel each other out and no order will be raised
- the strategy actively rejects the trade order due to logic you specify by using one of the relevant [entry](strategy-callbacks.md#trade-entry-buy-order-confirmation) or [exit](strategy-callbacks.md#trade-exit-sell-order-confirmation) callbacks

Read through the [strategy customization](strategy-customization.md) documentation for more details.

## Backtesting and forward testing

Strategy development can be a long and frustrating process, as turning our human "gut instincts" into a working computer-controlled
("algo") strategy is not always straightforward.

Therefore a strategy should be tested to verify that it is going to work as intended.

Freqtrade has two testing modes:

- **backtesting**: using historical data that you [download from an exchange](data-download.md), backtesting is a quick way to assess performance of a strategy. However, it can be very easy to distort results so a strategy will look a lot more profitable than it really is. Check the [backtesting documentation](backtesting.md) for more information. 
- **dry run**: often referred to as `forward testing`, dry runs use real time data from the exchange. However, any signals that would result in trades are tracked as normal by Freqtrade, but do not have any trades opened on the exchange itself. Forward testing runs in real time, so whilst it takes longer to get results it is a much more reliable indicator of **potential** performance then backtesting. 

Dry runs are enabled by setting `dry_run` to true in your [configuration](configuration.md#using-dry-run-mode).

!!! Warning "Backtests can be very inaccurate"
    There are many reasons why backtest results will not match reality. Please check the [backtesting assumptions](backtesting.md#assumptions-made-by-backtesting) and [common strategy mistakes](strategy-customization.md#common-mistakes-when-developing-strategies) documentation.

!!! Warning "Public strategies can have significant issues"
    There are many websites listing strategies with impressive backtest results. Do not assume these results are achieveable or realistic.

??? Hint "Useful commands"
    Freqtrade includes two useful commands to check for basic flaws in strategies: [lookahead-analysis](lookahead-analysis.md) and [recursive-analysis](recursive-analysis.md).

!!! Note "Always dry run first!"
    Always dry run your strategy after backtesting it to see if backtesting and dry run results match, giving you confidence that things are operating correctly.

## Controlling or monitoring a running bot

Once your bot is running in dry or live mode, Freqtrade has four mechanisms to control or monitor a running bot:

- **[FreqUI](freq-ui.md)**: The easiest to get started with, FreqUI is a web interface to see and control current activity of your bot.
- **[Telegram](telegram-usage.md)**: On mobile devices, Telegram integration is available to get alerts about your bot activity and to control certain aspects.
- **[FTUI](https://github.com/freqtrade/ftui)**: FTUI is a terminal (command line) interface to Freqtrade, and allows monitoring of a running bot only.
- **[REST API](rest-api.md)**: The REST API allows programmers to develop their own tools to interact with a Freqtrade bot.

## Conclusion

Developing a strategy in Freqtrade involves defining entry and exit signals based on technical indicators. By following the structure and methods outlined above, you can create and test your own trading strategies.

Common questions and answers are available on our [FAQ](faq.md).

To continue, refer to the more in-depth [Freqtrade strategy customization documentation](strategy-customization.md).
