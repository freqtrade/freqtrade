# Freqtrade Strategies 101: A Quick Start for Strategy Development

For the purposes of this quick start, we are assuming you are familiar with the basics of trading.

## Required Knowledge

A strategy in Freqtrade is a Python class that defines the logic for buying and selling cryptocurrency `assets`.

Assets are defined as `pairs`, which represent the `coin` and the `stake`. The coin is the asset you are trading using another currency as the stake.

Data is supplied by the exchange in the form of `candles`, which are made up of a six values: `date`, `open`, `high`, `low`, `close` and `volume`.

`Technical analysis` functions analyse the candle data using various computational and statistical formulae, and produce secondary values called `indicators`.

Indicators are analysed on the asset pair candles to generate `signals`.

Signals are turned into `orders` on a cryptocurrency `exchange`, i.e. `trades`.

We use the terms `entry` and `exit` instead of `buying` and `selling` because Freqtrade supports both `long` and `short` trades.

- long:
  - You buy the coin based on a stake, e.g. buying the coin BTC using USDT as your stake, and you make a profit by selling the coin at a higher rate than you paid for. Profits are made in long trades by the coin value going up versus the stake.
- short
  - You borrow capital from from the exchange in the form of the coin, and you pay back the stake value of the coin later. Profits are made in short trades by the coin value going down versus the stake (you pay the loan off at a lower rate).

For simplicity, here we will focus on spot (long) trades only.

## Structure of a Basic Strategy

### Main dataframe

Freqtrade strategies use a tabular data structure with rows and columns known as a `dataframe` to generate signals to enter and exit trades.

Freqtrade dataframes are organised by pair, and are indexed by the `date` column, e.g. `2024-06-31 12:00`.

The next 5 columns represent the `open`, `high`, `low`, `close` and `volume` (OHLCV) data.

### Populate indicator values
The `populate_indicators` function adds columns to the dataframe that represent the technical analysis indicator values.

Examples of common indicators include Relative Strength Index, Bollinger Bands, Money Flow Index, Moving Average, and Average True Range.

Columns are added to the dataframe by calling technical analysis functions, e.g. ta-lib's RSI function `ta.RSI()`, and assigning them to a column name, e.g. `rsi`

```python
    dataframe['rsi'] = ta.RSI(dataframe)
```

### Populate entry signals
The `populate_entry_trend` function defines conditions for an entry signal.

The dataframe column `enter_long` is added to the dataframe, and when a value of `1` is in this column, Freqtrade sees an entry signal.

??? Hint "Shorting"
    When shorting, this column is called `enter_short`.

### Populate exit signals
The `populate_exit_trend` function defines conditions for an exit signal.

The dataframe column `exit_long` is added to the dataframe, and when a value of `1` is in this column, Freqtrade sees an exit signal.

??? Hint "Shorting"
    When shorting, this column is called `exit_short`.

## A simple strategy

Here is a minimal example of a Freqtrade strategy:

```python
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

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
When a signal is found (a `1` in an entry or exit column), Freqtrade will attempt to make an order on the exchange, i.e. a `trade`.



## Conclusion
Developing a strategy in Freqtrade involves defining entry and exit signals based on technical indicators. By following the structure and methods outlined above, you can create and test your own trading strategies.

To continue, refer to the more in-depth [Freqtrade Strategy Documentation](strategy-customization.md).
