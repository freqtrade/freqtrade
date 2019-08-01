# Analyzing bot data

After performing backtests, or after running the bot for some time, it will be interesting to analyze the results your bot generated.

A good way for this is using Jupyter (notebook or lab) - which provides an interactive environment to analyze the data.

The following helpers will help you loading the data into Pandas DataFrames, and may also give you some starting points in analyzing the results.

## Strategy development problem analysis

Debugging a strategy (are there no buy signals, ...) can be very time-consuming.
FreqTrade tries to help you by exposing a few helper-functions, which can be very handy.

I recommend using Juptyer Notebooks for this analysis, since it offers a dynamic way to rerun certain parts.

The following is a full code-snippet, which will be explained by both comments, and step by step below.

```python
# Some necessary imports
from pathlib import Path

from freqtrade.data.history import load_pair_history
from freqtrade.resolvers import StrategyResolver
# Define some constants
ticker_interval = "5m"

# Name of the strategy class
strategyname = 'Awesomestrategy'
# Location of the strategy
strategy_location = '../xmatt/strategies'
# Location of the data
data_location = '../freqtrade/user_data/data/binance/'
# Only use one pair here
pair = "XRP_ETH"

# Load data
bt_data = load_pair_history(datadir=Path(data_location),
                            ticker_interval = ticker_interval,
                            pair=pair)
print(len(bt_data))


# Load strategy - best done in a new cell
# Needs to be ran each time the strategy-file is changed.
strategy = StrategyResolver({'strategy': strategyname,
                            'user_data_dir': Path.cwd(),
                            'strategy_path': location}).strategy

# Run strategy (just like in backtesting)
df = strategy.analyze_ticker(bt_data, {'pair': pair})
print(f"Generated {df['buy'].sum()} buy signals")

# Reindex data to be "nicer" and show data
data = df.set_index('date', drop=True)
data.tail()

```

### Explanation

#### Imports and constant definition

``` python
# Some necessary imports
from pathlib import Path

from freqtrade.data.history import load_pair_history
from freqtrade.resolvers import StrategyResolver
# Define some constants
ticker_interval = "5m"

# Name of the strategy class
strategyname = 'Awesomestrategy'
# Location of the strategy
strategy_location = 'user_data/strategies'
# Location of the data
data_location = 'user_data/data/binance'
# Only use one pair here
pair = "XRP_ETH"
```

This first section imports necessary modules, and defines some constants you'll probably need differently

#### Load candles

``` python
# Load data
bt_data = load_pair_history(datadir=Path(data_location),
                            ticker_interval = ticker_interval,
                            pair=pair)
print(len(bt_data))
```

This second section loads the historic data and prints the amount of candles in the data.

#### Run strategy and analyze results

Now, it's time to load and run your strategy.
For this, I recommend using a new cell in your notebook, since you'll want to repeat this until you're satisfied with your strategy.

``` python
# Load strategy - best done in a new cell
# Needs to be ran each time the strategy-file is changed.
strategy = StrategyResolver({'strategy': strategyname,
                            'user_data_dir': Path.cwd(),
                            'strategy_path': location}).strategy

# Run strategy (just like in backtesting)
df = strategy.analyze_ticker(bt_data, {'pair': pair})
print(f"Generated {df['buy'].sum()} buy signals")

# Reindex data to be "nicer" and show data
data = df.set_index('date', drop=True)
data.tail()
```

The code snippet loads and analyzes the strategy, prints the number of buy signals.

The last 2 lines serve to analyze the dataframe in detail.
This can be important if your strategy did not generate any buy signals.
Note that using `data.head()` would also work, however this is misleading since most indicators have some "startup" time at the start of a backtested dataframe.

There can be many things wrong, some signs to look for are:

* Columns with NaN values at the end of the dataframe
* Columns used in `crossed*()` functions with completely different units

## Backtesting

To analyze your backtest results, you can [export the trades](#exporting-trades-to-file).
You can then load the trades to perform further analysis.

Freqtrade provides the `load_backtest_data()` helper function to easily load the backtest results, which takes the path to the the backtest-results file as parameter.

``` python
from freqtrade.data.btanalysis import load_backtest_data
df = load_backtest_data("user_data/backtest-result.json")

# Show value-counts per pair
df.groupby("pair")["sell_reason"].value_counts()

```

This will allow you to drill deeper into your backtest results, and perform analysis which otherwise would make the regular backtest-output very difficult to digest due to information overload.

If you have some ideas for interesting / helpful backtest data analysis ideas, please submit a Pull Request so the community can benefit from it.

## Live data

To analyze the trades your bot generated, you can load them to a DataFrame as follows:

``` python
from freqtrade.data.btanalysis import load_trades_from_db

df = load_trades_from_db("sqlite:///tradesv3.sqlite")

df.groupby("pair")["sell_reason"].value_counts()

```

Feel free to submit an issue or Pull Request enhancing this document if you would like to share ideas on how to best analyze the data.
