# Analyzing bot data

You can analyze the results of backtests and trading history easily using Jupyter notebooks. A sample notebook is located at `user_data/notebooks/analysis_example.ipynb`. For usage instructions, see [jupyter.org](https://jupyter.org/documentation).

## Strategy debugging example

Debugging a strategy can be time-consuming. FreqTrade offers helper functions to visualize raw data.

### Import requirements and define variables used in the script

```python
# Imports
from pathlib import Path
import os
from freqtrade.data.history import load_pair_history
from freqtrade.resolvers import StrategyResolver
from freqtrade.data.btanalysis import load_backtest_data
from freqtrade.data.btanalysis import load_trades_from_db

# Define some constants
ticker_interval = "1m"
# Name of the strategy class
strategy_name = 'NewStrategy'
# Path to user data
user_data_dir = 'user_data'
# Location of the strategy
strategy_location = os.path.join(user_data_dir, 'strategies')
# Location of the data
data_location = os.path.join(user_data_dir, 'data', 'binance')
# Pair to analyze 
# Only use one pair here
pair = "BTC_USDT"
```

### Load exchange data

```python
# Load data using values set above
bt_data = load_pair_history(datadir=Path(data_location),
                            ticker_interval=ticker_interval,
                            pair=pair)

# Confirm success
print("Loaded " + str(len(bt_data)) + f" rows of data for {pair} from {data_location}")
```

### Load and run strategy  

* Rerun each time the strategy file is changed
* Display the trade details. Note that using `data.head()` would also work, however most indicators have some "startup" data at the top of the dataframe.

Some possible problems:

* Columns with NaN values at the end of the dataframe
* Columns used in `crossed*()` functions with completely different units

```python
# Load strategy using values set above
strategy = StrategyResolver({'strategy': strategy_name,
                            'user_data_dir': user_data_dir,
                            'strategy_path': strategy_location}).strategy

# Run strategy (just like in backtesting)
df = strategy.analyze_ticker(bt_data, {'pair': pair})

# Report results
print(f"Generated {df['buy'].sum()} buy signals")
data = df.set_index('date', drop=True)
data.tail()
```

### Load backtest results into a pandas dataframe

```python
# Load backtest results
df = load_backtest_data("user_data/backtest_data/backtest-result.json")

# Show value-counts per pair
df.groupby("pair")["sell_reason"].value_counts()
```

### Load live trading results into a pandas dataframe

``` python
# Fetch trades from database
df = load_trades_from_db("sqlite:///tradesv3.sqlite")

# Display results
df.groupby("pair")["sell_reason"].value_counts()
```

Feel free to submit an issue or Pull Request enhancing this document if you would like to share ideas on how to best analyze the data.
