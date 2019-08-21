# Analyzing bot data

You can analyze the results of backtests and trading history easily using Jupyter notebooks. A sample notebook is located at `user_data/notebooks/analysis_example.ipynb`. For usage instructions, see [jupyter.org](https://jupyter.org/documentation).

*Pro tip - Don't forget to start a jupyter notbook server from within your conda or venv environment or use [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels)*

## Example snippets

### Load backtest results into a pandas dataframe

```python
from freqtrade.data.btanalysis import load_backtest_data
# Load backtest results
df = load_backtest_data("user_data/backtest_results/backtest-result.json")

# Show value-counts per pair
df.groupby("pair")["sell_reason"].value_counts()
```

This will allow you to drill deeper into your backtest results, and perform analysis which otherwise would make the regular backtest-output very difficult to digest due to information overload.

### Load live trading results into a pandas dataframe

``` python
from freqtrade.data.btanalysis import load_trades_from_db

# Fetch trades from database
df = load_trades_from_db("sqlite:///tradesv3.sqlite")

# Display results
df.groupby("pair")["sell_reason"].value_counts()
```

### Load multiple configuration files

This option can be usefull to inspect the results of passing in multiple configs in case of problems

``` python
from freqtrade.configuration import Configuration
config = Configuration.from_files(["config1.json", "config2.json"])
print(config)
```

## Strategy debugging example

Debugging a strategy can be time-consuming. FreqTrade offers helper functions to visualize raw data.

### Import requirements and define variables used in analyses

```python
# Imports
from pathlib import Path
import os
from freqtrade.data.history import load_pair_history
from freqtrade.resolvers import StrategyResolver

# You can override strategy settings as demonstrated below.
# Customize these according to your needs.

# Define some constants
ticker_interval = "5m"
# Name of the strategy class
strategy_name = 'AwesomeStrategy'
# Path to user data
user_data_dir = 'user_data'
# Location of the strategy
strategy_location = Path(user_data_dir, 'strategies')
# Location of the data
data_location = Path(user_data_dir, 'data', 'binance')
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
print(f"Loaded {len(bt_data)} rows of data for {pair} from {data_location}")
```

### Load and run strategy  

* Rerun each time the strategy file is changed

```python
# Load strategy using values set above
strategy = StrategyResolver({'strategy': strategy_name,
                            'user_data_dir': user_data_dir,
                            'strategy_path': strategy_location}).strategy

# Generate buy/sell signals using strategy
df = strategy.analyze_ticker(bt_data, {'pair': pair})
```

### Display the trade details

* Note that using `data.head()` would also work, however most indicators have some "startup" data at the top of the dataframe.

#### Some possible problems

* Columns with NaN values at the end of the dataframe
* Columns used in `crossed*()` functions with completely different units

#### Comparison with full backtest

having 200 buy signals as output for one pair from `analyze_ticker()` does not necessarily mean that 200 trades will be made during backtesting.

Assuming you use only one condition such as, `df['rsi'] < 30` as buy condition, this will generate multiple "buy" signals for each pair in sequence (until rsi returns > 29).
The bot will only buy on the first of these signals (and also only if a trade-slot ("max_open_trades") is still available), or on one of the middle signals, as soon as a "slot" becomes available.

```python
# Report results
print(f"Generated {df['buy'].sum()} buy signals")
data = df.set_index('date', drop=True)
data.tail()
```

Feel free to submit an issue or Pull Request enhancing this document if you would like to share ideas on how to best analyze the data.
