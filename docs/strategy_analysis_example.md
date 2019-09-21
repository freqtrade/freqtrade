# Strategy analysis example

Debugging a strategy can be time-consuming. FreqTrade offers helper functions to visualize raw data.

## Setup


```python
from pathlib import Path
# Customize these according to your needs.

# Define some constants
ticker_interval = "5m"
# Name of the strategy class
strategy_name = 'SampleStrategy'
# Path to user data
user_data_dir = Path('user_data')
# Location of the strategy
strategy_location = user_data_dir / 'strategies'
# Location of the data
data_location = Path(user_data_dir, 'data', 'binance')
# Pair to analyze - Only use one pair here
pair = "BTC_USDT"
```


```python
# Load data using values set above
from freqtrade.data.history import load_pair_history

candles = load_pair_history(datadir=data_location,
                            ticker_interval=ticker_interval,
                            pair=pair)

# Confirm success
print("Loaded " + str(len(candles)) + f" rows of data for {pair} from {data_location}")
candles.head()
```

## Load and run strategy
* Rerun each time the strategy file is changed


```python
# Load strategy using values set above
from freqtrade.resolvers import StrategyResolver
strategy = StrategyResolver({'strategy': strategy_name,
                            'user_data_dir': user_data_dir,
                            'strategy_path': strategy_location}).strategy

# Generate buy/sell signals using strategy
df = strategy.analyze_ticker(candles, {'pair': pair})
df.tail()
```

### Display the trade details

* Note that using `data.head()` would also work, however most indicators have some "startup" data at the top of the dataframe.
* Some possible problems
    * Columns with NaN values at the end of the dataframe
    * Columns used in `crossed*()` functions with completely different units
* Comparison with full backtest
    * having 200 buy signals as output for one pair from `analyze_ticker()` does not necessarily mean that 200 trades will be made during backtesting.
    * Assuming you use only one condition such as, `df['rsi'] < 30` as buy condition, this will generate multiple "buy" signals for each pair in sequence (until rsi returns > 29). The bot will only buy on the first of these signals (and also only if a trade-slot ("max_open_trades") is still available), or on one of the middle signals, as soon as a "slot" becomes available.  



```python
# Report results
print(f"Generated {df['buy'].sum()} buy signals")
data = df.set_index('date', drop=False)
data.tail()
```

## Load existing objects into a Jupyter notebook

The following cells assume that you have already generated data using the cli.  
They will allow you to drill deeper into your results, and perform analysis which otherwise would make the output very difficult to digest due to information overload.

### Load backtest results to pandas dataframe

Analyze a trades dataframe (also used below for plotting)


```python
from freqtrade.data.btanalysis import load_backtest_data

# Load backtest results
trades = load_backtest_data(user_data_dir / "backtest_results/backtest-result.json")

# Show value-counts per pair
trades.groupby("pair")["sell_reason"].value_counts()
```

### Load live trading results into a pandas dataframe

In case you did already some trading and want to analyze your performance


```python
from freqtrade.data.btanalysis import load_trades_from_db

# Fetch trades from database
trades = load_trades_from_db("sqlite:///tradesv3.sqlite")

# Display results
trades.groupby("pair")["sell_reason"].value_counts()
```

## Plot results

Freqtrade offers interactive plotting capabilities based on plotly.


```python
from freqtrade.plot.plotting import  generate_candlestick_graph
# Limit graph period to keep plotly quick and reactive

data_red = data['2019-06-01':'2019-06-10']
# Generate candlestick graph
graph = generate_candlestick_graph(pair=pair,
                                   data=data_red,
                                   trades=trades,
                                   indicators1=['sma20', 'ema50', 'ema55'],
                                   indicators2=['rsi', 'macd', 'macdsignal', 'macdhist']
                                  )



```


```python
# Show graph inline
# graph.show()

# Render graph in a seperate window
graph.show(renderer="browser")

```

Feel free to submit an issue or Pull Request enhancing this document if you would like to share ideas on how to best analyze the data.
