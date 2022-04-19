# Strategy analysis example

Debugging a strategy can be time-consuming. Freqtrade offers helper functions to visualize raw data.
The following assumes you work with SampleStrategy, data for 5m timeframe from Binance and have downloaded them into the data directory in the default location.

## Setup


```python
from pathlib import Path
from freqtrade.configuration import Configuration

# Customize these according to your needs.

# Initialize empty configuration object
config = Configuration.from_files([])
# Optionally, use existing configuration file
# config = Configuration.from_files(["config.json"])

# Define some constants
config["timeframe"] = "5m"
# Name of the strategy class
config["strategy"] = "SampleStrategy"
# Location of the data
data_location = Path(config['user_data_dir'], 'data', 'binance')
# Pair to analyze - Only use one pair here
pair = "BTC/USDT"
```


```python
# Load data using values set above
from freqtrade.data.history import load_pair_history

candles = load_pair_history(datadir=data_location,
                            timeframe=config["timeframe"],
                            pair=pair,
                            data_format = "hdf5",
                            )

# Confirm success
print("Loaded " + str(len(candles)) + f" rows of data for {pair} from {data_location}")
candles.head()
```

## Load and run strategy
* Rerun each time the strategy file is changed


```python
# Load strategy using values set above
from freqtrade.resolvers import StrategyResolver
from freqtrade.data.dataprovider import DataProvider
strategy = StrategyResolver.load_strategy(config)
strategy.dp = DataProvider(config, None, None)

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
print(f"Generated {df['enter_long'].sum()} entry signals")
data = df.set_index('date', drop=False)
data.tail()
```

## Load existing objects into a Jupyter notebook

The following cells assume that you have already generated data using the cli.  
They will allow you to drill deeper into your results, and perform analysis which otherwise would make the output very difficult to digest due to information overload.

### Load backtest results to pandas dataframe

Analyze a trades dataframe (also used below for plotting)


```python
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats

# if backtest_dir points to a directory, it'll automatically load the last backtest file.
backtest_dir = config["user_data_dir"] / "backtest_results"
# backtest_dir can also point to a specific file
# backtest_dir = config["user_data_dir"] / "backtest_results/backtest-result-2020-07-01_20-04-22.json"
```


```python
# You can get the full backtest statistics by using the following command.
# This contains all information used to generate the backtest result.
stats = load_backtest_stats(backtest_dir)

strategy = 'SampleStrategy'
# All statistics are available per strategy, so if `--strategy-list` was used during backtest, this will be reflected here as well.
# Example usages:
print(stats['strategy'][strategy]['results_per_pair'])
# Get pairlist used for this backtest
print(stats['strategy'][strategy]['pairlist'])
# Get market change (average change of all pairs from start to end of the backtest period)
print(stats['strategy'][strategy]['market_change'])
# Maximum drawdown ()
print(stats['strategy'][strategy]['max_drawdown'])
# Maximum drawdown start and end
print(stats['strategy'][strategy]['drawdown_start'])
print(stats['strategy'][strategy]['drawdown_end'])


# Get strategy comparison (only relevant if multiple strategies were compared)
print(stats['strategy_comparison'])

```


```python
# Load backtested trades as dataframe
trades = load_backtest_data(backtest_dir)

# Show value-counts per pair
trades.groupby("pair")["exit_reason"].value_counts()
```

## Plotting daily profit / equity line


```python
# Plotting equity line (starting with 0 on day 1 and adding daily profit for each backtested day)

from freqtrade.configuration import Configuration
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
import plotly.express as px
import pandas as pd

# strategy = 'SampleStrategy'
# config = Configuration.from_files(["user_data/config.json"])
# backtest_dir = config["user_data_dir"] / "backtest_results"

stats = load_backtest_stats(backtest_dir)
strategy_stats = stats['strategy'][strategy]

dates = []
profits = []
for date_profit in strategy_stats['daily_profit']:
    dates.append(date_profit[0])
    profits.append(date_profit[1])

equity = 0
equity_daily = []
for daily_profit in profits:
    equity_daily.append(equity)
    equity += float(daily_profit)


df = pd.DataFrame({'dates': dates,'equity_daily': equity_daily})

fig = px.line(df, x="dates", y="equity_daily")
fig.show()

```

### Load live trading results into a pandas dataframe

In case you did already some trading and want to analyze your performance


```python
from freqtrade.data.btanalysis import load_trades_from_db

# Fetch trades from database
trades = load_trades_from_db("sqlite:///tradesv3.sqlite")

# Display results
trades.groupby("pair")["exit_reason"].value_counts()
```

## Analyze the loaded trades for trade parallelism
This can be useful to find the best `max_open_trades` parameter, when used with backtesting in conjunction with `--disable-max-market-positions`.

`analyze_trade_parallelism()` returns a timeseries dataframe with an "open_trades" column, specifying the number of open trades for each candle.


```python
from freqtrade.data.btanalysis import analyze_trade_parallelism

# Analyze the above
parallel_trades = analyze_trade_parallelism(trades, '5m')

parallel_trades.plot()
```

## Plot results

Freqtrade offers interactive plotting capabilities based on plotly.


```python
from freqtrade.plot.plotting import  generate_candlestick_graph
# Limit graph period to keep plotly quick and reactive

# Filter trades to one pair
trades_red = trades.loc[trades['pair'] == pair]

data_red = data['2019-06-01':'2019-06-10']
# Generate candlestick graph
graph = generate_candlestick_graph(pair=pair,
                                   data=data_red,
                                   trades=trades_red,
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

## Plot average profit per trade as distribution graph


```python
import plotly.figure_factory as ff

hist_data = [trades.profit_ratio]
group_labels = ['profit_ratio']  # name of the dataset

fig = ff.create_distplot(hist_data, group_labels, bin_size=0.01)
fig.show()

```

Feel free to submit an issue or Pull Request enhancing this document if you would like to share ideas on how to best analyze the data.

## Analyse the buy/entry and sell/exit tags

It can be helpful to understand how a strategy behaves according to the buy/entry tags used to
mark up different buy conditions. You might want to see more complex statistics about each buy and
sell condition above those provided by the default backtesting output. You may also want to
determine indicator values on the signal candle that resulted in a trade opening.

We first need to enable the exporting of trades from backtesting:

```
freqtrade backtesting -c <config.json> --timeframe <tf> --strategy <strategy_name> --timerange=<timerange> --export=trades --export-filename=user_data/backtest_results/<name>-<timerange>
```

To analyse the buy tags, we need to use the buy_reasons.py script in the `scripts/`
folder. We need the signal candles for each opened trade so add the following option to your
config file:

```
'backtest_signal_candle_export_enable': true,
```

This will tell freqtrade to output a pickled dictionary of strategy, pairs and corresponding
DataFrame of the candles that resulted in buy signals. Depending on how many buys your strategy
makes, this file may get quite large, so periodically check your `user_data/backtest_results`
folder to delete old exports.

Before running your next backtest, make sure you either delete your old backtest results or run
backtesting with the `--cache none` option to make sure no cached results are used.

If all goes well, you should now see a `backtest-result-{timestamp}_signals.pkl` file in the
`user_data/backtest_results` folder.

Now run the buy_reasons.py script, supplying a few options:

```
./scripts/buy_reasons.py -c <config.json> -s <strategy_name> -t <timerange> -g0,1,2,3,4
```

The `-g` option is used to specify the various tabular outputs, ranging from the simplest (0)
to the most detailed per pair, per buy and per sell tag (4). More options are available by
running with the `-h` option.

### Tuning the buy tags and sell tags to display

To show only certain buy and sell tags in the displayed output, use the following two options:

```
--buy_reason_list : Comma separated list of buy signals to analyse. Default: "all"
--sell_reason_list : Comma separated list of sell signals to analyse. Default: "stop_loss,trailing_stop_loss"
```

For example:

```
./scripts/buy_reasons.py -c <config.json> -s <strategy_name> -t <timerange> -g0,1,2,3,4 --buy_reason_list "buy_tag_a,buy_tag_b" --sell_reason_list "roi,custom_sell_tag_a,stop_loss"
```

### Outputting signal candle indicators

The real power of the buy_reasons.py script comes from the ability to print out the indicator
values present on signal candles to allow fine-grained investigation and tuning of buy signal
indicators. To print out a column for a given set of indicators, use the `--indicator-list`
option:

```
./scripts/buy_reasons.py -c <config.json> -s <strategy_name> -t <timerange> -g0,1,2,3,4 --buy_reason_list "buy_tag_a,buy_tag_b" --sell_reason_list "roi,custom_sell_tag_a,stop_loss" --indicator_list "rsi,rsi_1h,bb_lowerband,ema_9,macd,macdsignal"
```

The indicators have to be present in your strategy's main dataframe (either for your main
timeframe or for informatives) otherwise they will simply be ignored in the script
output.
