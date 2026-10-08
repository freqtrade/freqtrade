<!-- 本文件为中文翻译版，由 AI 根据 docs/strategy_analysis_example.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 策略分析示例

调试策略可能非常耗时。Freqtrade 提供了辅助函数来可视化原始数据。
以下内容假设您使用 SampleStrategy、来自 Binance 的 5 分钟时间框架数据，并且已将数据下载到默认位置的 data 目录中。
请参阅[文档](https://www.freqtrade.io/en/stable/data-download/)了解更多详情。

## 环境准备

### 将工作目录更改为仓库根目录


```python
import os
from pathlib import Path


# Change directory
# Modify this cell to insure that the output shows the correct path.
# Define all paths relative to the project root shown in the cell output
project_root = "somedir/freqtrade"
i = 0
try:
    os.chdir(project_root)
    if not Path("LICENSE").is_file():
        i = 0
        while i < 4 and (not Path("LICENSE").is_file()):
            os.chdir(Path(Path.cwd(), "../"))
            i += 1
        project_root = Path.cwd()
except FileNotFoundError:
    print("Please define the project root relative to the current directory")
print(Path.cwd())
```

### 配置 Freqtrade 环境


```python
from freqtrade.configuration import Configuration


# Customize these according to your needs.

# Initialize empty configuration object
config = Configuration.from_files([])
# Optionally (recommended), use existing configuration file
# config = Configuration.from_files(["user_data/config.json"])

# Define some constants
config["timeframe"] = "5m"
# Name of the strategy class
config["strategy"] = "SampleStrategy"
# Location of the data
data_location = config["datadir"]
# Pair to analyze - Only use one pair here
pair = "BTC/USDT"
```


```python
# Load data using values set above
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType


candles = load_pair_history(
    datadir=data_location,
    timeframe=config["timeframe"],
    pair=pair,
    data_format="json",  # Make sure to update this to your data
    candle_type=CandleType.SPOT,
)

# Confirm success
print(f"Loaded {len(candles)} rows of data for {pair} from {data_location}")
candles.head()
```

## 加载并运行策略
* 每次策略文件更改后都需要重新运行此部分


```python
# Load strategy using values set above
from freqtrade.data.dataprovider import DataProvider
from freqtrade.resolvers import StrategyResolver


strategy = StrategyResolver.load_strategy(config)
strategy.dp = DataProvider(config, None, None)
strategy.ft_bot_start()

# Generate buy/sell signals using strategy
df = strategy.analyze_ticker(candles, {"pair": pair})
df.tail()
```

### 显示交易详情

* 请注意，使用 `data.head()` 也可以，但大多数指标在数据框顶部会有一些"启动"数据。
* 一些可能的问题
    * 数据框末尾存在 NaN 值的列
    * 在 `crossed*()` 函数中使用的列具有完全不同的单位
* 与完整回测的比较
    * `analyze_ticker()` 输出 200 个买入信号并不一定意味着回测期间会产生 200 笔交易。
    * 假设您只使用一个条件，例如 `df['rsi'] < 30` 作为买入条件，这将为每个交易对连续生成多个"买入"信号（直到 RSI 返回 > 29）。机器人只会在这类信号的第一个上执行买入（并且仅在交易槽位（`max_open_trades`）仍然可用的情况下），或者在某个中间信号上执行，一旦"槽位"变为可用。



```python
# Report results
print(f"Generated {df['enter_long'].sum()} entry signals")
data = df.set_index("date", drop=False)
data.tail()
```

## 将现有对象加载到 Jupyter notebook 中

以下单元格假设您已经使用 CLI 生成了数据。
它们将使您能够深入分析结果，并执行那些因信息过载而使输出难以理解的分析。

### 将回测结果加载到 pandas 数据框中

分析交易数据框（下面绘图时也会使用）


```python
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats


# if backtest_dir points to a directory, it'll automatically load the last backtest file.
backtest_dir = config["user_data_dir"] / "backtest_results"
# backtest_dir can also point to a specific file
# backtest_dir = (
#   config["user_data_dir"] / "backtest_results/backtest-result-2020-07-01_20-04-22.json"
# )
```


```python
# You can get the full backtest statistics by using the following command.
# This contains all information used to generate the backtest result.
stats = load_backtest_stats(backtest_dir)

strategy = "SampleStrategy"
# All statistics are available per strategy, so if `--strategy-list` was used during backtest,
# this will be reflected here as well.
# Example usages:
print(stats["strategy"][strategy]["results_per_pair"])
# Get pairlist used for this backtest
print(stats["strategy"][strategy]["pairlist"])
# Get market change (average change of all pairs from start to end of the backtest period)
print(stats["strategy"][strategy]["market_change"])
# Maximum drawdown ()
print(stats["strategy"][strategy]["max_drawdown_abs"])
# Maximum drawdown start and end
print(stats["strategy"][strategy]["drawdown_start"])
print(stats["strategy"][strategy]["drawdown_end"])


# Get strategy comparison (only relevant if multiple strategies were compared)
print(stats["strategy_comparison"])
```


```python
# Load backtested trades as dataframe
trades = load_backtest_data(backtest_dir)

# Show value-counts per pair
trades.groupby("pair")["exit_reason"].value_counts()
```

## 绘制每日收益/权益曲线


```python
# Plotting equity line (starting with 0 on day 1 and adding daily profit for each backtested day)

import pandas as pd
import plotly.express as px

from freqtrade.configuration import Configuration
from freqtrade.data.btanalysis import load_backtest_stats


# strategy = 'SampleStrategy'
# config = Configuration.from_files(["user_data/config.json"])
# backtest_dir = config["user_data_dir"] / "backtest_results"

stats = load_backtest_stats(backtest_dir)
strategy_stats = stats["strategy"][strategy]

df = pd.DataFrame(columns=["dates", "equity"], data=strategy_stats["daily_profit"])
df["equity_daily"] = df["equity"].cumsum()

fig = px.line(df, x="dates", y="equity_daily")
fig.show()
```

### 将实盘交易结果加载到 pandas 数据框中

如果您已经进行了一些交易并想分析您的表现


```python
from freqtrade.data.btanalysis import load_trades_from_db


# Fetch trades from database
trades = load_trades_from_db("sqlite:///tradesv3.sqlite")

# Display results
trades.groupby("pair")["exit_reason"].value_counts()
```

## 分析已加载交易的并行性

当与回测结合使用并设置非常高的 `max_open_trades` 时，这对于找到最佳的 `max_open_trades` 参数非常有用。

`analyze_trade_parallelism()` 返回一个时间序列数据框，其中包含一个 `open_trades` 列，指定每个 K 线的未平仓交易数量。


```python
from freqtrade.data.btanalysis import analyze_trade_parallelism


# Analyze the above
parallel_trades = analyze_trade_parallelism(trades, "5m")

parallel_trades.plot()
```

## 绘制结果

Freqtrade 提供了基于 plotly 的交互式绘图功能。


```python
from freqtrade.plot.plotting import generate_candlestick_graph


# Limit graph period to keep plotly quick and reactive

# Filter trades to one pair
trades_red = trades.loc[trades["pair"] == pair]

data_red = data["2019-06-01":"2019-06-10"]
# Generate candlestick graph
graph = generate_candlestick_graph(
    pair=pair,
    data=data_red,
    trades=trades_red,
    indicators1=["sma20", "ema50", "ema55"],
    indicators2=["rsi", "macd", "macdsignal", "macdhist"],
)
```


```python
# Show graph inline
# graph.show()

# Render graph in a separate window
graph.show(renderer="browser")
```

## 绘制每笔交易平均收益的分布图


```python
import plotly.figure_factory as ff


hist_data = [trades.profit_ratio]
group_labels = ["profit_ratio"]  # name of the dataset

fig = ff.create_distplot(hist_data, group_labels, bin_size=0.01)
fig.show()
```

欢迎提交 Issue 或 Pull Request 来改进本文档，如果您想分享关于如何更好地分析数据的想法。