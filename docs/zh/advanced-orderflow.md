<!-- 本文件为中文翻译版，由 AI 根据 docs/advanced-orderflow.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 订单流数据

本指南将介绍如何利用公开交易数据在 Freqtrade 中进行高级订单流分析。

!!! Warning "实验性功能"
    订单流功能目前处于测试阶段，在未来版本中可能会发生变化。请在 [Freqtrade GitHub 仓库](https://github.com/freqtrade/freqtrade/issues) 报告任何问题或反馈。
    目前该功能尚未与 freqAI 进行测试——将这两个功能结合使用在当前阶段被视为超出范围。

!!! Warning "性能提示"
    订单流需要原始交易数据。这些数据量相当大，当 freqtrade 需要下载最近 X 根 K 线的交易数据时，可能会导致初始启动缓慢。此外，启用此功能会增加内存使用量。请确保有足够的可用资源。

## 快速入门

### 启用公开交易数据

在你的 `config.json` 文件中，在 `exchange` 部分将 `use_public_trades` 选项设置为 true。

```json
"exchange": {
   ...
   "use_public_trades": true,
}
```

### 配置订单流处理

在 config.json 的 orderflow 部分定义订单流处理的设置。你可以调整以下参数：

- `cache_size`：保存到缓存中的前几根订单流 K 线数量，而不是每根新 K 线都重新计算
- `max_candles`：筛选你希望获取交易数据的 K 线数量。
- `scale`：控制足迹图的价格区间大小。
- `stacked_imbalance_range`：定义被视为有效的连续失衡价格级别的最小数量。
- `imbalance_volume`：过滤掉成交量低于此阈值的失衡。
- `imbalance_ratio`：过滤掉比率（买卖成交量之差）低于此值的失衡。

```json
"orderflow": {
    "cache_size": 1000,
    "max_candles": 1500,
    "scale": 0.5,
    "stacked_imbalance_range": 3,
    "imbalance_volume": 1,
    "imbalance_ratio": 3
  },
```

## 下载回测用的交易数据

要下载历史交易数据用于回测，请在 freqtrade download-data 命令中使用 --dl-trades 标志。

```bash
freqtrade download-data -p BTC/USDT:USDT --timerange 20230101- --trading-mode futures --timeframes 5m --dl-trades
```

!!! Warning "数据可用性"
    并非所有交易所都提供公开交易数据。对于支持的交易所，如果你使用 `--dl-trades` 标志开始下载数据，而公开交易数据不可用，freqtrade 会发出警告。

## 访问订单流数据

启用后，你的 dataframe 中会新增以下几列：

``` python

dataframe["trades"] # 包含每笔单独交易的信息。
dataframe["orderflow"] # 表示一个足迹图字典（见下文）
dataframe["imbalances"] # 包含订单流中失衡的信息。
dataframe["bid"] # 总买方成交量
dataframe["ask"] # 总卖方成交量
dataframe["delta"] # 卖方与买方成交量之差。
dataframe["min_delta"] # K 线内的最小 delta 值
dataframe["max_delta"] # K 线内的最大 delta 值
dataframe["total_trades"] # 总交易笔数
dataframe["stacked_imbalances_bid"] # 堆叠买方失衡区间起始价格级别列表
dataframe["stacked_imbalances_ask"] # 堆叠卖方失衡区间起始价格级别列表
```

你可以在策略代码中访问这些列进行进一步分析。以下是一个示例：

``` python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # 计算累积 delta
    dataframe["cum_delta"] = cumulative_delta(dataframe["delta"])
    # 访问总交易笔数
    total_trades = dataframe["total_trades"]
    ...

def cumulative_delta(delta: Series):
    cumdelta = delta.cumsum()
    return cumdelta

```

### 足迹图 (`dataframe["orderflow"]`)

此列提供了不同价格级别下买卖订单的详细分解，为订单流动态提供了有价值的洞察。配置中的 `scale` 参数决定了此表示的价格区间大小。

`orderflow` 列包含一个具有以下结构的字典：

``` output
{
    "price": {
        "bid_amount": 0.0,
        "ask_amount": 0.0,
        "bid": 0,
        "ask": 0,
        "delta": 0.0,
        "total_volume": 0.0,
        "total_trades": 0
    }
}
```

#### orderflow 列说明

- key：价格区间 - 按 `scale` 间隔进行分箱
- `bid_amount`：每个价格级别的总买入成交量。
- `ask_amount`：每个价格级别的总卖出成交量。
- `bid`：每个价格级别的买单数量。
- `ask`：每个价格级别的卖单数量。
- `delta`：每个价格级别的卖方与买方成交量之差。
- `total_volume`：每个价格级别的总成交量（卖方量 + 买方量）。
- `total_trades`：每个价格级别的总交易笔数（卖单 + 买单）。

通过利用这些功能，你可以获得关于市场情绪的有价值的洞察，并基于订单流分析发现潜在的交易机会。

### 原始交易数据 (`dataframe["trades"]`)

该列表包含 K 线期间发生的每笔单独交易。这些数据可用于对订单流动态进行更细粒度的分析。

每个条目包含一个具有以下键的字典：

- `timestamp`：交易时间戳。
- `date`：交易日期。
- `price`：交易价格。
- `amount`：交易量。
- `side`：买入或卖出。
- `id`：交易的唯一标识符。
- `cost`：交易总成本（价格 * 数量）。

### 失衡 (`dataframe["imbalances"]`)

此列提供了一个包含订单流失衡信息的字典。当某一价格级别的卖方和买方成交量之间存在显著差异时，就会出现失衡。

每行的结构如下——以价格为索引，对应的买卖失衡值为列

``` output
{
    "price": {
        "bid_imbalance": False,
        "ask_imbalance": False
    }
}
```