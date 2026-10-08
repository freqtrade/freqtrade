<!-- 本文件为中文翻译版，由 AI 根据 docs/strategy-customization.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件（如 exchanges.md），这些文件将在逐步翻译过程中补齐。 -->
<!-- 图片与 includes 引用使用 ../ 指向英文原文档资源，以保证显示正常。 -->

# 策略自定义

本页介绍如何自定义你的策略、添加新指标以及设置交易规则。

如果你还没有了解以下内容，请先熟悉：

- [Freqtrade 策略入门](strategy-101.md)，提供了策略开发的快速入门指南
- [Freqtrade 机器人基础](bot-basics.md)，提供了机器人整体运行方式的信息

## 开发你自己的策略

机器人包含一个默认的策略文件。

此外，[策略仓库](https://github.com/freqtrade/freqtrade-strategies)中还提供了多种其他策略。

不过，你很可能会有自己的策略想法。

本文档旨在帮助你将想法转化为可运行的策略。

### 生成策略模板

要开始，可以使用以下命令：

```bash
freqtrade new-strategy --strategy AwesomeStrategy
```

这将从模板创建一个名为 `AwesomeStrategy` 的新策略，文件将保存在 `user_data/strategies/AwesomeStrategy.py`。

!!! Note
    策略的*名称*与文件名是有区别的。在大多数命令中，Freqtrade 使用策略的*名称*，*而不是文件名*。

!!! Note
    `new-strategy` 命令生成的起始示例不会直接产生盈利。

??? Hint "Different template levels"
    `freqtrade new-strategy` 有一个额外参数 `--template`，用于控制创建策略中预构建信息的数量。使用 `--template minimal` 可以获得一个不包含任何指标示例的空策略，或使用 `--template advanced` 获得包含更复杂功能定义的模板。

### 策略结构解析

策略文件包含构建策略逻辑所需的全部信息：

- OHLCV 格式的 K 线数据
- 指标
- 入场逻辑
  - 信号
- 出场逻辑
  - 信号
  - 最小 ROI
  - 回调函数（"自定义函数"）
- 止损
  - 固定/绝对止损
  - 追踪止损
  - 回调函数（"自定义函数"）
- 定价 [可选]
- 仓位调整 [可选]

机器人包含一个名为 `SampleStrategy` 的示例策略，你可以将其作为基础：`user_data/strategies/sample_strategy.py`。
你可以使用参数 `--strategy SampleStrategy` 来测试它。请记住，使用的是策略类名，而不是文件名。

此外，还有一个名为 `INTERFACE_VERSION` 的属性，它定义了机器人应使用的策略接口版本。
当前版本是 3 — 如果在策略中未显式设置，这也是默认值。

你可能会看到旧版策略设置为接口版本 2，这些策略需要更新为 v3 术语，因为未来的版本将要求设置此值。

使用 `trade` 命令以模拟或实盘模式启动机器人：

```bash
freqtrade trade --strategy AwesomeStrategy
```

### 机器人模式

Freqtrade 策略可以在 5 种主要模式下由 Freqtrade 机器人处理：

- 回测
- 超参数优化
- 模拟交易（"前瞻测试"）
- 实盘交易
- FreqAI（此处不涉及）

请查看[配置文档](configuration.md)了解如何将机器人设置为模拟或实盘模式。

**在测试时请始终使用模拟模式，因为这可以让你了解策略在实际中的表现，而不会有资金风险。**

## 深入了解

**在以下部分中，我们将使用 [user_data/strategies/sample_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_strategy.py) 文件作为参考。**

!!! Note "Strategies and Backtesting"
    为避免回测与模拟/实盘模式之间出现问题和意外差异，请注意
    在回测期间，整个时间范围会一次性传递给 `populate_*()` 方法。
    因此，最好使用向量化操作（在整个 dataframe 上操作，而非循环），
    并避免使用索引引用（`df.iloc[-1]`），而是使用 `df.shift()` 获取前一根 K 线。

!!! Warning "Warning: Using future data"
    由于回测将整个时间范围传递给 `populate_*()` 方法，策略作者
    需要注意避免策略使用未来数据。
    本文档的[常见错误](#策略开发中的常见错误)部分列出了一些常见模式。

??? Hint "Lookahead and recursive analysis"
    Freqtrade 包含两个有用的命令，帮助评估常见的前瞻偏差（使用未来数据）和
    递归偏差（指标值的方差）问题。在以模拟或实盘模式运行策略之前，
    你应始终先使用这些命令。请查看[前瞻分析](lookahead-analysis.md)和[递归分析](recursive-analysis.md)的相关文档。

### DataFrame

Freqtrade 使用 [pandas](https://pandas.pydata.org/) 来存储/提供 K 线（OHLCV）数据。
Pandas 是一个优秀的库，专为处理大量表格格式数据而开发。

DataFrame 中的每一行对应图表上的一根 K 线，最新完成的 K 线始终是 DataFrame 中的最后一行（按日期排序）。

如果我们使用 pandas 的 `head()` 函数查看主 DataFrame 的前几行，会看到：

```output
> dataframe.head()
                       date      open      high       low     close     volume
0 2021-11-09 23:25:00+00:00  67279.67  67321.84  67255.01  67300.97   44.62253
1 2021-11-09 23:30:00+00:00  67300.97  67301.34  67183.03  67187.01   61.38076
2 2021-11-09 23:35:00+00:00  67187.02  67187.02  67031.93  67123.81  113.42728
3 2021-11-09 23:40:00+00:00  67123.80  67222.40  67080.33  67160.48   78.96008
4 2021-11-09 23:45:00+00:00  67160.48  67160.48  66901.26  66943.37  111.39292
```

DataFrame 是一个表格，其中的列不是单个值，而是一系列数据值。因此，以下简单的 Python 比较将不起作用：

``` python
    if dataframe['rsi'] > 30:
        dataframe['enter_long'] = 1
```

上述代码将报错 `The truth value of a Series is ambiguous [...]`。

必须改用 pandas 兼容的方式编写，使操作在整个 DataFrame 上执行，即 `向量化`。

``` python
    dataframe.loc[
        (dataframe['rsi'] > 30)
    , 'enter_long'] = 1
```

通过这段代码，你的 DataFrame 中会有一个新列，每当 RSI 高于 30 时，该列会被赋值为 `1`。

Freqtrade 将此新列用作入场信号，假设交易将在下一根 K 线开盘时开仓。

Pandas 提供了快速计算指标的方法，即"向量化"。为了充分利用这种速度，建议不要使用循环，而是使用向量化方法。

向量化操作在整个数据范围内执行计算，因此与逐行循环相比，在计算指标时速度要快得多。

??? Hint "Signals vs Trades"
    - 信号在 K 线收盘时由指标生成，是入场交易的意向。
    - 交易是（在实盘模式中于交易所）执行的订单，交易将尽可能在下一根 K 线开盘价附近开仓。

!!! Warning "Trade order assumptions"
    在回测中，信号在 K 线收盘时生成。交易随后在下一根 K 线开盘时立即执行。

    在模拟和实盘中，这可能会因为需要先分析所有交易对的 DataFrame，然后对每个交易对进行交易处理而有所延迟。
    这意味着在模拟/实盘中，你需要尽量保持较低的计算延迟，通常通过运行较少数量的交易对和拥有高时钟频率的 CPU 来实现。

#### 为什么我看不到"实时"K 线数据？

Freqtrade 不会在 DataFrame 中存储未完成/不完整的 K 线。

使用不完整数据进行策略决策被称为"重绘"，你可能会看到其他平台允许这样做。

Freqtrade 不会。DataFrame 中只有完成的/已结束的 K 线数据。

### 自定义指标

入场和出场信号需要指标。你可以通过扩展策略文件中 `populate_indicators()` 方法中包含的列表来添加更多指标。

你应该只添加在 `populate_entry_trend()`、`populate_exit_trend()` 中使用或用于填充其他指标的指标，否则性能可能会受到影响。

始终从这三个函数返回 DataFrame 且不删除/修改 `"open"、"high"、"low"、"close"、"volume"` 列非常重要，否则这些字段将包含意外内容。

示例：

```python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Adds several different TA indicators to the given DataFrame

    Performance Note: For the best performance be frugal on the number of indicators
    you are using. Let uncomment only the indicator you are using in your strategies
    or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
    :param dataframe: Dataframe with data from the exchange
    :param metadata: Additional information, like the currently traded pair
    :return: a Dataframe with all mandatory indicators for the strategies
    """
    dataframe['sar'] = ta.SAR(dataframe)
    dataframe['adx'] = ta.ADX(dataframe)
    stoch = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch['fastd']
    dataframe['fastk'] = stoch['fastk']
    dataframe['bb_lower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']
    dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
    dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
    dataframe['mfi'] = ta.MFI(dataframe)
    dataframe['rsi'] = ta.RSI(dataframe)
    dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
    dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
    dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
    dataframe['ao'] = awesome_oscillator(dataframe)
    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']
    hilbert = ta.HT_SINE(dataframe)
    dataframe['htsine'] = hilbert['sine']
    dataframe['htleadsine'] = hilbert['leadsine']
    dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
    dataframe['plus_di'] = ta.PLUS_DI(dataframe)
    dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
    dataframe['minus_di'] = ta.MINUS_DI(dataframe)

    # remember to always return the dataframe
    return dataframe
```

!!! Note "Want more indicator examples?"
    查看 [user_data/strategies/sample_strategy.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_strategy.py)。
    然后取消注释你需要的指标。

#### 指标库

开箱即用，freqtrade 安装了以下技术指标库：

- [ta-lib](https://ta-lib.github.io/ta-lib-python/)（包含函数的详细文档：[Ta-Lib](https://ta-lib.org/)）
- [pandas-ta](https://twopirllc.github.io/pandas-ta/)
- [technical](https://technical.freqtrade.io)

可以根据需要安装额外的技术指标库，或者策略作者可以自行编写/发明自定义指标。

### 策略启动周期

某些指标在启动期间是不稳定的，此时没有足够的 K 线数据来计算任何值（NaN），或者计算结果不正确。这可能导致不一致，因为 Freqtrade 不知道这个不稳定期有多长，并使用 DataFrame 中的指标值。

为了解决这个问题，可以为策略分配 `startup_candle_count` 属性。

该值应设置为策略计算稳定指标所需的最大 K 线数。如果用户在信息交易对中包含更高时间周期，`startup_candle_count` 不一定需要更改。该值是任何信息时间周期计算稳定指标所需的最大周期（以 K 线为单位）。

你可以使用[递归分析](recursive-analysis.md)来检查并找到正确的 `startup_candle_count` 值。当递归分析显示方差为 0% 时，你就可以确认有足够的启动 K 线数据。

在此示例策略中，应设置为 400（`startup_candle_count = 400`），因为 ema100 计算所需的最小历史数据量为 400 根 K 线，以确保值正确。

``` python
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
```

通过让机器人知道需要多少历史数据，回测交易可以在回测和超参数优化期间从指定的时间范围开始。

!!! Warning "Using x calls to get OHLCV"
    如果你收到类似 `WARNING - Using 3 calls to get OHLCV. This can result in slower operations for the bot. Please check if you really need 1500 candles for your strategy` 的警告 - 你应考虑是否真的需要这么多历史数据用于信号。
    这会导致 Freqtrade 对同一交易对发出多次请求，显然比单次网络请求更慢。
    因此，Freqtrade 刷新 K 线的时间会更长 - 应尽可能避免这种情况。
    最多限制为 5 次调用，以避免使交易所过载或使 freqtrade 太慢。

!!! Warning
    `startup_candle_count` 应低于 `ohlcv_candle_limit * 5`（大多数交易所为 500 * 5）- 因为在模拟/实盘交易运行期间，只有这个数量的 K 线可用。

#### 示例

让我们尝试使用上述包含 EMA100 的示例策略来回测 1 个月（2019 年 1 月）的 5 分钟 K 线。

``` bash
freqtrade backtesting --timerange 20190101-20190201 --timeframe 5m
```

假设 `startup_candle_count` 设置为 400，回测知道需要 400 根 K 线来生成有效的入场信号。它将从 `20190101 - (400 * 5m)` 加载数据 - 约为 2018-12-30 11:40:00。

如果这些数据可用，指标将使用这个扩展的时间范围进行计算。不稳定的启动期（到 2019-01-01 00:00:00）将在回测执行之前被移除。

!!! Note "Unavailable startup candle data"
    如果启动期的数据不可用，时间范围将被调整以考虑此启动期。在我们的示例中，回测将从 2019-01-02 09:20:00 开始。

### 入场信号规则

编辑策略文件中的 `populate_entry_trend()` 方法来更新你的入场策略。

始终返回 DataFrame 且不删除/修改 `"open"、"high"、"low"、"close"、"volume"` 列非常重要，否则这些字段将包含意外内容。策略可能会产生无效值，或完全停止工作。

此方法还将定义一个新列 `"enter_long"`（做空时为 `"enter_short"`），其中入场需要包含 `1`，"无操作"为 `0`。`enter_long` 是必须设置的列，即使策略只做空也必须设置。

你可以使用 `"enter_tag"` 列来命名你的入场信号，这有助于稍后调试和评估策略。

来自 `user_data/strategies/sample_strategy.py` 的示例：

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Based on TA indicators, populates the buy signal for the given dataframe
    :param dataframe: DataFrame populated with indicators
    :param metadata: Additional information, like the currently traded pair
    :return: DataFrame with buy column
    """
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
            (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard
            (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        ['enter_long', 'enter_tag']] = (1, 'rsi_cross')

    return dataframe
```

??? Note "Enter short trades"
    可以通过设置 `enter_short` 来创建做空入场（对应做多交易的 `enter_long`）。
    `enter_tag` 列保持不变。
    做空需要你的交易所和市场配置支持！
    此外，如果你打算做空，请确保在策略中正确设置 [`can_short`](#can_short)。

    ```python
    # allow both long and short trades
    can_short = True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
                (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard
                (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, 'rsi_cross')

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['rsi'], 70)) &  # Signal: RSI crosses below 70
                (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard
                (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['enter_short', 'enter_tag']] = (1, 'rsi_cross')

        return dataframe
    ```

!!! Note
    买入需要有卖方才能成交。因此成交量需要大于 0（`dataframe['volume'] > 0`），以确保机器人不会在无活动时段买入/卖出。

### 出场信号规则

编辑策略文件中的 `populate_exit_trend()` 方法来更新你的出场策略。

可以通过在配置或策略中将 `use_exit_signal` 设置为 false 来抑制出场信号。

`use_exit_signal` 不会影响[信号冲突规则](#信号冲突) - 这些规则仍然适用，并可能阻止入场。

始终返回 DataFrame 且不删除/修改 `"open"、"high"、"low"、"close"、"volume"` 列非常重要，否则这些字段将包含意外内容。策略可能会产生无效值，或完全停止工作。

此方法还将定义一个新列 `"exit_long"`（做空时为 `"exit_short"`），其中出场需要包含 `1`，"无操作"为 `0`。

你可以使用 `"exit_tag"` 列来命名你的出场信号，这有助于稍后调试和评估策略。

来自 `user_data/strategies/sample_strategy.py` 的示例：

```python
def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Based on TA indicators, populates the exit signal for the given dataframe
    :param dataframe: DataFrame populated with indicators
    :param metadata: Additional information, like the currently traded pair
    :return: DataFrame with buy column
    """
    dataframe.loc[
        (
            (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
            (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard
            (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard
            (dataframe['volume'] > 0)  # Make sure Volume is not 0
        ),
        ['exit_long', 'exit_tag']] = (1, 'rsi_too_high')
    return dataframe
```

??? Note "Exit short trades"
    可以通过设置 `exit_short` 来创建做空出场（对应 `exit_long`）。
    `exit_tag` 列保持不变。
    做空需要你的交易所和市场配置支持！
    此外，如果你打算做空，请确保在策略中正确设置 [`can_short`](#can_short)。

    ```python
    # allow both long and short trades
    can_short = True

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
                (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard
                (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, 'rsi_too_high')
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['rsi'], 30)) &  # Signal: RSI crosses below 30
                (dataframe['tema'] < dataframe['bb_middleband']) &  # Guard
                (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            ['exit_short', 'exit_tag']] = (1, 'rsi_too_low')
        return dataframe
    ```

### 最小 ROI

`minimal_roi` 策略变量定义了交易在出场前应达到的最小投资回报率（ROI），与出场信号无关。

其格式如下，即一个 Python `dict`，字典键（冒号左侧）是交易开仓后经过的分钟数，值（冒号右侧）是百分比。

```python
minimal_roi = {
    "40": 0.0,
    "30": 0.01,
    "20": 0.02,
    "0": 0.04
}
```

上述配置的含义是：

- 达到 4% 利润时出场
- 达到 2% 利润时出场（20 分钟后生效）
- 达到 1% 利润时出场（30 分钟后生效）
- 交易不亏损时出场（40 分钟后生效）

计算包含手续费。

#### 禁用最小 ROI

要完全禁用 ROI，将其设置为空字典：

```python
minimal_roi = {}
```

#### 在最小 ROI 中使用计算

要使用基于 K 线周期（时间周期）的时间，以下代码片段会很有用。

这将允许你更改策略的时间周期，但最小 ROI 时间仍以 K 线为单位设置，例如 3 根 K 线之后。

``` python
from freqtrade.exchange import timeframe_to_minutes

class AwesomeStrategy(IStrategy):

    timeframe = "1d"
    timeframe_mins = timeframe_to_minutes(timeframe)
    minimal_roi = {
        "0": 0.05,                      # 5% for the first 3 candles
        str(timeframe_mins * 3): 0.02,  # 2% after 3 candles
        str(timeframe_mins * 6): 0.01,  # 1% After 6 candles
    }
```

??? info "Orders that don't fill immediately"
    `minimal_roi` 将以 `trade.open_date` 作为参考，即交易初始化的时间，也就是该交易的第一笔订单下达的时间。
    对于未立即成交的限价订单（通常与通过 `custom_entry_price()` 设定的"偏离点位"价格结合使用），以及通过 `adjust_entry_price()` 替换初始订单价格的情况，这一点同样适用。
    使用的时间仍然是初始 `trade.open_date`（初始订单首次下达的时间），而不是新下达或调整后的订单日期。

### 止损

强烈建议设置止损，以保护你的资金免受不利走势的影响。

设置 10% 止损的示例：

``` python
stoploss = -0.10
```

有关止损功能的完整文档，请查看专门的[止损页面](stoploss.md)。

### 时间周期

这是机器人在策略中应使用的 K 线周期。

常见值包括 `"1m"`、`"5m"`、`"15m"`、`"1h"`，但交易所支持的所有值都应该有效。

请注意，相同的入场/出场信号可能在一个时间周期下表现良好，但在其他时间周期下可能不行。

此设置可以通过策略方法中的 `self.timeframe` 属性访问。

### Can short

要在期货市场中使用做空信号，你需要设置 `can_short = True`。

启用此功能的策略在现货市场中将无法加载。

如果你在 `enter_short` 列中有 `1` 值来触发做空信号，设置 `can_short = False`（默认值）将意味着这些做空信号会被忽略，即使你在配置中指定了期货市场。

### Metadata 字典

`metadata` 字典（可用于 `populate_entry_trend`、`populate_exit_trend`、`populate_indicators`）包含额外信息。
目前其中包含 `pair`，可以使用 `metadata['pair']` 访问，返回格式为 `XRP/BTC`（期货市场为 `XRP/BTC:BTC`）的交易对。

metadata 字典不应被修改，也不会在策略的多个函数之间持久保存信息。

请查看[存储信息](strategy-advanced.md#storing-information-persistent)部分。

--8<-- "includes/strategy-imports.md"

## 策略文件加载

默认情况下，freqtrade 将尝试从 `userdir`（默认为 `user_data/strategies`）中的所有 `.py` 文件加载策略。

假设你的策略名为 `AwesomeStrategy`，存储在文件 `user_data/strategies/AwesomeStrategy.py` 中，那么你可以使用以下命令以模拟（或实盘，取决于你的配置）模式启动 freqtrade：

```bash
freqtrade trade --strategy AwesomeStrategy
```

请注意，我们使用的是类名，而不是文件名。

你可以使用 `freqtrade list-strategies` 查看 Freqtrade 能够加载的所有策略列表（正确文件夹中的所有策略）。
它还将包含一个"状态"字段，突出显示潜在问题。

??? Hint "Customize strategy directory"
    你可以使用 `--strategy-path user_data/otherPath` 来指定不同的目录。此参数可用于所有需要策略的命令。

## 信息交易对

### 获取非交易对的数据

额外的、信息性交易对（参考交易对）的数据对于某些策略查看更广时间周期的数据是有益的。

这些交易对的 OHLCV 数据将作为常规白名单刷新过程的一部分被下载，并可通过 `DataProvider` 与其他交易对一样使用（见下文）。

这些交易对**不会**被交易，除非它们也在交易对白名单中指定，或已被动态白名单（如 `VolumePairList`）选中。

交易对需要以元组格式指定为 `("pair", "timeframe")`，其中 pair 作为第一个参数，timeframe 作为第二个参数。

示例：

``` python
def informative_pairs(self):
    return [("ETH/USDT", "5m"),
            ("BTC/TUSD", "15m"),
            ]
```

完整示例可以在 [DataProvider 部分](#完整的-dataprovider-示例)中找到。

!!! Warning
    由于这些交易对将作为常规白名单刷新的一部分进行刷新，最好保持此列表简短。
    所有时间周期和所有交易对都可以指定，只要它们在所用交易所上可用（且活跃）。
    但是，最好尽可能使用重采样到更长时间周期，
    以避免向交易所发送过多请求而面临被封锁的风险。

??? Note "Alternative candle types"
    informative_pairs 还可以提供第三个元组元素，用于显式定义 K 线类型。
    替代 K 线类型的可用性取决于交易模式和交易所。
    一般来说，现货交易对不能用于期货市场，期货 K 线也不能用作现货机器人的信息交易对。
    具体细节可能有所不同，如有差异，请查看交易所文档。

    ``` python
    def informative_pairs(self):
        return [
            ("ETH/USDT", "5m", ""),   # Uses default candletype, depends on trading_mode (recommended)
            ("ETH/USDT", "5m", "spot"),   # Forces usage of spot candles (only valid for bots running on spot markets).
            ("BTC/TUSD", "15m", "futures"),  # Uses futures candles (only bots with `trading_mode=futures`)
            ("BTC/TUSD", "15m", "mark"),  # Uses mark candles (only bots with `trading_mode=futures`)
        ]
    ```
***

### 信息交易对装饰器（`@informative()`）

要轻松定义信息交易对，请使用 `@informative` 装饰器。所有被装饰的 `populate_indicators_*` 方法都是独立运行的，
无法访问其他信息交易对的数据。但是，每个交易对的所有信息 DataFrame 都会被合并并传递给主 `populate_indicators()` 方法。

!!! Note
    如果你需要在生成一个信息交易对时使用另一个信息交易对的数据，请不要使用 `@informative` 装饰器。应按照 [DataProvider 部分](#完整的-dataprovider-示例)中描述的方式手动定义信息交易对。

在超参数优化时，不支持使用超参数的 `.value` 属性。请使用 `.range` 属性。有关更多信息，请参阅[优化指标参数](hyperopt.md#optimizing-an-indicator-parameter)。

??? info "Full documentation"
    ``` python
    def informative(
        timeframe: str,
        asset: str = "",
        fmt: str | Callable[[Any], str] | None = None,
        *,
        candle_type: CandleType | str | None = None,
        ffill: bool = True,
    ) -> Callable[[PopulateIndicators], PopulateIndicators]:
        """
        A decorator for populate_indicators_Nn(self, dataframe, metadata), allowing these functions to
        define informative indicators.

        Example usage:

            @informative('1h')
            def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
                dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
                return dataframe

        :param timeframe: Informative timeframe. Must always be equal or higher than strategy timeframe.
        :param asset: Informative asset, for example BTC, BTC/USDT, ETH/BTC. Do not specify to use
                    current pair. Also supports limited pair format strings (see below)
        :param fmt: Column format (str) or column formatter (callable(name, asset, timeframe)). When not
        specified, defaults to:
        * {base}_{quote}_{column}_{timeframe} if asset is specified.
        * {column}_{timeframe} if asset is not specified.
        Pair format supports these format variables:
        * {base} - base currency in lower case, for example 'eth'.
        * {BASE} - same as {base}, except in upper case.
        * {quote} - quote currency in lower case, for example 'usdt'.
        * {QUOTE} - same as {quote}, except in upper case.
        Format string additionally supports this variables.
        * {asset} - full name of the asset, for example 'BTC/USDT'.
        * {column} - name of dataframe column.
        * {timeframe} - timeframe of informative dataframe.
        :param ffill: ffill dataframe after merging informative pair.
        :param candle_type: '', mark, index, premiumIndex, or funding_rate
        """
    ```

??? Example "Fast and easy way to define informative pairs"

    大多数时候我们不需要 `merge_informative_pair()` 提供的强大和灵活性，因此可以使用装饰器快速定义信息交易对。

    ``` python

    from datetime import datetime
    from freqtrade.persistence import Trade
    from freqtrade.strategy import IStrategy, informative

    class AwesomeStrategy(IStrategy):
        
        # This method is not required. 
        # def informative_pairs(self): ...

        # Define informative upper timeframe for each pair. Decorators can be stacked on same 
        # method. Available in populate_indicators as 'rsi_30m' and 'rsi_1h'.
        @informative('30m')
        @informative('1h')
        def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe

        # Define BTC/STAKE informative pair. Available in populate_indicators and other methods as
        # 'btc_rsi_1h'. Current stake currency should be specified as {stake} format variable 
        # instead of hard-coding actual stake currency. Available in populate_indicators and other 
        # methods as 'btc_usdt_rsi_1h' (when stake currency is USDT).
        @informative('1h', 'BTC/{stake}')
        def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe

        # Define BTC/ETH informative pair. You must specify quote currency if it is different from
        # stake currency. Available in populate_indicators and other methods as 'eth_btc_rsi_1h'.
        @informative('1h', 'ETH/BTC')
        def populate_indicators_eth_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe
    
        # Define BTC/STAKE informative pair. A custom formatter may be specified for formatting
        # column names. A callable `fmt(**kwargs) -> str` may be specified, to implement custom
        # formatting. Available in populate_indicators and other methods as 'rsi_upper_1h'.
        @informative('1h', 'BTC/{stake}', '{column}_{timeframe}')
        def populate_indicators_btc_1h_2(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            dataframe['rsi_upper'] = ta.RSI(dataframe, timeperiod=14)
            return dataframe
    
        def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            # Strategy timeframe indicators for current pair.
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            # Informative pairs are available in this method.
            dataframe['rsi_less'] = dataframe['rsi'] < dataframe['rsi_1h']
            return dataframe

    ```

!!! Note
    在访问其他交易对的信息 DataFrame 时使用字符串格式化。这样可以轻松在配置中更改质押货币，而无需调整策略代码。

    ``` python
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stake = self.config['stake_currency']
        dataframe.loc[
            (
                (dataframe[f'btc_{stake}_rsi_1h'] < 35)
                &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'buy_signal_rsi')
    
        return dataframe
    ```

    或者可以使用列重命名来从列名中移除质押货币：`@informative('1h', 'BTC/{stake}', fmt='{base}_{column}_{timeframe}')`。

!!! Warning "Duplicate method names"
    使用 `@informative()` 装饰器标记的方法必须始终具有唯一的名称！重用相同的名称（例如复制粘贴已定义的信息方法时）将覆盖先前定义的方法，并且由于 Python 编程语言的限制不会产生任何错误。在这种情况下，你会发现策略文件中较高位置创建的指标在 DataFrame 中不可用。请仔细检查方法名称并确保它们是唯一的！

### *merge_informative_pair()*

此方法帮助你安全且一致地将信息交易对合并到常规主 DataFrame 中，不带前瞻偏差。

选项：

- 重命名列以创建唯一的列
- 合并 DataFrame 而不带前瞻偏差
- 前向填充（可选）

有关完整示例，请参考下面的[完整 DataProvider 示例](#完整的-dataprovider-示例)。

信息 DataFrame 的所有列都将以重命名后的形式在返回的 DataFrame 中可用：

!!! Example "Column renaming"
    假设 `inf_tf = '1d'`，结果列将为：

    ``` python
    'date', 'open', 'high', 'low', 'close', 'rsi'                     # from the original dataframe
    'date_1d', 'open_1d', 'high_1d', 'low_1d', 'close_1d', 'rsi_1d'   # from the informative dataframe
    ```

??? Example "Column renaming - 1h"
    假设 `inf_tf = '1h'`，结果列将为：

    ``` python
    'date', 'open', 'high', 'low', 'close', 'rsi'                     # from the original dataframe
    'date_1h', 'open_1h', 'high_1h', 'low_1h', 'close_1h', 'rsi_1h'   # from the informative dataframe
    ```

??? Example "Custom implementation"
    可以自定义实现，方法如下：

    ``` python

    # Shift date by 1 candle
    # This is necessary since the data is always the "open date"
    # and a 15m candle starting at 12:15 should not know the close of the 1h candle from 12:00 to 13:00
    minutes = timeframe_to_minutes(inf_tf)
    # Only do this if the timeframes are different:
    informative['date_merge'] = informative["date"] + pd.to_timedelta(minutes, 'm')

    # Rename columns to be unique
    informative.columns = [f"{col}_{inf_tf}" for col in informative.columns]
    # Assuming inf_tf = '1d' - then the columns will now be:
    # date_1d, open_1d, high_1d, low_1d, close_1d, rsi_1d

    # Combine the 2 dataframes
    # all indicators on the informative sample MUST be calculated before this point
    dataframe = pd.merge(dataframe, informative, left_on='date', right_on=f'date_merge_{inf_tf}', how='left')
    # FFill to have the 1d value available in every row throughout the day.
    # Without this, comparisons would only work once per day.
    dataframe = dataframe.ffill()

    ```

!!! Warning "Informative timeframe < timeframe"
    不建议使用此方法时将信息时间周期设置为小于主 DataFrame 的时间周期，因为它不会使用这可能提供的任何额外信息。
    要正确使用更详细的信息，应采用更高级的方法（这超出了本文档的范围）。

## 额外数据（DataProvider）

策略提供了对 `DataProvider` 的访问。这允许你获取额外的数据用于策略。

所有方法在失败时返回 `None`，即失败不会引发异常。

请始终检查操作模式以选择正确的获取数据方法（参见下面的示例）。

!!! Warning "Hyperopt Limitations"
    DataProvider 在超参数优化期间可用，但它只能在**策略中的** `populate_indicators()` 中使用，而不能在超参数优化类文件中使用。
    它在 `populate_entry_trend()` 和 `populate_exit_trend()` 方法中也不可用。

### DataProvider 的可用选项

- [`available_pairs`](#available_pairs) - 属性，列出缓存的交易对及其时间周期的元组（pair, timeframe）。
- [`current_whitelist()`](#current_whitelist) - 返回当前白名单交易对列表。对于访问动态白名单（如 VolumePairlist）很有用。
- [`get_pair_dataframe(pair, timeframe)`](#get_pair_dataframepair-timeframe) - 这是一个通用方法，返回历史数据（用于回测）或缓存的实时数据（用于模拟和实盘模式）。
- [`get_analyzed_dataframe(pair, timeframe)`](#get_analyzed_dataframepair-timeframe) - 返回分析后的 DataFrame（在调用 `populate_indicators()`、`populate_buy()`、`populate_sell()` 之后）和最新分析的时间。
- `historic_ohlcv(pair, timeframe)` - 返回存储在磁盘上的历史数据。
- `market(pair)` - 返回交易对的市场数据：手续费、限制、精度、活动标志等。有关 Market 数据结构的更多详细信息，请参阅 [ccxt 文档](https://github.com/ccxt/ccxt/wiki/Manual#markets)。
- `ohlcv(pair, timeframe)` - 交易对当前缓存的 K 线（OHLCV）数据，返回 DataFrame 或空 DataFrame。
- [`orderbook(pair, maximum)`](#orderbookpair-maximum) - 返回交易对的最新订单簿数据，一个包含 bids/asks 的字典，总共 `maximum` 个条目。
- [`ticker(pair)`](#tickerpair) - 返回交易对的当前行情数据。有关 Ticker 数据结构的更多详细信息，请参阅 [ccxt 文档](https://github.com/ccxt/ccxt/wiki/Manual#price-tickers)。
- [`check_delisting(pair)`](#check_delistingpair) - 如果有，返回交易对退市计划的日期时间，否则返回 None。
- [`funding_rate(pair)`](#funding_ratepair) - 返回交易对的当前资金费率数据。
- `runmode` - 包含当前运行模式的属性。

### 使用示例

### *available_pairs*

``` python
for pair, timeframe in self.dp.available_pairs:
    print(f"available {pair}, {timeframe}")
```

### *current_whitelist()*

假设你开发了一个使用 `5m` 时间周期的策略，信号来自按交易量排名前 10 的交易所交易对的 `1d` 时间周期。

策略逻辑可能类似于这样：

*每 5 分钟使用 `VolumePairList` 扫描按交易量排名前 10 的交易对，并使用 14 天 RSI 进行入场和出场。*

由于可用数据有限，将 `5m` K 线重采样为日 K 线用于 14 天 RSI 非常困难。大多数交易所将用户限制在仅 500-1000 根 K 线，这实际上只给我们约 1.74 根日 K 线。我们至少需要 14 天！

由于我们无法重采样数据，我们将不得不使用信息交易对，而且由于白名单是动态的，我们不知道使用哪个交易对！我们遇到了问题！

这时调用 `self.dp.current_whitelist()` 就派上用场了，它只检索白名单中的交易对。

```python
    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign timeframe to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        return informative_pairs
```

??? Note "Plotting with current_whitelist"
    当前白名单不支持 `plot-dataframe`，因为此命令通常通过提供显式的交易对列表来使用，因此会使此方法的返回值产生误导。
    它也不支持 [webserver 模式](utils.md#webserver-mode)下的 FreqUI 可视化，因为 webserver 模式的配置不需要设置交易对列表。

### *get_pair_dataframe(pair, timeframe)*

``` python
# fetch live / historical candle (OHLCV) data for the first informative pair
inf_pair, inf_timeframe = self.informative_pairs()[0]
informative = self.dp.get_pair_dataframe(pair=inf_pair,
                                         timeframe=inf_timeframe)
```

!!! Warning "Warning about backtesting"
    在回测中，`dp.get_pair_dataframe()` 的行为取决于调用位置。
    在 `populate_*()` 方法中，`dp.get_pair_dataframe()` 返回完整的时间范围。请确保不要"窥探未来"以避免在模拟/实盘模式下运行时出现意外。
    在[回调函数](strategy-callbacks.md)中，你将获得截至当前（模拟）K 线的完整时间范围。

### *get_analyzed_dataframe(pair, timeframe)*

此方法由 freqtrade 内部使用以确定最后的信号。
它也可以在特定回调函数中使用，以获取导致操作的信号（有关可用回调函数的更多详细信息，请参阅[高级策略文档](strategy-advanced.md)）。

``` python
# fetch current dataframe
dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=metadata['pair'],
                                                         timeframe=self.timeframe)
```

!!! Note "No data available"
    如果请求的交易对未被缓存，则返回空 DataFrame。
    你可以使用 `if dataframe.empty:` 检查并相应处理此情况。
    使用白名单交易对时不应出现此情况。

### *orderbook(pair, maximum)*

检索交易对的当前订单簿。

``` python
if self.dp.runmode.value in ('live', 'dry_run'):
    ob = self.dp.orderbook(metadata['pair'], 1)
    dataframe['best_bid'] = ob['bids'][0][0]
    dataframe['best_ask'] = ob['asks'][0][0]
```

订单簿结构与 [ccxt](https://github.com/ccxt/ccxt/wiki/Manual#order-book-structure) 的订单结构一致，因此结果格式如下：

``` js
{
    'bids': [
        [ price, amount ], // [ float, float ]
        [ price, amount ],
        ...
    ],
    'asks': [
        [ price, amount ],
        [ price, amount ],
        //...
    ],
    //...
}
```

因此，如上所示使用 `ob['bids'][0][0]` 将使用最佳买价。`ob['bids'][0][1]` 将查看该订单簿位置的数量。

!!! Warning "Warning about backtesting"
    订单簿不是历史数据的一部分，这意味着如果使用此方法，回测和超参数优化将无法正常工作，因为该方法将返回最新值。

### *ticker(pair)*

``` python
if self.dp.runmode.value in ('live', 'dry_run'):
    ticker = self.dp.ticker(metadata['pair'])
    dataframe['last_price'] = ticker['last']
    dataframe['volume24h'] = ticker['quoteVolume']
    dataframe['vwap'] = ticker['vwap']
```

!!! Warning
    尽管 ticker 数据结构是 ccxt 统一接口的一部分，但此方法返回的值可能因交易所而异。
    例如，许多交易所不返回 `vwap` 值，有些交易所并不总是填充 `last` 字段（因此可能为 None），等等。因此你需要仔细验证从交易所返回的 ticker 数据，并添加适当的错误处理/默认值。

!!! Warning "Warning about backtesting"
    此方法将始终返回最新/实时值。因此，在回测/超参数优化期间使用而不检查运行模式将导致错误结果，例如你的整个 DataFrame 所有行将包含相同的单个值。

### *check_delisting(pair)*

如果有，返回交易对退市计划的日期时间，否则返回 None。

```python
def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
    if self.dp.runmode.value in ('live', 'dry_run'):
        delisting_dt = self.dp.check_delisting(pair)
        if delisting_dt is not None:
            return "delist"
```

!!! Note "Availabiity of delisting information"
    此方法仅适用于某些交易所，在不可用或交易对未安排退市的情况下将返回 `None`。

!!! Warning "Warning about backtesting"
    此方法将始终返回最新/实时值。因此，在回测/超参数优化期间使用而不检查运行模式将导致错误结果，例如你的整个 DataFrame 所有行将包含相同的单个值。

### *funding_rate(pair)*

检索交易对的当前资金费率，仅适用于格式为 `base/quote:settle` 的期货交易对（例如 `ETH/USDT:USDT`）。

``` python
if self.dp.runmode.value in ('live', 'dry_run'):
    funding_rate = self.dp.funding_rate(metadata['pair'])
    dataframe['current_funding_rate'] = funding_rate['fundingRate']
    dataframe['next_funding_timestamp'] = funding_rate['fundingTimestamp']
    dataframe['next_funding_datetime'] = funding_rate['fundingDatetime']
```

资金费率结构与 [ccxt](https://github.com/ccxt/ccxt/wiki/Manual#funding-rate-structure) 的资金费率结构一致，因此结果格式如下：

``` python
{
    "info": {
        # ... 
    },
    "symbol": "BTC/USDT:USDT",
    "markPrice": 110730.7,
    "indexPrice": 110782.52,
    "interestRate": 0.0001,
    "estimatedSettlePrice": 110822.67200153,
    "timestamp": 1757146321001,
    "datetime": "2025-09-06T08:12:01.001Z",
    "fundingRate": 5.609e-05,
    "fundingTimestamp": 1757174400000,
    "fundingDatetime": "2025-09-06T16:00:00.000Z",
    "nextFundingRate": None,
    "nextFundingTimestamp": None,
    "nextFundingDatetime": None,
    "previousFundingRate": None,
    "previousFundingTimestamp": None,
    "previousFundingDatetime": None,
    "interval": None,
}
```

因此，如上所示使用 `funding_rate['fundingRate']` 将使用当前资金费率。
实际可用数据因交易所而异，因此此代码可能无法在所有交易所中按预期工作。

!!! Warning "Warning about backtesting"
    当前资金费率不是历史数据的一部分，这意味着如果使用此方法，回测和超参数优化将无法正常工作，因为该方法将返回最新值。
    我们建议在回测中使用历史可用资金费率（它会自动下载，频率取决于交易所提供的数据，通常为 4 小时或 8 小时）。
    `self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='8h', candle_type="funding_rate")`

### 发送通知

DataProvider 的 `.send_msg()` 函数允许你从策略中发送自定义通知。
相同的通知每根 K 线只发送一次，除非第二个参数（`always_send`）设置为 True。

``` python
    self.dp.send_msg(f"{metadata['pair']} just got hot!")

    # Force send this notification, avoid caching (Please read warning below!)
    self.dp.send_msg(f"{metadata['pair']} just got hot!", always_send=True)
```

通知只在交易模式（实盘/模拟）下发送 - 因此此方法可以在回测中无条件调用。

!!! Warning "Spamming"
    通过在此方法中设置 `always_send=True`，你可能会给自己发送大量垃圾消息。请谨慎使用，仅在你确定不会在整个 K 线期间发生的条件下使用，以避免每 5 秒收到一条消息。

### 完整的 DataProvider 示例

```python
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame

class SampleStrategy(IStrategy):
    # strategy init stuff...

    timeframe = '5m'

    # more strategy init stuff..

    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1d') for pair in pairs]
        # Optionally Add additional "static" pairs
        informative_pairs += [("ETH/USDT", "5m"),
                              ("BTC/TUSD", "15m"),
                            ]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        inf_tf = '1d'
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        # Get the 14 day rsi
        informative['rsi'] = ta.RSI(informative, timeperiod=14)

        # Use the helper function merge_informative_pair to safely merge the pair
        # Automatically renames the columns and merges a shorter timeframe dataframe and a longer timeframe informative pair
        # use ffill to have the 1d value available in every row throughout the day.
        # Without this, comparisons between columns of the original and the informative pair would only work once per day.
        # Full documentation of this method, see below
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        # Calculate rsi of the original dataframe (5m timeframe)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Do other stuff
        # ...

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
                (dataframe['rsi_1d'] < 30) &                     # Ensure daily RSI is < 30
                (dataframe['volume'] > 0)                        # Ensure this candle had volume (important for backtesting)
            ),
            ['enter_long', 'enter_tag']] = (1, 'rsi_cross')

```

***

## 额外数据（Wallets）

策略提供了对 `wallets` 对象的访问。它包含你在交易所的钱包/账户的当前余额。

!!! Note "Backtesting / Hyperopt"
    Wallets 的行为取决于调用它的函数。
    在 `populate_*()` 方法中，它将返回配置的完整钱包。
    在[回调函数](strategy-callbacks.md)中，你将获得与模拟过程中该时点实际模拟钱包对应的钱包状态。

始终检查 `wallets` 是否可用，以避免回测期间失败。

``` python
if self.wallets:
    free_eth = self.wallets.get_free('ETH')
    used_eth = self.wallets.get_used('ETH')
    total_eth = self.wallets.get_total('ETH')
```

### Wallets 的可用选项

- `get_free(asset)` - 当前可用于交易的余额
- `get_used(asset)` - 当前被占用的余额（未完成订单）
- `get_total(asset)` - 总可用余额 - 上述两者的总和

***

## 额外数据（Trades）

可以通过查询数据库在策略中检索交易历史。

在文件顶部，导入所需的对象：

```python
from freqtrade.persistence import Trade
```

以下示例查询当前交易对（`metadata['pair']`）今天以来的交易。可以轻松添加其他过滤条件。

``` python
trades = Trade.get_trades_proxy(pair=metadata['pair'],
                                open_date=datetime.now(timezone.utc) - timedelta(days=1),
                                is_open=False,
            ]).order_by(Trade.close_date).all()
# Summarize profit for this pair.
curdayprofit = sum(trade.close_profit for trade in trades)
```

有关可用方法的完整列表，请参阅 [Trade 对象](trade-object.md)文档。

!!! Warning
    在回测或超参数优化期间，交易历史在 `populate_*` 方法中不可用，将返回空结果。

## 阻止特定交易对的交易发生

Freqtrade 在交易对退出时自动锁定该交易对的当前 K 线（直到该 K 线结束），防止该交易对立即重新入场。

这是为了防止在单根 K 线内发生大量频繁交易的"瀑布"效应。

被锁定的交易对将显示消息 `Pair <pair> is currently locked.`。

### 从策略内部锁定交易对

有时可能需要在某些事件发生后锁定交易对（例如连续多笔亏损交易）。

Freqtrade 提供了一种从策略内部轻松实现此操作的方法，即调用 `self.lock_pair(pair, until, [reason])`。
`until` 必须是未来的 datetime 对象，之后该交易对将重新启用交易，而 `reason` 是可选字符串，详细说明锁定交易对的原因。

也可以手动解除锁定，通过调用 `self.unlock_pair(pair)` 或 `self.unlock_reason(<reason>)`，提供解锁交易对的原因。
`self.unlock_reason(<reason>)` 将解锁当前因提供的所有被锁定的交易对。

要验证交易对当前是否被锁定，请使用 `self.is_pair_locked(pair)`。

!!! Note
    被锁定的交易对将始终向上舍入到下一根 K 线。因此，假设 `5m` 时间周期，`until` 设置为 10:18 的锁定将把交易对锁定到 10:15-10:20 的 K 线完成为止。

!!! Warning
    在回测期间，手动锁定交易对不可用。只允许通过保护机制进行锁定。

#### 交易对锁定示例

``` python
from freqtrade.persistence import Trade
from datetime import timedelta, datetime, timezone
# Put the above lines at the top of the strategy file, next to all the other imports
# --------

# Within populate indicators (or populate_entry_trend):
if self.config['runmode'].value in ('live', 'dry_run'):
    # fetch closed trades for the last 2 days
    trades = Trade.get_trades_proxy(
        pair=metadata['pair'], is_open=False, 
        open_date=datetime.now(timezone.utc) - timedelta(days=2))
    # Analyze the conditions you'd like to lock the pair .... will probably be different for every strategy
    sumprofit = sum(trade.close_profit for trade in trades)
    if sumprofit < 0:
        # Lock pair for 12 hours
        self.lock_pair(metadata['pair'], until=datetime.now(timezone.utc) + timedelta(hours=12))
```

## 打印主 DataFrame

要检查当前的主 DataFrame，你可以在 `populate_entry_trend()` 或 `populate_exit_trend()` 中发出 print 语句。
你可能还想打印交易对，以便清楚地知道当前显示的是什么数据。

``` python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            #>> whatever condition<<<
        ),
        ['enter_long', 'enter_tag']] = (1, 'somestring')

    # Print the Analyzed pair
    print(f"result for {metadata['pair']}")

    # Inspect the last 5 rows
    print(dataframe.tail())

    return dataframe
```

也可以使用 `print(dataframe)` 代替 `print(dataframe.tail())` 来打印更多行。但不建议这样做，因为可能产生大量输出（每个交易对每 5 秒约 500 行）。

## 策略开发中的常见错误

### 回测时窥探未来

回测出于性能原因会一次性分析整个 DataFrame 的时间范围。因此，策略作者需要确保策略不会前瞻窥探未来，即使用在模拟或实盘模式中不可用的数据。

这是一个常见的痛点，可能导致回测和模拟/实盘运行方法之间的巨大差异。窥探未来的策略在回测期间表现良好，通常具有令人难以置信的利润或胜率，但在实际条件下会失败或表现不佳。

以下列表包含一些应避免的常见模式，以防止挫败感：

- 不要使用 `shift(-1)` 或其他负值。这在回测中使用了未来数据，在模拟或实盘模式中不可用。
- 不要在 `populate_` 函数中使用 `.iloc[-1]` 或 DataFrame 中的任何其他绝对位置，因为在模拟运行和回测之间这会不同。然而，在回调函数中使用绝对 `iloc` 索引是安全的 - 参见[策略回调函数](strategy-callbacks.md)。
- 不要使用使用所有 DataFrame 或列值的函数，例如 `dataframe['mean_volume'] = dataframe['volume'].mean()`。由于回测使用完整的 DataFrame，在 DataFrame 的任何位置，`'mean_volume'` 系列将包含来自未来的数据。请改用 rolling() 计算，例如 `dataframe['volume'].rolling(<window>).mean()`。
- 不要使用 `.resample('1h')`。这使用周期区间的左边界，因此将数据从小时边界移动到小时的开始。请改用 `.resample('1h', label='right')`。
- 不要使用 `.merge()` 将更长时间周期合并到更短时间周期上。应使用[信息交易对](#信息交易对)辅助方法。（普通合并可能隐式导致前瞻偏差，因为日期指的是开盘日期，而不是收盘日期）。

!!! Tip "Identifying problems"
    你应始终使用两个辅助命令 [lookahead-analysis](lookahead-analysis.md) 和 [recursive-analysis](recursive-analysis.md)，它们各自可以帮助你以不同方式找出策略的问题。
    请将它们视为识别最常见问题的辅助工具。每个的负面结果并不能保证没有包含上述任何错误。

### 信号冲突

当冲突的信号碰撞时（例如 `'enter_long'` 和 `'exit_long'` 都设置为 `1`），freqtrade 将不执行任何操作并忽略入场信号。这将避免交易入场后立即出场。显然，这可能导致错过入场。

如果 3 个信号中有超过一个被设置，以下规则适用，入场信号将被忽略：

- `enter_long` -> `exit_long`, `enter_short`
- `enter_short` -> `exit_short`, `enter_long`

## 更多策略思路

要获取更多策略思路，请访问[策略仓库](https://github.com/freqtrade/freqtrade-strategies)。欢迎将它们用作示例，但结果将取决于当前市场状况、使用的交易对等。因此，这些策略应仅被视为学习目的，而非实际交易。请先针对你的交易所/期望的交易对回测策略，然后进行模拟运行以仔细评估，使用风险自负。

欢迎将其中任何策略作为你自己策略的灵感来源。我们很乐意接受包含新策略的 Pull Request 到仓库中。

## 后续步骤

现在你有了一个完美的策略，你可能想要回测它。
你的下一步是学习[如何使用回测](backtesting.md)。