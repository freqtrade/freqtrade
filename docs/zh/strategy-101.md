<!-- 本文件为中文翻译版，由 AI 根据 docs/strategy-101.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件（如 exchanges.md），这些文件将在逐步翻译过程中补齐。 -->
<!-- 图片与 includes 引用使用 ../ 指向英文原文档资源，以保证显示正常。 -->

# Freqtrade 策略 101：策略开发快速入门

在本快速入门中，我们假设你已经熟悉交易的基础知识，并已阅读了 [Freqtrade 基础](bot-basics.md) 页面。

## 必备知识

Freqtrade 中的策略是一个 Python 类，用于定义买入和卖出加密货币`资产`的逻辑。

资产被定义为`交易对`，代表`币种`和`计价资产`。币种是你正在交易的资产，而计价资产则是用于交易的另一种货币。

数据以`K线`的形式由交易所提供，每根 K 线由六个值组成：`date`（日期）、`open`（开盘价）、`high`（最高价）、`low`（最低价）、`close`（收盘价）和 `volume`（成交量）。

`技术分析`函数通过各种计算和统计公式对 K 线数据进行分析，并产生称为`指标`的二级数值。

指标在资产交易对的 K 线数据上进行分析，以生成`信号`。

信号在加密货币`交易所`上转化为`订单`，即`交易`。

我们使用`入场`和`出场`这两个术语来代替`买入`和`卖出`，因为 Freqtrade 同时支持`做多`和`做空`交易。

- **做多（long）**：你基于计价资产买入币种，例如使用 USDT 作为计价资产买入 BTC，并通过以高于买入价的价格卖出该币种来获利。在做多交易中，利润来自于币种相对于计价资产的升值。
- **做空（short）**：你从交易所借入币种形式的资金，并在之后偿还该币种的计价资产价值。在做空交易中，利润来自于币种相对于计价资产的贬值（你以更低的价格偿还借款）。

虽然 Freqtrade 对某些交易所支持现货和期货市场，但为简单起见，我们将仅关注现货（做多）交易。

## 基本策略的结构

### 主数据框

Freqtrade 策略使用一种称为`数据框（dataframe）`的表格数据结构（包含行和列）来生成入场和出场信号。

你配置的交易对列表中的每个交易对都有自己的数据框。数据框以 `date` 列作为索引，例如 `2024-06-31 12:00`。

接下来的 5 列代表 `open`（开盘价）、`high`（最高价）、`low`（最低价）、`close`（收盘价）和 `volume`（成交量），即 OHLCV 数据。

### 填充指标值

`populate_indicators` 函数向数据框中添加代表技术分析指标值的列。

常见指标的示例包括相对强弱指数（RSI）、布林带（Bollinger Bands）、资金流量指数（MFI）、移动平均线（MA）和平均真实波幅（ATR）。

通过调用技术分析函数将列添加到数据框中，例如 ta-lib 的 RSI 函数 `ta.RSI()`，并将其赋值给一个列名，例如 `rsi`。

```python
dataframe['rsi'] = ta.RSI(dataframe)
```

??? Hint "技术分析库"
    不同的库生成指标值的方式各不相同。请查阅每个库的文档以了解如何将其集成到你的策略中。你也可以查看 [Freqtrade 示例策略](https://github.com/freqtrade/freqtrade-strategies) 以获取灵感。

### 填充入场信号

`populate_entry_trend` 函数定义入场信号的条件。

数据框的 `enter_long` 列被添加到数据框中，当该列中的值为 `1` 时，Freqtrade 会识别为一个入场信号。

??? Hint "做空"
    要进入做空交易，请使用 `enter_short` 列。

### 填充出场信号

`populate_exit_trend` 函数定义出场信号的条件。

数据框的 `exit_long` 列被添加到数据框中，当该列中的值为 `1` 时，Freqtrade 会识别为一个出场信号。

??? Hint "做空"
    要退出做空交易，请使用 `exit_short` 列。

## 一个简单的策略

以下是一个 Freqtrade 策略的最小示例：

```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class MyStrategy(IStrategy):

    timeframe = '15m'

    # set the initial stoploss to -10%
    stoploss = -0.10

    # exit profitable positions at any time when the profit is greater than 1%
    minimal_roi = {"0": 0.01}

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

## 进行交易

当发现信号（在入场或出场列中出现 `1`）时，Freqtrade 将尝试创建订单，即一笔`交易`或`仓位`。

每个新的交易仓位占用一个`槽位`。槽位代表可以同时开设的新交易的最大数量。

槽位的数量由 `max_open_trades` [配置](configuration.md) 选项定义。

然而，在某些情况下，生成信号并不一定会创建交易订单。这些情况包括：

- 没有足够的剩余计价资产来买入资产，或钱包中没有足够的资金来卖出资产（包括任何手续费）
- 没有足够的空闲槽位来开设新交易（你当前持有的仓位数量等于 `max_open_trades` 选项）
- 该交易对已有一个未平仓交易（Freqtrade 无法堆叠仓位——但可以[调整现有仓位](strategy-callbacks.md#adjust-trade-position)）
- 如果在同一根 K 线上同时出现入场和出场信号，它们被视为[冲突](strategy-customization.md#colliding-signals)，不会产生订单
- 策略通过使用相关的[入场](strategy-callbacks.md#trade-entry-buy-order-confirmation)或[出场](strategy-callbacks.md#trade-exit-sell-order-confirmation)回调中的逻辑，主动拒绝交易订单

请阅读[策略自定义](strategy-customization.md)文档以了解更多细节。

## 回测和模拟测试

策略开发可能是一个漫长且令人沮丧的过程，因为将我们的"直觉"转化为可运行的计算机控制（"算法"）策略并不总是一帆风顺的。

因此，应该对策略进行测试，以验证它是否能按预期工作。

Freqtrade 有两种测试模式：

- **回测（backtesting）**：使用你[从交易所下载的历史数据](data-download.md)，回测是评估策略表现的快速方法。然而，很容易使结果失真，使策略看起来比实际情况更有利可图。请查看[回测文档](backtesting.md)以了解更多信息。
- **模拟运行（dry run）**：通常称为_模拟测试_，模拟运行使用交易所的实时数据。Freqtrade 正常跟踪所有会产生交易的信号，但不会在交易所上实际开设任何交易。模拟测试是实时运行的，因此虽然获得结果需要更长时间，但它是比回测更可靠的**潜在**表现指标。

通过在[配置](configuration.md#using-dry-run-mode)中将 `dry_run` 设置为 true 来启用模拟运行。

!!! Warning "回测可能非常不准确"
    回测结果可能与实际情况不一致的原因有很多。请查看[回测假设](backtesting.md#assumptions-made-by-backtesting)和[常见策略错误](strategy-customization.md#common-mistakes-when-developing-strategies)文档。
    一些列出和排名 Freqtrade 策略的网站展示了令人印象深刻的回测结果。不要假设这些结果是可实现或真实的。

??? Hint "有用的命令"
    Freqtrade 包含两个有用的命令来检查策略中的基本缺陷：[前瞻分析](lookahead-analysis.md)和[递归分析](recursive-analysis.md)。

### 评估回测和模拟运行结果

在回测策略后，务必进行模拟运行，以查看回测和模拟运行的结果是否足够相似。

如果存在任何显著差异，请验证你的入场和出场信号在两种模式之间是否一致且出现在相同的 K 线上。但是，模拟运行和回测之间总会存在差异：

- 回测假设所有订单都会成交。在模拟运行中，如果使用限价单或交易所没有成交量，情况可能并非如此。
- 在 K 线收盘时跟随入场信号，回测假设交易在下一根 K 线的开盘价进入（除非你的策略中有自定义定价回调）。在模拟运行中，信号和交易开设之间通常存在延迟。这是因为当新 K 线在你的主时间框架到来时（例如每 5 分钟），Freqtrade 需要时间来分析所有交易对的数据框。因此，Freqtrade 将在 K 线开盘后几秒（理想情况下尽可能短的延迟）尝试开设交易。
- 由于模拟运行中的入场价格可能与回测不匹配，利润计算也会有所不同。因此，如果 ROI、止损、追踪止损和回调退出不完全一致，这是正常的。
- 新 K 线到来、信号产生和交易开设之间的计算"延迟"越大，价格的不可预测性就越大。确保你的计算机有足够的处理能力，在合理的时间内处理你交易对列表中的所有交易对的数据。如果存在显著的数据处理延迟，Freqtrade 会在日志中发出警告。

## 控制或监控正在运行的机器人

当你的机器人在模拟或实盘模式下运行时，Freqtrade 提供六种机制来控制或监控正在运行的机器人：

- **[FreqUI](freq-ui.md)**：最容易上手的方式，FreqUI 是一个 Web 界面，用于查看和控制机器人的当前活动。
- **[Telegram](telegram-usage.md)**：在移动设备上，可以使用 Telegram 集成来获取机器人活动的警报并控制某些方面。
- **[FTUI](https://github.com/freqtrade/ftui)**：FTUI 是 Freqtrade 的终端（命令行）界面，仅允许监控正在运行的机器人。
- **[freqtrade-client](rest-api.md#consuming-the-api)**：REST API 的 Python 实现，方便你从 Python 应用程序或命令行发送请求和获取机器人响应。
- **[REST API 端点](rest-api.md#available-endpoints)**：REST API 允许程序员开发自己的工具来与 Freqtrade 机器人进行交互。
- **[Webhooks](webhook-config.md)**：Freqtrade 可以通过 webhook 将信息发送到其他服务，例如 Discord。

### 日志

Freqtrade 生成大量的调试日志来帮助你了解正在发生的事情。请熟悉你可能在机器人日志中看到的信息和错误消息。

默认情况下，日志输出到标准输出（命令行）。如果你想将日志写入文件，许多 Freqtrade 命令（包括 `trade` 命令）都接受 `--logfile` 选项来写入文件。

请查看[常见问题](faq.md#how-do-i-search-the-bot-logs-for-something)以获取示例。

## 最后的思考

算法交易是困难的，大多数公开策略的表现并不好，因为让策略在多种场景下盈利需要大量的时间和精力。

因此，使用公开策略并以回测作为评估表现的方式通常是有问题的。然而，Freqtrade 提供了有用的方式来帮助你做出决策并进行尽职调查。

实现盈利有许多不同的方法，没有单一的技巧、诀窍或配置选项能够修复表现不佳的策略。

Freqtrade 是一个拥有庞大且乐于助人的社区的开源平台——请务必访问我们的 [Discord 频道](https://discord.gg/p7nuUNVfP7) 与其他人讨论你的策略！

一如既往，只投资你能承受损失的金额。

## 结论

在 Freqtrade 中开发策略涉及基于技术指标定义入场和出场信号。通过遵循上述结构和方法，你可以创建和测试你自己的交易策略。

常见问题和解答可在我们的[常见问题](faq.md)页面上找到。

如需进一步了解，请参阅更深入的 [Freqtrade 策略自定义文档](strategy-customization.md)。