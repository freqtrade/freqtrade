# Hyperopt（超参优化）

本页介绍如何通过寻找最优参数来调整你的策略，这一过程称为超参数优化（hyperparameter optimization）。机器人使用 `optuna` 包中内置的算法来完成这项工作。
搜索过程会占满你的所有 CPU 核心，让你的笔记本听起来像一架战斗机，而且仍然需要很长的时间。

一般来说，寻找最佳参数的过程从几个随机组合开始（详见[下文](#可复现的结果)），然后使用 optuna 的某一个采样器算法（目前是 NSGAIIISampler）在搜索超空间中快速找到一组能够最小化[损失函数](#损失函数-loss-functions)的参数组合。

与回测一样，Hyperopt 需要有历史数据可用（hyperopt 会使用不同的参数多次运行回测）。
要了解如何为你感兴趣的币对和交易所获取数据，请参阅文档的[数据下载](data-download.md)章节。

!!! Bug
    Hyperopt 在仅使用 1 个 CPU 核心时可能会崩溃，详见 [Issue #1133](https://github.com/freqtrade/freqtrade/issues/1133)。

!!! Note
    自 2021.4 版本起，你不再需要编写单独的 hyperopt 类，而是可以直接在策略中配置参数。
    旧方法一直支持到 2021.8，并在 2021.9 中被移除。

## 安装 hyperopt 依赖

由于运行机器人本身不需要 hyperopt 依赖，而且这些依赖很重、在某些平台（如树莓派）上不容易构建，因此它们默认不会被安装。在运行 Hyperopt 之前，你需要按照本节所述安装相应的依赖。

!!! Note
    由于 Hyperopt 是一个资源密集型过程，不建议也不支持在树莓派上运行。

### Docker

docker 镜像已包含 hyperopt 依赖，无需任何额外操作。

### 简易安装脚本（setup.sh）/ 手动安装

```bash
source .venv/bin/activate
pip install -r requirements-hyperopt.txt
```

## Hyperopt 命令参考

--8<-- "commands/hyperopt.md"

### Hyperopt 检查清单

关于 hyperopt 中所有任务 / 可能性的检查清单

根据你想要优化的空间（space），下面只有部分是必需的：

* 用 `space='buy'` 定义参数 - 用于入场信号优化
* 用 `space='sell'` 定义参数 - 用于出场信号优化
* 用 `space='enter'` 定义参数 - 用于入场信号优化
* 用 `space='exit'` 定义参数 - 用于出场信号优化
* 用 `space='protection'` 定义参数 - 用于保护机制优化
* 用 `space='random_spacename'` 定义参数 - 用于更好地控制哪些参数一起被优化

选择最适合该参数的空间名称。为了清晰起见，我们推荐使用 `buy` / `sell` 或 `enter` / `exit`（不过在这方面没有技术限制）。

!!! Note
    `populate_indicators` 需要创建所有空间可能使用的指标，否则 hyperopt 将无法工作。

少数情况下你可能还需要创建一个名为 `HyperOpt` 的[嵌套类](advanced-hyperopt.md#覆盖预定义空间)，并实现：

* `roi_space` - 用于自定义 ROI 优化（如果你需要的 ROI 参数取值范围与默认不同）
* `generate_roi_table` - 用于自定义 ROI 优化（如果你需要的 ROI 表中的值或条目数（步数）与默认不同，默认是 4 步）
* `stoploss_space` - 用于自定义 stoploss 优化（如果你需要的 stoploss 参数取值范围与默认不同）
* `trailing_space` - 用于自定义 trailing stop 优化（如果你需要的 trailing stop 参数取值范围与默认不同）
* `max_open_trades_space` - 用于自定义 max_open_trades 优化（如果你需要的 max_open_trades 参数取值范围与默认不同）

!!! Tip "快速优化 ROI、stoploss 和 trailing stoploss"
    你可以在不修改策略任何内容的情况下，快速优化 `roi`、`stoploss` 和 `trailing` 空间。

    ``` bash
    # 手头有一个能用的策略。
    freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --spaces roi stoploss trailing --strategy MyWorkingStrategy --config config.json -e 100
    ```

### Hyperopt 执行逻辑

除非指定了 `--analyze-per-epoch`，否则 Hyperopt 会先将你的数据加载到内存中，然后对每个币对运行一次 `populate_indicators()` 以生成所有指标。

随后 Hyperopt 会派生出不同的进程（处理器数量，或 `-j <n>`），并一遍又一遍地运行回测，改变 `--spaces` 中定义的参数。

对于每一组新参数，freqtrade 会先运行 `populate_entry_trend()`，然后运行 `populate_exit_trend()`，再运行常规回测过程以模拟交易。

回测之后，结果会被传入[损失函数](#损失函数-loss-functions)，该函数会评估这个结果比之前的结果是更好还是更差。
根据损失函数的结果，hyperopt 会决定下一轮回测要尝试的下一组参数。

### 配置你的 Guards 和 Triggers

要在策略文件中添加新的 hyperopt 优化参数，你需要修改两个地方：

* 在类级别定义 hyperopt 要优化的参数。
* 在 `populate_entry_trend()` 内部 - 使用定义的参数值，而不是原始常量。

这里有两种不同类型的指标：1. `guards`（守卫）和 2. `triggers`（触发条件）。

1. Guards 是诸如“如果 ADX < 10 则绝不入场”，或“如果当前价格高于 EMA10 则绝不入场”这样的条件。
2. Triggers 是在特定时刻实际触发入场的指标，例如“当 EMA5 上穿 EMA10 时入场”或“当收盘价触及布林带下轨时入场”。

!!! Hint "Guards 和 Triggers"
    从技术上讲，Guards 和 Triggers 之间没有区别。
    不过，本指南做此区分是为了明确：信号不应“粘连”（sticking）。
    粘连信号是指持续多个蜡烛的信号。这可能导致在信号即将消失之前才入场（这意味着成功率远低于信号刚出现时）。

超优化（hyper-optimization）在每一轮 epoch 中，会挑选一个 trigger 以及可能的多个 guards。

#### 出场信号优化

与上面的入场信号类似，出场信号也可以被优化。
将相应的设置放入以下方法中：

* 在类级别定义 hyperopt 要优化的参数，可以命名为 `sell_*`，或显式定义 `space='sell'`。
* 在 `populate_exit_trend()` 内部 - 使用定义的参数值，而不是原始常量。

其配置和规则与买入信号相同。

## 解开一个谜题

假设你很好奇：到底应该用 MACD 交叉还是布林带下轨来触发你的多头入场。
你还想知道该用 RSI 还是 ADX 来辅助这些决策。
如果你决定使用 RSI 或 ADX，那应该取什么值？

那么让我们使用超参数优化来解这个谜题。

### 定义要使用的指标

我们从计算策略要使用的指标开始。

``` python
class MyAwesomeStrategy(IStrategy):

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成策略使用的所有指标
        """
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_upperband'] = bollinger['upperband']
        return dataframe
```

### 可超参优化的参数

我们继续定义可超参优化的参数：

```python
class MyAwesomeStrategy(IStrategy):
    buy_adx = DecimalParameter(20, 40, decimals=1, default=30.1, space="buy")
    buy_rsi = IntParameter(20, 40, default=30, space="buy")
    buy_adx_enabled = BooleanParameter(default=True, space="buy")
    buy_rsi_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_trigger = CategoricalParameter(["bb_lower", "macd_cross_signal"], default="bb_lower", space="buy")
```

上面的定义表示：我有五个参数，想要随机组合以找到最佳组合。
`buy_rsi` 是一个整数参数，将在 20 到 40 之间测试。该空间大小为 20。
`buy_adx` 是一个小数参数，将在 20 到 40 之间以 1 位小数进行求值（即值为 20.1、20.2、... 等）。该空间大小为 200。
然后我们有三个类别变量。前两个是 `True` 或 `False`。
我们用它们来启用或禁用 ADX 和 RSI 守卫。
最后一个我们称为 `trigger`，用它来决定使用哪个买入触发条件。

!!! Note "参数空间分配"
    - 参数必须被分配给名为 `buy_*`、`sell_*`、`enter_*` 或 `exit_*` 或 `protection_*` 的变量，或者显式地通过参数分配空间（`space='buy'`、`space='sell'`、`space='protection'`）。
    - 存在冲突分配的参数（例如 `buy_adx = IntParameter(4, 24, default=14, space='sell')`）将使用显式空间分配。
    - 如果某个空间没有可用参数，在运行 hyperopt 时你会收到找不到该空间的错误。
    空间分配不明确的参数（例如 `adx_period = IntParameter(4, 24, default=14)` - 既无显式也无隐式空间）将不会被检测到，因此会被忽略。
    空间也可以自定义命名（例如 `space='my_custom_space'`），唯一的限制是空间名称不能是 `all`、`default`，并且必须是一个有效的 python 标识符。

那么让我们用这些值来编写买入策略：

```python
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # GUARDS AND TRENDS
        if self.buy_adx_enabled.value:
            conditions.append(dataframe['adx'] > self.buy_adx.value)
        if self.buy_rsi_enabled.value:
            conditions.append(dataframe['rsi'] < self.buy_rsi.value)

        # TRIGGERS
        if self.buy_trigger.value == 'bb_lower':
            conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
        if self.buy_trigger.value == 'macd_cross_signal':
            conditions.append(qtpylib.crossed_above(
                dataframe['macd'], dataframe['macdsignal']
            ))

        # 检查成交量不为 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe
```

Hyperopt 现在会以不同的数值组合多次（`epochs`）调用 `populate_entry_trend()`。
它会使用给定的历史数据，并基于上面函数生成的买入信号来模拟买入。
基于结果，hyperopt 会告诉你哪个参数组合产生了最佳结果（基于配置的[损失函数](#损失函数-loss-functions)）。

!!! Note
    上面的设置期望在已填充的指标中找到 ADX、RSI 和布林带。
    当你想测试一个机器人当前未使用的指标时，请记得将它添加到策略或 hyperopt 文件的 `populate_indicators()` 方法中。

## 参数类型

共有四种参数类型，各自适用于不同的目的。

* `IntParameter` - 定义一个整数参数，带有搜索空间的上限和下限。
* `DecimalParameter` - 定义一个浮点参数，带有有限的小数位数（默认 3）。在大多数情况下应优先于 `RealParameter`。
* `RealParameter` - 定义一个浮点参数，带有上限和下限，且没有精度限制。很少使用，因为它会创建一个近乎无限可能性的空间。
* `CategoricalParameter` - 定义一个具有预定数量选项的参类。
* `BooleanParameter` - 是 `CategoricalParameter([True, False])` 的简写 - 非常适合“启用”类参数。

### 参数选项

有两个参数选项可以帮助你快速测试各种想法：

* `optimize` - 当设置为 `False` 时，该参数将不会被包含在优化过程中。（默认：True）
* `load` - 当设置为 `False` 时，之前 hyperopt 运行的结果（在 `buy_params` 和 `sell_params` 中，无论是在你的策略还是 JSON 输出文件中）将不会被用作后续 hyperopt 的起始值。将改用参数中指定的默认值。（默认：True）

!!! Tip "对回测的影响（`load=False`）"
    请注意，将 `load` 选项设置为 `False` 意味着回测也会使用参数中指定的默认值，而*不是*通过超优化找到的值。

!!! Warning
    可超参优化的参数不能在 `populate_indicators` 中使用 - 因为 hyperopt 不会为每个 epoch 重新计算指标，所以在那种情况下会使用起始值。

## 优化一个指标参数

假设你脑海中有一个简单的策略 - 一个 EMA 交叉策略（2 条移动均线交叉），并且你想为这个策略找到理想的参数。
默认情况下，我们假设 stoploss 为 5%，止盈（`minimal_roi`）为 10% - 这意味着一旦达到 10% 的利润，freqtrade 就会卖出该交易。

``` python
from pandas import DataFrame
from functools import reduce

import talib.abstract as ta

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MyAwesomeStrategy(IStrategy):
    stoploss = -0.05
    timeframe = '15m'
    minimal_roi = {
        "0":  0.10
    }
    # 定义参数空间
    buy_ema_short = IntParameter(3, 50, default=5)
    buy_ema_long = IntParameter(15, 200, default=50)


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """生成策略使用的所有指标"""

        # 计算所有 ema_short 值
        for val in self.buy_ema_short.range:
            dataframe[f'ema_short_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # 计算所有 ema_long 值
        for val in self.buy_ema_long.range:
            dataframe[f'ema_long_{val}'] = ta.EMA(dataframe, timeperiod=val)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(qtpylib.crossed_above(
                dataframe[f'ema_short_{self.buy_ema_short.value}'], dataframe[f'ema_long_{self.buy_ema_long.value}']
            ))

        # 检查成交量不为 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(qtpylib.crossed_above(
                dataframe[f'ema_long_{self.buy_ema_long.value}'], dataframe[f'ema_short_{self.buy_ema_short.value}']
            ))

        # 检查成交量不为 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        return dataframe
```

分解来看：

使用 `self.buy_ema_short.range` 将返回一个 range 对象，包含参数低值和高值之间的所有条目。
在本例中（`IntParameter(3, 50, default=5)`），循环将针对 3 到 50 之间的所有数字运行（`[3, 4, 5, ... 49, 50]`）。
通过在循环中使用它，hyperopt 将生成 48 个新列（`['buy_ema_3', 'buy_ema_4', ... , 'buy_ema_50']`）。

Hyperopt 本身随后将使用选定的值来创建买入和卖出信号。

虽然这个策略很可能过于简单而无法提供稳定的利润，但它应该作为一个如何优化指标参数的示例。

!!! Note
    `self.buy_ema_short.range` 在 hyperopt 和其他模式之间的行为会有所不同。对于 hyperopt，上面的示例可能生成 48 个新列，而对于所有其他模式（回测、dry/live），它只会生成选定值对应的列。因此你应该避免将生成的列与显式值（除 `self.buy_ema_short.value` 之外的值）一起使用。

!!! Note
    `range` 属性也可以与 `DecimalParameter` 和 `CategoricalParameter` 一起使用。`RealParameter` 由于无限搜索空间而不提供此属性。

??? Hint "性能提示"
    在常规 hyperopt 过程中，指标只计算一次并提供给每个 epoch，随着核心数量的增加线性增加 RAM 使用量。由于这也有性能方面的影响，有两个替代方案可以减少 RAM 使用量：

    * 将 `ema_short` 和 `ema_long` 的计算从 `populate_indicators()` 移到 `populate_entry_trend()`。由于 `populate_entry_trend()` 会在每个 epoch 计算，你不需要使用 `.range` 功能。
    * hyperopt 提供了 `--analyze-per-epoch`，它会将 `populate_indicators()` 的执行移到 epoch 进程中，每个 epoch 每个参数只计算一个值，而不是使用 `.range` 功能。在这种情况下，`.range` 功能只会返回实际被使用的值。

    这些替代方案会减少 RAM 使用量，但会增加 CPU 使用量。不过，你的 hyperopt 运行将不太可能因为内存不足（OOM）问题而失败。

    无论你使用 `.range` 功能还是上述替代方案，你都应该尽量使用尽可能小的空间范围，因为这会改善 CPU/RAM 使用量。

## 优化保护机制（protections）

Freqtrade 也可以优化保护机制。如何优化保护机制由你决定，以下内容仅供参考。

策略只需要将 "protections" 条目定义为返回保护配置列表的属性（property）。

``` python
from pandas import DataFrame
from functools import reduce

import talib.abstract as ta

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MyAwesomeStrategy(IStrategy):
    stoploss = -0.05
    timeframe = '15m'
    # 定义参数空间
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)


    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 4,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ...

```

然后你可以按如下方式运行 hyperopt：
`freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy MyAwesomeStrategy --spaces protection`

!!! Note
    保护空间不是默认空间的一部分，并且仅通过参数 Hyperopt 接口可用，而不通过旧版 hyperopt 接口（需要单独的 hyperopt 文件）提供。
    如果选择了保护空间，Freqtrade 也会自动更改 "--enable-protections" 标志。

!!! Warning
    如果保护机制被定义为属性（property），则配置中定义的条目将被忽略。
    因此建议不要在配置中定义保护机制。

### 从之前的属性设置迁移

从之前的设置迁移非常简单，可以通过将 protections 条目转换为一个属性来完成。简单来说，以下配置将被转换为下面的形式。

``` python
class MyAwesomeStrategy(IStrategy):
    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 4
        }
    ]
```

结果：

``` python
class MyAwesomeStrategy(IStrategy):

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 4
            }
        ]
```

然后你显然也会将潜在有趣的条目更改为参数，以允许超优化。

### 优化 `max_entry_position_adjustment`

虽然 `max_entry_position_adjustment` 不是一个独立的空间，但通过使用上面展示的属性方法，它仍然可以在 hyperopt 中使用。

``` python
from pandas import DataFrame
from functools import reduce

import talib.abstract as ta

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MyAwesomeStrategy(IStrategy):
    stoploss = -0.05
    timeframe = '15m'

    # 定义参数空间
    max_epa = CategoricalParameter([-1, 0, 1, 3, 5, 10], default=1, space="buy", optimize=True)

    @property
    def max_entry_position_adjustment(self):
        return self.max_epa.value


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ...
```

??? Tip "使用 `IntParameter`"
    你也可以为此优化使用 `IntParameter`，但你必须显式返回一个整数：
    ``` python
    max_epa = IntParameter(-1, 10, default=1, space="buy", optimize=True)

    @property
    def max_entry_position_adjustment(self):
        return int(self.max_epa.value)
    ```

## 损失函数（Loss-functions）

每次超参数调优都需要一个目标。这通常被定义为一个损失函数（有时也称为目标函数），它应该随着更期望的结果而减小，随着坏结果而增大。

损失函数必须通过 `--hyperopt-loss <类名>` 参数（或可选地通过配置中的 `"hyperopt_loss"` 键）指定。
这个类应该位于 `user_data/hyperopts/` 目录中自己的文件里。

目前，内置了以下损失函数：

* `ShortTradeDurHyperOptLoss` - （默认的旧版 Freqtrade 超优化损失函数）- 主要针对短期交易时长并避免亏损。
* `OnlyProfitHyperOptLoss` - 只考虑利润多少。
* `SharpeHyperOptLoss` - 优化基于交易收益相对于标准差计算的夏普比率（Sharpe Ratio）。
* `SharpeHyperOptLossDaily` - 优化基于**每日**交易收益相对于标准差计算的夏普比率。
* `SortinoHyperOptLoss` - 优化基于交易收益相对于**下行**标准差计算的索提诺比率（Sortino Ratio）。
* `SortinoHyperOptLossDaily` - 优化基于**每日**交易收益相对于**下行**标准差计算的索提诺比率。
* `MaxDrawDownHyperOptLoss` - 优化最大绝对回撤。
* `MaxDrawDownRelativeHyperOptLoss` - 优化最大绝对回撤，同时调整最大相对回撤。
* `MaxDrawDownPerPairHyperOptLoss` - 计算每个币对的利润/回撤比，并将最差结果作为目标返回，迫使 hyperopt 为币对列表中的所有币对优化参数。这样，我们防止了一个或多个结果良好的币对夸大指标，而结果较差的币对未被代表，因此未被优化。
* `CalmarHyperOptLoss` - 优化基于交易收益相对于最大回撤计算的卡玛比率（Calmar Ratio）。
* `ProfitDrawDownHyperOptLoss` - 通过最大利润 & 最小回撤目标进行优化。hyperoptloss 文件中的 `DRAWDOWN_MULT` 变量可以调整，以使回撤要求更严格或更宽松。
* `MultiMetricHyperOptLoss` - 通过几个关键指标进行优化，以实现均衡的表现。主要关注最大化利润和最小化回撤，同时考虑其他指标，如利润因子（Profit Factor）、期望比率（Expectancy Ratio）和胜率（Winrate）。此外，它会对交易次数较少的 epoch 施加惩罚，鼓励具有足够交易频率的策略。

自定义损失函数的创建在文档的[高级 Hyperopt](advanced-hyperopt.md)部分中介绍。

## 执行 Hyperopt

一旦你更新了 hyperopt 配置，就可以运行它。
因为 hyperopt 会尝试大量组合来寻找最佳参数，所以获得好的结果需要时间。

我们强烈建议使用 `screen` 或 `tmux` 来防止任何连接中断。

```bash
freqtrade hyperopt --config config.json --hyperopt-loss <hyperoptlossname> --strategy <strategyname> -e 500 --spaces all
```

`-e` 选项将设置 hyperopt 要执行的评估次数。由于 hyperopt 使用贝叶斯搜索，一次运行太多 epoch 可能不会产生更好的结果。经验表明，500-1000 个 epoch 之后结果通常不会有太大改善。
`--early-stop` 选项将设置在没有改善的情况下经过多少个 epoch 后 hyperopt 会停止。一个好的值是总 epoch 数的 20-30%。任何大于 0 且小于 20 的值将被替换为 20。早停默认是禁用的（`--early-stop=0`）。

使用少量几千个 epoch 和不同的随机状态进行多次运行（执行）很可能会产生不同的结果。

`--spaces all` 选项决定了应该优化所有可能参数。可能性如下。

!!! Note
    Hyperopt 会以 hyperopt 开始时间的时间戳存储 hyperopt 结果。
    读取命令（`hyperopt-list`、`hyperopt-show`）可以使用 `--hyperopt-filename <filename>` 来读取和显示较早的 hyperopt 结果。
    你可以用 `ls -l user_data/hyperopt_results/` 找到文件名列表。

### 使用不同的历史数据源执行 Hyperopt

如果你想使用磁盘上的备用历史数据集来优化参数，请使用 `--datadir PATH` 选项。默认情况下，hyperopt 使用来自 `user_data/data` 目录的数据。

### 使用较小的测试集运行 Hyperopt

使用 `--timerange` 参数来更改你想要使用的测试集数量。
例如，要使用一个月的数据，请向 hyperopt 调用传递 `--timerange 20210101-20210201`（从 2021 年 1 月到 2021 年 2 月）。

完整命令：

```bash
freqtrade hyperopt --strategy <strategyname> --timerange 20210101-20210201
```

### 使用较小的搜索空间运行 Hyperopt

使用 `--spaces` 选项来限制 hyperopt 使用的搜索空间。
让 Hyperopt 优化所有东西通常是一个巨大的搜索空间。
通常，先只搜索初始入场算法可能更有意义。
或者，你只是想为你那个很棒的新策略优化你的 stoploss 或 roi 表。

合法的值有：

* `all`：优化所有内容（包括自定义空间）
* `buy`：只搜索一个新的买入策略
* `sell`：只搜索一个新的卖出策略
* `enter`：只搜索一个新的入场逻辑
* `exit`：只搜索一个新的出场逻辑
* `roi`：只优化你策略的最小利润表
* `stoploss`：搜索最佳的 stoploss 值
* `trailing`：搜索最佳的 trailing stop 值
* `trades`：搜索最佳的 max open trades 值
* `protection`：搜索最佳的保护参数（阅读[保护机制章节](#优化保护机制-protections)了解如何正确定义它们）
* `default`：`all` 除了 `trailing`、`trades` 和 `protection`
* `custom_space_name`：你的策略中任何参数使用的任何自定义空间
* 上述任意值的空格分隔列表，例如 `--spaces roi stoploss`

默认情况下使用 Hyperopt 搜索空间（当未指定 `--space` 命令行选项时），不包含 `trailing` 超空间。我们建议你单独运行 `trailing` 超空间的优化，在找到其他超空间的最佳参数、验证并将其粘贴到你的自定义策略后。

## 理解 Hyperopt 结果

一旦 Hyperopt 完成，你可以使用结果来更新你的策略。
给定以下来自 hyperopt 的结果：

```
Best result:

    44/100:    135 trades. Avg profit  0.57%. Total profit  0.03871918 BTC (0.7722%). Avg duration 180.4 mins. Objective: 1.94367

    # Buy hyperspace params:
    buy_params = {
        'buy_adx': 44,
        'buy_rsi': 29,
        'buy_adx_enabled': False,
        'buy_rsi_enabled': True,
        'buy_trigger': 'bb_lower'
    }
```

你应该这样理解这个结果：

* 效果最好的买入触发条件是 `bb_lower`。
* 你不应该使用 ADX，因为 `'buy_adx_enabled': False`。
* 你应该**考虑**使用 RSI 指标（`'buy_rsi_enabled': True`），最佳值是 `29.0`（`'buy_rsi': 29.0`）

### 自动将参数应用到策略

使用可超参优化的参数时，你的 hyperopt 运行结果将被写入策略旁边的一个 json 文件（因此对于 `MyAwesomeStrategy.py`，文件将是 `MyAwesomeStrategy.json`）。
除非向这两个命令中的任何一个提供 `--disable-param-export`，否则在使用 `hyperopt-show` 子命令时也会更新此文件。

你的策略类也可以显式包含这些结果。只需复制 hyperopt 结果块并将其粘贴到类级别，替换旧参数（如果有）。下次执行策略时会自动加载新参数。

将你的整个 hyperopt 结果转移到你的策略中，看起来像这样：

```python
class MyAwesomeStrategy(IStrategy):
    # Buy hyperspace params:
    buy_params = {
        'buy_adx': 44,
        'buy_rsi': 29,
        'buy_adx_enabled': False,
        'buy_rsi_enabled': True,
        'buy_trigger': 'bb_lower'
    }
```

!!! Note
    配置文件中的值将覆盖参数文件级别的参数 - 而两者都会覆盖策略中的参数。
    因此优先级为：config > 参数文件 > 策略 `*_params` > 参数默认值

### 理解 Hyperopt ROI 结果

如果你正在优化 ROI（即如果优化搜索空间包含 'all'、'default' 或 'roi'），你的结果将如下所示，并包含一个 ROI 表：

```
Best result:

    44/100:    135 trades. Avg profit  0.57%. Total profit  0.03871918 BTC (0.7722%). Avg duration 180.4 mins. Objective: 1.94367

    # ROI table:
    minimal_roi = {
        0: 0.10674,
        21: 0.09158,
        78: 0.03634,
        118: 0
    }
```

为了在回测以及实盘交易/dry-run 中使用 Hyperopt 找到的这个最佳 ROI 表，请将它复制粘贴为你的自定义策略的 `minimal_roi` 属性的值：

```
    # 为策略设计的最小 ROI。
    # 如果配置文件包含 "minimal_roi"，此属性将被覆盖
    minimal_roi = {
        0: 0.10674,
        21: 0.09158,
        78: 0.03634,
        118: 0
    }
```

如注释中所述，你也可以将它用作配置文件中 `minimal_roi` 设置的值。

#### 默认 ROI 搜索空间

如果你正在优化 ROI，Freqtrade 会为你创建 'roi' 优化超空间 -- 它是 ROI 表各组成部分的超空间。默认情况下，Freqtrade 生成的每个 ROI 表由 4 行（步数）组成。Hyperopt 为 ROI 表实现了自适应范围，其值范围取决于所用的时间周期（timeframe）。默认情况下，值按以下范围变化（针对一些最常用的时间周期，值四舍五入到小数点后 3 位）：

| # step | 1m     |               | 5m       |             | 1h         |               | 1d           |               |
| ------ | ------ | ------------- | -------- | ----------- | ---------- | ------------- | ------------ | ------------- |
| 1      | 0      | 0.011...0.119 | 0        | 0.03...0.31 | 0          | 0.068...0.711 | 0            | 0.121...1.258 |
| 2      | 2...8  | 0.007...0.042 | 10...40  | 0.02...0.11 | 120...480  | 0.045...0.252 | 2880...11520 | 0.081...0.446 |
| 3      | 4...20 | 0.003...0.015 | 20...100 | 0.01...0.04 | 240...1200 | 0.022...0.091 | 5760...28800 | 0.040...0.162 |
| 4      | 6...44 | 0.0           | 30...220 | 0.0         | 360...2640 | 0.0           | 8640...63360 | 0.0           |

这些范围在大多数情况下应该足够。步骤中的分钟数（ROI 字典的键）根据你使用的时间周期线性缩放。步骤中的 ROI 值（ROI 字典的值）根据你使用的时间周期对数缩放。

如果你在自定义 hyperopt 中有 `generate_roi_table()` 和 `roi_space()` 方法，请移除它们，以便利用 Freqtrade 默认生成的这些自适应 ROI 表和 ROI 超优化空间。

如果你需要 ROI 表的组成部分在其他范围内变化，请覆盖 `roi_space()` 方法。如果你需要 ROI 表的不同结构或其他行数（步数），请覆盖 `generate_roi_table()` 和 `roi_space()` 方法，并实现你自己的自定义方法来生成 ROI 表。

这些方法的示例可以在[覆盖预定义空间章节](advanced-hyperopt.md#覆盖预定义空间)中找到。

!!! Note "缩减的搜索空间"
    为了进一步限制搜索空间，小数被限制为 3 位小数（精度 0.001）。这通常在大多数情况下已足够，任何比这更精确的值通常会导致过拟合结果。不过你可以[覆盖预定义空间](advanced-hyperopt.md#覆盖预定义空间)来根据需要进行更改。

### 理解 Hyperopt Stoploss 结果

如果你正在优化 stoploss 值（即如果优化搜索空间包含 'all'、'default' 或 'stoploss'），你的结果将如下所示，并包含 stoploss：

```
Best result:

    44/100:    135 trades. Avg profit  0.57%. Total profit  0.03871918 BTC (0.7722%). Avg duration 180.4 mins. Objective: 1.94367

    # Buy hyperspace params:
    buy_params = {
        'buy_adx': 44,
        'buy_rsi': 29,
        'buy_adx_enabled': False,
        'buy_rsi_enabled': True,
        'buy_trigger': 'bb_lower'
    }

    stoploss: -0.27996
```

为了在回测以及实盘交易/dry-run 中使用 Hyperopt 找到的这个最佳 stoploss 值，请将它复制粘贴为你的自定义策略的 `stoploss` 属性的值：

``` python
    # 为策略设计的最佳 stoploss
    # 如果配置文件包含 "stoploss"，此属性将被覆盖
    stoploss = -0.27996
```

如注释中所述，你也可以将它用作配置文件中 `stoploss` 设置的值。

#### 默认 Stoploss 搜索空间

如果你正在优化 stoploss 值，Freqtrade 会为你创建 'stoploss' 优化超空间。默认情况下，该超空间中的 stoploss 值变化范围在 -0.35...-0.02 之间，在大多数情况下已足够。

如果你在自定义 hyperopt 文件中有 `stoploss_space()` 方法，请移除它，以便利用 Freqtrade 默认生成的 Stoploss 超优化空间。

如果需要 stoploss 值在超优化过程中以其他范围变化，请覆盖 `stoploss_space()` 方法并在其中定义所需范围。此方法的示例可以在[覆盖预定义空间章节](advanced-hyperopt.md#覆盖预定义空间)中找到。

!!! Note "缩减的搜索空间"
    为了进一步限制搜索空间，小数被限制为 3 位小数（精度 0.001）。这通常在大多数情况下已足够，任何比这更精确的值通常会导致过拟合结果。不过你可以[覆盖预定义空间](advanced-hyperopt.md#覆盖预定义空间)来根据需要进行更改。

### 理解 Hyperopt Trailing Stop 结果

如果你正在优化 trailing stop 值（即如果优化搜索空间包含 'all' 或 'trailing'），你的结果将如下所示，并包含 trailing stop 参数：

```
Best result:

    45/100:    606 trades. Avg profit  1.04%. Total profit  0.31555614 BTC ( 630.48%). Avg duration 150.3 mins. Objective: -1.10161

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.02001
    trailing_stop_positive_offset = 0.06038
    trailing_only_offset_is_reached = True
```

为了在回测以及实盘交易/dry-run 中使用 Hyperopt 找到的这些最佳 trailing stop 参数，请将它们复制粘贴为你的自定义策略相应属性的值：

``` python
    # Trailing stop
    # 如果配置文件包含相应的值，这些属性将被覆盖。
    trailing_stop = True
    trailing_stop_positive = 0.02001
    trailing_stop_positive_offset = 0.06038
    trailing_only_offset_is_reached = True
```

如注释中所述，你也可以将它们用作配置文件中相应设置的值。

#### 默认 Trailing Stop 搜索空间

如果你正在优化 trailing stop 值，Freqtrade 会为你创建 'trailing' 优化超空间。默认情况下，该超空间中的 `trailing_stop` 参数始终设置为 True，`trailing_only_offset_is_reached` 的值在 True 和 False 之间变化，`trailing_stop_positive` 和 `trailing_stop_positive_offset` 参数的值分别在 0.02...0.35 和 0.01...0.1 范围内变化，在大多数情况下已足够。

如果需要 trailing stop 参数的值在超优化过程中以其他范围变化，请覆盖 `trailing_space()` 方法并在其中定义所需范围。此方法的示例可以在[覆盖预定义空间章节](advanced-hyperopt.md#覆盖预定义空间)中找到。

!!! Note "缩减的搜索空间"
    为了进一步限制搜索空间，小数被限制为 3 位小数（精度 0.001）。这通常在大多数情况下已足够，任何比这更精确的值通常会导致过拟合结果。不过你可以[覆盖预定义空间](advanced-hyperopt.md#覆盖预定义空间)来根据需要进行更改。

### 可复现的结果

寻找最优参数的过程从超参数空间中几个（目前是 30 个）随机组合开始，即随机的 Hyperopt epoch。这些随机 epoch 在 Hyperopt 输出的第一列中用星号（`*`）标记。

这些随机值（随机状态）生成的初始状态由 `--random-state` 命令行选项的值控制。你可以将它设置为你选择的任意值，以获得可复现的结果。

如果你没有在命令行选项中显式设置此值，Hyperopt 会为你用一个随机值初始化随机状态。每次 Hyperopt 运行的随机状态值都会显示在日志中，因此你可以将它复制粘贴到 `--random-state` 命令行选项中，以重复所使用的一组初始随机 epoch。

如果你没有更改命令行选项、配置、timerange、策略和 Hyperopt 类、历史数据以及损失函数 -- 你应该使用相同的随机状态值获得相同的超优化结果。

## 输出格式化

默认情况下，hyperopt 会打印彩色结果 -- 具有正利润（profit）的 epoch 会以绿色打印。这种高亮有助于你找到以后分析中可能有趣的 epoch。总利润为零或为负利润（亏损）的 epoch 会以正常颜色打印。如果你不需要结果彩色化（例如，当你将 hyperopt 输出重定向到文件时），可以通过在命令行指定 `--no-color` 选项来关闭彩色化。

如果你希望看到 hyperopt 输出中的所有结果，而不仅仅是最好的一些，可以使用 `--print-all` 命令行选项。当使用 `--print-all` 时，当前最佳结果默认也会被彩色化 -- 它们以粗体（明亮）样式打印。这也可以通过 `--no-color` 命令行选项关闭。

!!! Note "Windows 和彩色输出"
    Windows 本身不支持彩色输出，因此它会自动禁用。要让在 Windows 下运行的 hyperopt 拥有彩色输出，请考虑使用 WSL。

## 仓位堆叠（position stacking）和禁用最大市场仓位

在某些情况下，你可能需要使用 `--eps`/`--enable-position-stacking` 参数运行 Hyperopt（和回测），或者你需要将 `max_open_trades` 设置为一个非常高的数字，以禁用对开放交易数量的限制。

默认情况下，hyperopt 模拟 Freqtrade 实盘运行/Dry Run 的行为，其中每个币对只允许一个开放交易。所有币对的开放交易总数也受到 `max_open_trades` 设置的限制。在 Hyperopt/回测期间，这可能导致潜在交易被已经开放的交易所隐藏（或掩盖）。

`--eps`/`--enable-position-stacking` 参数允许模拟多次买入同一币对。
使用 `--max-open-trades` 并配合非常高的数字将禁用对开放交易数量的限制。

!!! Note
    Dry/live 运行**不会**使用仓位堆叠 - 因此以不启用此功能的方式验证策略也是有意义的，因为这更接近现实。

你也可以通过显式设置 `"position_stacking"=true` 在配置文件中启用仓位堆叠。

## 内存不足（Out of Memory）错误

由于 hyperopt 消耗大量内存（每个并行回测进程需要一次性将完整数据加载到内存中），你很可能会遇到“内存不足”错误。
要解决这些问题，你有多种选择：

* 减少币对数量。
* 减少使用的 timerange（`--timerange <timerange>`）。
* 避免使用 `--timeframe-detail`（这会将大量额外数据加载到内存中）。
* 减少并行进程的数量（`-j <n>`）。
* 增加你机器的内存。
* 如果你正在使用大量带有 `.range` 功能的参数，请使用 `--analyze-per-epoch`。

## The objective has been evaluated at this point before.

如果你看到 `The objective has been evaluated at this point before.` - 那么这表明你的空间已经耗尽，或接近耗尽。
基本上你空间中的所有点都已被命中（或已命中局部最小值） - 并且 hyperopt 不再找到它在多维空间中尚未尝试过的点。
在这种情况下，Freqtrade 会尝试通过使用新的、随机化的点来对抗“局部最小值”问题。

示例：

``` python
buy_ema_short = IntParameter(5, 20, default=10, space="buy", optimize=True)
# 这是买入空间中唯一的参数
```

`buy_ema_short` 空间有 15 个可能的值（`5, 6, ... 19, 20`）。如果你现在为买入空间运行 hyperopt，hyperopt 在选项耗尽之前只有 15 个值可尝试。
因此，你的 epoch 应该与可能的值对齐 - 或者你应该准备好在注意到大量 `The objective has been evaluated at this point before.` 警告时中断运行。

## 显示 Hyperopt 结果详情

在你为所需数量的 epoch 运行 Hyperopt 之后，你以后可以列出所有结果进行分析，只选择最好或盈利的结果，并显示之前评估过的任何 epoch 的详情。这可以使用 `hyperopt-list` 和 `hyperopt-show` 子命令来完成。这些子命令的用法在 [Utils](utils.md#列出-hyperopt-结果) 章节中描述。

## 从你的策略输出调试消息

如果你想从你的策略输出调试消息，你可以使用 `logging` 模块。默认情况下，Freqtrade 会输出级别为 `INFO` 或更高级别的所有消息。

``` python
import logging


logger = logging.getLogger(__name__)


class MyAwesomeStrategy(IStrategy):
    ...

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.info("This is a debug message")
        ...

```

!!! Note "使用 print"
    除非禁用并行（`-j 1`），否则通过 `print()` 打印的消息不会显示在 hyperopt 输出中。
    建议使用 `logging` 模块代替。

## 验证回测结果

一旦将优化后的策略实现到你的策略中，你应该回测该策略，以确保一切按预期工作。

为了获得与 Hyperopt 期间相同的结果（交易数量、持续时间、利润等），请使用与 Hyperopt 相同的配置和参数（timerange、timeframe 等）进行回测。

### 为什么我的回测结果与 hyperopt 结果不匹配？

如果结果不匹配，请检查以下因素：

* 你可能已将参数添加到 hyperopt 的 `populate_indicators()` 中，在那里它们**为所有 epoch** 只计算一次。例如，如果你试图优化多个 SMA timeperiod 值，可超参优化的 timeperiod 参数应放在 `populate_entry_trend()` 中，后者会在每个 epoch 计算。参见[优化指标参数](https://www.freqtrade.io/en/stable/hyperopt/#optimizing-an-indicator-parameter)。
* 如果你禁用了将 hyperopt 参数自动导出到 JSON 参数文件，请仔细检查以确保你已将所有的 hyperopt 值正确转移到你的策略中。
* 检查日志以验证正在设置哪些参数以及正在使用哪些值。
* 特别留意 stoploss、max_open_trades 和 trailing stoploss 参数，因为它们经常在配置文件中设置，这会覆盖对策略的更改。检查回测的日志，确保没有被配置无意中设置任何参数（如 `stoploss`、`max_open_trades` 或 `trailing_stop`）。
* 验证你没有一个意外的参数 JSON 文件覆盖你策略中的参数或默认的 hyperopt 设置。
* 验证回测中启用的任何保护机制在 hyperopt 时也已启用，反之亦然。使用 `--space protection` 时，保护机制会自动为 hyperopt 启用。
