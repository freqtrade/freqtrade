<!-- 本文件为中文翻译版，由 AI 根据 docs/advanced-hyperopt.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，图片与 includes 引用使用 ../ 指向英文原文档资源。 -->

# 进阶 Hyperopt

本页讲解一些进阶的 Hyperopt（超参数优化）主题，这些主题可能比创建一个普通的超参数优化类需要更高的编码技巧和 Python 知识。

## 创建并使用自定义损失函数

要使用自定义的损失函数类，请确保你的自定义 hyperopt 损失类中定义了 `hyperopt_loss_function` 函数。对于下面的示例，你需要在 hyperopt 调用中添加命令行参数 `--hyperopt-loss SuperDuperHyperOptLoss`，这样该函数才会被使用。

下面的示例与默认的 Hyperopt 损失实现完全相同。完整示例可在 [userdata/hyperopts](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt_loss.py) 中找到。

``` python
from datetime import datetime
from typing import Any, Dict

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.optimize.hyperopt import IHyperOptLoss

TARGET_TRADES = 600
EXPECTED_MAX_PROFIT = 3.0
MAX_ACCEPTED_TRADE_DURATION = 300

class SuperDuperHyperOptLoss(IHyperOptLoss):
    """
    Defines the default loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(
        *,
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        backtest_stats: dict[str, Any],
        starting_balance: float,
        **kwargs,
    ) -> float:
        """
        Objective function, returns smaller number for better results
        This is the legacy algorithm (used until now in freqtrade).
        Weights are distributed as follows:
        * 0.4 to trade duration
        * 0.25: Avoiding trade loss
        * 1.0 to total profit, compared to the expected value (`EXPECTED_MAX_PROFIT`) defined above
        """
        total_profit = results['profit_ratio'].sum()
        trade_duration = results['trade_duration'].mean()

        trade_loss = 1 - 0.25 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5.8)
        profit_loss = max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
        duration_loss = 0.4 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1)
        result = trade_loss + profit_loss + duration_loss
        return result
```

目前，这些参数如下：

* `results`：包含所得交易的 DataFrame。
    results 中可用的列如下（对应于使用 `--export trades` 时回测的输出文件）：  
    `pair, profit_ratio, profit_abs, open_date, open_rate, fee_open, close_date, close_rate, fee_close, amount, trade_duration, is_open, exit_reason, stake_amount, min_rate, max_rate, stop_loss_ratio, stop_loss_abs`
* `trade_count`：交易数量（等同于 `len(results)`）
* `min_date`：所用时间范围的开始日期
* `min_date`：所用时间范围的结束日期
* `config`：所使用的配置对象（注意：如果策略相关参数属于 hyperopt 空间的一部分，并非所有这些参数都会在此更新）。
* `processed`：以交易对为键、包含用于回测的数据的 DataFrame 字典。
* `backtest_stats`：回测统计，格式与回测文件中 "strategy" 子结构相同。可用字段可在 `optimize_reports.py` 的 `generate_strategy_stats()` 中查看。
* `starting_balance`：用于回测的起始余额。

该函数需要返回一个浮点数（`float`）。数值越小表示结果越好。具体的参数和权衡由你自己决定。

!!! Note
    该函数在每个 epoch 会被调用一次——因此请务必尽可能优化此函数，以免不必要地拖慢 hyperopt 的速度。

!!! Note "`*args` 和 `**kwargs`"
    请在接口中保留 `*args` 和 `**kwargs` 参数，以便我们未来能够扩展此接口。

## 覆盖预定义的搜索空间

要覆盖预定义的空间（`roi_space`、`generate_roi_table`、`stoploss_space`、`trailing_space`、`max_open_trades_space`），请定义一个名为 Hyperopt 的嵌套类，并按如下方式定义所需的空间：

```python
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

class MyAwesomeStrategy(IStrategy):
    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.05, -0.01, decimals=3, name='stoploss')]

        # Define custom ROI space
        def roi_space() -> List[Dimension]:
            return [
                Integer(10, 120, name='roi_t1'),
                Integer(10, 60, name='roi_t2'),
                Integer(10, 40, name='roi_t3'),
                SKDecimal(0.01, 0.04, decimals=3, name='roi_p1'),
                SKDecimal(0.01, 0.07, decimals=3, name='roi_p2'),
                SKDecimal(0.01, 0.20, decimals=3, name='roi_p3'),
            ]

        def generate_roi_table(params: Dict) -> dict[int, float]:

            roi_table = {}
            roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
            roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
            roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
            roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

            return roi_table

        def trailing_space() -> List[Dimension]:
            # All parameters here are mandatory, you can only modify their type or the range.
            return [
                # Fixed to true, if optimizing trailing_stop we assume to use trailing stop at all times.
                Categorical([True], name='trailing_stop'),

                SKDecimal(0.01, 0.35, decimals=3, name='trailing_stop_positive'),
                # 'trailing_stop_positive_offset' should be greater than 'trailing_stop_positive',
                # so this intermediate parameter is used as the value of the difference between
                # them. The value of the 'trailing_stop_positive_offset' is constructed in the
                # generate_trailing_params() method.
                # This is similar to the hyperspace dimensions used for constructing the ROI tables.
                SKDecimal(0.001, 0.1, decimals=3, name='trailing_stop_positive_offset_p1'),

                Categorical([True, False], name='trailing_only_offset_is_reached'),
        ]

        # Define a custom max_open_trades space
        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(-1, 10, name='max_open_trades'),
            ]
```

!!! Note
    所有覆盖都是可选的，可按需混合/搭配使用。

## 动态参数

参数也可以动态定义，但必须在 [`bot_start()` 回调](strategy-callbacks.md#bot-start) 被调用后对实例可用。

``` python

class MyAwesomeStrategy(IStrategy):

    def bot_start(self, **kwargs) -> None:
        self.buy_adx = IntParameter(20, 30, default=30, optimize=True)

    # ...
```

!!! Warning
    以这种方式创建的参数不会显示在 `list-strategies` 的参数计数中。

## 覆盖基础估计器（Base estimator）

你可以通过在 Hyperopt 子类中实现 `generate_estimator()` 来为 Hyperopt 定义自己的 optuna sampler。

```python
class MyAwesomeStrategy(IStrategy):
    class HyperOpt:
        def generate_estimator(dimensions: List['Dimension'], **kwargs):
            return "NSGAIIISampler"

```

可选值可以是以下之一："NSGAIISampler"、"TPESampler"、"GPSampler"、"CmaEsSampler"、"NSGAIIISampler"、"QMCSampler"（详情见 [optuna-samplers 文档](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)），或"一个继承自 `optuna.samplers.BaseSampler` 的类的实例"。

你可能需要做一些研究才能找到额外的 Sampler（例如来自 optunahub）。

!!! Note
    虽然可以提供自定义的估计器，但作为用户，你需要自行研究可能的参数，并分析/理解应该使用哪些。
    如果你对此不确定，最好使用默认值之一（"`NSGAIIISampler`" 已被证明最通用）且不再附加额外参数。

??? Example "使用来自 Optunahub 的 `AutoSampler`"

    [AutoSampler 文档](https://hub.optuna.org/samplers/auto_sampler/)
    
    安装必要的依赖
    ``` bash
    pip install optunahub cmaes torch scipy
    ```
    在你的策略中实现 `generate_estimator()`

    ``` python
    # ...
    from freqtrade.strategy.interface import IStrategy
    from typing import List
    import optunahub
    # ... 

    class my_strategy(IStrategy):
        class HyperOpt:
            def generate_estimator(dimensions: List["Dimension"], **kwargs):
                if "random_state" in kwargs.keys():
                    return optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=kwargs["random_state"])
                else:
                    return optunahub.load_module("samplers/auto_sampler").AutoSampler()

    ```

    显然，同样的方法适用于 optuna 支持的所有其他 Sampler。

## 空间选项

对于额外的空间，scikit-optimize（与 Freqtrade 组合使用）提供以下空间类型：

* `Categorical` - 从一组类别中选取（例如 `Categorical(['a', 'b', 'c'], name="cat")`）
* `Integer` - 从一组整数范围中选取（例如 `Integer(1, 10, name='rsi')`）
* `SKDecimal` - 从具有有限精度的十进制数范围中选取（例如 `SKDecimal(0.1, 0.5, decimals=3, name='adx')`）。*仅 Freqtrade 提供*。
* `Real` - 从具有完整精度的十进制数范围中选取（例如 `Real(0.1, 0.5, name='adx')`）

你可以从 `freqtrade.optimize.space` 导入所有这些类型，不过 `Categorical`、`Integer` 和 `Real` 只是其对应 scikit-optimize 空间（Space）的别名。`SKDecimal` 由 freqtrade 提供，以实现更快的优化。

``` python
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real  # noqa
```

!!! Hint "SKDecimal vs. Real"
    我们建议几乎在所有情况下都使用 `SKDecimal` 而不是 `Real` 空间。虽然 Real 空间提供完整精度（约 16 位小数）——但这种精度很少需要，并且会导致不必要的超长 hyperopt 时间。

    假设定义了一个相当小的空间（`SKDecimal(0.10, 0.15, decimals=2, name='xxx')`）——SKDecimal 会有 5 种可能（`[0.10, 0.11, 0.12, 0.13, 0.14, 0.15]`）。

    而对应的 real 空间 `Real(0.10, 0.15 name='xxx')` 则具有几乎无限多的可能（`[0.10, 0.010000000001, 0.010000000002, ... 0.014999999999, 0.01500000000]`）。
