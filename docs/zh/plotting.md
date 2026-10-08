<!-- 本文件为中文翻译版，由 AI 根据 docs/plotting.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 绘图

本页面介绍如何绘制价格、指标和收益图表。

!!! Warning "已弃用"
    本页面中描述的命令（`plot-dataframe`、`plot-profit`）应被视为已弃用，并处于维护模式。
    这主要是因为即使是中等规模的图表也可能导致性能问题，同时也因为"保存文件然后在浏览器中打开"从 UI 角度来看并不直观。

    虽然目前没有立即移除它们的计划，但它们不再被积极维护——如果需要进行重大更改才能维持其正常运行，可能会在短期内被移除。

    请使用 [FreqUI](freq-ui.md) 来满足绘图需求，它不会遇到同样的性能问题。

## 安装 / 配置

绘图模块使用 Plotly 库。您可以通过运行以下命令来安装/升级它：

``` bash
pip install -U -r requirements-plot.txt
```

## 绘制价格和指标

`freqtrade plot-dataframe` 子命令显示一个包含三个子图的交互式图表：

* 主图，包含蜡烛图和跟随价格的指标（sma/ema）
* 成交量柱状图
* 由 `--indicators2` 指定的附加指标

![plot-dataframe](../assets/plot-dataframe.png)

可用参数：

--8<-- "../commands/plot-dataframe.md"

示例：

``` bash
freqtrade plot-dataframe -p BTC/ETH --strategy AwesomeStrategy
```

`-p/--pairs` 参数可用于指定您想要绘制的交易对。

!!! Note
    `freqtrade plot-dataframe` 子命令会为每个交易对生成一个图表文件。

指定自定义指标。
使用 `--indicators1` 用于主图，使用 `--indicators2` 用于下方的子图（如果指标值与价格范围不同）。

``` bash
freqtrade plot-dataframe --strategy AwesomeStrategy -p BTC/ETH --indicators1 sma ema --indicators2 macd
```

### 更多使用示例

要绘制多个交易对，请用空格分隔它们：

``` bash
freqtrade plot-dataframe --strategy AwesomeStrategy -p BTC/ETH XRP/ETH
```

要绘制特定时间范围（用于放大查看）

``` bash
freqtrade plot-dataframe --strategy AwesomeStrategy -p BTC/ETH --timerange=20180801-20180805
```

要绘制存储在数据库中的交易记录，请将 `--db-url` 与 `--trade-source DB` 组合使用：

``` bash
freqtrade plot-dataframe --strategy AwesomeStrategy --db-url sqlite:///tradesv3.dry_run.sqlite -p BTC/ETH --trade-source DB
```

要绘制回测结果中的交易记录，请使用 `--export-filename <filename>`

``` bash
freqtrade plot-dataframe --strategy AwesomeStrategy --export-filename user_data/backtest_results/backtest-result.json -p BTC/ETH
```

### 绘图数据框基础

![plot-dataframe2](../assets/plot-dataframe2.png)

`plot-dataframe` 子命令需要回测数据、一个策略以及一个回测结果文件或包含与该策略对应交易记录的数据库。

生成的图表将包含以下元素：

* 绿色三角形：策略的买入信号。（注意：并非每个买入信号都会产生交易，可与青色圆圈进行比较。）
* 红色三角形：策略的卖出信号。（同样，并非每个卖出信号都会终止交易，可与红色和绿色方块进行比较。）
* 青色圆圈：交易入场点。
* 红色方块：亏损或 0% 收益交易的出场点。
* 绿色方块：盈利交易的出场点。
* 值与蜡烛图刻度对应的指标（如 SMA/EMA），通过 `--indicators1` 指定。
* 成交量（主图底部的柱状图）。
* 值在不同刻度上的指标（如 MACD、RSI），显示在成交量柱状图下方，通过 `--indicators2` 指定。

!!! Note "布林带"
    如果 DataFrame 中存在 `bb_lowerband` 和 `bb_upperband` 列，布林带会自动添加到图表中，并以浅蓝色区域显示在下轨和上轨之间。

#### 高级绘图配置

可以在策略的 `plot_config` 参数中指定高级绘图配置。

使用 `plot_config` 时的附加功能包括：

* 为每个指标指定颜色
* 指定附加子图
* 指定指标对以填充它们之间的区域

下面的示例绘图配置为指标指定了固定颜色。否则，连续的图表每次可能产生不同的配色方案，使比较变得困难。
它还允许使用多个子图同时显示 MACD 和 RSI。

可以使用 `type` 键配置图表类型。可选类型有：

* `scatter` 对应散点图。
* `bar` 对应柱状图。

`plotly` 字典中可以指定 `plotly.graph_objects.*` 构造函数的额外参数——这些仅在使用 plotly 作为绘图库时才支持，在使用 freq-ui 时将被忽略。

带有行内注释说明流程的示例配置：

``` python
@property
def plot_config(self):
    """
        有很多方法可以构建返回字典。
        唯一重要的是返回值。
        示例：
            plot_config = {'main_plot': {}, 'subplots': {}}

    """
    plot_config = {}
    plot_config['main_plot'] = {
        # 主图指标配置。
        # 假设指定了 2 个参数：emashort 和 emalong。
        f'ema_{self.emashort.value}': {'color': 'red'},
        f'ema_{self.emalong.value}': {'color': '#CCCCCC'},
        # 省略颜色将随机选择一种颜色。
        'sar': {},
        # 填充 senkou_a 和 senkou_b 之间的区域
        'senkou_a': {
            'color': 'green', #可选
            'fill_to': 'senkou_b',
            'fill_label': 'Ichimoku Cloud', #可选
            'fill_color': 'rgba(255,76,46,0.2)', #可选
        },
        # 同时绘制 senkou_b，不仅仅是它所在的区域。
        'senkou_b': {}
    }
    plot_config['subplots'] = {
         # 创建 MACD 子图
        "MACD": {
            'macd': {'color': 'blue', 'fill_to': 'macdhist'},
            'macdsignal': {'color': 'orange'},
            'macdhist': {'type': 'bar', 'plotly': {'opacity': 0.9}}
        },
        # 附加 RSI 子图
        "RSI": {
            'rsi': {'color': 'red'}
        }
    }

    return plot_config
```

??? Note "作为属性（旧方法）"
    `plot_config` 也可以作为属性赋值（这曾经是默认方式）。
    这种方式的缺点是策略参数不可用，导致某些配置无法工作。

    ``` python
        plot_config = {
            'main_plot': {
                # 主图指标配置。
                # 指定 `ema10` 为红色，`ema50` 为灰色
                'ema10': {'color': 'red'},
                'ema50': {'color': '#CCCCCC'},
                # 省略颜色将随机选择一种颜色。
                'sar': {},
            # 填充 senkou_a 和 senkou_b 之间的区域
            'senkou_a': {
                'color': 'green', #可选
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud', #可选
                'fill_color': 'rgba(255,76,46,0.2)', #可选
            },
            # 同时绘制 senkou_b，不仅仅是它所在的区域。
            'senkou_b': {}
            },
            'subplots': {
                # 创建 MACD 子图
                "MACD": {
                    'macd': {'color': 'blue', 'fill_to': 'macdhist'},
                    'macdsignal': {'color': 'orange'},
                    'macdhist': {'type': 'bar', 'plotly': {'opacity': 0.9}}
                },
                # 附加 RSI 子图
                "RSI": {
                    'rsi': {'color': 'red'}
                }
            }
        }

    ```


!!! Note
    上述配置假设 `ema10`、`ema50`、`senkou_a`、`senkou_b`、
    `macd`、`macdsignal`、`macdhist` 和 `rsi` 是策略创建的 DataFrame 中的列。

!!! Warning
    `plotly` 参数仅在使用 plotly 库时支持，与 freq-ui 配合使用时将不起作用。

!!! Note "交易仓位调整"
    如果使用了 `position_adjustment_enable` / `adjust_trade_position()`，交易的初始买入价格会通过多个订单取平均值，交易起始价格很可能会显示在蜡烛图范围之外。

## 绘制收益图

![plot-profit](../assets/plot-profit.png)

`plot-profit` 子命令显示一个包含四个图表的交互式图形：

* 所有交易对的平均收盘价。
* 回测产生的汇总收益。
请注意，这不是实际收益，而更像是一种估算。
* 每个交易对各自的收益。
* 交易并行度。
* 水下图（回撤期间）。

第一个图表有助于了解整体市场的走势。

第二个图表将显示您的算法是否有效。
也许您想要一个稳定赚取小额利润的算法，或者一个操作较少但能获取大幅收益的算法。
该图表还会突出显示最大回撤期的开始（和结束）。

第三个图表可用于发现异常值——交易对中导致收益突增的事件。

第四个图表可以帮助您分析交易并行度，显示 max_open_trades 被充分利用的频率。

`freqtrade plot-profit` 子命令的可用选项：

--8<-- "../commands/plot-profit.md"

`-p/--pairs` 参数可用于限制参与计算的交易对。

示例：

使用自定义回测导出文件

``` bash
freqtrade plot-profit  -p LTC/BTC --export-filename user_data/backtest_results/backtest-result.json
```

使用自定义数据库

``` bash
freqtrade plot-profit  -p LTC/BTC --db-url sqlite:///tradesv3.sqlite --trade-source DB
```

``` bash
freqtrade --datadir user_data/data/binance_save/ plot-profit -p LTC/BTC
```