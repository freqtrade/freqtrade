<!-- 本文件为中文翻译版，由 AI 根据 docs/advanced-backtesting.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，图片与 includes 引用使用 ../ 指向英文原文档资源。 -->

# 进阶回测分析

## 分析买入/入场标签与卖出/离场标签

理解策略在不同买入条件下（用来标记不同买入情形的 buy/entry 标签）的行为方式会很有帮助。你可能想在默认回测输出之上，查看关于每个买入和卖出条件的更复杂统计。你可能也想确定触发开仓的信号蜡烛（signal candle）上的指标数值。

!!! Note
    以下买入原因（buy reason）分析仅适用于回测，*不适用于 hyperopt*。

我们需要在使用 `--export` 选项并将其设为 `signals` 的情况下运行回测，以导出信号**以及**交易：

``` bash
freqtrade backtesting -c <config.json> --timeframe <tf> --strategy <strategy_name> --timerange=<timerange> --export=signals
```

这会指示 freqtrade 输出一个 pickled 字典，包含策略、交易对，以及触发入场和离场信号所对应的蜡烛（candle）的 DataFrame。

根据你策略发出的入场信号数量，这个文件可能会变得相当大，因此请定期清理 `user_data/backtest_results` 文件夹，删除旧的导出文件。

在运行下一次回测之前，请确保要么删掉旧的回测结果，要么使用 `--cache none` 选项运行回测，以确保不会使用到缓存的结果。

如果一切顺利，你现在应该在 `user_data/backtest_results` 文件夹中看到 `backtest-result-{timestamp}_signals.pkl` 和 `backtest-result-{timestamp}_exited.pkl` 这两个文件。

为了分析入场/离场标签，我们现在需要使用 `freqtrade backtesting-analysis` 命令，并通过 `--analysis-groups` 选项传入以空格分隔的参数：

``` bash
freqtrade backtesting-analysis -c <config.json> --analysis-groups 0 1 2 3 4 5
```

该命令会读取最近一次的回测结果。`--analysis-groups` 选项用于指定各种表格化输出，展示每个分组或每笔交易的盈亏情况，范围从最简单（0）到最详细（按交易对、按买入、按卖出标签，4）：

* 0：按 enter_tag 划分的总体胜率和盈亏汇总
* 1：按 enter_tag 分组的盈亏汇总
* 2：按 enter_tag 和 exit_tag 分组的盈亏汇总
* 3：按交易对和 enter_tag 分组的盈亏汇总
* 4：按交易对、enter_tag 和 exit_tag 分组的盈亏汇总（这个可能会非常大）
* 5：按 exit_tag 分组的盈亏汇总

更多选项可通过 `-h` 选项查看。

### 使用 backtest-filename

默认情况下，`backtesting-analysis` 会处理 `user_data/backtest_results` 目录中最新的回测结果。如果你想分析更早一次回测的结果，可以使用 `--backtest-filename` 选项来指定目标文件。这样，你随时可以通过提供相关回测结果的文件名，重新查看并重新分析历史回测输出：

``` bash
freqtrade backtesting -c <config.json> --strategy <strategy_name> --timerange <timerange> --export signals --backtest-filename backtest-result-2025-03-05_20-38-34.zip
```

你应该会在日志中看到类似下面的输出，其中带有导出的带时间戳的文件名：

```
2022-06-14 16:28:32,698 - freqtrade.misc - INFO - dumping json to "mystrat_backtest-2022-06-14_16-28-32.json"
```

然后你可以在 `backtesting-analysis` 中使用该文件名：

``` bash
freqtrade backtesting-analysis -c <config.json> --backtest-filename=backtest-result-2025-03-05_20-38-34.zip
```

如果要使用来自不同结果目录的结果，可以使用 `--backtest-directory` 来指定目录：

``` bash
freqtrade backtesting-analysis -c <config.json> --backtest-directory custom_results/ --backtest-filename backtest-result-2025-03-05_20-38-34.zip
```

### 调整要显示的买入标签和卖出标签

要仅在显示输出中展示某些买入和卖出标签，可使用以下两个选项：

```
--enter-reason-list : 要分析的入场信号列表，以空格分隔。默认值："all"
--exit-reason-list : 要分析的离场信号列表，以空格分隔。默认值："all"
```

例如：

``` bash
freqtrade backtesting-analysis -c <config.json> --analysis-groups 0 2 --enter-reason-list enter_tag_a enter_tag_b --exit-reason-list roi custom_exit_tag_a stop_loss
```

### 输出信号蜡烛的指标

`freqtrade backtesting-analysis` 真正的强大之处在于，它能够打印出信号蜡烛上存在的指标数值，从而让你对买入信号指标进行细致的调查和调优。要为一组给定的指标打印出对应的列，可使用 `--indicator-list` 选项：

``` bash
freqtrade backtesting-analysis -c <config.json> --analysis-groups 0 2 --enter-reason-list enter_tag_a enter_tag_b --exit-reason-list roi custom_exit_tag_a stop_loss --indicator-list rsi rsi_1h bb_lowerband ema_9 macd macdsignal
```

这些指标必须存在于你策略的主 DataFrame 中（可以是主时间周期的，也可以是信息型时间周期 informative timeframes 的），否则它们在脚本输出中会被直接忽略。

!!! Note "Indicator List"
    指标数值会同时显示在入场点和离场点。如果指定了 `--indicator-list all`，则只会显示入场点的指标，以避免列表过大——具体大小取决于策略。

分析中还包含一系列蜡烛和交易相关的字段，因此通过将它们加入 indicator-list 即可自动访问，这些字段包括：

* **open_date     :** 交易开仓时间
* **close_date    :** 交易平仓时间
* **min_rate      :** 持仓期间见过的最低价格
* **max_rate      :** 持仓期间见过的最高价格
* **open          :** 信号蜡烛开盘价
* **close         :** 信号蜡烛收盘价
* **high          :** 信号蜡烛最高价
* **low           :** 信号蜡烛最低价
* **volume        :** 信号蜡烛成交量
* **profit_ratio  :** 交易盈亏比例
* **profit_abs    :** 交易的绝对盈亏金额

#### 指标数值的示例输出

``` bash
freqtrade backtesting-analysis -c user_data/config.json --analysis-groups 0 --indicator-list chikou_span tenkan_sen 
```

在本例中，我们旨在展示交易入场点和离场点上的 `chikou_span` 和 `tenkan_sen` 指标数值。

指标的一个示例输出可能如下所示：

| pair      | open_date                 | enter_reason | exit_reason | chikou_span (entry) | tenkan_sen (entry) | chikou_span (exit) | tenkan_sen (exit) |
|-----------|---------------------------|--------------|-------------|---------------------|--------------------|--------------------|-------------------|
| DOGE/USDT | 2024-07-06 00:35:00+00:00 |              | exit_signal | 0.105               | 0.106              | 0.105              | 0.107             |
| BTC/USDT  | 2024-08-05 14:20:00+00:00 |              | roi         | 54643.440           | 51696.400          | 54386.000          | 52072.010         |

如表中所示，`chikou_span (entry)` 表示交易入场时的指标数值，而 `chikou_span (exit)` 反映的则是离场时的数值。这种对指标数值的详细视图增强了分析能力。

会在指标后添加 `(entry)` 和 `(exit)` 后缀，以区分交易入场点和离场点时的数值。

!!! Note "交易级指标（Trade-wide Indicators）"
    某些交易级指标不带 `(entry)` 或 `(exit)` 后缀。这些指标包括：`pair`、`stake_amount`、
    `max_stake_amount`、`amount`、`open_date`、`close_date`、`open_rate`、`close_rate`、`fee_open`、`fee_close`、`trade_duration`、
    `profit_ratio`、`profit_abs`、`exit_reason`、`initial_stop_loss_abs`、`initial_stop_loss_ratio`、`stop_loss_abs`、`stop_loss_ratio`、
    `min_rate`、`max_rate`、`is_open`、`enter_tag`、`leverage`、`is_short`、`open_timestamp`、`close_timestamp` 和 `orders`

#### 根据入场或离场信号筛选指标

默认情况下，`--indicator-list` 选项会同时显示入场信号和离场信号的指标数值。要仅筛选入场信号的指标数值，可使用 `--entry-only` 参数。类似地，要仅显示离场信号的指标数值，可使用 `--exit-only` 参数。

示例：仅显示入场信号的指标数值：

``` bash
freqtrade backtesting-analysis -c user_data/config.json --analysis-groups 0 --indicator-list chikou_span tenkan_sen --entry-only
```

示例：仅显示离场信号的指标数值：

``` bash
freqtrade backtesting-analysis -c user_data/config.json --analysis-groups 0 --indicator-list chikou_span tenkan_sen --exit-only
```

!!! note 
    使用这些筛选器时，指标名称不会附加 `(entry)` 或 `(exit)` 后缀。

### 按日期筛选交易输出

要仅显示回测时间范围内、某个日期区间内的交易，请使用常规的 `timerange` 选项，格式为 `YYYYMMDD-[YYYYMMDD]`：

```
--timerange : 用于筛选输出交易的回测时间范围，开始日期包含，结束日期不包含。例如 20220101-20221231
```

例如，如果你的回测时间范围是 `20220101-20221231`，但你只想输出 1 月份的交易：

``` bash
freqtrade backtesting-analysis -c <config.json> --timerange 20220101-20220201
```

### 打印被拒绝的信号（rejected signals）

使用 `--rejected-signals` 选项来打印被拒绝的信号。

``` bash
freqtrade backtesting-analysis -c <config.json> --rejected-signals
```

### 将表格写入 CSV

某些表格化输出可能会变得很大，因此打印到终端并不理想。使用 `--analysis-to-csv` 选项关闭向标准输出打印表格，并将其写入 CSV 文件。

``` bash
freqtrade backtesting-analysis -c <config.json> --analysis-to-csv
```

默认情况下，它会为你在该 `backtesting-analysis` 命令中指定的每个输出表格各写一个文件，例如：

``` bash
freqtrade backtesting-analysis -c <config.json> --analysis-to-csv --rejected-signals --analysis-groups 0 1
```

这会写入 `user_data/backtest_results`：

* rejected_signals.csv
* group_0.csv
* group_1.csv

要覆盖文件写入位置，还可指定 `--analysis-csv-path` 选项。

``` bash
freqtrade backtesting-analysis -c <config.json> --analysis-to-csv --analysis-csv-path another/data/path/
```
