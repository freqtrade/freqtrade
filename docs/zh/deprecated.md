<!-- 本文件为中文翻译版，由 AI 根据 docs/deprecated.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 已弃用的功能

本页包含已被机器人开发团队声明为**已弃用（DEPRECATED）**且不再受支持的命令行参数、配置参数以及机器人功能的说明。请避免在你的配置中使用它们。

## 已移除的功能

### `--refresh-pairs-cached` 命令行选项

在回测、hyperopt 和 edge 的上下文中，`--refresh-pairs-cached` 允许刷新用于回测的蜡烛数据。
由于这会导致很多困惑，并且拖慢回测速度（同时它又不是回测的一部分），因此它已被拆分成一个独立的 freqtrade 子命令 `freqtrade download-data`。

该命令行选项在 2019.7-dev（develop 分支）中被弃用，并在 2019.9 中移除。

### `--dynamic-whitelist` 命令行选项

该命令行选项在 2018 年被弃用，并在 freqtrade 2019.6-dev（develop 分支）和 freqtrade 2019.7 中移除。
请改用 [pairlists](plugins.md#pairlists-and-pairlist-handlers)。

### `--live` 命令行选项

在回测上下文中，`--live` 允许下载最新的 tick 数据用于回测。
它只下载了最新的 500 根蜡烛，因此在获取良好的回测数据方面是无效的。
在 2019-7-dev（develop 分支）和 freqtrade 2019.8 中移除。

### `ticker_interval`（现为 `timeframe`）

对 `ticker_interval` 术语的支持在 2020.6 中被弃用，让位于 `timeframe`——兼容代码在 2022.3 中被移除。

### 允许按顺序运行多个 pairlist

配置中前者的 `"pairlist"` 段已被移除，并被 `"pairlists"`（一个用于指定一系列 pairlist 的列表）取代。

旧的配置参数段（`"pairlist"`）在 2019.11 中被弃用，并在 2020.4 中移除。

### 弃用 volume-pairlist 中的 bidVolume 和 askVolume

由于只有 quoteVolume 可以在资产之间进行比较，其他选项（bidVolume、askVolume）在 2020.4 中被弃用，并在 2020.9 中移除。

### 使用订单簿步进确定离场价格

使用 `order_book_min` 和 `order_book_max` 曾经允许步进订单簿并尝试寻找下一个 ROI 槽位——试图尽早放置卖单。
然而，由于这会增加风险且没有任何好处，出于可维护性的考虑，它在 2021.7 中被移除。

### 旧版 Hyperopt 模式

使用单独的 hyperopt 文件在 2021.4 中被弃用，并在 2021.9 中移除。
请切换到新的 [参数化策略](hyperopt.md) 以受益于新的 hyperopt 接口。

## V2 与 V3 之间的策略变化

隔离合约（Isolated Futures）/ 做空交易在 2022.4 中引入。这需要对配置设置、策略接口等进行重大更改。

我们已尽最大努力保持与现有策略的兼容性，所以如果你只是想在现货市场继续使用 freqtrade，则不需要做任何更改。
虽然我们可能在未来的某个时候放弃对当前接口的支持，但我们会单独宣布这一点，并安排适当的过渡期。

请遵循[策略迁移](strategy_migration.md)指南将你的策略迁移到新格式，以开始使用新功能。

### webhooks - 2022.4 的变化

#### `buy_tag` 已被重命名为 `enter_tag`

这应该只适用于你的策略，可能也适用于 webhooks。
我们会保留 1-2 个版本的兼容层（因此 `buy_tag` 和 `enter_tag` 都仍会工作），但在此之后 webhooks 中对它的支持将消失。

#### 命名变化

Webhook 的术语从 "sell" 改为 "exit"，从 "buy" 改为 "entry"，并在此过程中移除了 "webhook"。

* `webhookbuy`、`webhookentry` -> `entry`
* `webhookbuyfill`、`webhookentryfill` -> `entry_fill`
* `webhookbuycancel`、`webhookentrycancel` -> `entry_cancel`
* `webhooksell`、`webhookexit` -> `exit`
* `webhooksellfill`、`webhookexitfill` -> `exit_fill`
* `webhooksellcancel`、`webhookexitcancel` -> `exit_cancel`

## 移除 `populate_any_indicators`

2023.3 版本移除了 `populate_any_indicators`，转而使用用于特征工程和目标的分离方法。完整细节请阅读 [迁移文档](strategy_migration.md#freqai-strategy)。

## 从配置中移除 `protections`

通过 `"protections": [],` 从配置中设置 protections 在 2024.10 中被移除，此前已经发出了超过 3 年的弃用警告。

## hdf5 数据存储

使用 hdf5 作为数据存储已在 2024.12 中被弃用，并在 2025.1 中移除。我们建议切换到 feather 数据格式。

请在更新之前使用 [`convert-data` 子命令](data-download.md#sub-command-convert-data) 将你现有的数据转换为受支持的格式之一。

## 通过配置配置高级日志

通过 `--logfile systemd` 和 `--logfile journald` 分别配置 syslog 和 journald 已在 2025.3 中被弃用。
请改用基于配置的 [日志设置](advanced-setup.md#advanced-logging)。

## 移除 edge 模块

edge 模块在 2023.9 中被弃用，并在 2025.6 中移除。
edge 的所有功能都已被移除，配置了 edge 会导致错误。

## 对动态资金费率处理的调整

在 2025.12 版本中，动态资金费率的处理进行了调整，以同时支持低至 1h 资金间隔的动态资金费率。
因此，mark 和资金费率的时间周期已更改为每个受支持的合约交易所的 1h。

由于 mark 和 funding_fee 蜡烛的时间周期已更改（通常从 8h 改为 1h）——已经下载的数据将必须被调整或部分重新下载。
你可以重新下载所有内容（`freqtrade download-data [...] --erase` - :warning: 可能耗时很长）——或者有选择地下载更新的数据。

### 策略

大多数策略应该不需要调整即可继续按预期工作——但是，使用 `@informative("8h", candle_type="funding_rate")` 或类似写法的策略必须将时间周期切换为 1h。
对于 `dp.get_pair_dataframe(metadata["pair"], "8h", candle_type="funding_rate")` 也是如此——需要切换到 1h。

freqtrade 会自动调整时间周期并返回 `funding_rates`，尽管给定的时间周期是错误的。它会发出警告——并且可能仍然会破坏你的策略。

### 有选择地重新下载数据

下面的脚本应作为一个示例——你可能需要根据你的需要调整时间周期和交易所！

``` bash
# 清理不再需要的数据
rm user_data/data/<exchange>/futures/*-mark*
rm user_data/data/<exchange>/futures/*-funding_rate*

# 下载新数据（只需运行一次以修复 mark 和资金费率数据）
freqtrade download-data -t 1h --trading-mode futures --candle-types funding_rate mark [...] --timerange <你拥有其他数据的完整时间范围>

```

上述操作的结果是你的 funding_rates 和 mark 数据将具有 1h 时间周期。
你可以用 `freqtrade list-data --exchange <yourexchange> --show` 来验证这一点。

!!! Note "附加参数"
    上述命令可能需要附加参数，例如配置文件或显式的 user_data（如果它们与默认值不同）。

**Hyperliquid** 现在是一个特例——它将不再需要 1h mark 数据——而是使用常规蜡烛（这些数据从未存在过，并且与 1h 合约蜡烛相同）。由于我们不支缓为 hyperliquid 下载数据（它们不提供历史数据）——因此 hyperliquid 用户无需采取任何操作。

## FreqAI 中的 Catboost 模型

CatBoost 模型已在 2025.12 版本中移除，不再主动支持。
如果你有使用 CatBoost 模型的现有机器人，你仍然可以通过从 git 历史记录（如下链接）中复制/粘贴它们并手动安装 Catboost 库，在你的自定义模型中使用它们。
但是我们建议切换到其他受支持的模型库，如 LightGBM 或 XGBoost，以获得更好的支持以及未来的兼容性。

* [CatboostRegressor](https://github.com/freqtrade/freqtrade/blob/c6f3b0081927e161a16b116cc47fb663f7831d30/freqtrade/freqai/prediction_models/CatboostRegressor.py)
* [CatboostClassifier](https://github.com/freqtrade/freqtrade/blob/c6f3b0081927e161a16b116cc47fb663f7831d30/freqtrade/freqai/prediction_models/CatboostClassifier.py)
* [CatboostClassifierMultiTarget](https://github.com/freqtrade/freqtrade/blob/c6f3b0081927e161a16b116cc47fb663f7831d30/freqtrade/freqai/prediction_models/CatboostClassifierMultiTarget.py)
