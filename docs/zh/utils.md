<!-- 本文件为中文翻译版，由 AI 根据 docs/utils.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 实用子命令

除了实时交易和模拟运行模式、`backtesting` 和 `hyperopt` 优化子命令，以及用于准备历史数据的 `download-data` 子命令之外，机器人还包含许多实用子命令。本节将对这些子命令进行说明。

## 创建用户目录

创建用于存放 freqtrade 文件的目录结构。
同时会为你创建策略和 hyperopt 示例文件，帮助你快速上手。
可以多次使用——使用 `--reset` 会将示例策略和 hyperopt 文件重置为默认状态。

--8<-- "commands/create-userdir.md"

!!! Warning
    使用 `--reset` 可能会导致数据丢失，因为这会覆盖所有示例文件而不会再次询问。

```
├── backtest_results
├── data
├── hyperopt_results
├── hyperopts
│   ├── sample_hyperopt_loss.py
├── notebooks
│   └── strategy_analysis_example.ipynb
├── plot
└── strategies
    └── sample_strategy.py
```

## 创建新配置

创建一个新的配置文件，过程中会询问一些对配置来说很重要的选择项。

--8<-- "commands/new-config.md"

!!! Warning
    只会询问关键问题。Freqtrade 提供了更多的配置选项，详见[配置文档](configuration.md#configuration-parameters)。

### 创建配置示例

```
$ freqtrade new-config --config user_data/config_binance.json

? Do you want to enable Dry-run (simulated trades)?  Yes
? Please insert your stake currency: BTC
? Please insert your stake amount: 0.05
? Please insert max_open_trades (Integer or -1 for unlimited open trades): 3
? Please insert your desired timeframe (e.g. 5m): 5m
? Please insert your display Currency (for reporting): USD
? Select exchange  binance
? Do you want to enable Telegram?  No
```

## 显示配置

显示配置文件（默认会对敏感值进行脱敏处理）。
对于[分离配置文件](configuration.md#multiple-configuration-files)或[环境变量](configuration.md#environment-variables)特别有用，此命令会显示合并后的配置。

![Show config output](../assets/show-config-output.png)

--8<-- "commands/show-config.md"

``` output
Your combined configuration is:
{
  "exit_pricing": {
    "price_side": "other",
    "use_order_book": true,
    "order_book_top": 1
  },
  "stake_currency": "USDT",
  "exchange": {
    "name": "binance",
    "key": "REDACTED",
    "secret": "REDACTED",
    "ccxt_config": {},
    "ccxt_async_config": {},
  }
  // ...
}
```

!!! Warning "分享此命令提供的信息"
    我们会尝试从默认输出（不使用 `--show-sensitive` 时）中移除所有已知的敏感信息。
    但请务必仔细检查输出中的敏感值，确保你不会意外暴露某些私人信息。

## 创建新策略

从类似于 SampleStrategy 的模板创建新策略。
文件将根据你的类名命名，不会覆盖已有文件。

结果将保存在 `user_data/strategies/<strategyclassname>.py`。

--8<-- "commands/new-strategy.md"

### new-strategy 使用示例

```bash
freqtrade new-strategy --strategy AwesomeStrategy
```

使用自定义用户目录

```bash
freqtrade new-strategy --userdir ~/.freqtrade/ --strategy AwesomeStrategy
```

使用高级模板（填充所有可选函数和方法）

```bash
freqtrade new-strategy --strategy AwesomeStrategy --template advanced
```

## 列出策略

使用 `list-strategies` 子命令查看特定目录中的所有策略。

此子命令对于发现环境中加载策略的问题很有用：包含错误且加载失败的策略模块会以红色显示（LOAD FAILED），而名称重复的策略会以黄色显示（DUPLICATE NAME）。

--8<-- "commands/list-strategies.md"

!!! Warning
    使用这些命令会尝试加载目录中的所有 Python 文件。如果该目录中存在不受信任的文件，这可能会带来安全风险，因为所有模块级代码都会被执行。

示例：搜索默认策略目录（在默认用户目录中）。

``` bash
freqtrade list-strategies
```

示例：搜索用户目录中的策略目录。

``` bash
freqtrade list-strategies --userdir ~/.freqtrade/
```

示例：搜索指定的策略路径。

``` bash
freqtrade list-strategies --strategy-path ~/.freqtrade/strategies/
```

## 列出 Hyperopt 损失函数

使用 `list-hyperoptloss` 子命令查看所有可用的 hyperopt 损失函数。

它提供了环境中所有可用损失函数的快速列表。

此子命令对于发现环境中加载损失函数的问题很有用：包含错误且加载失败的 Hyperopt 损失函数模块会以红色显示（LOAD FAILED），而名称重复的 Hyperopt 损失函数会以黄色显示（DUPLICATE NAME）。

--8<-- "commands/list-hyperoptloss.md"

## 列出 FreqAI 模型

使用 `list-freqaimodels` 子命令查看所有可用的 freqAI 模型。

此子命令对于发现环境中加载 freqAI 模型的问题很有用：包含错误且加载失败的模型模块会以红色显示（LOAD FAILED），而名称重复的模型会以黄色显示（DUPLICATE NAME）。

--8<-- "commands/list-freqaimodels.md"

## 列出交易所

使用 `list-exchanges` 子命令查看机器人可用的交易所。

--8<-- "commands/list-exchanges.md"

示例：查看机器人可用的交易所：

```
$ freqtrade list-exchanges
Exchanges available for Freqtrade:
Exchange name       Supported    Markets                 Reason
------------------  -----------  ----------------------  ------------------------------------------------------------------------
binance             Official     spot, isolated futures
bitmart             Official     spot
bybit                            spot, isolated futures
gate                Official     spot, isolated futures
htx                 Official     spot
huobi                            spot
kraken              Official     spot
okx                 Official     spot, isolated futures
```

!!! info ""
    为清晰起见，输出已精简——受支持的可用交易所可能会随时间变化。

!!! Note "missing opt exchanges"
    值为 "missing opt:" 的交易所可能需要特殊配置（例如，如果缺少 `fetchTickers` 则使用 orderbook）——但理论上应该可以工作（尽管我们无法保证它们一定会正常工作）。

示例：查看 ccxt 库支持的所有交易所（包括"有问题的"交易所，即已知无法与 Freqtrade 配合使用的交易所）

```
$ freqtrade list-exchanges -a
All exchanges supported by the ccxt library:
Exchange name       Valid    Supported    Markets                 Reason
------------------  -------  -----------  ----------------------  ---------------------------------------------------------------------------------
binance             True     Official     spot, isolated futures
bitflyer            False                 spot                    missing: fetchOrder. missing opt: fetchTickers.
bitmart             True     Official     spot
bybit               True                  spot, isolated futures
gate                True     Official     spot, isolated futures
htx                 True     Official     spot
kraken              True     Official     spot
okx                 True     Official     spot, isolated futures
```

!!! info ""
    输出已精简——受支持的可用交易所可能会随时间变化。

## 列出时间周期

使用 `list-timeframes` 子命令查看交易所可用的时间周期列表。

--8<-- "commands/list-timeframes.md"

* 示例：查看配置文件中设置的 'binance' 交易所的时间周期：

```
$ freqtrade list-timeframes -c config_binance.json
...
Timeframes available for the exchange `binance`: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
```

* 示例：枚举 Freqtrade 可用的交易所并打印每个交易所支持的时间周期：
```
$ for i in `freqtrade list-exchanges -1`; do freqtrade list-timeframes --exchange $i; done
```

## 列出交易对 / 列出市场

`list-pairs` 和 `list-markets` 子命令允许查看交易所上可用的交易对/市场。

交易对是市场符号中基础货币部分和报价货币部分之间带有 '/' 字符的市场。
例如，在 'ETH/BTC' 交易对中，'ETH' 是基础货币，而 'BTC' 是报价货币。

对于 Freqtrade 交易的交易对，报价货币由 `stake_currency` 配置设置的值定义。

你可以使用这些子命令打印任何交易对/市场的信息——还可以使用 `--quote BTC` 按报价货币过滤输出，或使用 `--base ETH` 按基础货币过滤输出。

这些子命令具有相同的用法和相同的可用选项集：

--8<-- "commands/list-pairs.md"

默认情况下，只显示活跃的交易对/市场。活跃的交易对/市场是当前可以在交易所进行交易的那些。
你可以使用 `-a`/`-all` 选项查看所有交易对/市场的列表，包括不活跃的那些。
如果市场上最小可交易价格非常小（即小于 `1e-11`（`0.00000000001`）），交易对可能会被列为不可交易。

交易对/市场在打印输出中按其符号字符串排序。

### 示例

* 以 JSON 格式打印默认配置文件中指定交易所（即 "Binance" 交易所）上报价货币为 USD 的活跃交易对列表：

```
$ freqtrade list-pairs --quote USD --print-json
```

* 打印 `config_binance.json` 配置文件中指定交易所（即 "Binance" 交易所）上所有交易对的列表，筛选基础货币为 BTC 或 ETH、报价货币为 USDT 或 USD 的交易对，以人类可读的列表形式显示并附带摘要：

```
$ freqtrade list-pairs -c config_binance.json --all --base BTC ETH --quote USDT USD --print-list
```

* 以表格格式打印 "Kraken" 交易所上的所有市场：

```
$ freqtrade list-markets --exchange kraken --all
```

## 测试交易对列表

使用 `test-pairlist` 子命令测试[动态交易对列表](plugins.md#pairlists)的配置。

需要一个指定了 `pairlists` 属性的配置。
可用于生成在回测/hyperopt 期间使用的静态交易对列表。

--8<-- "commands/test-pairlist.md"

### 示例

使用[动态交易对列表](plugins.md#pairlists)时显示白名单。

```
freqtrade test-pairlist --config config.json --quote USDT BTC
```

## 转换数据库

`freqtrade convert-db` 可用于将数据库从一个系统转换到另一个系统（sqlite -> postgres，postgres -> 其他 postgres），迁移所有交易、订单和 Pairlock。

请参阅[相关文档](advanced-setup.md#use-a-different-database-system)了解不同数据库系统的要求。

--8<-- "commands/convert-db.md"

!!! Warning
    请确保仅在空的目标数据库上使用此命令。Freqtrade 将执行常规迁移，但如果已存在条目可能会失败。

## Webserver 模式

!!! Warning "实验性功能"
    Webserver 模式是一种实验性模式，旨在提高回测和策略开发的生产力。
    可能仍存在 bug——如果你碰巧遇到了这些问题，请将它们作为 github issue 报告，谢谢。

以 webserver 模式运行 freqtrade。
Freqtrade 将启动 web 服务器，允许 FreqUI 启动和控制回测过程。
这样做的优势在于，数据不会在回测运行之间重新加载（只要时间周期和时间范围保持不变）。
FreqUI 还会显示回测结果。

--8<-- "commands/webserver.md"

### Webserver 模式 - Docker

你也可以通过 Docker 使用 webserver 模式。
启动一次性容器需要显式配置端口，因为默认情况下端口不会暴露。
你可以使用 `docker compose run --rm -p 127.0.0.1:8080:8080 freqtrade webserver` 来启动一个一次性容器，停止后会自动移除。这假设端口 8080 仍然可用且没有其他机器人在该端口上运行。

或者，你可以重新配置 docker-compose 文件来更新命令：

``` yml
    command: >
      webserver
      --config /freqtrade/user_data/config.json
```

你现在可以使用 `docker compose up` 来启动 web 服务器。
这假设配置中已启用 web 服务器并为 Docker 进行了配置（监听端口 = `0.0.0.0`）。

!!! Tip
    如果你想启动实时交易或模拟运行的机器人，别忘了将命令重置回 trade 命令。

## 显示之前的回测结果

允许你显示之前的回测结果。
添加 `--show-pair-list` 会输出一个排序后的交易对列表，你可以轻松复制粘贴到配置中（省略表现不佳的交易对）。

??? Warning "策略过拟合"
    只使用盈利的交易对可能导致策略过拟合，该策略在未来数据上表现不佳。在冒真金白银的风险之前，请务必在模拟运行中对策略进行充分测试。

--8<-- "commands/backtesting-show.md"

## 详细的回测分析

高级回测结果分析。

更多详情请参阅[回测分析](advanced-backtesting.md#analyze-the-buyentry-and-sellexit-tags)章节。

--8<-- "commands/backtesting-analysis.md"

## 列出 Hyperopt 结果

你可以使用 `hyperopt-list` 子命令列出 Hyperopt 模块之前评估的超优化迭代。

--8<-- "commands/hyperopt-list.md"

!!! Note
    `hyperopt-list` 会自动使用最新的可用 hyperopt 结果文件。
    你可以使用 `--hyperopt-filename` 参数覆盖此设置，指定另一个可用的文件名（不带路径！）。

### 示例

列出所有结果，最后打印最佳结果的详细信息：
```
freqtrade hyperopt-list
```

只列出有正利润的迭代。不打印最佳迭代的详细信息，以便可以在脚本中迭代列表：
```
freqtrade hyperopt-list --profitable --no-details
```

## 显示 Hyperopt 结果的详细信息

你可以使用 `hyperopt-show` 子命令显示 Hyperopt 模块之前评估的任何超优化迭代的详细信息。

--8<-- "commands/hyperopt-show.md"

!!! Note
    `hyperopt-show` 会自动使用最新的可用 hyperopt 结果文件。
    你可以使用 `--hyperopt-filename` 参数覆盖此设置，指定另一个可用的文件名（不带路径！）。

### 示例

打印迭代 168 的详细信息（迭代编号由 `hyperopt-list` 子命令或 Hyperopt 在超优化运行期间显示）：

```
freqtrade hyperopt-show -n 168
```

以 JSON 格式打印最后一个最佳迭代（即所有迭代中的最佳者）的详细信息：

```
freqtrade hyperopt-show --best -n -1 --print-json --no-header
```

## 显示交易

从数据库中打印选定的（或所有）交易到屏幕。

--8<-- "commands/show-trades.md"

### 示例

以 JSON 格式打印 ID 为 2 和 3 的交易

``` bash
freqtrade show-trades --db-url sqlite:///tradesv3.sqlite --trade-ids 2 3 --print-json
```

## 策略更新器

将列出的策略或策略文件夹中的所有策略更新为 v3 兼容版本。
如果命令运行时未使用 --strategy-list，则策略文件夹内的所有策略都将被转换。
你的原始策略将保留在 `user_data/strategies_orig_updater/` 目录中。

!!! Warning "转换结果"
    策略更新器将以"尽力而为"的方式工作。请务必尽职调查并验证转换结果。
    我们还建议运行 Python 格式化工具（例如 `ruff format`）以合理的方式格式化结果。

--8<-- "commands/strategy-updater.md"