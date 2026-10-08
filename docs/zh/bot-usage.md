<!-- 本文件为中文翻译版，由 AI 根据 docs/bot-usage.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 启动机器人

本页讲解机器人的各种参数以及如何运行它。

!!! Note
    如果你使用了 `setup.sh`，在运行 freqtrade 命令之前，别忘了激活你的虚拟环境（`source .venv/bin/activate`）。

!!! Warning "时钟准确"
    运行机器人的系统其时钟必须准确，并且要足够频繁地与 NTP 服务器同步，以避免与交易所通信时出现问题。

## 机器人命令

--8<-- "../commands/main.md"

### 机器人交易命令

--8<-- "../commands/trade.md"

### 如何指定要使用的配置文件？

机器人允许你通过 `-c/--config` 命令行选项来选择要使用的配置文件：

```bash
freqtrade trade -c path/far/far/away/config.json
```

默认情况下，机器人会从当前工作目录加载 `config.json` 配置文件。

### 如何使用多个配置文件？

机器人允许你在命令行中指定多个 `-c/--config` 选项来使用多个配置文件。在后面的配置文件中定义的配置参数，会覆盖前面命令行中较早指定的同名配置参数。

例如，你可以创建一个单独的配置文件，存放你用于交易的交易所的 key 和 secret，在 Dry 模式（实际并不需要使用它们）下运行时使用默认的、key 和 secret 为空的配置文件：

```bash
freqtrade trade -c ./config.json
```

而在正常的实盘交易模式（Live Trade Mode）下运行时，同时指定这两个配置文件：

```bash
freqtrade trade -c ./config.json -c path/to/secrets/keys.config.json
```

这样做可以通过为包含真实密钥的文件设置适当的文件权限，从而在你的本地机器上隐藏交易所的私有 key 和 secret，并且还能防止你在项目 issue 或互联网上发布配置示例时，意外泄露敏感的私有数据。

更多细节和示例，请参阅[配置](configuration.md)文档页。

### 自定义数据存放在哪里？

Freqtrade 允许通过 `freqtrade create-userdir --userdir someDirectory` 创建用户数据目录。该目录结构如下：

```
user_data/
├── backtest_results
├── data
├── hyperopts
├── hyperopt_results
├── plot
└── strategies
```

你可以将 `user_data_dir` 设置项添加到你的配置中，让机器人始终指向该目录。或者，在每条命令中都传入 `--userdir`。

如果该目录不存在，机器人将启动失败，但会自动创建必要的子目录。

该目录应当包含你的自定义策略、自定义 hyperopt 以及 hyperopt 损失函数、回测历史数据（使用 backtesting 命令或下载脚本下载得到），以及绘图输出。

建议对策略的改动使用版本控制进行跟踪。

### 如何使用 **--strategy**？

该参数允许你加载你的自定义策略类。要测试机器人安装，你可以使用由 `create-userdir` 子命令安装的 `SampleStrategy`（通常位于 `user_data/strategy/sample_strategy.py`）。

机器人会在 `user_data/strategies` 中搜索你的策略文件。要使用其他目录，请阅读下一节关于 `--strategy-path` 的内容。

要加载一个策略，只需在此参数中传入类名（例如：`CustomStrategy`）。

**示例：**
在 `user_data/strategies` 中你有一个文件 `my_awesome_strategy.py`，其中有一个名为 `AwesomeStrategy` 的策略类，要加载它：

```bash
freqtrade trade --strategy AwesomeStrategy
```

如果机器人没有找到你的策略文件，它会显示一条错误信息说明原因（文件未找到，或你的代码有错误）。

更多关于策略文件的内容，请参阅[策略定制](strategy-customization.md)。

### 如何使用 **--strategy-path**？

该参数允许你添加一个额外的策略查找路径，它会在默认位置（传入的路径必须是一个目录！）之前被检查：

```bash
freqtrade trade --strategy AwesomeStrategy --strategy-path /some/directory
```

#### 如何安装一个策略？

这非常简单。将你的策略文件复制粘贴到 `user_data/strategies` 目录，或使用 `--strategy-path`。然后，机器人就可以使用它了。

### 如何使用 **--db-url**？

当你以 Dry-run（模拟运行）模式运行机器人时，默认不会将任何交易存储到数据库中。如果你想通过 `--db-url` 将机器人的操作存储到数据库，也可以用它来在生产模式下指定自定义数据库。示例命令：

```bash
freqtrade trade -c config.json --db-url sqlite:///tradesv3.dry_run.sqlite
```

## 下一步

机器人的最优策略会随市场趋势的变化而改变。下一步是[策略定制](strategy-customization.md)。
