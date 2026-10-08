# Freqtrade 常见问题（FAQ）

## 支持的市场

Freqtrade 支持现货交易，以及对某些选定交易所的合约交易。请参考[文档起始页](index.md#supported-futures-exchanges) 以获取支持的交易所的最新列表。

### 我的机器人可以开空仓吗？

Freqtrade 可以在合约市场开空仓。这需要为该目的制作策略 - 并且在配置中使用 `"trading_mode": "futures"`。请务必先阅读[相关文档页面](leverage.md)。

在现货市场中，在某些情况下你可以使用杠杆现货代币，它反映了一个反向币对（例如 BTCUP/USD、BTCDOWN/USD、ETHBULL/USD、ETHBEAR/USD...），这些可以用 Freqtrade 交易。

### 我的机器人可以交易期权或期货吗？

合约交易受选定交易所支持。请参考[文档起始页](index.md#supported-futures-exchanges) 以获取支持的交易所的最新列表。

## 初学者提示与技巧

* 当你处理你的策略和 hyperopt 文件时，你应该使用一个合适的代码编辑器，如 VSCode 或 PyCharm。一个好的代码编辑器将提供语法高亮以及行号，使查找语法错误变得容易（Freqtrade 在启动期间很可能会指出这些错误）。

## Freqtrade 常见问题

### Freqtrade 可以在同一币对上并行开启多个仓位吗？

不可以。Freqtrade 一次只会为每个币对开启一个仓位。但是你可以使用 [`adjust_trade_position()` 回调](strategy-callbacks.md#adjust-trade-position) 来调整一个已开启的仓位。

回测通过 `--eps` 提供了此选项 - 然而这仅用于突出“隐藏”信号，并且在 live 模式下不起作用。

### Freqtrade 支持沙盒账户吗？

不支持，但你可以使用 dry-run 模式来模拟交易，而无需冒真实资金的风险。

沙盒市场是独立的、模拟的市场 - 不适合在真实环境中测试你的策略。这些市场通常有不同的订单簿、流动性和交易行为（通常参与者非常少）- 这使得它们不适合对你的策略进行真实测试。

### 机器人无法启动

运行 `freqtrade trade --config config.json` 显示输出 `freqtrade: command not found`（找不到命令）。

这可能是由于以下原因造成的：

* 虚拟环境未激活。
  * 运行 `source .venv/bin/activate` 来激活虚拟环境。
* 安装未成功完成。
  * 请查看 [安装文档](installation.md)。

### 机器人启动了，但处于 STOPPED（已停止）模式

确保你在 config.json 中将 `initial_state` 配置选项设置为 `"running"`。

### 我已经等了 5 分钟，为什么机器人还没有做任何交易？

* 取决于入场策略、白名单币的数量、市场状况等，找到一笔好的入场仓位可能需要数小时或数天。请耐心等待！

* 回测会大致告诉你预期有多少笔交易 - 但这并不能保证它们会随时间均匀分布 - 所以你可能一天有 20 笔交易，而本周其余时间都是 0。

* 这可能是因为配置错误。最好检查日志，它们通常会告诉你机器人是否只是没有收到买入信号（只有心跳消息），或者是否有问题（日志中的错误 / 异常）。

### 我已经做了 12 笔交易，为什么我的总利润是负的？

我理解你的失望，但不幸的是 12 笔交易确实不足以说明任何问题。如果你运行回测，你可以看到当前的算法确实让你处于盈利一方，但那是在数千笔交易之后，即使在那之后，你也会在特定币上留下亏损，这些币你交易了数十次甚至数百次。我们当然会不断努力改进机器人，但它将 _始终_ 是一场赌博，这应该让你在每月基础上获得 modest 的收益，但你无法从少数交易中说太多。

### 我想对配置进行更改。我可以在不终止机器人的情况下做到吗？

可以。你可以编辑你的配置并使用 `/reload_config` 命令重新加载配置。机器人将停止，重新加载配置和策略，并以新的配置和策略重新启动。

### 为什么我的机器人不卖出它买入的所有东西？

这被称为“coin dust”（零碎币），可能在所有交易所发生。发生这种情况是因为许多交易所从“接收货币”中扣除手续费 - 所以你买入 100 个 COIN - 但你只得到 99.9 个 COIN。由于 COIN 以整手大小（1COIN 步长）交易，你无法卖出 0.9 COIN（或 99.9 COIN）- 但你需要向下取整到 99 COIN。

这不是机器人的问题，在手动交易时也会发生。

虽然 freqtrade 可以处理这个问题（它将卖出 99 COIN），但手续费通常低于最小可交易手数（你只能交易整 COIN，不能交易 0.9 COIN）。将零碎币（0.9 COIN）留在交易所通常是有意义的，因为下次 freqtrade 买入 COIN 时，它会消耗剩余的少量余额，这次卖出它买入的所有东西，从而慢慢减少零碎币余额（尽管它很可能永远不会恰好达到 0）。

在可能的情况下（例如在 binance 上），使用交易所的专用手续费货币将解决这个问题。在 binance 上，在你的账户中拥有 BNB 并在你的个人资料中启用“用 BNB 支付手续费”就足够了。你的 BNB 余额将慢慢减少（因为它被用来支付手续费）- 但你将不再遇到零碎币问题（Freqtrade 会将手续费包含在利润计算中）。其他交易所不提供这种可能性，在这种情况下，这要么是你必须接受的事情，要么换到不同的交易所。

### 我向交易所存入了更多资金，但我的机器人没有识别出来

Freqtrade 将在必要时更新交易所余额（在下单之前）。RPC 调用（Telegram 的 `/balance`、对 `/balance` 的 API 调用）最多每小时触发一次更新。

如果启用了 `adjust_trade_position`（并且机器人有符合仓位调整条件的未平仓交易）- 那么钱包将每小时刷新一次。要强制立即更新，你可以使用 `/reload_config` - 这将重新启动机器人。

### 我想使用不完整的蜡烛

Freqtrade 不会向策略提供不完整的蜡烛。使用不完整蜡烛会导致 repainting（重绘），从而导致具有“幽灵”买入的策略，这些买入既无法回测，也无法在发生后验证。

你可以通过使用[数据提供者](strategy-customization.md#orderbookpair-maximum) 的 orderbook 或 ticker 方法来使用“当前”市场数据 - 但这些在回测期间无法使用。

### 是否有一个设置只出场已持有的交易而不执行任何新的入场？

你可以使用 Telegram 中的 `/stopentry` 命令来防止未来的交易入场，然后跟随 `/forceexit all`（卖出所有未平仓交易）。

### 我卖出了机器人的本金，现在日志中有错误

Freqtrade 假设它开启的交易仅通过机器人管理。如果你（意外地）卖出了机器人的本金，freqtrade 将尝试通过尝试重新查找交易所订单来恢复。

这是一种尽力而为的方法，并且不会在所有情况下都起作用，特别是在使用 freqtrade 不支持的订单类型（OCO、iceberg 等）时，或者在使用较旧的交易时（交易所不再提供完整的订单信息）。确切的限制将在交易所之间有所不同 - 细节通常在交易所的 API 文档中记录。

### 我想在同一台机器上运行多个机器人

请查看[高级设置文档页面](advanced-setup.md#running-multiple-instances-of-freqtrade)。

### 启动机器人时显示 "Impossible to load Strategy"（无法加载策略）

当机器人无法加载策略时会显示此错误消息。通常，你可以使用 `freqtrade list-strategies` 来列出所有可用的策略。此命令的输出还将包含一个状态列，显示策略是否可以加载。

请检查以下内容：

* 你使用的是正确的策略名称吗？策略名称区分大小写，并且必须对应于 Strategy 类名（不是文件名！）。
* 策略是否在 `user_data/strategies` 目录中，并且文件扩展名为 `.py`？
* 机器人是否在此错误之前显示了其他警告？也许你缺少策略的某些依赖 - 这将在日志中突出显示。
* 在 docker 的情况下 - 策略目录是否正确挂载（检查 docker-compose 文件的 volumes 部分）？

### 日志中显示 "Missing data fillup"（缺失数据填充）消息

此消息只是一个警告，表示最新的蜡烛中包含了缺失的蜡烛。根据交易所的不同，这可能表明该币对你使用的时间周期没有发生交易 - 并且交易所只返回有成交量的蜡烛。在低成交量币对上，这是相当常见的现象。

如果这种情况发生在 pairlist 中的所有币对上，这可能表明交易所最近的停机时间。请查看你交易所的公开频道以获取详细信息。

无论原因如何，Freqtrade 都会用“空”蜡烛填充这些蜡烛，其中 open、high、low 和 close 都设置为前一根蜡烛的收盘价 - 而 volume 为空。在图表中，这看起来像 `_` - 并且与交易所通常表示 0 成交量蜡烛的方式一致。

### 我收到 "Price jump between 2 candles detected"（检测到 2 根蜡烛之间的价格跳变）

此消息是一个警告，表示蜡烛出现了 > 30% 的价格跳变。这可能是币对停止交易并且发生了一些代币交换的迹象（例如 2021 年的 COCOS - 价格从 0.0000154 跳到 0.01621）。此消息通常伴随着 ["Missing data fillup"](#im-getting-missing-data-fillup-messages-in-the-log) - 因为在此类币对上的交易通常停止一段时间。

### 我想重置机器人的数据库

要重置机器人的数据库，你可以删除数据库（默认为 `tradesv3.sqlite` 或 `tradesv3.dryrun.sqlite`），或者通过 `--db-url` 使用不同的数据库 url（例如 `sqlite:///mynewdatabase.sqlite`）。

### 日志中显示 "Outdated history for pair xxx"（币对 xxx 的历史数据过期）

机器人试图告诉你，它得到了一个过期的最后一根蜡烛（不是最后一根完整的蜡烛）。因此，Freqtrade 不会为该币对进入交易 - 因为在旧信息上交易通常不是所期望的。

此警告可能指向以下一个或多个问题：

* 交易所停机 -> 查看你交易所的状态页面 / 博客 / twitter feed 以获取详细信息。
* 系统时间错误 -> 确保你的系统时间正确。
* 几乎不交易的币对 -> 在交易所网页上查看该币对，查看你的策略使用的时间周期。如果该币对在某些蜡烛中没有成交量（通常可视化为“volume 0”柱和作为蜡烛的“_”），则该币对在此时间周期内没有任何交易。这些币对最好避免，因为它们可能导致订单填充问题。
* API 问题 -> API 返回错误数据（此处仅为完整性而列出，在受支持的交易所不应发生）。

### 我在日志中收到 "Couldn't reuse watch for xxx"（无法重用 xxx 的 watch）消息

这是一条信息性消息，表示机器人尝试使用来自 websocket 的蜡烛，但交易所没有提供正确的信息。如果 websocket 连接中断 - 或者该币对你使用的时间周期内没有任何交易发生，则可能发生这种情况。

Freqtrade 将通过回退到 REST API 来优雅地处理此问题。虽然这会使迭代稍微变慢（由于 REST API 调用）- 但它不会对机器人的操作造成任何问题。

### 我收到 "Exchange XXX does not support market orders."（交易所 XXX 不支持市价订单）消息，并且无法运行我的策略

正如消息所说，你的交易所不支持市价订单，并且你将 [订单类型](configuration.md/#understand-order_types) 之一设置为 "market"。你的策略可能是为其他交易所编写的，并为 "stoploss" 订单设置了 "market" 订单，这对于大多数支持市价订单的交易所来说是正确的且更可取（但不适用于 Gate.io）。

要修复此问题，请在策略中将订单类型重新定义为使用 "limit" 而不是 "market"：

``` python
    order_types = {
        ...
        "stoploss": "limit",
        ...
    }
```

如果在配置文件中定义了订单类型（而不是在策略中），则应在配置文件中应用相同的修复。

### 我试图启动机器人的 live 模式，但收到 API 权限错误

像 `Invalid API-key, IP, or permissions for action`（API 密钥、IP 或操作权限无效）这样的错误正是它们实际所说的意思。你的 API 密钥要么无效（复制/粘贴错误？检查配置中的前导/尾随空格），要么已过期，要么是你运行机器人的 IP 未在交易所的 API 控制台中启用。通常，权限 "Spot Trading"（现货交易）（或你使用的交易所中的等效权限）将是必要的。合约通常需要专门启用。

### 如何搜索机器人日志中的某些内容？

默认情况下，机器人将其日志写入 stderr 流。这样实现是为了让你能够轻松地将机器人的诊断消息与回测、Edge 和 Hyperopt 结果、来自其他各种 Freqtrade 实用程序子命令的输出，以及你可能插入到策略中的自定义 `print()` 的输出分开。因此，如果你需要使用 grep 实用程序搜索日志消息，你需要将 stderr 重定向到 stdout 并忽略 stdout。

* 在 unix shell 中，这通常可以简单地完成：
```shell
$ freqtrade --some-options 2>&1 >/dev/null | grep 'something'
```
（注意，`2>&1` 和 `>/dev/null` 应按此顺序编写）

* Bash 解释器还支持所谓的进程替换语法，你可以用它来 grep 日志中的字符串：
```shell
$ freqtrade --some-options 2> >(grep 'something') >/dev/null
```
或
```shell
$ freqtrade --some-options 2> >(grep -v 'something' 1>&2)
```

* 你也可以使用 `--logfile` 选项将 Freqtrade 日志消息的副本写入文件：
```shell
$ freqtrade --logfile /path/to/mylogfile.log --some-options
```
然后像这样 grep 它：
```shell
$ cat /path/to/mylogfile.log | grep 'something'
```
或者甚至动态进行，随着机器人的工作以及日志文件的增长：
```shell
$ tail -f /path/to/mylogfile.log | grep 'something'
```
从单独的终端窗口。

在 Windows 上，Freqtrade 也支持 `--logfile` 选项，你可以使用 `findstr` 命令在日志中搜索感兴趣的字符串：
```
> type \path\to\mylogfile.log | findstr "something"
```

## Hyperopt 模块

### 为什么 freqtrade 没有 GPU 支持？

首先，大多数指标库没有 GPU 支持 - 因此，对指标计算几乎没有好处。GPU 改进将仅适用于 pandas 原生计算 - 或你自己编写的计算。

GPU 只擅长处理数字（浮点运算）。对于 hyperopt，我们既需要数字处理（寻找下一个参数）又需要运行 python 代码（运行回测）。因此，GPU 不太适合 hyperopt 的大部分内容。

因此，使用 GPU 的好处将相当微小 - 并且无法证明尝试添加 GPU 支持所带来的复杂性是合理的。

然而，如果你认为必须使用 GPU 启用的指标，没有什么能阻止你在策略内使用它们 - 但你可能会对被赋予的微小收益（与复杂性相比）所失望。

### 我需要多少个 epoch 才能获得好的 Hyperopt 结果？

默认情况下，Hyperopt 在没有 `-e`/`--epochs` 命令行选项调用时只会运行 100 个 epoch，意味着对你的触发器、守卫等进行 100 次评估。太少了，无法找到很好的结果（除非你非常幸运），所以你可能必须运行 10000 或更多。但这将需要永恒的时间来计算。

由于 hyperopt 使用贝叶斯搜索，运行过多的 epoch 可能不会产生更好的结果。

因此，建议反复运行 500-1000 个 epoch，直到你总共达到至少 10000 个 epoch（或者你对结果满意）。你最好通过查看结果来判断 - 如果机器人不断发现更好的策略，最好继续下去。

```bash
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy SampleStrategy -e 1000
```

### 为什么运行 hyperopt 需要很长时间？

* 使用 Hyperopt 发现一个好策略需要时间。研究 www.freqtrade.io、Freqtrade 文档页面，加入 Freqtrade [discord 社区](https://discord.gg/p7nuUNVfP7)。当你耐心等待世界上最先进、免费的加密机器人，为你专门设计的一个可能的黄金策略时。

* 如果你想知道为什么在这里做 1000 个 epoch 可能需要 20 分钟到几天，这里有一些答案：

此答案是在 0.15.1 版本发布期间编写的，当时我们有：

* 8 个触发器
* 9 个守卫：假设我们从每个中评估 10 个值
* 1 个止损计算：假设我们也想从中评估 10 个值

以下计算仍然非常粗略且不精确，但它会给出这个想法。仅使用这些触发器和守卫，就已经有 8\*10^9\*10 次评估。大约总共 800 亿次评估。你运行了 100000 次评估？恭喜，你完成了搜索空间的约 1 / 100000，假设机器人从未多次测试相同的参数。

* 运行 1000 个 hyperopt epoch 所需的时间取决于诸如：可用的 cpu、硬盘、ram、时间周期、timerange、指标设置、指标数量、hyperopt 测试策略所基于的币数量以及由此产生的交易数量 - 这可以是一年 650 笔交易或 100000 笔交易，取决于策略是旨在通过很少交易获得大利润，还是通过许多低利润交易。

示例：一年 4% 利润 650 次 vs 每笔交易 0.3% 利润 10000 次。如果我们假设你将 --timerange 设置为 365 天。

示例：
`freqtrade --config config.json --strategy SampleStrategy --hyperopt SampleHyperopt -e 1000 --timerange 20190601-20200601`

## 官方频道

Freqtrade 专门使用以下官方频道：

* [Freqtrade discord 服务器](https://discord.gg/p7nuUNVfP7)
* [Freqtrade 文档 (https://freqtrade.io)](https://freqtrade.io)
* [Freqtrade github 组织](https://github.com/freqtrade)

任何与 freqtrade 项目有关联的人都不会向你要你的交易所密钥或任何其他会暴露你的资金以供利用的东西。如果你被要求暴露你的交易所密钥或将资金发送到某个随机钱包，请不要遵循这些指示。

未能遵循这些准则将不是 freqtrade 的责任。

## 支持政策

我们在我们的 [Discord 服务器](https://discord.gg/p7nuUNVfP7) 和通过 GitHub issues 为 Freqtrade 提供免费支持。我们只支持最新的发布版本（例如 2025.8）和当前的开发分支（例如 2025.9-dev）。

如果你使用的是较旧的版本，请遵循[升级说明](updating.md)，看看你的问题是否已经被解决。

## “Freqtrade token”

Freqtrade 没有提供加密代币发行。

你在互联网上找到的提及 Freqtrade、FreqAI 或 freqUI 的代币发行必须被视为骗局，试图利用 freqtrade 的知名度来谋取他们自己的、邪恶的收益。
