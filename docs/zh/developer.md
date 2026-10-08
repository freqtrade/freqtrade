<!-- 本文件为中文翻译版，由 AI 根据 docs/developer.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 开发帮助

本页面向 Freqtrade 的开发者、想要为 Freqtrade 代码库或文档做出贡献的人，或者想要理解他们所运行应用程序源代码的人。

我们欢迎所有的贡献、bug 报告、bug 修复、文档改进、增强功能和想法。我们在 [GitHub](https://github.com) 上[跟踪 issue](https://github.com/freqtrade/freqtrade/issues)，并且在 [discord](https://discord.gg/p7nuUNVfP7) 上也有一个开发频道，你可以在那里提问。

## 文档

文档可在 [https://freqtrade.io](https://www.freqtrade.io/) 获取，并且每个新功能 PR 都需要提供文档。

文档的特殊字段（如 Note 框等）可以在[这里](https://squidfunk.github.io/mkdocs-material/reference/admonitions/)找到。

要在本地测试文档，请使用以下命令。

``` bash
pip install -r docs/requirements-docs.txt
mkdocs serve
```

这将启动一个本地服务器（通常在 8000 端口），这样你就可以看到一切是否如你所愿。

## 开发者环境搭建

要配置开发环境，你可以使用提供的 [DevContainer](#devcontainer-setup)，或者使用 `setup.sh` 脚本并在被问到 "Do you want to install dependencies for dev [y/N]? " 时回答 "y"。
或者（例如，如果你的系统不被 setup.sh 脚本支持），请遵循手动安装过程并运行 `pip3 install -r requirements-dev.txt`——接着运行 `pip3 install -e .[all]`。

这将安装开发所需的所有工具，包括 `pytest`、`ruff`、`mypy` 和 `coveralls`。

运行以下命令来安装 git hook 脚本：

``` bash
pre-commit install
```

这些 pre-commit 脚本会在每次提交前自动检查你的改动。
如果发现任何格式问题，提交将失败并提示修复。
这减少了不必要的 CI 失败，降低了维护负担，并提高了代码质量。

你可以在必要时用 `pre-commit run -a` 手动运行这些检查。

在开启 pull request 之前，请先熟悉我们的[贡献指南](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)。

### Devcontainer 搭建

最快最简单的方法是使用带有 Remote container 扩展的 [VSCode](https://code.visualstudio.com/)。
这使开发者能够启动机器人及其所有必需的依赖项，***无需***在本地机器上安装任何 freqtrade 特定的依赖项。

#### Devcontainer 依赖

* [VSCode](https://code.visualstudio.com/)
* [docker](https://docs.docker.com/install/)
* [Remote container 扩展文档](https://code.visualstudio.com/docs/remote)

关于 [Remote container 扩展](https://code.visualstudio.com/docs/remote) 的更多信息，最好查阅文档。

### 测试

新代码应该被基础的单元测试覆盖。根据功能的复杂程度，审查者可能会要求更深入的单元测试。
如有必要，Freqtrade 团队可以协助并指导编写良好的测试（但请不要指望有人会为你编写测试）。

#### 如何运行测试

在根目录使用 `pytest` 来运行所有可用的测试用例，并确认你的本地环境已正确搭建。

!!! Note "feature 分支"
    测试应在 `develop` 和 `stable` 分支上通过。其他分支可能仍是进行中的工作，测试可能尚未正常工作。

#### 在测试中检查日志内容

Freqtrade 使用 2 种主要方法来检查测试中的日志内容：`log_has()` 和 `log_has_re()`（用于使用正则检查，在日志消息是动态的情况下）。这些可从 `conftest.py` 获取，并可在任何测试模块中导入。

一个示例检查如下所示：

``` python
from tests.conftest import log_has, log_has_re

def test_method_to_test(caplog):
    method_to_test()

    assert log_has("This event happened", caplog)
    # 用正则检查带尾随数字的消息 ...
    assert log_has_re(r"This dynamic event happened and produced \d+", caplog)

```

### 调试配置

要调试 freqtrade，我们推荐使用 VSCode（带 Python 扩展）以及以下启动配置（位于 `.vscode/launch.json`）。
细节显然会因搭建方式而异——但这应该能让你上手。

``` json
{
    "name": "freqtrade trade",
    "type": "debugpy",
    "request": "launch",
    "module": "freqtrade",
    "console": "integratedTerminal",
    "args": [
        "trade",
        // 可选：
        // "--userdir", "user_data",
        "--strategy", 
        "MyAwesomeStrategy",
    ]
},
```

命令行参数可以添加在 `"args"` 数组中。
此方法也可用于调试策略，方法是在策略内设置断点。

对于 Pycharm 也可以采取类似的设置——使用 `freqtrade` 作为模块名，并将命令行参数设置为 "parameters"。

??? Tip "正确使用 venv"
    当使用虚拟环境时（你应该这么做），请确保你的编辑器使用的是正确的虚拟环境，以避免问题或 "unknown import" 错误。

    #### Vscode

    你可以在 VSCode 中用 "Python: Select Interpreter" 命令选择正确的环境——它会显示扩展检测到的环境。
    如果你的环境未被检测到，你也可以手动选择路径。

    #### Pycharm

    在 pycharm 中，你可以在 "Run/Debug Configurations" 窗口中选择适当的环境。
    ![Pycharm 调试配置](assets/pycharm_debug.png)

!!! Note "启动目录"
    这假设你已经检出了仓库，并且编辑器是在仓库根目录级别启动的（所以 pyproject.toml 位于仓库的最顶层）。

## 错误处理（ErrorHandling）

Freqtrade 的异常全部继承自 `FreqtradeException`。
然而，这个通用的错误类不应被直接使用。相反，存在多个专门化的子异常。

下面是异常继承层级的概述：

```
+ FreqtradeException
|
+---+ OperationalException
|   |
|   +---+ ConfigurationError
|
+---+ DependencyException
|   |
|   +---+ PricingError
|   |
|   +---+ ExchangeError
|       |
|       +---+ TemporaryError
|       |
|       +---+ DDosProtection
|       |
|       +---+ InvalidOrderException
|           |
|           +---+ RetryableOrderError
|           |
|           +---+ InsufficientFundsError
|
+---+ StrategyError
```

---

## 插件（Plugins）

### 交易对列表（Pairlists）

你有一个想要尝试的新交易对选择算法的好点子？太好了。
希望你也会想把这个贡献回上游。

无论你的动机是什么——这应该能让你开始尝试开发一个新的 Pairlist Handler。

首先，看看 [VolumePairList](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/plugins/pairlist/VolumePairList.py) Handler，并最好复制这个文件，将其命名为你的新 Pairlist Handler 的名称。

这是一个简单的 Handler，但作为如何开始开发的一个好例子。

接下来，修改 Handler 的类名（最好与模块文件名对齐）。

基类提供了交易所的一个实例（`self._exchange`）、pairlist 管理器（`self._pairlistmanager`），以及主配置（`self._config`）、pairlist 专用配置（`self._pairlistconfig`）和在整个 pairlist 列表中的绝对位置。

```python
        self._exchange = exchange
        self._pairlistmanager = pairlistmanager
        self._config = config
        self._pairlistconfig = pairlistconfig
        self._pairlist_pos = pairlist_pos
```

!!! Tip
    别忘了在 `constants.py` 下的 `AVAILABLE_PAIRLISTS` 变量中注册你的 pairlist——否则它将无法被选择。

现在，让我们逐步过一遍需要操作的方法：

#### Pairlist 配置

Pairlist Handler 链的配置是在机器人配置文件的 `"pairlists"` 元素中完成的，它是一个包含链中每个 Pairlist Handler 的配置参数的数组。

按照惯例，`"number_assets"` 用于指定要保留在 pairlist 中的最大交易对数量。请遵循这一点以确保一致的用户体验。

可以根据需要配置额外的参数。例如，`VolumePairList` 使用 `"sort_key"` 来指定排序值——但是请随意指定你的伟大算法成功和动态所需的任何内容。

#### short_desc

返回用于 Telegram 消息的描述。

这应该包含 Pairlist Handler 的名称，以及包含交易对数量的简短描述。请遵循 `"PairlistName - top/bottom X pairs"` 格式。

#### gen_pairlist

如果你的 Pairlist Handler 可以用作链中的主导 Pairlist Handler，定义初始 pairlist（然后由链中的所有 Pairlist Handler 处理），则覆盖此方法。例子有 `StaticPairList` 和 `VolumePairList`。

机器人每次迭代都会调用它（仅当 Pairlist Handler 位于第一个位置时）——因此请考虑为计算/网络密集型计算实现缓存。

它必须返回结果 pairlist（然后可能会传入 Pairlist Handler 链）。

验证是可选的，父类暴露了一个 `verify_blacklist(pairlist)` 和 `_whitelist_for_active_markets(pairlist)` 来做默认过滤。如果你将结果限制为一定数量的交易对，请使用它——这样最终结果不会比预期的短。

#### filter_pairlist

pairlist 管理器会为链中的每个 Pairlist Handler 调用此方法。

机器人每次迭代都会调用它——因此请考虑为计算/网络密集型计算实现缓存。

它会传入一个 pairlist（可以是先前 pairlist 的结果）以及 `tickers`，即 `get_tickers()` 的预取版本。

基类中的默认实现只是对 pairlist 中的每个交易对调用 `_validate_pair()` 方法，但你可以覆盖它。因此，你应该要么在你的 Pairlist Handler 中实现 `_validate_pair()`，要么覆盖 `filter_pairlist()` 来做其他事情。

如果被覆盖，它必须返回结果 pairlist（然后可能会传入链中的下一个 Pairlist Handler）。

验证是可选的，父类暴露了一个 `verify_blacklist(pairlist)` 和 `_whitelist_for_active_markets(pairlist)` 来做默认过滤。如果你将结果限制为一定数量的交易对，请使用它——这样最终结果不会比预期的短。

在 `VolumePairList` 中，这实现了不同的排序方法，进行早期验证，以便只返回预期数量的交易对。

##### 示例

``` python
    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> List[str]:
        # 生成动态白名单
        pairs = self._calculate_pairlist(pairlist, tickers)
        return pairs
```

### 保护措施（Protections）

最好阅读 [Protection 文档](plugins.md#protections) 来理解 protections。
本指南面向想要开发新 protection 的开发者。

任何 protection 都不应直接使用 datetime，而应使用提供的 `date_now` 变量进行日期计算。这保留了回测 protections 的能力。

!!! Tip "编写一个新的 Protection"
    最好复制现有某个 Protection 作为一个好例子。

#### 新 protection 的实现

所有 Protection 实现都必须以 `IProtection` 作为父类。
因此，它们必须实现以下方法：

* `short_desc()`
* `global_stop()`
* `stop_per_pair()`

`global_stop()` 和 `stop_per_pair()` 必须返回一个 ProtectionReturn 对象，它包含：

* lock pair - 布尔值
* lock until - 日期时间 - 该交易对应被锁定到何时（会向上取整到下一根新蜡烛）
* reason - 字符串，用于日志记录和存储在数据库中
* lock_side - long、short 或 '*'。

`until` 部分应使用提供的 `calculate_lock_end()` 方法计算。

所有 Protections 应使用 `"stop_duration"` / `"stop_duration_candles"` 来定义一对（或所有交易对）应被锁定多长时间。
此内容作为 `self._stop_duration` 提供给每个 Protection。

如果你的 protection 需要一个回溯（look-back）期，请使用 `"lookback_period"` / `"lockback_period_candles"` 来保持所有 protections 对齐。

#### 全局与本地停止

Protections 可以有 2 种不同的方式来在有限时间内停止交易：

* 每对（本地）
* 对所有交易对（全局）

##### Protections - 每对

实现每对方法的 Protections 必须设置 `has_local_stop=True`。
每次交易平仓（离场订单完成）时都会调用 `stop_per_pair()` 方法。

##### Protections - 全局保护

这些 Protections 应该跨所有交易对进行评估，因此也会锁定所有交易对进行交易（称为全局 PairLock）。
全局保护必须设置 `has_global_stop=True` 才能进行全局停止的评估。
每次交易平仓（离场订单完成）时都会调用 `global_stop()` 方法。

##### Protections - 计算锁定结束时间

Protections 应该根据它考虑的最后一笔交易来计算锁定结束时间。
这避免了在回溯期长于实际锁定周期时重新锁定。

`IProtection` 父类在 `calculate_lock_end()` 中为此提供了一个辅助方法。

---

## 实现新的交易所（WIP，进行中）

!!! Note
    本节是一项进行中的工作，并不是关于如何用 Freqtrade 测试新交易所的完整指南。

!!! Note
    在运行以下任何测试之前，请确保使用最新版本的 CCXT。
    你可以在激活虚拟环境后运行 `pip install -U ccxt` 来获取最新版本的 ccxt。
    这些测试不支持原生 docker，但可用的 dev-container 将支持所有必需的操作以及最终必要的更改。

大多数 CCXT 支持的交易所应该开箱即用。

如果你需要实现特定的交易所类，这些类位于 `freqtrade/exchange` 源代码文件夹中。你还需要将导入添加到 `freqtrade/exchange/__init__.py`，以使加载逻辑感知到新的交易所。
我们建议查看现有的交易所实现，以了解可能需要什么。

!!! Warning
    实现和测试一个交易所可能有很多试错，所以请记住这一点。
    你也应该有一些开发经验，因为这不是一个初学者的任务。

要快速测试一个交易所的公共端点，请将你交易所的配置添加到 `tests/exchange_online/conftest.py`，并使用 `pytest --longrun tests/exchange_online/test_ccxt_compat.py` 运行这些测试。
成功完成这些测试是一个良好的基础点（实际上它是一个要求），但是这些并不能保证交易所功能正确，因为这只测试公共端点，而不测试私有端点（如生成订单或类似操作）。

还要尝试使用 `freqtrade download-data` 下载一个扩展的时间范围（多个月），并验证下载的数据是否正确（没有空洞，指定的时间范围确实被下载了）。

这些是让一个交易所被列为受支持或社区已测试（列在首页）的先决条件。
以下是"额外项"，它们会让一个交易所更好（功能完整）——但对于这两个分类都不是绝对必要的。

需要完成的额外测试 / 步骤：

* 验证 `fetch_ohlcv()` 提供的数据——并可能为该交易所调整 `ohlcv_candle_limit`
* 检查 L2 订单簿限制范围（API 文档）——并视需要设置
* 检查余额是否显示正确（*）
* 创建市价单（*）
* 创建限价单（*）
* 取消订单（*）
* 完成交易（入场 + 离场）（*）
  * 比较交易所与机器人之间的结果计算
  * 确保手续费被正确应用（对照交易所核对数据库）

（*）需要交易所的 API key 和余额。

### 交易所端止损（Stoploss On Exchange）

检查新交易所是否通过其 API 支持交易所端止损订单。

由于 CCXT 尚未为交易所端止损提供统一实现，我们需要自己实现交易所特定的参数。最好查看 `binance.py` 作为一个示例实现。你需要深入研究该交易所 API 的文档，了解具体如何做到这一点。[CCXT Issues](https://github.com/ccxt/ccxt/issues) 也可能提供很大帮助，因为其他人可能已经为他们的项目实现了类似的功能。

### 不完整的蜡烛（Incomplete candles）

在获取蜡烛（OHLCV）数据时，我们可能会最终获取到不完整的蜡烛（取决于交易所）。
为了演示这一点，我们将使用日线蜡烛（`"1d"`）以简化问题。
我们查询 api（`ct.fetch_ohlcv()`）获取时间周期，并查看最后一个条目的日期。如果这个条目发生变化或显示了一个"不完整"蜡烛的日期，那么我们应该丢弃它，因为拥有不完整的蜡烛是有问题的，因为指标假定只有完整的蜡烛被传递给它们，并且会产生大量虚假的买入信号。因此，默认情况下我们会移除最后一根蜡烛，假设它是不完整的。

要检查新交易所的行为，你可以使用以下代码片段：

``` python
import ccxt
from datetime import datetime, timezone
from freqtrade.data.converter import ohlcv_to_dataframe
ct = ccxt.binance()  # 使用你正在测试的交易所
timeframe = "1d"
pair = "BTC/USDT"  # 确保使用在该交易所存在的交易对！
raw = ct.fetch_ohlcv(pair, timeframe=timeframe)

# 转换为 dataframe
df1 = ohlcv_to_dataframe(raw, timeframe, pair=pair, drop_incomplete=False)

print(df1.tail(1))
print(datetime.now(timezone.utc))
```

``` output
                         date      open      high       low     close  volume  
499 2019-06-08 00:00:00+00:00  0.000007  0.000007  0.000007  0.000007   26264344.0  
2019-06-09 12:30:27.873327
```

输出将显示交易所的最后一个条目以及当前的 UTC 日期。
如果日期显示的是同一天，那么可以假设最后一根蜡烛是不完整的，应该被丢弃（保持交易所类中的 `"ohlcv_partial_candle"` 设置不变 / True）。否则，将 `"ohlcv_partial_candle"` 设置为 `False` 以不丢弃蜡烛（如上面的例子所示）。
另一种方法是连续多次运行此命令，并观察成交量是否在变化（而日期保持不变）。

### 更新 binance 缓存的杠杆层级（leverage tiers）

更新杠杆层级应该定期进行——并且需要一个启用了合约的已认证账户。

``` python
import ccxt
import json
from pathlib import Path

exchange = ccxt.binance({
    'apiKey': '<apikey>',
    'secret': '<secret>',
    'options': {'defaultType': 'swap'}
    })
_ = exchange.load_markets()

lev_tiers = exchange.fetch_leverage_tiers()

# 假设此脚本运行在仓库的根目录。
file = Path('freqtrade/exchange/binance_leverage_tiers.json')
json.dump(dict(sorted(lev_tiers.items())), file.open('w'), indent=2)

```

然后这个文件应该被贡献到上游，这样其他人也可以从中受益。

## 更新示例 notebook

为了保持 jupyter notebook 与文档一致，在更新一个示例 notebook 后应运行以下内容。

``` bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace freqtrade/templates/strategy_analysis_example.ipynb
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to markdown freqtrade/templates/strategy_analysis_example.ipynb --stdout > docs/strategy_analysis_example.md
```

## 回测文档结果

要生成回测输出，请使用以下命令：

``` bash
# 假设为此输出使用一个专用的用户目录
freqtrade create-userdir --userdir user_data_bttest/
# 设置 can_short = True
sed -i "s/can_short: bool = False/can_short: bool = True/" user_data_bttest/strategies/sample_strategy.py

freqtrade download-data --timerange 20250625-20250801 --config tests/testdata/config.tests.usdt.json --userdir user_data_bttest/ -t 5m

freqtrade backtesting --config tests/testdata/config.tests.usdt.json -s SampleStrategy --userdir user_data_bttest/ --cache none --timerange 20250701-20250801
```

## 持续集成（Continuous integration）

这记录了为 CI 流水线所做的一些决策。

* CI 运行在所有 OS 变体上，Linux（ubuntu）、macOS 和 Windows。
* Docker 镜像为 `stable` 和 `develop` 分支构建，并作为多架构（multiarch）构建，通过同一标签支持多个平台。
* 包含 Plot 依赖的 Docker 镜像也可作为 `stable_plot` 和 `develop_plot` 使用。
* Docker 镜像包含一个文件 `/freqtrade/freqtrade_commit`，其中包含该镜像所基于的提交。
* 完整的 Docker 镜像重建每周通过计划任务运行一次。
* 部署运行在 ubuntu 上。
* 所有测试必须通过，PR 才能被合并到 `stable` 或 `develop`。

## 创建发布（release）

文档的这一部分面向维护者，展示了如何创建一个发布。

### 创建发布分支

!!! Note
    确保 `stable` 分支是最新的！

首先，选择一个大约一周前（大约一周旧的）的提交（以不包含发布的最新添加内容）。

``` bash
# 创建新分支
git checkout -b new_release <commitid>
```

确定这个提交和当前状态之间是否做了关键的 bug 修复，并最终 cherry-pick 这些修复。

* 将该发布分支（stable）合并到这个分支。
* 编辑 `freqtrade/__init__.py` 并添加与当前日期匹配的版本（例如 2025 年 7 月为 `2025.7`）。如果需要那个月做第二次发布，小版本可以是 `2025.7.1`。版本号必须遵循 PEP0440 允许的版本，以避免推送到 pypi 时失败。
* 提交这部分。
* 将该分支推送到远程，并创建一个针对 **stable 分支** 的 PR。
* 将 develop 版本更新为遵循 `2025.8-dev` 模式的下一个版本。

### 从 git 提交创建 changelog

``` bash
# 需要在合并 / 拉取该分支之前完成。
git log --oneline --no-decorate --no-merges stable..new_release
```

为了使发布日志保持简短，最好将完整的 git changelog 包裹在一个可折叠的 details 段中。

```markdown
<details>
<summary>展开完整 changelog</summary>

... 完整的 git changelog

</details>
```

### FreqUI 发布

如果 FreqUI 已经实质性更新，请确保在合并发布分支之前创建一个发布。
确保在合并发布之前，发布上的 freqUI CI 已完成并通过。

### 创建 github 发布 / 标签

一旦针对 stable 的 PR 被合并（最好在合并后立即）：

* 在 Github UI（releases 子节）中使用 "Draft a new release" 按钮。
* 使用指定的版本号作为标签。
* 使用 "stable" 作为参考（此步骤在上述 PR 合并之后进行）。
* 使用上述 changelog 作为发布评论（作为代码块）。
* 使用下面的片段作为新发布

??? Tip "发布模板"
    ````
    --8<-- "includes/release_template.md"
    ````

## 发布

### pypi

!!! Warning "手动发布"
    此过程是作为 Github Actions 的一部分自动化的。
    手动推送 pypi 不应是必要的。

??? example "手动发布"
    要手动创建 pypi 发布，请运行以下命令：

    附加要求：`wheel`、`twine`（用于上传）、在 pypi 上拥有适当权限的账户。

    ``` bash
    pip install -U build
    python -m build --sdist --wheel

    # 对于 pypi 测试（检查安装是否有一些更改能正常工作）
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

    # 对于生产环境：
    twine upload dist/*
    ```

    请不要将非发布版本推送到生产 / 真实的 pypi 实例。
