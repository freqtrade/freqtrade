<!-- 本文件为中文翻译版，由 AI 根据 docs/freq-ui.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# FreqUI

Freqtrade 提供了一个内置的 Web 服务器，可以为 [FreqUI](https://github.com/freqtrade/frequi)（freqtrade 的前端界面）提供服务。

默认情况下，UI 会在安装过程中自动安装（通过安装脚本或 Docker）。
freqUI 也可以通过 `freqtrade install-ui` 命令手动安装。
同一命令也可用于将 freqUI 更新到新版本。

当机器人以交易模式或模拟交易模式启动后（使用 `freqtrade trade`），UI 将在配置的 API 端口上可用（默认为 `http://127.0.0.1:8080`）。

??? Note "Looking to contribute to freqUI?"
    开发者不应使用此方法，而应克隆相应的仓库并使用 [freqUI 仓库](https://github.com/freqtrade/frequi) 中描述的方法获取 freqUI 的源代码。构建前端需要已安装可用的 Node.js 环境。

!!! tip "freqUI is not required to run freqtrade"
    freqUI 是 freqtrade 的可选组件，运行机器人并不需要它。
    它是一个可用于监控机器人并与之交互的前端界面——但即使没有它，freqtrade 本身也能正常运行。

## 配置

FreqUI 没有自己的配置文件，但需要 [REST API](rest-api.md) 已正确配置。
请参阅相应的文档页面来完成 freqUI 的设置。

## 界面

FreqUI 是一个现代化的响应式 Web 应用程序，可用于监控和操作您的机器人。

FreqUI 提供亮色和暗色两种主题。
可以通过页面顶部醒目的按钮轻松切换主题。
本页面上的截图主题会随文档主题的选择而变化，因此如需查看暗色（或亮色）版本，请切换文档的主题。

### 登录

以下截图显示了 freqUI 的登录界面。

![FreqUI - login](assets/frequi-login-CORS.png#only-dark)
![FreqUI - login](assets/frequi-login-CORS-light.png#only-light)

!!! Hint "CORS"
    此截图中显示的 CORS 错误是由于 UI 运行在与 API 不同的端口上，且 [CORS](#cors) 尚未正确配置。

### 交易视图

交易视图允许您可视化机器人正在进行的交易并与机器人进行交互。
在此页面上，您还可以通过启动和停止机器人来与之交互，并且——如果已配置——还可以强制执行交易入场和出场。

![FreqUI - trade view](assets/freqUI-trade-pane-dark.png#only-dark)
![FreqUI - trade view](assets/freqUI-trade-pane-light.png#only-light)

### 仪表盘

仪表盘视图提供了机器人性能和状态的概览。
如果连接了多个机器人，仪表盘将显示所有已连接机器人的概览，使您可以轻松地在它们之间切换或只显示部分可用的机器人。

#### 钱包余额

freqtrade 2026.4 新增功能：显示机器人余额随时间的变化。

与"累计利润"图表相比，此图表将显示机器人随时间变化的实际余额，包括未实现的盈亏以及存款和取款。

历史数据已根据可用的交易所数据重新填充——但请注意，这属于尽力而为，可能不是 100% 准确。
更具体地说，它不会涵盖存款和取款，并且会假设起始余额为当前余额减去盈亏。

为了清晰起见，图表上会显示一条"开始记录"标记线，表示迁移到新的钱包余额跟踪系统的时间点。
只有在此时间点之后，钱包余额才被认为是准确的。

### 绘图配置器

FreqUI 的绘图配置可以通过策略中的 `plot_config` 配置对象来设置（可通过"从策略加载"按钮加载），也可以通过 UI 界面进行配置。
可以创建多个绘图配置并随意切换——为您的图表提供灵活多样的视图。

可以通过交易视图右上角的"绘图配置器"（齿轮图标）按钮来访问绘图配置。

![FreqUI - plot configuration](assets/freqUI-plot-configurator-dark.png#only-dark)
![FreqUI - plot configuration](assets/freqUI-plot-configurator-light.png#only-light)

### 设置

可以通过访问设置页面来更改多项与 UI 相关的设置。

可更改的设置包括（但不限于）：

* UI 的时区
* 在浏览器标签页的 favicon 中显示当前未平仓交易
* K 线颜色（涨/跌 -> 红/绿）
* 启用/禁用应用内通知类型

![FreqUI - Settings view](assets/frequi-settings-dark.png#only-dark)
![FreqUI - Settings view](assets/frequi-settings-light.png#only-light)

## Web 服务器模式

当 freqtrade 以 [Web 服务器模式](utils.md#webserver-mode) 启动时（使用 `freqtrade webserver` 命令），Web 服务器将以特殊模式运行，允许使用额外的功能，例如：

* 下载数据
* 测试交易对列表
* [回测策略](#回测)
* ...更多功能将陆续添加

### 回测

当 freqtrade 以 [Web 服务器模式](utils.md#webserver-mode) 启动时（使用 `freqtrade webserver` 命令），回测视图将变为可用。
此视图允许您对策略进行回测并可视化结果。

您还可以加载和可视化之前的回测结果，以及对结果进行比较。

![FreqUI - Backtesting](assets/freqUI-backtesting-dark.png#only-dark)
![FreqUI - Backtesting](assets/freqUI-backtesting-light.png#only-light)


--8<-- "../includes/cors.md"