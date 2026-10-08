<!-- 本文件为中文翻译版，由 AI 根据 docs/index.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件（如 exchanges.md），这些文件将在逐步翻译过程中补齐。 -->
<!-- 图片与 includes 引用使用 ../ 指向英文原文档资源，以保证显示正常。 -->

![freqtrade](../assets/freqtrade_poweredby.svg)

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![codecov](https://codecov.io/gh/freqtrade/freqtrade/branch/develop/graph/badge.svg?token=AD5BG3ATKI)](https://codecov.io/gh/freqtrade/freqtrade)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Discord Server](https://img.shields.io/badge/Freqtrade_Discord-4E4E4E?logo=discord)](https://discord.gg/p7nuUNVfP7)

<!-- GitHub action buttons -->
[:octicons-star-16: Star](https://github.com/freqtrade/freqtrade){ .md-button .md-button--sm }
[:octicons-repo-forked-16: Fork](https://github.com/freqtrade/freqtrade/fork){ .md-button .md-button--sm }
[:octicons-download-16: Download](https://github.com/freqtrade/freqtrade/archive/stable.zip){ .md-button .md-button--sm }

## 简介

Freqtrade 是一个免费、开源的加密货币交易机器人，使用 Python 编写。它设计为支持各大主流交易所，并可通过 Telegram 或 WebUI 进行控制。它包含回测、绘图和资金管理等工具，还支持通过机器学习进行策略优化。

!!! Danger "免责声明"
    本软件仅供学习用途。请勿投入你无法承受损失的资金。使用本软件风险自负。作者及所有关联方对你交易结果不承担任何责任。

    请始终先在 Dry-run（模拟运行）模式下运行交易机器人，在了解其工作原理以及你应预期的盈亏之前，不要投入真实资金。

    我们强烈建议你具备基本的编程能力和 Python 知识。请务必阅读源代码，理解该机器人实现的机制、算法和技术。

![freqtrade 截图](../assets/freqtrade-screenshot.png)

## 功能特性

- **开发你的策略**：使用 [pandas](https://pandas.pydata.org/) 以 Python 编写你的策略。你可以在 [策略仓库](https://github.com/freqtrade/freqtrade-strategies) 中找到可供参考的示例策略。
- **下载市场数据**：下载交易所及你可能想要交易的市场的历史数据。
- **回测**：在历史下载数据上测试你的策略。
- **优化**：使用采用机器学习方法的超参数优化（hyperoptimization）为你的策略寻找最佳参数。你可以优化买入、卖出、止盈（ROI）、止损和追踪止损等参数。
- **选择市场**：创建静态列表，或基于成交量/价格排名的自动列表（回测期间不可用）。你也可以明确将不想交易的货币对加入黑名单。
- **运行**：使用模拟资金测试策略（Dry-Run 模式），或使用真实资金部署（Live-Trade 模式）。
- **控制 / 监控**：使用 Telegram 或 WebUI（启动/停止机器人、显示盈亏、每日汇总、当前未平仓交易结果等）。
- **分析**：可以对回测数据或 Freqtrade 交易历史（SQL 数据库）进行进一步分析，包括自动化标准图表，以及将数据加载到 [交互式环境](data-analysis.md) 的方法。

## 支持的交易所市场

请阅读 [交易所特定说明](exchanges.md) 以了解各交易所可能需要的特殊配置。

### 支持的现货交易所

- [X] [Binance](https://www.binance.com/)
- [X] [BingX](https://bingx.com/invite/0EM9RX)
- [X] [Bitget](https://www.bitget.com/)
- [X] [Bitmart](https://bitmart.com/)
- [X] [Bybit EU](https://bybit.eu/)
- [X] [Bybit](https://bybit.com/)
- [X] [Gate EU](https://www.gate.com/en-eu)
- [X] [Gate](https://www.gate.com/ref/6266643)
- [X] [HTX](https://www.htx.com/)
- [X] [Hyperliquid](https://hyperliquid.xyz/)（去中心化交易所，即 DEX）
- [X] [Kraken](https://kraken.com/)
- [X] [MyOKX](https://okx.com/)（OKX EEA）
- [X] [OKX](https://okx.com/)
- [ ] [可能还有更多](https://github.com/ccxt/ccxt/)。*（我们无法保证它们都能正常工作）*

### 支持的合约交易所

- [X] [Binance](https://www.binance.com/)
- [X] [Bitget](https://www.bitget.com/)
- [X] [Bybit](https://bybit.com/)
- [X] [Gate](https://www.gate.com/ref/6266643)
- [X] [Hyperliquid](https://hyperliquid.xyz/)（去中心化交易所，即 DEX）
- [X] [Kraken](https://www.kraken.com/features/futures)
- [X] [OKX](https://okx.com/)

在深入之前，请务必阅读 [交易所特定说明](exchanges.md) 以及 [杠杆交易](leverage.md) 文档。

### 社区验证

经社区确认可用的交易所：

- [X] [Bitvavo](https://bitvavo.com/)
- [X] [Kucoin](https://www.kucoin.com/)

## 社区展示

--8<-- "../includes/showcase.md"

## 环境要求

### 硬件要求

运行本机器人，我们建议使用 Linux 云服务器，最低配置：

- 2GB 内存（RAM）
- 1GB 磁盘空间
- 2 个 vCPU

### 软件要求

- Docker（推荐）

或者：

- Python 3.11+
- pip（pip3）
- git
- TA-Lib
- virtualenv（推荐）

## 支持

### 帮助 / Discord

对于文档未涵盖的任何问题，或想进一步了解本机器人，亦或只是想结识志同道合的人，我们鼓励你加入 Freqtrade 的 [Discord 服务器](https://discord.gg/p7nuUNVfP7)。

## 准备开始？

先从阅读安装指南开始：使用 [Docker 安装](docker_quickstart.md)（推荐），或不使用 Docker 的 [原生安装](installation.md)。
