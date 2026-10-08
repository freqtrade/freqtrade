<!-- 本文件为中文翻译版，由 AI 根据 docs/freqai.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

![freqai-logo](../assets/freqai_doc_logo.svg)

# FreqAI

## 简介

FreqAI 是一款自动化软件，旨在完成与训练预测性机器学习模型相关的各种任务，根据一组输入信号生成市场预测。总体而言，FreqAI 致力于成为一个方便在实时数据上部署强大机器学习库的沙盒（[详情](#freqai-在开源机器学习领域中的定位)）。

!!! Note
    FreqAI 现在是、将来也永远是一个非营利的开源项目。FreqAI *没有*加密代币，FreqAI *不*出售信号，FreqAI 除了当前的 [Freqtrade 文档](https://www.freqtrade.io/en/stable/freqai/) 外没有其他域名。

主要功能包括：

* **自适应再训练** - 在[实时部署](freqai-running.md#live-deployments)期间对模型进行再训练，以监督学习的方式自适应市场变化
* **快速特征工程** - 基于用户自定义策略创建大型丰富的[特征集](freqai-feature-engineering.md#feature-engineering)（10k+ 特征）
* **高性能** - 多线程允许在独立线程（或 GPU，如果可用）上进行自适应模型再训练，与模型推理（预测）和交易操作并行执行。最新的模型和数据保存在内存中，以实现快速推理
* **逼真的回测** - 使用[回测模块](freqai-running.md#backtesting)模拟自适应训练，自动化再训练过程，在历史数据上回测策略
* **可扩展性** - 通用且健壮的架构允许集成 Python 中可用的任何[机器学习库/方法](freqai-configuration.md#using-different-prediction-models)。目前提供八种示例，包括分类器、回归器和卷积神经网络
* **智能异常值移除** - 使用多种[异常检测技术](freqai-feature-engineering.md#outlier-detection)从训练和预测数据集中移除异常值
* **崩溃恢复能力** - 将训练好的模型存储到磁盘，使崩溃后的重新加载快速便捷，并[清除过时文件](freqai-running.md#purging-old-model-data)以维持模拟/实时运行
* **自动数据归一化** - 以智能且统计安全的方式[归一化数据](freqai-feature-engineering.md#building-the-data-pipeline)
* **自动数据下载** - 计算数据下载的时间范围并更新历史数据（在实时部署中）
* **传入数据清洗** - 在训练和模型推理前安全处理 NaN 值
* **降维** - 通过[主成分分析](freqai-feature-engineering.md#data-dimensionality-reduction-with-principal-component-analysis)减少训练数据的规模
* **部署机器人集群** - 设置一个机器人负责训练模型，同时一组[消费者](producer-consumer.md)使用信号进行交易

## 快速开始

测试 FreqAI 最简单的方法是在模拟模式下运行以下命令：

```bash
freqtrade trade --config config_examples/config_freqai.example.json --strategy FreqaiExampleStrategy --freqaimodel LightGBMRegressor --strategy-path freqtrade/templates
```

你将看到自动数据下载的启动过程，随后是同步训练和交易。

!!! danger "不适用于生产环境"
    Freqtrade 源代码中提供的示例策略旨在展示/测试 FreqAI 的各种功能。它还被设计为在小型计算机上运行，以便作为开发者和用户之间的基准。它*不*适合在生产环境中运行。

作为起点的示例策略、预测模型和配置文件分别位于
`freqtrade/templates/FreqaiExampleStrategy.py`、`freqtrade/freqai/prediction_models/LightGBMRegressor.py` 和
`config_examples/config_freqai.example.json`。

## 总体方法

你向 FreqAI 提供一组自定义*基础指标*（与[典型的 Freqtrade 策略](strategy-customization.md)中的方式相同）以及目标值（*标签*）。对于白名单中的每个交易对，FreqAI 训练一个模型，根据自定义指标的输入来预测目标值。然后以预定频率持续对模型进行再训练，以适应市场条件。FreqAI 既提供了回测策略的能力（在历史数据上通过定期再训练来模拟真实情况），也支持模拟/实时运行。在模拟/实时条件下，FreqAI 可以设置为在后台线程中持续再训练，以使模型尽可能保持最新。

下面展示了算法的概览，解释了数据处理管道和模型使用方式。

![freqai-algo](../assets/freqai_algo.jpg)

### 重要的机器学习术语

**特征（Features）** - 基于历史数据的参数，模型在这些参数上进行训练。单根 K 线的所有特征存储为一个向量。在 FreqAI 中，你可以从策略中能构造的任何内容构建特征数据集。

**标签（Labels）** - 模型训练所针对的目标值。每个特征向量都与你在策略中定义的单个标签相关联。这些标签有意地展望未来，是你训练模型要能够预测的内容。

**训练（Training）** - "教导"模型将特征集与相关标签匹配的过程。不同类型的模型以不同的方式"学习"，这意味着在特定应用中某种模型可能比另一种更好。关于 FreqAI 中已实现的不同模型的更多信息，请参见[此处](freqai-configuration.md#using-different-prediction-models)。

**训练数据（Train data）** - 特征数据集的一个子集，在训练期间提供给模型以"教导"模型如何预测目标。这些数据直接影响模型中的权重连接。

**测试数据（Test data）** - 特征数据集的一个子集，用于在训练后评估模型的性能。这些数据不会影响模型中的节点权重。

**推理（Inferencing）** - 将训练好的模型应用于新的未见数据并进行预测的过程。

## 安装前提条件

正常的 Freqtrade 安装过程会询问你是否要安装 FreqAI 依赖项。如果你想使用 FreqAI，应该回答"是"。如果你当初没有选择"是"，可以在安装后手动安装这些依赖项：

``` bash
pip install -r requirements-freqai.txt
```

!!! Note
    Catboost 不会在低功耗 ARM 设备（如树莓派）上安装，因为该平台没有提供预编译的 wheel 包。

### Docker 使用

如果你使用 Docker，有一个包含 FreqAI 依赖项的专用标签 `:freqai`。因此 - 你可以在 docker compose 文件中将镜像行替换为 `image: freqtradeorg/freqtrade:stable_freqai`。此镜像包含常规的 FreqAI 依赖项。与本地安装类似，Catboost 在基于 ARM 的设备上不可用。如果你想使用 PyTorch 或强化学习，应使用 torch 或 RL 标签，即 `image: freqtradeorg/freqtrade:stable_freqaitorch`、`image: freqtradeorg/freqtrade:stable_freqairl`。

!!! note "docker-compose-freqai.yml"
    我们在 `docker/docker-compose-freqai.yml` 中提供了一个专门的 docker-compose 文件 - 可以通过 `docker compose -f docker/docker-compose-freqai.yml run ...` 使用 - 也可以复制替换原始的 docker 文件。此 docker-compose 文件还包含一个（已禁用的）部分，用于在 Docker 容器中启用 GPU 资源。这当然假设系统有可用的 GPU 资源。

### FreqAI 在开源机器学习领域中的定位

预测基于混沌时间序列的系统（如股票/加密货币市场），需要一套广泛的工具来测试各种假设。幸运的是，近年来成熟的强大机器学习库（如 `scikit-learn`）开辟了广阔的研究可能性。来自不同领域的科学家现在可以轻松地在众多成熟的机器学习算法上原型化他们的研究。同样，这些用户友好的库使得"公民科学家"能够利用基本的 Python 技能进行数据探索。然而，在历史和实时混沌数据源上利用这些机器学习库在逻辑上可能既困难又昂贵。此外，稳健的数据收集、存储和处理也是一项截然不同的挑战。[`FreqAI`](#freqai) 旨在提供一个通用且可扩展的开源框架，面向市场预测的自适应建模的实时部署。`FreqAI` 框架实际上是开源机器学习库丰富世界的沙盒。在 `FreqAI` 沙盒中，用户可以组合各种第三方库，在免费的 24/7 实时混沌数据源 - 加密货币交易所数据 - 上测试创造性的假设。

### 引用 FreqAI

FreqAI 已[发表在 Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.04864)。如果你觉得 FreqAI 对你的研究有帮助，请使用以下引用：

```bibtex
@article{Caulk2022, 
    doi = {10.21105/joss.04864},
    url = {https://doi.org/10.21105/joss.04864},
    year = {2022}, publisher = {The Open Journal},
    volume = {7}, number = {80}, pages = {4864},
    author = {Robert A. Caulk and Elin Törnquist and Matthias Voppichler and Andrew R. Lawless and Ryan McMullan and Wagner Costa Santos and Timothy C. Pogue and Johan van der Vlugt and Stefan P. Gehring and Pascal Schmidt},
    title = {FreqAI: generalizing adaptive modeling for chaotic time-series market forecasts},
    journal = {Journal of Open Source Software} } 
```

## 常见陷阱

FreqAI 不能与动态 `VolumePairlists`（或任何动态添加和移除交易对的 pairlist 过滤器）组合使用。
这是出于性能考虑 - FreqAI 依赖于快速进行预测/再训练。为了有效地做到这一点，
它需要在模拟/实时实例开始时下载所有训练数据。FreqAI 会自动存储和追加
新的 K 线数据以供未来的再训练使用。这意味着如果由于 volume pairlist 导致新交易对在模拟运行后期才出现，FreqAI 将无法准备好数据。不过，FreqAI 可以与 `ShufflePairlist` 或保持总交易对列表不变（但根据成交量重新排序）的 `VolumePairlist` 配合使用。

## 补充学习资料

这里我们汇编了一些外部资料，深入介绍了 FreqAI 的各个组件：

- [Real-time head-to-head: Adaptive modeling of financial market data using XGBoost and CatBoost](https://emergentmethods.medium.com/real-time-head-to-head-adaptive-modeling-of-financial-market-data-using-xgboost-and-catboost-995a115a7495)
- [FreqAI - from price to prediction](https://emergentmethods.medium.com/freqai-from-price-to-prediction-6fadac18b665)


## 支持

你可以在多个地方找到 FreqAI 的支持，包括 [Freqtrade Discord](https://discord.gg/Jd8JYeWHc4)、专门的 [FreqAI Discord](https://discord.gg/7AMWACmbjT) 以及 [GitHub Issues](https://github.com/freqtrade/freqtrade/issues)。

## 致谢

FreqAI 由一群各自为项目贡献特定技能的个人共同开发。

构思和软件开发：
Robert Caulk @robcaulk

理论研讨和数据分析：
Elin Törnquist @th0rntwig

代码审查和软件架构研讨：
@xmatthias

软件开发：
Wagner Costa @wagnercosta
Emre Suzen @aemr3
Timothy Pogue @wizrds

Beta 测试和错误报告：
Stefan Gehring @bloodhunter4rc, @longyu, Andrew Lawless @paranoidandy, Pascal Schmidt @smidelis, Ryan McMullan @smarmau, Juha Nykänen @suikula, Johan van der Vlugt @jooopiert, Richárd Józsa @richardjosza