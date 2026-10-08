<!-- 本文件为中文翻译版，由 AI 根据 docs/freqai-running.md 翻译。 -->
<!-- 文档内部链接指向同目录下的中文同名文件，includes 引用使用 ../ 指向英文原文档资源。 -->

# 运行 FreqAI

有两种方式可以训练和部署自适应机器学习模型——实时部署和历史回测。在这两种情况下，FreqAI 都会运行/模拟模型的定期重新训练，如下图所示：

![freqai-window](../assets/freqai_moving-window.jpg)

## 实时部署

FreqAI 可以使用以下命令进行模拟盘/实盘运行：

```bash
freqtrade trade --strategy FreqaiExampleStrategy --config config_freqai.example.json --freqaimodel LightGBMRegressor
```

启动后，FreqAI 将根据配置设置开始训练一个具有新 `identifier` 的新模型。训练完成后，该模型将用于对传入的 K 线进行预测，直到新模型可用。新模型通常会尽可能频繁地生成，FreqAI 内部管理一个交易对队列，以尽量保持所有模型的更新程度一致。FreqAI 始终使用最近训练的模型对传入的实时数据进行预测。如果您不希望 FreqAI 尽可能频繁地重新训练新模型，可以设置 `live_retrain_hours` 来告诉 FreqAI 在训练新模型之前至少等待该小时数。此外，您可以设置 `expired_hours` 来告诉 FreqAI 避免对超过该小时数的模型进行预测。

训练好的模型默认保存到磁盘，以便在回测期间或崩溃后重新使用。您可以通过在配置中设置 `"purge_old_models": true` 来选择[清除旧模型](#清除旧模型数据)以节省磁盘空间。

要从保存的回测模型（或从之前崩溃的模拟盘/实盘会话）启动模拟盘/实盘运行，您只需指定特定模型的 `identifier`：

```json
    "freqai": {
        "identifier": "example",
        "live_retrain_hours": 0.5
    }
```

在这种情况下，尽管 FreqAI 将使用预训练模型启动，但它仍会检查自模型训练以来经过了多少时间。如果自加载模型结束以来已经过了完整的 `live_retrain_hours`，FreqAI 将开始训练新模型。

### 自动数据下载

FreqAI 会自动下载所需数量的数据，以确保通过定义的 `train_period_days` 和 `startup_candle_count` 来训练模型（有关这些参数的详细说明，请参阅[参数表](freqai-parameter-table.md)）。

### 保存预测数据

在特定 `identifier` 模型的生命周期内所做的所有预测都存储在 `historic_predictions.pkl` 中，以便在崩溃或对配置进行更改后重新加载。

### 清除旧模型数据

FreqAI 在每次成功训练后都会存储新的模型文件。随着新模型的生成以适应新的市场条件，这些文件会变得过时。如果您计划长时间运行 FreqAI 并进行高频重新训练，应在配置中启用 `purge_old_models`：

```json
    "freqai": {
        "purge_old_models": 4,
    }
```

这将自动清除所有超过最近训练的四个模型的旧模型，以节省磁盘空间。输入 "0" 将永远不会清除任何模型。

## 回测

FreqAI 回测模块可以使用以下命令执行：

```bash
freqtrade backtesting --strategy FreqaiExampleStrategy --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --freqaimodel LightGBMRegressor --timerange 20210501-20210701
```

如果此命令从未使用现有配置文件执行过，FreqAI 将为扩展的 `--timerange` 内的每个回测窗口、每个交易对训练一个新模型。

回测模式需要在部署之前[下载必要的数据](#下载数据以覆盖完整的回测周期)（与 FreqAI 自动处理数据下载的模拟盘/实盘模式不同）。您应该注意，下载数据的时间范围要大于回测时间范围。这是因为 FreqAI 需要在期望的回测时间范围之前的数据，以便训练模型为在设定的回测时间范围的第一根 K 线上进行预测做好准备。有关如何计算要下载的数据的更多详细信息，请参阅[此处](#确定滑动训练窗口和回测持续时间的大小)。

!!! Note "模型重用"
    一旦训练完成，您可以使用相同的配置文件再次执行回测，
    FreqAI 将找到已训练的模型并加载它们，而不是花费时间进行训练。如果您想调整（甚至超参数优化）策略中的买入和卖出标准，这将非常有用。如果您*想要*使用相同的配置文件重新训练新模型，只需更改 `identifier` 即可。
    这样，您只需指定 `identifier` 即可返回使用您希望的任何模型。

!!! Note
    回测对每个回测窗口调用一次 `set_freqai_targets()`（窗口数量为完整回测时间范围除以 `backtest_period_days` 参数）。这样做意味着目标模拟了模拟盘/实盘行为，没有前瞻偏差。然而，`feature_engineering_*()` 中特征的定义是在整个训练时间范围上执行一次的。这意味着您应该确保特征不会前瞻未来。
    有关前瞻偏差的更多详细信息，请参阅[常见错误](strategy-customization.md#common-mistakes-when-developing-strategies)。

---

### 保存回测预测数据

为了允许调整您的策略（**不是**特征！），FreqAI 将在回测期间自动保存预测，以便它们可以在使用相同 `identifier` 模型的未来回测和实盘运行中重用。这提供了针对**高级超参数优化**入场/出场标准的性能增强。

一个名为 `backtesting_predictions` 的附加目录将在 `unique-id` 文件夹中创建，其中包含以 `feather` 格式存储的所有预测。

要更改您的**特征**，您**必须**在配置中设置新的 `identifier` 以通知 FreqAI 训练新模型。

要保存在特定回测期间生成的模型，以便您可以从其中一个模型开始实时部署而不是训练新模型，您必须在配置中将 `save_backtest_models` 设置为 `True`。

!!! Note
    为确保模型可以重用，FreqAI 将使用长度为 1 的 dataframe 调用您的策略。
    如果您的策略需要比这更多的数据来生成相同的特征，则您不能将回测预测用于实时部署，需要为每个新回测更新您的 `identifier`。

!!! Danger "安全提示"
    从磁盘加载保存的模型可能会导致安全问题，如果使用远程模型文件（您从互联网下载的文件或从不受信任的来源收到的文件），因为需要设置 `weights_only=False`，这可能会导致安全问题。
    只要您只加载自己训练的模型，就没有风险。

### 回测实盘收集的预测

FreqAI 允许您通过回测参数 `--freqai-backtest-live-models` 重用实盘历史预测。当您想重用在模拟盘/实盘中生成的预测进行比较或其他研究时，这非常有用。

`--timerange` 参数不应提供，因为它将通过历史预测文件中的数据自动计算。

### 下载数据以覆盖完整的回测周期

对于实时/模拟盘部署，FreqAI 将自动下载必要的数据。但是，要使用回测功能，您需要使用 `download-data` 下载必要的数据（详情请参阅[此处](data-download.md#data-downloading)）。您需要仔细理解需要下载多少*额外*的数据，以确保在回测时间范围开始*之前*有足够的训练数据。额外数据量可以通过将时间范围的开始日期向后移动 `train_period_days` 和 `startup_candle_count`（有关这些参数的详细说明，请参阅[参数表](freqai-parameter-table.md)）从期望的回测时间范围的开始来粗略估计。

例如，要使用[示例配置](freqai-configuration.md#setting-up-the-configuration-file)回测 `--timerange 20210501-20210701`，该配置将 `train_period_days` 设置为 30，加上 `startup_candle_count: 40`，最大 `include_timeframes` 为 1h，则下载数据的开始日期需要是 `20210501` - 30 天 - 40 * 1h / 24 小时 = 20210330（比期望训练时间范围的开始早 31.7 天）。

### 确定滑动训练窗口和回测持续时间的大小

回测时间范围使用配置文件中典型的 `--timerange` 参数定义。滑动训练窗口的持续时间由 `train_period_days` 设置，而 `backtest_period_days` 是滑动回测窗口，均以天数为单位（`backtest_period_days` 可以是浮点数，以指示在模拟盘/实盘模式下进行亚日重新训练）。在所提供的[示例配置](freqai-configuration.md#setting-up-the-configuration-file)（位于 `config_examples/config_freqai.example.json`）中，用户要求 FreqAI 使用 30 天的训练期，并在随后的 7 天内进行回测。在模型训练完成后，FreqAI 将在随后的 7 天内进行回测。然后"滑动窗口"向前移动一周（模拟 FreqAI 在实盘模式下每周重新训练一次），新模型使用前 30 天（包括前一个模型用于回测的 7 天）进行训练。重复此过程直到 `--timerange` 结束。这意味着如果您设置 `--timerange 20210501-20210701`，FreqAI 将在 `--timerange` 结束时训练 8 个独立的模型（因为完整范围包含 8 周）。

!!! Note
    虽然允许使用小数 `backtest_period_days`，但您应该注意，`--timerange` 除以该值以确定 FreqAI 需要训练多少个模型来完成完整范围的回测。例如，通过设置 10 天的 `--timerange` 和 0.1 的 `backtest_period_days`，FreqAI 将需要为每个交易对训练 100 个模型才能完成完整回测。因此，对 FreqAI 自适应训练的真正回测将需要*非常*长的时间。完全测试模型的最佳方法是运行模拟盘并让它持续训练。在这种情况下，回测所需的时间与模拟盘运行完全相同。

## 定义模型过期时间

在模拟盘/实盘模式下，FreqAI 按顺序（在与主 Freqtrade 机器人不同的线程/GPU 上）训练每个交易对。这意味着模型之间总是存在时间差异。如果您在 50 个交易对上进行训练，每个交易对需要 5 分钟来训练，那么最旧的模型将超过 4 小时。如果策略的特征时间尺度（交易持续时间目标）小于 4 小时，这可能是不可取的。您可以通过在配置文件中设置 `expiration_hours` 来决定仅在模型小时数少于特定小时数时才进行交易入场：

```json
    "freqai": {
        "expiration_hours": 0.5,
    }
```

在所提供的示例配置中，用户将仅允许对少于 1/2 小时的模型进行预测。

## 控制模型学习过程

模型训练参数对于所选的机器学习库是唯一的。FreqAI 允许您使用配置中的 `model_training_parameters` 字典为任何库设置任何参数。示例配置（位于 `config_examples/config_freqai.example.json`）显示了与 `Catboost` 和 `LightGBM` 相关的一些示例参数，但您可以添加这些库或您选择实现的任何其他机器学习库中可用的任何参数。

数据拆分参数在 `data_split_parameters` 中定义，可以是与 scikit-learn 的 `train_test_split()` 函数相关的任何参数。`train_test_split()` 有一个名为 `shuffle` 的参数，允许对数据进行打乱或保持不打乱。这对于避免在时间自相关数据上产生训练偏差特别有用。有关这些参数的更多详细信息，请访问 [scikit-learn 网站](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)（外部网站）。

FreqAI 特定参数 `label_period_candles` 定义了用于 `labels` 的偏移量（未来 K 线的数量）。在所提供的[示例配置](freqai-configuration.md#setting-up-the-configuration-file)中，用户要求 `labels` 为未来 24 根 K 线。

## 持续学习

您可以通过在配置中设置 `"continual_learning": true` 来选择采用持续学习方案。通过启用 `continual_learning`，在从头开始训练初始模型后，后续训练将从前一次训练的最终模型状态开始。这使新模型具有先前状态的"记忆"。默认情况下，此设置为 `False`，这意味着所有新模型都是从头开始训练的，没有来自先前模型的输入。

???+ danger "持续学习强制保持恒定的参数空间"
    由于 `continual_learning` 意味着模型参数空间*不能*在训练之间更改，因此当启用 `continual_learning` 时，`principal_component_analysis` 会自动禁用。提示：PCA 会更改参数空间和特征数量，请在[此处](freqai-feature-engineering.md#data-dimensionality-reduction-with-principal-component-analysis)了解有关 PCA 的更多信息。

???+ danger "实验性功能"
    请注意，这目前是一种朴素的增量学习方法，在市场偏离您的模型时，它很有可能过拟合/陷入局部最小值。我们在 FreqAI 中提供此机制主要用于实验目的，以便为加密货币市场等混沌系统中的更成熟的持续学习方法做好准备。

## 超参数优化

您可以使用与[典型 Freqtrade 超参数优化](hyperopt.md)相同的命令进行超参数优化：

```bash
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy FreqaiExampleStrategy --freqaimodel LightGBMRegressor --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --timerange 20220428-20220507
```

`hyperopt` 要求您以与进行[回测](#回测)相同的方式预先下载数据。此外，在尝试对 FreqAI 策略进行超参数优化时，您必须考虑一些限制：

- `--analyze-per-epoch` 超参数优化参数与 FreqAI 不兼容。
- 无法在 `feature_engineering_*()` 和 `set_freqai_targets()` 函数中对指标进行超参数优化。这意味着您无法使用超参数优化来优化模型参数。除了此例外，可以优化所有其他[空间](hyperopt.md#running-hyperopt-with-smaller-search-space)。
- 回测说明也适用于超参数优化。

结合超参数优化和 FreqAI 的最佳方法是专注于对入场/出场阈值/标准进行超参数优化。您需要专注于对特征中未使用的参数进行超参数优化。例如，您不应尝试对特征创建中的滚动窗口长度或更改预测的 FreqAI 配置的任何部分进行超参数优化。为了有效地对 FreqAI 策略进行超参数优化，FreqAI 将预测存储为 dataframe 并重用它们。因此要求仅对入场/出场阈值/标准进行超参数优化。

FreqAI 中可进行超参数优化的参数的一个好例子是[相异度指数 (DI)](freqai-feature-engineering.md#identifying-outliers-with-the-dissimilarity-index-di) `DI_values` 的阈值，超过该阈值我们将数据点视为异常值：

```python
di_max = IntParameter(low=1, high=20, default=10, space='buy', optimize=True, load=True)
dataframe['outlier'] = np.where(dataframe['DI_values'] > self.di_max.value/10, 1, 0)
```

此特定超参数优化将帮助您了解特定参数空间的适当 `DI_values`。

## 使用 Tensorboard

!!! note "可用性"
    FreqAI 包含适用于多种模型的 Tensorboard，包括 XGBoost、所有 PyTorch 模型、强化学习和 Catboost。如果您希望看到 Tensorboard 集成到其他模型类型中，请在 [Freqtrade GitHub](https://github.com/freqtrade/freqtrade/issues) 上提交 issue。

!!! danger "要求"
    Tensorboard 日志记录需要 FreqAI torch 安装/镜像。


使用 Tensorboard 的最简单方法是确保配置文件中的 `freqai.activate_tensorboard` 设置为 `True`（默认设置），运行 FreqAI，然后打开另一个 shell 并运行：

```bash
cd freqtrade
tensorboard --logdir user_data/models/unique-id
```

其中 `unique-id` 是在 `freqai` 配置文件中设置的 `identifier`。如果您希望在浏览器中查看输出（127.0.0.1:6060），此命令必须在单独的 shell 中运行（6060 是 Tensorboard 使用的默认端口）。

![tensorboard](../assets/tensorboard.jpg)


!!! note "停用以提高性能"
    Tensorboard 日志记录会减慢训练速度，应在生产环境中停用。