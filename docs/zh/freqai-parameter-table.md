# 参数表（Parameter table）

下表将列出 FreqAI 可用的所有配置参数。部分参数在 `config_examples/config_freqai.example.json` 中有示例。

必需参数标记为 **Required**（必需）并且必须通过建议的方式之一进行设置。

### 通用配置参数

|  参数 | 描述 |
|------------|-------------|
|  |  **`config.freqai` 树内的通用配置参数**
| `freqai` | **必需。** <br> 包含控制 FreqAI 的所有参数的父字典。 <br> **数据类型：** Dictionary（字典）。
| `train_period_days` | **必需。** <br> 用于训练数据的天数（滑动窗口的宽度）。 <br> **数据类型：** 正整数。
| `backtest_period_days` | **必需。** <br> 在滑动上面定义的 `train_period_days` 窗口并重新训练模型之前，从训练模型进行推断的天数（有关回测的更多信息[此处](freqai-running.md#backtesting)）。这可以是小数天，但要注意提供的 `timerange` 将被此数字整除，以得出完成回测所需的训练次数。 <br> **数据类型：** 浮点数。
| `identifier` | **必需。** <br> 当前模型的唯一 ID。如果模型被保存到磁盘，则 `identifier` 允许重新加载特定的预训练模型/数据。 <br> **数据类型：** 字符串。
| `live_retrain_hours` | dry/live 运行期间重新训练的频率。 <br> **数据类型：** 浮点数 > 0。 <br> 默认值：`0`（模型尽可能频繁地重新训练）。
| `expiration_hours` | 如果模型比 `expiration_hours` 更旧，则避免进行预测。 <br> **数据类型：** 正整数。 <br> 默认值：`0`（模型永不过期）。
| `purge_old_models` | 磁盘上保留的模型数量（与回测无关）。默认值为 2，这意味着 dry/live 运行将在磁盘上保留最新的 2 个模型。设置为 0 会保留所有模型。此参数也接受布尔值以保持向后兼容性。 <br> **数据类型：** 整数。 <br> 默认值：`2`。
| `save_backtest_models` | 运行回测时将模型保存到磁盘。回测通过保存预测数据并直接为后续运行（当你想要调整入场/出场参数时）重用它们来最高效地运行。将回测模型保存到磁盘还允许使用相同的模型文件以相同的模型 `identifier` 启动 dry/live 实例。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`（不保存模型）。
| `fit_live_predictions_candles` | 用于从预测数据（而不是从训练数据集）计算目标（标签）统计的历史蜡烛数量（更多信息可在此处找到[此处](freqai-configuration.md#creating-a-dynamic-target-threshold)）。 <br> **数据类型：** 正整数。
| `continual_learning` | 使用最近训练的模型的最终状态作为新模型的起点，允许增量学习（更多信息可在此处找到[此处](freqai-running.md#continual-learning)）。注意，目前这是一种天真的增量学习方法，当市场偏离你的模型时，它很可能过拟合/陷入局部最小值。我们提供这些连接主要是为了实验目的，以便为混沌系统（如加密市场）中更成熟的增量学习方法做好准备。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。
| `write_metrics_to_disk` | 在 json 文件中收集训练计时、推断计时和 cpu 使用率。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`
| `data_kitchen_thread_count` | <br> 指定要用于数据处理（离群值方法、归一化等）的线程数。这对用于训练的线程数没有影响。如果用户未设置（默认），FreqAI 将使用最大线程数 - 2（为 Freqtrade 机器人和 FreqUI 留出 1 个物理核心） <br> **数据类型：** 正整数。
| `activate_tensorboard` | <br> 指示是否为启用了 tensorboard 的模块（目前是强化学习、XGBoost、Catboost 和 PyTorch）激活 tensorboard。Tensorboard 需要安装 Torch，这意味着你需要 torch/RL docker 镜像，或者你需要在关于是否希望安装 Torch 的安装问题中回答“yes”。 <br> **数据类型：** 布尔值。 <br> 默认值：`True`。
| `wait_for_training_iteration_on_reload` | <br> 使用 /reload 或 ctrl-c 时，等待当前训练迭代完成后再完成优雅关闭。如果设置为 `False`，FreqAI 将中断当前的训练迭代，让你更快地优雅关闭，但你会丢失当前的训练迭代。 <br> **数据类型：** 布尔值。 <br> 默认值：`True`。

### 特征参数

|  参数 | 描述 |
|------------|-------------|
|  |  **`freqai.feature_parameters` 子字典内的特征参数**
| `feature_parameters` | 包含用于工程化特征集的参数的字典。详细信息和示例显示[此处](freqai-feature-engineering.md)。 <br> **数据类型：** 字典。
| `include_timeframes` | 将为 `feature_engineering_expand_*()` 中的所有指标创建的时间周期列表。该列表作为特征添加到基础指标数据集。 <br> **数据类型：** 时间周期列表（字符串）。
| `include_corr_pairlist` | FreqAI 将作为附加特征添加到所有 `pair_whitelist` 币对的相关币的列表。在特征工程期间设置的所有指标（详细信息[此处](freqai-feature-engineering.md)）将为每个相关币创建。`include_corr_pairlist` 相关币的特征被添加到基础指标数据集。 <br> **数据类型：** 资产列表（字符串）。
| `label_period_candles` | 创建标签所针对的未来蜡烛数量。这可以在 `set_freqai_targets()` 中使用（有关详细用法，请参见 `templates/FreqaiExampleStrategy.py`）。此参数不一定是必需的，你可以创建自定义标签并选择是否使用此参数。请参见 `templates/FreqaiExampleStrategy.py` 查看示例用法。 <br> **数据类型：** 正整数。
| `include_shifted_candles` | 从先前蜡烛向后续蜡烛添加特征，目的是添加历史信息。如果使用，FreqAI 将复制并移动来自 `include_shifted_candles` 先前蜡烛的所有特征，以便这些信息对后续蜡烛可用。 <br> **数据类型：** 正整数。
| `weight_factor` | 根据训练数据点的时间近因对其进行加权（详细信息[此处](freqai-feature-engineering.md#weighting-features-for-temporal-importance)）。 <br> **数据类型：** 正浮点数（通常 < 1）。
| `indicator_max_period_candles` | **不再使用（#7325）**。由在[策略](freqai-configuration.md#building-a-freqai-strategy) 中设置的 `startup_candle_count` 替代。`startup_candle_count` 与时间周期无关，并定义了在 `feature_engineering_*()` 中用于指标创建的最大 *period*。FreqAI 将此参数与 `include_time_frames` 中的最大时间周期一起使用，以计算应下载多少个数据点，使得第一个数据点不包含 NaN。 <br> **数据类型：** 正整数。
| `indicator_periods_candles` | 计算指标所用的时间周期。这些指标被添加到基础指标数据集。 <br> **数据类型：** 正整数列表。
| `principal_component_analysis` | 使用主成分分析自动降低数据集的维度。有关其工作原理的详细信息[此处](freqai-feature-engineering.md#data-dimensionality-reduction-with-principal-component-analysis) <br> **数据类型：** 布尔值。 <br> 默认值：`False`。
| `plot_feature_importances` | 为前/后 `plot_feature_importances` 个特征中的每个模型创建特征重要性图。图表存储在 `user_data/models/<identifier>/sub-train-<COIN>_<timestamp>.html`。 <br> **数据类型：** 整数。 <br> 默认值：`0`。
| `DI_threshold` | 当设置为 > 0 时，激活使用相异指数进行离群值检测。有关其工作原理的详细信息[此处](freqai-feature-engineering.md#identifying-outliers-with-the-dissimilarity-index-di)。 <br> **数据类型：** 正浮点数（通常 < 1）。
| `use_SVM_to_remove_outliers` | 训练一个支持向量机以检测和移除训练数据集以及传入数据点中的离群值。有关其工作原理的详细信息[此处](freqai-feature-engineering.md#identifying-outliers-using-a-support-vector-machine-svm)。 <br> **数据类型：** 布尔值。
| `svm_params` | Sklearn 的 `SGDOneClassSVM()` 中可用的所有参数。有关某些选定参数的详细信息[此处](freqai-feature-engineering.md#identifying-outliers-using-a-support-vector-machine-svm)。 <br> **数据类型：** 字典。
| `use_DBSCAN_to_remove_outliers` | 使用 DBSCAN 算法对训练数据和预测数据进行聚类，以识别和移除离群值。有关其工作原理的详细信息[此处](freqai-feature-engineering.md#identifying-outliers-with-dbscan)。 <br> **数据类型：** 布尔值。
| `noise_standard_deviation` | 如果设置，FreqAI 会向训练特征添加噪声，目的是防止过拟合。FreqAI 从标准偏差为 `noise_standard_deviation` 的高斯分布中生成随机偏差，并将它们添加到所有数据点。`noise_standard_deviation` 应相对于归一化空间保持，即在 -1 和 1 之间。换句话说，由于 FreqAI 中的数据总是归一化到 -1 和 1 之间，`noise_standard_deviation: 0.05` 会导致 32% 的数据被随机增加/减少超过 2.5%（即落在第一个标准偏差内的数据百分比）。 <br> **数据类型：** 整数。 <br> 默认值：`0`。
| `outlier_protection_percentage` | 启用以防止离群值检测方法丢弃过多数据。如果超过 `outlier_protection_percentage` % 的点被 SVM 或 DBSCAN 检测为离群值，FreqAI 将记录警告消息并忽略离群值检测，即原始数据集将保持完整。如果触发了离群值保护，将不会基于训练数据集进行任何预测。 <br> **数据类型：** 浮点数。 <br> 默认值：`30`。
| `reverse_train_test_order` | 拆分特征数据集（见下文），并使用最新的数据拆分进行训练，在历史数据拆分上进行测试。这允许模型训练到最近的数据点，同时避免过度拟合。但是，在使用此参数之前，你应该小心理解此参数的非正统性质。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`（不反转）。
| `shuffle_after_split` | 将数据拆分为训练集和测试集，然后分别打乱两个集合。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。
| `buffer_train_data_candles` | 在指标填充*之后*，从训练数据的开头和结尾切除 `buffer_train_data_candles`。主要的示例使用场景是在预测最大值和最小值时，argrelextrema 函数无法知道 timerange 边缘的最大值/最小值。为了提高模型准确性，最好在整个 timerange 上计算 argrelextrema，然后使用此函数通过内核切除边缘（缓冲区）。在另一种情况下，如果目标设置为偏移的价格变动，则此缓冲区是不必要的，因为 timerange 末尾的偏移蜡烛将是 NaN，FreqAI 将自动将它们从训练数据集中切除。<br> **数据类型：** 整数。 <br> 默认值：`0`。

### 数据拆分参数

|  参数 | 描述 |
|------------|-------------|
|  |  **`freqai.data_split_parameters` 子字典内的数据拆分参数**
| `data_split_parameters` | 包含来自 scikit-learn `test_train_split()` 的任何可用附加参数，显示[此处](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)（外部网站）。 <br> **数据类型：** 字典。
| `test_size` | 应用于测试而不是训练的数据比例。 <br> **数据类型：** 小于 1 的正浮点数。
| `shuffle` | 在训练期间打乱训练数据点。通常，为了不移除时间序列预测中数据的时序顺序，此设置为 `False`。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。

### 模型训练参数

|  参数 | 描述 |
|------------|-------------|
|  |  **`freqai.model_training_parameters` 子字典内的模型训练参数**
| `model_training_parameters` | 一个灵活的字典，包含所选模型库可用的所有参数。例如，如果你使用 `LightGBMRegressor`，此字典可以包含 `LightGBMRegressor` [此处](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)（外部网站）可用的任何参数。如果你选择不同的模型，此字典可以包含该模型的任何参数。当前可用模型的列表可在此处找到[此处](freqai-configuration.md#using-different-prediction-models)。  <br> **数据类型：** 字典。
| `n_estimators` | 在模型训练中拟合的提升树的数量。 <br> **数据类型：** 整数。
| `learning_rate` | 模型训练期间的提升学习率。 <br> **数据类型：** 浮点数。
| `n_jobs`, `thread_count`, `task_type` | 设置并行处理的线程数和 `task_type`（`gpu` 或 `cpu`）。不同的模型库使用不同的参数名称。 <br> **数据类型：** 浮点数。

### 强化学习参数

|  参数 | 描述 |
|------------|-------------|
|  |  **`freqai.rl_config` 子字典内的强化学习参数**
| `rl_config` | 包含强化学习模型控制参数的字典。 <br> **数据类型：** 字典。
| `train_cycles` | 训练时间步将基于 `train_cycles * 训练数据点数量` 设置。 <br> **数据类型：** 整数。
| `max_trade_duration_candles`| 引导智能体训练将交易保持在期望长度以下。在 `prediction_models/ReinforcementLearner.py` 内的可自定义 `calculate_reward()` 函数中显示示例用法。 <br> **数据类型：** int。
| `model_type` | 来自 stable_baselines3 或 SBcontrib 的模型字符串。可用字符串包括：`'TRPO', 'ARS', 'RecurrentPPO', 'MaskablePPO', 'PPO', 'A2C', 'DQN'`。用户应确保 `model_training_parameters` 与相应 stable_baselines3 模型可用的参数匹配，方法是访问它们的文档。 [PPO 文档](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)（外部网站） <br> **数据类型：** 字符串。
| `policy_type` | 来自 stable_baselines3 的可用策略类型之一 <br> **数据类型：** 字符串。
| `max_training_drawdown_pct` | 智能体在训练期间被允许经历的最大回撤。 <br> **数据类型：** 浮点数。 <br> 默认值：0.8
| `cpu_count` | 专用于强化学习训练过程的线程/cpu 数量（取决于是否选择了 `ReinforcementLearner_multiproc`）。建议保持不动，默认情况下，此值设置为物理核心总数减 1。 <br> **数据类型：** int。
| `model_reward_parameters` | 在 `ReinforcementLearner.py` 内的可自定义 `calculate_reward()` 函数中使用的参数 <br> **数据类型：** int。
| `add_state_info` | 告诉 FreqAI 在用于训练和推断的特征集中包含状态信息。当前状态变量包括交易持续时间、当前利润、交易仓位。这仅在 dry/live 运行中可用，并且针对回测自动切换为 false。 <br> **数据类型：** bool。 <br> 默认值：`False`。
| `net_arch` | 网络架构，在 [`stable_baselines3` 文档](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#examples) 中有详细描述。总结：`[<shared layers>, dict(vf=[<non-shared value network layers>], pi=[<non-shared policy network layers>])]`。默认情况下，它设置为 `[128, 128]`，它定义了 2 个共享隐藏层，每个层有 128 个单元。
| `randomize_starting_position` | 随机化每个回合的起点以避免过拟合。 <br> **数据类型：** bool。 <br> 默认值：`False`。
| `drop_ohlc_from_features` | 不要在传递给智能体进行训练的特征集中包含归一化的 ohlc 数据（ohlc 在所有情况下仍用于驱动环境） <br> **数据类型：** 布尔值。 <br> **默认值：** `False`
| `progress_bar` | 显示带有当前进度、已用时间和估计剩余时间的进度条。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。

### PyTorch 参数

#### 通用

|  参数 | 描述 |
|------------|-------------|
|  |  **`freqai.model_training_parameters` 子字典内的模型训练参数**
| `learning_rate` | 要传递给优化器的学习率。 <br> **数据类型：** 浮点数。 <br> 默认值：`3e-4`。
| `model_kwargs` | 要传递给模型类的参数。 <br> **数据类型：** 字典。 <br> 默认值：`{}`。
| `trainer_kwargs` | 要传递给训练器类的参数。 <br> **数据类型：** 字典。 <br> 默认值：`{}`。

#### trainer_kwargs

| 参数    | 描述 |
|--------------|-------------|
|              |  **`freqai.model_training_parameters.model_kwargs` 子字典内的模型训练参数**
| `n_epochs`   | `n_epochs` 参数是 PyTorch 训练循环中的一个关键设置，它决定了整个训练数据集将被用于更新模型参数的次数。一个 epoch 表示对整个训练数据集的一次完整遍历。覆盖 `n_steps`。必须设置 `n_epochs` 或 `n_steps` 之一。 <br><br> **数据类型：** int。可选。 <br> 默认值：`10`。
| `n_steps`    | 设置 `n_epochs` 的替代方法 - 要运行的训练迭代次数。这里的迭代指的是我们调用 `optimizer.step()` 的次数。如果设置了 `n_epochs`，则忽略。函数的简化版本： <br><br> n_epochs = n_steps / (n_obs / batch_size) <br><br> 这里的动机是 `n_steps` 更容易优化并跨不同的 n_obs（数据点数量）保持稳定。  <br> <br> **数据类型：** int。可选。 <br> 默认值：`None`。
| `batch_size` | 训练期间使用的批次大小。 <br><br> **数据类型：** int。 <br> 默认值：`64`。
| `early_stopping_patience` | 在训练提前停止之前，验证损失没有改善的 epoch 数量。这有助于通过在模型停止改进时停止训练来防止过拟合。设置为 `0` 以禁用提前停止。需要一个测试/验证拆分（`test_size > 0`）。 <br><br> **数据类型：** int。 <br> 默认值：`0`（禁用）。

### 附加参数

|  参数 | 描述 |
|------------|-------------|
|  |  **无关参数**
| `freqai.keras` | 如果所选模型使用 Keras（典型的基于 TensorFlow 的预测模型），需要激活此标志，以便模型的保存/加载遵循 Keras 标准。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。
| `freqai.conv_width` | 神经网络输入张量的宽度。这通过将要历史数据点作为张量的第二维输入，取代了移动蜡烛（`include_shifted_candles`）的需要。从技术上讲，此参数也可用于回归器，但它只会增加计算开销，而不会改变模型的训练/预测。 <br> **数据类型：** 整数。 <br> 默认值：`2`。
| `freqai.reduce_df_footprint` | 将所有数值列重新转换为 float32/int32，目的是减少 ram/磁盘使用量并减少训练/推断时间。此参数设置在 Freqtrade 配置文件的主层级（不在 FreqAI 内部）。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。
| `freqai.override_exchange_check` | 覆盖交易所检查以强制 FreqAI 使用可能没有足够历史数据的交易所。如果你知道你的 FreqAI 模型和策略不需要历史数据，请将此设置为 True。 <br> **数据类型：** 布尔值。 <br> 默认值：`False`。
