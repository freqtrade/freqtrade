# 配置（Configuration）

FreqAI 通过典型的 [Freqtrade 配置文件](configuration.md) 和标准的 [Freqtrade 策略](strategy-customization.md) 进行配置。FreqAI 配置和策略文件的示例可以在 `config_examples/config_freqai.example.json` 和 `freqtrade/templates/FreqaiExampleStrategy.py` 中分别找到。

## 设置配置文件

虽然有许多额外的参数可供选择，如[参数表](freqai-parameter-table.md#parameter-table) 中突出显示的那样，但 FreqAI 配置必须至少包含以下参数（参数值仅为示例）：

```json
    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 30,
        "backtest_period_days": 7,
        "identifier" : "unique-id",
        "feature_parameters" : {
            "include_timeframes": ["5m","15m","4h"],
            "include_corr_pairlist": [
                "ETH/USD",
                "LINK/USD",
                "BNB/USD"
            ],
            "label_period_candles": 24,
            "include_shifted_candles": 2,
            "indicator_periods_candles": [10, 20]
        },
        "data_split_parameters" : {
            "test_size": 0.25
        }
    }
```

完整的示例配置可在 `config_examples/config_freqai.example.json` 中获得。

!!! Note
    `identifier` 通常被新手忽略，但是，这个值在你的配置中扮演着重要角色。这个值是你选择用来描述你的一次运行的唯一 ID。保持它不变可以让你保持崩溃弹性以及更快的回测。一旦你想尝试一次新的运行（新特征、新模型等），你应该更改这个值（或删除 `user_data/models/unique-id` 文件夹）。更多详细信息可在[参数表](freqai-parameter-table.md#feature-parameters)中找到。

## 构建 FreqAI 策略

FreqAI 策略需要在标准的 [Freqtrade 策略](strategy-customization.md) 中包含以下代码行：

```python
    # 用户应该定义最大启动蜡烛计数（传递给任何单个指标的最大蜡烛数）
    startup_candle_count: int = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # 模型将返回用户在 `set_freqai_targets()` 中创建的所有标签
        #（& 附加的目标），一个指示是否应该接受预测的指示，
        # 以及用户在 `set_freqai_targets()` 中为每个训练周期创建的每个标签的目标 mean/std 值。

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def feature_engineering_expand_all(self, dataframe: DataFrame, period, **kwargs) -> DataFrame:
        """
        *仅在启用了 FreqAI 的策略中可用*
        此函数将在配置定义的
        `indicator_periods_candles`、`include_timeframes`、`include_shifted_candles` 和
        `include_corr_pairs` 上自动扩展定义的特征。换句话说，在此函数中定义的单个特征
        将自动扩展为
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` 个特征添加到模型中。

        所有特征必须以 `%` 为前缀才能被 FreqAI 内部识别。

        :param df: 将接收特征的战略 dataframe
        :param period: 指标的 period - 使用示例：
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *仅在启用了 FreqAI 的策略中可用*
        此函数将在配置定义的
        `include_timeframes`、`include_shifted_candles` 和 `include_corr_pairs` 上自动扩展定义的特征。
        换句话说，在此函数中定义的单个特征
        将自动扩展为
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        个特征添加到模型中。

        此处定义的特征将 *不会* 在用户定义的
        `indicator_periods_candles` 上自动复制

        所有特征必须以 `%` 为前缀才能被 FreqAI 内部识别。

        :param df: 将接收特征的战略 dataframe
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *仅在启用了 FreqAI 的策略中可用*
        这个可选函数将使用基础时间周期的 dataframe 被调用一次。
        这是要调用的最后一个函数，这意味着进入此函数的 dataframe
        将包含由所有其他
        freqai_feature_engineering_* 函数创建的所有特征和列。

        这个函数是做自定义奇异特征提取（例如 tsfresh）的好地方。
        这个函数是任何不应该被自动扩展（例如星期几）的特征的好地方。

        所有特征必须以 `%` 为前缀才能被 FreqAI 内部识别。

        :param df: 将接收特征的战略 dataframe
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        *仅在启用了 FreqAI 的策略中可用*
        设置模型目标所需的函数。
        所有目标必须以 `&` 为前缀才能被 FreqAI 内部识别。

        :param df: 将接收目标的战略 dataframe
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
            )
        return dataframe
```

注意 `feature_engineering_*()` 是添加[特征](freqai-feature-engineering.md#feature-engineering) 的地方。同时 `set_freqai_targets()` 添加标签/目标。完整的示例策略可在 `templates/FreqaiExampleStrategy.py` 中获得。

!!! Note
    `self.freqai.start()` 函数不能在 `populate_indicators()` 之外调用。

!!! Note
    特征**必须**在 `feature_engineering_*()` 中定义。在 `populate_indicators()` 中定义 FreqAI 特征
    将导致算法在 live/dry 模式下失败。为了添加不与特定币对或时间周期关联的通用特征，你应该使用 `feature_engineering_standard()`
    （如 `freqtrade/templates/FreqaiExampleStrategy.py` 中所例示）。

## 重要的 dataframe 键模式

以下是你可以在典型策略 dataframe（`df[]`）中包含/使用的期望值：

|  DataFrame 键 | 描述 |
|------------|-------------|
| `df['&*']` | 在 `set_freqai_targets()` 中以 `&` 为前缀的任何 dataframe 列都被 FreqAI 视为训练目标（标签）（通常遵循 `&-s*` 命名约定）。例如，要预测 40 根蜡烛之后的收盘价，你会设置 `df['&-s_close'] = df['close'].shift(-self.freqai_info["feature_parameters"]["label_period_candles"])`，配置中 `"label_period_candles": 40`。FreqAI 进行预测，并在相同的键下（`df['&-s_close']`）将它们返回，以便在 `populate_entry/exit_trend()` 中使用。<br> **数据类型：** 取决于模型的输出。
| `df['&*_std/mean']` | 训练期间（或使用 `fit_live_predictions_candles` 的 live 跟踪）定义的标签的标准差和平均值。通常用于理解预测的稀有度（使用 z-score，如 `templates/FreqaiExampleStrategy.py` 所示，并在[此处](#creating-a-dynamic-target-threshold) 解释，以评估在训练期间或历史上使用 `fit_live_predictions_candles` 观察到特定预测的频率）。<br> **数据类型：** Float（浮点数）。
| `df['do_predict']` | 离群数据点的指示。返回值是 -2 到 2 之间的整数，让你知道预测是否可信。`do_predict==1` 意味着预测是可信的。如果输入数据点的DI（Dissimilarity Index，相异指数）（见配置中定义的阈值，FreqAI 将从 `do_predict` 中减去 1，导致 `do_predict==0`。如果 `use_SVM_to_remove_outliers` 处于活动状态，支持向量机（SVM，详见[此处](freqai-feature-engineering.md#identifying-outliers-using-a-support-vector-machine-svm)）也可能在训练和预测数据中检测到离群值。在这种情况下，SVM 也会从 `do_predict` 中减去 1。如果输入数据点被 SVM 视为离群值但未被 DI 视为，反之亦然，结果将是 `do_predict==0`。如果 DI 和 SVM 都认为输入数据点是离群值，结果将是 `do_predict==-1`。与 SVM 一样，如果 `use_DBSCAN_to_remove_outliers` 处于活动状态，DBSCAN（详见[此处](freqai-feature-engineering.md#identifying-outliers-with-dbscan)）也可能检测到离群值并从 `do_predict` 中减去 1。因此，如果 SVM 和 DBSCAN 都处于活动状态，并且识别出高于 DI 阈值的离群数据点，结果将是 `do_predict==-2`。一个特殊情况是 `do_predict == 2`，这意味着模型由于超过 `expired_hours` 而已过期。<br> **数据类型：** -2 到 2 之间的整数。
| `df['DI_values']` | 相异指数（DI）值是 FreqAI 对预测信心的代理。较低的 DI 意味着预测接近训练数据，即较高的预测置信度。有关 DI 的详细信息，请参见[此处](freqai-feature-engineering.md#identifying-outliers-with-the-dissimilarity-index-di)。<br> **数据类型：** Float。
| `df['%*']` | 在 `feature_engineering_*()` 中以 `%` 为前缀的任何 dataframe 列都被视为训练特征。例如，你可以通过将 `df['%-rsi']` 包含在训练特征集中（类似于 `templates/FreqaiExampleStrategy.py`）。有关如何执行此操作的更多详细信息，请参见[此处](freqai-feature-engineering.md)。<br> **注意：** 由于以 `%` 为前缀的特征数量可以非常快速地相乘（使用例如 `include_shifted_candles` 和 `include_timeframes` 的乘法功能，如[参数表](freqai-parameter-table.md) 中所述，很容易设计出数万个特征），这些特征会从从 FreqAI 返回到策略的 dataframe 中移除。要保留特定类型的特征用于绘图目的，你可以以 `%%` 为前缀（见下面的详细信息）。<br> **数据类型：** 取决于用户创建的特征。
| `df['%%*']` | 在 `feature_engineering_*()` 中以 `%%` 为前缀的任何 dataframe 列都被视为训练特征，与上述 `%` 前缀完全相同。但是，在这种情况下，特征被返回到策略，用于 FreqUI/plot-dataframe 绘图和 Dry/Live/Backtesting 中的监控 <br> **数据类型：** 取决于用户创建的特征。<br>*请注意* 在 `feature_engineering_expand()` 中创建的特征将具有根据你的扩展配置的自动 FreqAI 命名方案（即 `include_timeframes`、`include_corr_pairlist`、`indicators_periods_candles`、`include_shifted_candles`）。因此，如果你想从 `feature_engineering_expand_all()` 绘制 `%%-rsi`，你的绘图配置的最终命名方案将是：`%%-rsi-period_10_ETH/USDT:USDT_1h`（对于 `period=10`、`timeframe=1h` 和 `pair=ETH/USDT:USDT` 的 `rsi` 特征（如果你使用的是合约币对，则添加 `:USDT`）。在 `self.freqai.start()` 之后在 `populate_indicators()` 中添加 `print(dataframe.columns)` 以查看返回到策略用于绘图目的的可用特征完整列表是非常有用的。

## 设置 `startup_candle_count`

FreqAI 策略中的 `startup_candle_count` 需要与标准 Freqtrade 策略相同的方式设置（详见[此处](strategy-customization.md#strategy-startup-period)）。此值被 Freqtrade 用于确保调用 `dataprovider` 时提供足够的数据量，以避免第一次训练开始时出现任何 NaN。你可以通过识别传递给指标创建函数（例如，TA-Lib 函数）的最长周期（以蜡烛为单位）来轻松设置此值。在所示的示例中，`startup_candle_count` 是 20，因为这是 `indicators_periods_candles` 中的最大值。

!!! Note
    在某些情况下，TA-Lib 函数实际上需要比传递的 `period` 更多的数据，否则特征数据集会被 NaN 填充。据传闻，将 `startup_candle_count` 乘以 2 总是会产生一个完全没有 NaN 的训练数据集。因此，通常最安全的方法是将预期的 `startup_candle_count` 乘以 2。留意此日志消息以确认数据是干净的：

    ```
    2022-08-31 15:14:04 - freqtrade.freqai.data_kitchen - INFO - dropped 0 training points due to NaNs in populated dataset 4319.
    ```

## 创建动态目标阈值

决定何时进入或退出交易可以以反映当前市场条件的动态方式完成。FreqAI 允许你从模型的训练返回附加信息（更多信息[此处](freqai-feature-engineering.md#returning-additional-info-from-training)）。例如，`&*_std/mean` 返回值描述了目标/标签 *在最近一次训练期间* 的统计分布。将给定预测与这些值进行比较可以让你知道预测的稀有度。在 `templates/FreqaiExampleStrategy.py` 中，`target_roi` 和 `sell_roi` 被定义为距离均值 1.25 个 z-score，这会导致接近均值的预测被过滤掉。

```python
dataframe["target_roi"] = dataframe["&-s_close_mean"] + dataframe["&-s_close_std"] * 1.25
dataframe["sell_roi"] = dataframe["&-s_close_mean"] - dataframe["&-s_close_std"] * 1.25
```

要考虑 *历史预测* 的总体来创建动态目标，而不是如上所述的训练信息，你可以在配置中将 `fit_live_predictions_candles` 设置为你想要用于生成目标统计的历史预测蜡烛数量。

```json
    "freqai": {
        "fit_live_predictions_candles": 300,
    }
```

如果设置了此值，FreqAI 最初将使用来自训练数据的预测，随后开始引入生成的真实预测数据。FreqAI 将保存这些历史数据以便重新加载，如果你停止并使用相同的 `identifier` 重新启动模型。

## 使用不同的预测模型

FreqAI 有多个示例预测模型库，可以通过 `--freqaimodel` 标志直接使用。这些库包括 `LightGBM` 和 `XGBoost` 回归、分类和多目标模型，可以在 `freqai/prediction_models/` 中找到。

回归和分类模型在它们预测的目标上有所不同 - 回归模型将预测连续值的目标，例如明天 BTC 的价格将是多少，而分类器将预测离散值的目标，例如明天 BTC 的价格是否会上涨。这意味着你必须根据你使用的模型类型不同地指定你的目标（详见[下文](#setting-model-targets)）。

上述所有模型库都实现了梯度提升决策树算法。它们都基于集成学习（ensemble learning）的原理工作，其中来自多个简单学习器的预测被组合以获得更稳定和泛化的的最终预测。在这种情况下，简单学习器是决策树。梯度提升指的是学习的方法，其中每个简单学习器按顺序构建 - 后续的学习器用于改进先前学习器的错误。如果你想了解有关不同模型库的更多信息，你可以在它们各自的文档中找到信息：

* LightGBM: <https://lightgbm.readthedocs.io/en/v3.3.2/#>
* XGBoost: <https://xgboost.readthedocs.io/en/stable/#>
* CatBoost: <https://catboost.ai/en/docs/>（自 2025.12 起不再积极支持）

还有大量在线文章描述和比较这些算法。一些相对轻量的例子是 [CatBoost vs. LightGBM vs. XGBoost — Which is the best algorithm?](https://towardsdatascience.com/catboost-vs-lightgbm-vs-xgboost-c80f40662924#:~:text=In%20CatBoost%2C%20symmetric%20trees%2C%20or,the%20same%20depth%20can%20differ.) 和 [XGBoost, LightGBM or CatBoost — which boosting algorithm should I use?](https://medium.com/riskified-technology/xgboost-lightgbm-or-catboost-which-boosting-algorithm-should-i-use-e7fda7bb36bc)。请记住，每个模型的性能高度依赖于应用，因此任何报告的指标可能不适用于你对模型的特定使用。

除了 FreqAI 中已经可用的模型之外，还可以使用 `IFreqaiModel` 类自定义并创建你自己的预测模型。我们鼓励你继承 `fit()`、`train()` 和 `predict()` 来自定义训练过程的各个方面。你可以将自定义 FreqAI 模型放在 `user_data/freqaimodels` - freqtrade 将根据提供的 `--freqaimodel` 名称从中选取它们 - 这必须对应于你的自定义模型的类名。确保使用唯一的名称以避免覆盖内置模型。

### 设置模型目标

#### 回归器（Regressors）

如果你使用的是回归器，你需要指定一个具有连续值的目标。FreqAI 包含各种回归器，例如通过 `--freqaimodel LightGBMRegressor` 标志的 `LightGBMRegressor`。如何设置回归目标以预测未来 100 根蜡烛价格的一个例子是

```python
df['&s-close_price'] = df['close'].shift(-100)
```

如果你想预测多个目标，你需要使用与上述相同的语法定义多个标签。

#### 分类器（Classifiers）

如果你使用的是分类器，你需要指定一个具有离散值的目标。FreqAI 包含各种分类器，例如通过 `--freqaimodel LightGBMClassifier` 标志的 `LightGBMClassifier`。如果你选择使用分类器，需要使用字符串设置类。例如，如果你想预测未来 100 根蜡烛的价格是上涨还是下跌，你会设置

```python
df['&s-up_or_down'] = np.where( df["close"].shift(-100) > df["close"], 'up', 'down')
```

如果你想预测多个目标，你必须在同一个标签列中指定所有标签。例如，你可以通过设置添加标签 `same` 来定义价格不变的位置

```python
df['&s-up_or_down'] = np.where( df["close"].shift(-100) > df["close"], 'up', 'down')
df['&s-up_or_down'] = np.where( df["close"].shift(-100) == df["close"], 'same', df['&s-up_or_down'])
```

## PyTorch 模块

### 快速开始

快速运行 pytorch 模型的最简单方法是使用以下命令（用于回归任务）：

```bash
freqtrade trade --config config_examples/config_freqai.example.json --strategy FreqaiExampleStrategy --freqaimodel PyTorchMLPRegressor --strategy-path freqtrade/templates 
```

!!! Note "安装/docker"
    PyTorch 模块需要像 `torch` 这样的大包，应该在 `./setup.sh -i` 期间通过回答 "y" 给 "Do you also want dependencies for freqai-rl or PyTorch (~700mb additional space required) [y/N]?" 这个问题来显式请求。
    偏好 docker 的用户应该确保他们使用附加了 `_freqaitorch` 的 docker 镜像。
    我们在 `docker/docker-compose-freqai.yml` 中为此提供了一个明确的 docker-compose 文件 - 可以通过 `docker compose -f docker/docker-compose-freqai.yml run ...` 使用 - 或者可以复制以替换原始 docker 文件。
    这个 docker-compose 文件还包含一个（禁用的）部分来在 docker 容器内启用 GPU 资源。这显然假设系统有可用的 GPU 资源。

    PyTorch 在 2.3 版本中放弃了对 macOS x64（基于 intel 的 Apple 设备）的支持。随后，freqtrade 也放弃了对该平台上 PyTorch 的支持。

!!! Danger "安全通知"
    从磁盘加载保存的模型可能会导致安全问题，如果使用远程模型文件（你从互联网下载或从不受信任的来源收到的文件），因为必须设置 `weights_only=False`，这可能导致安全问题。
    只要你只加载你自己训练的模型，就没有风险。

### 结构

#### 模型

你可以通过在你的自定义 [`IFreqaiModel` 文件](#using-different-prediction-models) 中简单地定义你的 `nn.Module` 类，然后在你的 `def train()` 函数中使用该类来构建你自己的 PyTorch 神经网络架构。这是一个使用 PyTorch 的逻辑回归模型实现（应该与 nn.BCELoss 准则一起用于分类任务）的例子。

```python

class LogisticRegression(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        # 定义你的层
        self.linear = nn.Linear(input_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 定义前向传播
        out = self.linear(x)
        out = self.activation(out)
        return out

class MyCoolPyTorchClassifier(BasePyTorchClassifier):
    """
    这是一个自定义的 IFreqaiModel，展示了用户可能如何设置他们自己的
    自定义神经网络架构用于训练。
    """

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = config.get("learning_rate",  3e-4)
        self.model_kwargs: dict[str, Any] = config.get("model_kwargs",  {})
        self.trainer_kwargs: dict[str, Any] = config.get("trainer_kwargs",  {})

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        用户在这里设置训练和测试数据以适应他们想要的模型
        :param data_dictionary: 保存所有用于 train、test、
            labels、weights 的数据的字典
        :param dk: 当前币/模型的 datakitchen 对象
        """

        class_names = self.get_class_names()
        self.convert_label_column_to_int(data_dictionary, dk, class_names)
        n_features = data_dictionary["train_features"].shape[-1]
        model = LogisticRegression(
            input_dim=n_features
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        init_model = self.get_init_model(dk.pair)
        trainer = PyTorchModelTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            model_meta_data={"class_names": class_names},
            device=self.device,
            init_model=init_model,
            data_convertor=self.data_convertor,
            **self.trainer_kwargs,
        )
        trainer.fit(data_dictionary, self.splits)
        return trainer

```

#### 训练器

`PyTorchModelTrainer` 执行惯用的 PyTorch 训练循环：
定义我们的模型、损失函数和优化器，然后将它们移动到适当的设备（GPU 或 CPU）。在循环内部，我们遍历 dataloader 中的批次，将数据移动到设备，计算预测和损失，反向传播，并使用优化器更新模型参数。

此外，训练器负责以下事项：
 - 保存和加载模型
 - 将数据从 `pandas.DataFrame` 转换为 `torch.Tensor`。

#### 与 Freqai 模块的集成

像所有 freqai 模型一样，PyTorch 模型继承 `IFreqaiModel`。`IFreqaiModel` 声明了三个抽象方法：`train`、`fit` 和 `predict`。我们在三个层次结构中实现这些方法。从顶部到底部：

1. `BasePyTorchModel` - 实现 `train` 方法。所有 `BasePyTorch*` 都继承它。负责一般的数据准备（例如，数据归一化）并调用 `fit` 方法。设置子类中使用的 `device` 属性。设置父类中使用的 `model_type` 属性。
2. `BasePyTorch*` - 实现 `predict` 方法。这里的 `*` 代表一组算法，例如分类器或回归器。负责数据预处理、预测和后处理（如果需要）。
3. `PyTorch*Classifier` / `PyTorch*Regressor` - 实现 `fit` 方法。负责主要的训练流程，我们在这里初始化训练器和模型对象。

![image](../assets/freqai_pytorch-diagram.png)

#### 完整示例

使用 MLP（多层感知机）模型、MSELoss 准则和 AdamW 优化器构建 PyTorch 回归器。

```python
class PyTorchMLPRegressor(BasePyTorchRegressor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = config.get("learning_rate",  3e-4)
        self.model_kwargs: dict[str, Any] = config.get("model_training_parameters", {})
        self.trainer_kwargs: dict[str, Any] = config.get("trainer_kwargs",  {})

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        n_features = data_dictionary["train_features"].shape[-1]
        model = PyTorchMLPModel(
            input_dim=n_features,
            output_dim=1,
            **self.model_kwargs
        )
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        init_model = self.get_init_model(dk.pair)
        trainer = PyTorchModelTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            init_model=init_model,
            target_tensor_type=torch.float,
            **self.trainer_kwargs,
        )
        trainer.fit(data_dictionary)
        return trainer
```

在这里我们创建了一个实现 `fit` 方法的 `PyTorchMLPRegressor` 类。`fit` 方法指定了训练的构建块：模型、优化器、准则和训练器。我们继承了 `BasePyTorchRegressor` 和 `BasePyTorchModel`，前者实现了适合我们回归任务的 `predict` 方法，后者实现了 train 方法。

??? Note "为分类器设置类名"
    使用分类器时，用户必须通过将 `IFreqaiModel.class_names` 属性设置为类名称（或目标）来声明类名。这是通过在 FreqAI 策略中 `set_freqai_targets` 方法内设置 `self.freqai.class_names` 来实现的。
    
    例如，如果你使用二进制分类器将价格变动预测为上涨或下跌，你可以如下设置类名：
    ```python
    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        self.freqai.class_names = ["down", "up"]
        dataframe['&s-up_or_down'] = np.where(dataframe["close"].shift(-100) >
                                                  dataframe["close"], 'up', 'down')
    
        return dataframe
    ```
    要查看完整示例，你可以参考 [分类器测试策略类](https://github.com/freqtrade/freqtrade/blob/develop/tests/strategy/strats/freqai_test_classifier.py)。


#### 使用 `torch.compile()` 提升性能

Torch 提供了一个 `torch.compile()` 方法，可用于针对特定 GPU 硬件提升性能。更多详细信息可在此处找到[此处](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)。简而言之，你只需将你的 `model` 包装在 `torch.compile()` 中：

```python
        model = PyTorchMLPModel(
            input_dim=n_features,
            output_dim=1,
            **self.model_kwargs
        )
        model.to(self.device)
        model = torch.compile(model)
```

然后像往常一样使用模型。请记住，这样做将删除 eager 执行，这意味着错误和 traceback 将不具信息性。
