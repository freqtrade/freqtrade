# 特征工程（Feature engineering）

## 定义特征

底层特征工程在用户策略内部的一组名为 `feature_engineering_*` 的函数中执行。这些函数设置 `基础特征`（base features），例如 `RSI`、`MFI`、`EMA`、`SMA`、一天中的时间、成交量等。这些 `基础特征` 可以是自定义指标，也可以从你能找到的任何技术分析库中导入。FreqAI 配备了一组函数来简化快速大规模的特征工程：

|  函数 | 描述 |
|---------------|-------------|
| `feature_engineering_expand_all()` | 这个可选函数将在配置定义的 `indicator_periods_candles`、`include_timeframes`、`include_shifted_candles` 和 `include_corr_pairs` 上自动扩展定义的特征。
| `feature_engineering_expand_basic()` | 这个可选函数将在配置定义的 `include_timeframes`、`include_shifted_candles` 和 `include_corr_pairs` 上自动扩展定义的特征。注意：此函数*不会*在用户定义的 `indicator_periods_candles` 上扩展。
| `feature_engineering_standard()` | 这个可选函数将使用基础时间周期的 dataframe 被调用一次。这是将被调用的最后一个函数，这意味着进入此函数的 dataframe 将包含由其他 `feature_engineering_expand` 函数创建的所有特征和列。此函数是做自定义奇异特征提取（例如 tsfresh）的好地方。此函数也适用于任何不应被自动扩展的特征（例如星期几）。
| `set_freqai_targets()` | 设置模型目标所需的函数。所有目标必须以 `&` 为前缀才能被 FreqAI 内部识别。

与此同时，高层特征工程在 FreqAI 配置中的 `"feature_parameters":{}` 内处理。在此文件中，可以决定在 `基础特征` 之上的大规模特征扩展，例如“包含相关币对”或“包含信息类时间周期”甚至“包含最近蜡烛”。

建议从提供的示例策略（位于 `templates/FreqaiExampleStrategy.py`）中的模板 `feature_engineering_*` 函数开始，以确保特征定义遵循正确的约定。以下是如何在策略中设置指标和标签的示例：

```python
    def feature_engineering_expand_all(self, dataframe: DataFrame, period, metadata, **kwargs) -> DataFrame:
        """
        *仅在启用了 FreqAI 的策略中可用*
        此函数将在配置定义的
        `indicator_periods_candles`、`include_timeframes`、`include_shifted_candles` 和
        `include_corr_pairs` 上自动扩展定义的特征。换句话说，在此函数中定义的单个特征
        将自动扩展为
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` 个特征添加到模型中。

        所有特征必须以 `%` 为前缀才能被 FreqAI 内部识别。

        通过以下方式访问诸如当前币对/时间周期/周期之类的元数据：

        `metadata["pair"]` `metadata["tf"]`  `metadata["period"]`

        :param df: 将接收特征的战略 dataframe
        :param period: 指标的 period - 使用示例：
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"]
            - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
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

        通过以下方式访问诸如当前币对/时间周期之类的元数据：

        `metadata["pair"]` `metadata["tf"]`

        所有特征必须以 `%` 为前缀才能被 FreqAI 内部识别。

        :param df: 将接收特征的战略 dataframe
        :param metadata: 当前币对的元数据
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        """
        *仅在启用了 FreqAI 的策略中可用*
        这个可选函数将使用基础时间周期的 dataframe 被调用一次。
        这是将被调用的最后一个函数，这意味着进入此函数的 dataframe
        将包含由所有其他
        freqai_feature_engineering_* 函数创建的所有特征和列。

        这个函数是做自定义奇异特征提取（例如 tsfresh）的好地方。
        这个函数是任何不应该被自动扩展（例如星期几）的特征的好地方。

        通过以下方式访问诸如当前币对之类的元数据：

        `metadata["pair"]`

        所有特征必须以 `%` 为前缀才能被 FreqAI 内部识别。

        :param df: 将接收特征的战略 dataframe
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata, **kwargs) -> DataFrame:
        """
        *仅在启用了 FreqAI 的策略中可用*
        设置模型目标所需的函数。
        所有目标必须以 `&` 为前缀才能被 FreqAI 内部识别。

        通过以下方式访问诸如当前币对之类的元数据：

        `metadata["pair"]`

        :param df: 将接收目标的战略 dataframe
        :param metadata: 当前币对的元数据
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

在展示的示例中，用户不希望将 `bb_lowerband` 作为一个特征传递给模型，因此没有用 `%` 为它添加前缀。然而，用户的确希望将 `bb_width` 传递给模型进行训练/预测，因此用 `%` 为它添加了前缀。

在定义了 `基础特征` 之后，下一步是使用配置文件中强大的 `feature_parameters` 来扩展它们：

```json
    "freqai": {
        //...
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
        //...
    }
```

上面配置中的 `include_timeframes` 是对策略中每次调用 `feature_engineering_expand_*()` 的时间周期（`tf`）。在展示的情况下，用户要求将 `rsi`、`mfi`、`roc` 和 `bb_width` 的 `5m`、`15m` 和 `4h` 时间周期包含在特征集中。

你可以要求将每个定义的特征也包含给信息类币对，使用 `include_corr_pairlist`。这意味着特征集将包含来自 `feature_engineering_expand_*()` 的所有特征，位于配置中定义的每个相关币对（`ETH/USD`、`LINK/USD` 和 `BNB/USD` 在展示的示例中）的所有 `include_timeframes` 上。

`include_shifted_candles` 表示要包含在特征集中的先前蜡烛的数量。例如，`include_shifted_candles: 2` 告诉 FreqAI 为特征集中的每个特征包含过去 2 根蜡烛。

总之，展示的示例策略用户创建的特征数量为：`include_timeframes` 的长度 * `feature_engineering_expand_*()` 中的特征数量 * `include_corr_pairlist` 的长度 * `include_shifted_candles` 的数量 * `indicator_periods_candles` 的长度
 $= 3 * 3 * 3 * 2 * 2 = 108$。

!!! note "了解有关创造性特征工程的更多信息"
    查看我们的 [medium 文章](https://emergentmethods.medium.com/freqai-from-price-to-prediction-6fadac18b665)，旨在帮助用户学习如何创造性地工程化特征。

### 使用 `metadata` 对 `feature_engineering_*` 函数获得更精细的控制

所有 `feature_engineering_*` 和 `set_freqai_targets()` 函数都传递了一个 `metadata` 字典，其中包含有关 FreqAI 正在自动化以进行特征构建的 `pair`、`tf`（时间周期）和 `period` 的信息。因此，用户可以在 `feature_engineering_*` 函数内部使用 `metadata` 作为阻止/保留某些时间周期、周期、币对等的特征的标准。

```python
def feature_engineering_expand_all(self, dataframe: DataFrame, period, metadata, **kwargs) -> DataFrame:
    if metadata["tf"] == "1h":
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
```

这将阻止 `ta.ROC()` 被添加到除 `"1h"` 之外的任何时间周期。

### 从训练中返回附加信息

重要的指标可以通过在自定义预测模型类内部将它们赋值给 `dk.data['extra_returns_per_train']['my_new_value'] = XYZ` 来在每次模型训练结束时返回到策略。

FreqAI 获取在此字典中赋值的 `my_new_value` 并将其扩展以适应返回到策略的 dataframe。然后你可以通过 `dataframe['my_new_value']` 在你的策略中使用返回的指标。FreqAI 中返回值如何被使用的一个例子是用于[创建动态目标阈值](freqai-configuration.md#creating-a-dynamic-target-threshold) 的 `&*_mean` 和 `&*_std` 值。

另一个例子，用户想要使用来自交易数据库的 live 指标，如下所示：

```json
    "freqai": {
        "extra_returns_per_train": {"total_profit": 4}
    }
```

你需要在配置中设置标准字典，以便 FreqAI 可以返回正确的 dataframe 形状。这些值很可能被预测模型覆盖，但在模型尚未设置它们，或需要一个默认初始值的情况下，将返回预设值。

### 对时间重要性进行特征加权

FreqAI 允许你设置一个 `weight_factor`，通过指数函数对最近的数据比过去的数据进行更强的加权：

$$ W_i = \exp(\frac{-i}{\alpha*n}) $$

其中 $W_i$ 是总共 $n$ 个数据点中数据点 $i$ 的权重。下面是一个图，显示了不同权重因子对特征集中数据点的影响。

![weight-factor](../assets/freqai_weight-factor.jpg)

## 构建数据管道

默认情况下，FreqAI 根据用户配置设置构建一个动态管道。默认设置是健壮的，并设计为适用于各种方法。这两个步骤是 `MinMaxScaler(-1,1)` 和一个 `VarianceThreshold`，后者移除任何方差为 0 的列。用户可以通过更多的配置参数激活其他步骤。例如，如果用户在 `freqai` 配置中添加 `use_SVM_to_remove_outliers: true`，那么 FreqAI 将自动将 [`SVMOutlierExtractor`](#identifying-outliers-using-a-support-vector-machine-svm) 添加到管道中。同样，用户可以添加 `principal_component_analysis: true` 到 `freqai` 配置来激活 PCA。[相异指数（Dissimilarity Index）](#identifying-outliers-with-the-dissimilarity-index-di) 通过 `DI_threshold: 1` 激活。最后，也可以通过 `noise_standard_deviation: 0.1` 向数据中添加噪声。最后，用户可以通过 `use_DBSCAN_to_remove_outliers: true` 添加 [DBSCAN](#identifying-outliers-with-dbscan) 离群值移除。

!!! note "更多信息可用"
    请查看[参数表](freqai-parameter-table.md) 以获取有关这些参数的更多信息。

### 自定义管道

鼓励用户通过构建自己的数据管道，根据需要自定义数据管道。这可以通过在 `IFreqaiModel` 的 `train()` 函数内部简单地将 `dk.feature_pipeline` 设置为他们想要的 `Pipeline` 对象来完成，或者如果他们不想触及 `train()` 函数，他们可以在他们的 `IFreqaiModel` 中覆盖 `define_data_pipeline`/`define_label_pipeline` 函数：

!!! note "更多信息可用"
    FreqAI 使用 [`DataSieve`](https://github.com/emergentmethods/datasieve) 管道，它遵循 SKlearn 管道 API，但增加了 X、y 和 sample_weight 向量点移除之间的一致性、特征移除、特征名称跟踪等功能。

```python
from datasieve.transforms import SKLearnWrapper, DissimilarityIndex
from datasieve.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from freqai.base_models import BaseRegressionModel


class MyFreqaiModel(BaseRegressionModel):
    """
    一些很酷的自定义模型
    """
    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        我的自定义 fit 函数
        """
        model = cool_model.fit()
        return model

    def define_data_pipeline(self) -> Pipeline:
        """
        用户在此定义他们自定义的 feature 管道（如果他们愿意）
        """
        feature_pipeline = Pipeline([
            ('qt', SKLearnWrapper(QuantileTransformer(output_distribution='normal'))),
            ('di', ds.DissimilarityIndex(di_threshold=1))
        ])

        return feature_pipeline

    def define_label_pipeline(self) -> Pipeline:
        """
        用户在此定义他们自定义的 label 管道（如果他们愿意）
        """
        label_pipeline = Pipeline([
            ('qt', SKLearnWrapper(StandardScaler())),
        ])

        return label_pipeline
```

在这里，你正在定义将在训练和预测期间用于你的特征集的确切管道。你可以使用 `*大多数*` SKLearn 转换步骤，方法是像上面所示将它们包装在 `SKLearnWrapper` 类中。此外，你可以使用 [`DataSieve` 库](https://github.com/emergentmethods/datasieve) 中可用的任何转换。

你可以通过创建一个继承自 datasieve `BaseTransform` 并实现你的 `fit()`、`transform()` 和 `inverse_transform()` 方法的类，轻松添加你自己的转换：

```python
from datasieve.transforms.base_transform import BaseTransform
# import whatever else you need

class MyCoolTransform(BaseTransform):
    def __init__(self, **kwargs):
        self.param1 = kwargs.get('param1', 1)

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        # do something with X, y, sample_weight, or/and feature_list
        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None,
                  feature_list=None, outlier_check=False, **kwargs):
        # do something with X, y, sample_weight, or/and feature_list
        return X, y, sample_weight, feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        # do/dont do something with X, y, sample_weight, or/and feature_list
        return X, y, sample_weight, feature_list
```

!!! note "Hint"
    你可以将这个自定义类定义在与你的 `IFreqaiModel` 相同的文件中。

### 将自定义 `IFreqaiModel` 迁移到新的管道

如果你创建了自己的带有自定义 `train()`/`predict()` 函数的 `IFreqaiModel`，*并且* 你仍然依赖 `data_cleaning_train/predict()`，那么你将需要迁移到新的管道。如果你的模型*不*依赖 `data_cleaning_train/predict()`，那么你不需要担心这个迁移。

有关迁移的更多详细信息可以在[此处](strategy_migration.md#freqai-new-data-pipeline) 找到。

## 离群值检测（Outlier detection）

股票和加密市场受到高水平无模式噪声的影响，这些噪声以离群数据点的形式存在。FreqAI 实现了多种方法来识别此类离群值，从而降低风险。

### 使用相异指数（DI）识别离群值

相异指数（Dissimilarity Index，DI）旨在量化与模型做出的每个预测相关的不确定性。

你可以通过在配置中包含以下语句来告诉 FreqAI 使用 DI 从训练/测试数据集中移除离群数据点：

```json
    "freqai": {
        "feature_parameters" : {
            "DI_threshold": 1
        }
    }
```

这会将 `DissimilarityIndex` 步骤添加到你的 `feature_pipeline` 中，并将阈值设置为 1。DI 允许那些是离群值（不存在于模型特征空间中）的预测由于低置信度而被丢弃。为此，FreqAI 测量每个训练数据点（特征向量）$X_{a}$ 与所有其他训练数据点之间的距离：

$$ d_{ab} = \sqrt{\sum_{j=1}^p(X_{a,j}-X_{b,j})^2} $$

其中 $d_{ab}$ 是归一化点 $a$ 和 $b$ 之间的距离，$p$ 是特征的数量，即向量 $X$ 的长度。一组训练数据点的特征距离 $\overline{d}$ 只是平均距离的平均值：

$$ \overline{d} = \sum_{a=1}^n(\sum_{b=1}^n(d_{ab}/n)/n) $$

$\overline{d}$ 量化了训练数据的分布，它与一个新预测特征向量 $X_k$ 和所有训练数据之间的距离进行比较：

$$ d_k = \arg \min d_{k,i} $$

这使得能够估计相异指数为：

$$ DI_k = d_k/\overline{d} $$

你可以通过 `DI_threshold` 来调整 DI，以增加或减少训练模型的推断范围。更高的 `DI_threshold` 意味着 DI 更宽松，允许使用远离训练数据的预测，而更低的 `DI_threshold` 具有相反的效果，因此会丢弃更多的预测。

下面是一个描述 3D 数据集 DI 的图。

![DI](../assets/freqai_DI.jpg)

### 使用支持向量机（SVM）识别离群值

你可以通过在配置中包含以下语句来告诉 FreqAI 使用支持向量机（SVM）从训练/测试数据集中移除离群数据点：

```json
    "freqai": {
        "feature_parameters" : {
            "use_SVM_to_remove_outliers": true
        }
    }
```

这会将 `SVMOutlierExtractor` 步骤添加到你的 `feature_pipeline` 中。SVM 将在训练数据上进行训练，并且 SVM 认为超出特征空间的任何数据点都将被移除。

你可以选择通过配置中的 `feature_parameters.svm_params` 字典为 SVM 提供附加参数，例如 `shuffle` 和 `nu`。

参数 `shuffle` 默认设置为 `False` 以确保结果一致。如果设置为 `True`，由于 `max_iter` 太低导致算法无法达到要求的 `tol`，对同一数据集多次运行 SVM 可能会导致不同的结果。增加 `max_iter` 可以解决这个问题，但会导致过程花费更长时间。

参数 `nu` 在*非常*广泛的意义上，是应该被视为离群值的数据点的数量，并且应该在 0 到 1 之间。

### 使用 DBSCAN 识别离群值

你可以通过在配置中激活 `use_DBSCAN_to_remove_outliers` 来配置 FreqAI 使用 DBSCAN 对训练/测试数据集中的离群值进行聚类并移除，或从预测中移除传入的离群值：

```json
    "freqai": {
        "feature_parameters" : {
            "use_DBSCAN_to_remove_outliers": true
        }
    }
```

这会将 `DataSieveDBSCAN` 步骤添加到你的 `feature_pipeline` 中。这是一种无监督机器学习算法，可在不知道应该有多少个聚类的情况下对数据进行聚类。

给定一定数量的数据点 $N$ 和距离 $\varepsilon$，DBSCAN 通过将具有 $N-1$ 个其他数据点距离在 $\varepsilon$ 内的所有数据点设置为*核心点* 来对数据集进行聚类。一个数据点如果距离某个*核心点* 在 $\varepsilon$ 内，但自身没有 $N-1$ 个其他数据点距离在 $\varepsilon$ 内，则被视为*边缘点*。一个聚类随后是*核心点* 和*边缘点* 的集合。没有任何其他数据点距离 $<\varepsilon$ 的数据点被视为离群值。下图显示了一个 $N = 3$ 的聚类。

![dbscan](../assets/freqai_dbscan.jpg)

FreqAI 使用 `sklearn.cluster.DBSCAN`（详细信息可在 scikit-learn 的网页[此处](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) 找到），其中 `min_samples` ($N$) 取为 特征集中时间点数（蜡烛数）的 1/4。`eps` ($\varepsilon$) 通过计算特征集中所有数据点的成对距离的最近邻的 *k-distance 图* 中的拐点自动计算。

### 使用主成分分析进行数据降维

你可以通过在配置中激活 principal_component_analysis 来减少特征的维度：

```json
    "freqai": {
        "feature_parameters" : {
            "principal_component_analysis": true
        }
    }
```

这将对特征执行 PCA 并减少它们的维度，使得数据集的解释方差 >= 0.999。减少数据维度使模型训练更快，从而允许更最新的模型。
