# Configuration

## Setting up the configuration file

The user interface is isolated to the typical Freqtrade config file. Although there are plenty of additional parameters that a user can choose from, as highlighted in the [parameter table](#parameter-table), a FreqAI config must at minimum include the following parameters (the example inputs are not requires):

```json
    "freqai": {
        "enabled": true,
        "purge_old_models": true,
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
        },
        "model_training_parameters" : {
            "n_estimators": 100
        },
    }
```

## Building a FreqAI strategy

The FreqAI strategy requires the user to include the following lines of code in the standard Freqtrade strategy:

```python
    # user should define the maximum startup candle count (the largest number of candles
    # passed to any single indicator)
    startup_candle_count: int = 20

    def informative_pairs(self):
        whitelist_pairs = self.dp.current_whitelist()
        corr_pairs = self.config["freqai"]["feature_parameters"]["include_corr_pairlist"]
        informative_pairs = []
        for tf in self.config["freqai"]["feature_parameters"]["include_timeframes"]:
            for pair in whitelist_pairs:
                informative_pairs.append((pair, tf))
            for pair in corr_pairs:
                if pair in whitelist_pairs:
                    continue  # avoid duplication
                informative_pairs.append((pair, tf))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # the model will return all labels created by user in `populate_any_indicators`
        # (& appended targets), an indication of whether or not the prediction should be accepted,
        # the target mean/std values for each of the labels created by user in
        # `populate_any_indicators()` for each training period.

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):
        """
        Function designed to automatically generate, name and merge features
        from user indicated timeframes in the configuration file. User controls the indicators
        passed to the training/prediction by prepending indicators with `'%-' + coin `
        (see convention below). I.e. user should not prepend any supporting metrics
        (e.g. bb_lowerband below) with % unless they explicitly want to pass that metric to the
        model.
        :param pair: pair to be used as informative
        :param df: strategy dataframe which will receive merges from informatives
        :param tf: timeframe of the dataframe which will modify the feature names
        :param informative: the dataframe associated with the informative pair
        :param coin: the name of the coin which will modify the feature names.
        """

        coin = pair.split('/')[0]

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:
            t = int(t)
            informative[f"%-{coin}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%-{coin}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            informative[f"%-{coin}adx-period_{t}"] = ta.ADX(informative, window=t)

        indicators = [col for col in informative if col.startswith("%")]
        # This loop duplicates and shifts all indicators to add a sense of recency to data
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        df = df.drop(columns=skip_columns)

        # Add generalized indicators here (because in live, it will call this
        # function to populate indicators during training). Notice how we ensure not to
        # add them multiple times
        if set_generalized_indicators:

            # user adds targets here by prepending them with &- (see convention below)
            # If user wishes to use multiple targets, a multioutput prediction model
            # needs to be used such as templates/CatboostPredictionMultiModel.py
            df["&-s_close"] = (
                df["close"]
                .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
                .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
                .mean()
                / df["close"]
                - 1
            )

        return df


```

Notice how the `populate_any_indicators()` is where the user adds their own features ([more information](freqai-feature-engineering.md#feature-engineering)) and labels ([more information](freqai-running.md#setting-classifier-targets)). See a full example at `templates/FreqaiExampleStrategy.py`. 

Another structure to consider is the location of the labels at the bottom of the example function (below `if set_generalized_indicators:`). This is where the user will add single features and labels to their feature set to avoid duplication of them from various configuration parameters that multiply the feature set, such as `include_timeframes`.

!!! Note
    Features **must** be defined in `populate_any_indicators()`. Defining FreqAI features in `populate_indicators()`
    will cause the algorithm to fail in live/dry mode. If the user wishes to add generalized features that are not associated with
    a specific pair or timeframe, they should use the following structure inside `populate_any_indicators()`
    (as exemplified in `freqtrade/templates/FreqaiExampleStrategy.py`):

    ```python
        def populate_any_indicators(self, metadata, pair, df, tf, informative=None, coin="", set_generalized_indicators=False):

            ...

            # Add generalized indicators here (because in live, it will call only this function to populate
            # indicators for retraining). Notice how we ensure not to add them multiple times by associating
            # these generalized indicators to the basepair/timeframe
            if set_generalized_indicators:
                df['%-day_of_week'] = (df["date"].dt.dayofweek + 1) / 7
                df['%-hour_of_day'] = (df['date'].dt.hour + 1) / 25

                # user adds targets here by prepending them with &- (see convention below)
                # If user wishes to use multiple targets, a multioutput prediction model
                # needs to be used such as templates/CatboostPredictionMultiModel.py
                df["&-s_close"] = (
                    df["close"]
                    .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
                    .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
                    .mean()
                    / df["close"]
                    - 1
                    )
    ```

    (Please see the example script located in `freqtrade/templates/FreqaiExampleStrategy.py` for a full example of `populate_any_indicators()`.)

*Important*: The `self.freqai.start()` function cannot be called outside the `populate_indicators()`.

## Setting the `startup_candle_count`
Users need to take care to set the `startup_candle_count` in their strategy the same way they would for any normal Freqtrade strategy (see details [here](strategy-customization.md#strategy-startup-period)). This value is used by Freqtrade to ensure that a sufficient amount of data is provided when calling on the `dataprovider` to avoid any NaNs at the beginning of the first training. Users can easily set this value by identifying the longest period (in candle units) that they pass to their indicator creation functions (e.g. talib functions). In the present example, the user would pass 20 to as this value (since it is the maximum value in their `indicators_periods_candles`).

!!! Note
    Typically it is best for users to be safe and multiply their expected `startup_candle_count` by 2. There are instances where the talib functions actually require more data than just the passed `period`. Anecdotally, multiplying the `startup_candle_count` by 2 always leads to a fully NaN free training dataset. Look out for this log message to confirm that your data is clean:

    ```
    2022-08-31 15:14:04 - freqtrade.freqai.data_kitchen - INFO - dropped 0 training points due to NaNs in populated dataset 4319.
    ```

## Creating a dynamic target threshold

The `&*_std/mean` return values describe the statistical fit of the user defined label *during the most recent training*. This value allows the user to know the rarity of a given prediction. For example, `templates/FreqaiExampleStrategy.py`, creates a `target_roi` which is based on filtering out predictions that are below a given z-score of 1.25.

```python
dataframe["target_roi"] = dataframe["&-s_close_mean"] + dataframe["&-s_close_std"] * 1.25
dataframe["sell_roi"] = dataframe["&-s_close_mean"] - dataframe["&-s_close_std"] * 1.25
```

If the user wishes to consider the population
of *historical predictions* for creating the dynamic target instead of the trained labels, (as discussed above) the user
can do so by setting `fit_live_prediction_candles` in the config to the number of historical prediction candles
the user wishes to use to generate target statistics.

```json
    "freqai": {
        "fit_live_prediction_candles": 300,
    }
```

If the user sets this value, FreqAI will initially use the predictions from the training data
and subsequently begin introducing real prediction data as it is generated. FreqAI will save
this historical data to be reloaded if the user stops and restarts a model with the same `identifier`.


## Parameter table

The table below will list all configuration parameters available for FreqAI, presented in the same order as `config_examples/config_freqai.example.json`.

Mandatory parameters are marked as **Required**, which means that they are required to be set in one of the possible ways.

|  Parameter | Description |
|------------|-------------|
|  |  **General configuration parameters**
| `freqai` | **Required.** <br> The parent dictionary containing all the parameters for controlling FreqAI. <br> **Datatype:** Dictionary.
| `purge_old_models` | Delete obsolete models (otherwise, all historic models will remain on disk). <br> **Datatype:** Boolean. Default: `False`.
| `train_period_days` | **Required.** <br> Number of days to use for the training data (width of the sliding window). <br> **Datatype:** Positive integer.
| `backtest_period_days` | **Required.** <br> Number of days to inference from the trained model before sliding the window defined above, and retraining the model. This can be fractional days, but beware that the user-provided `timerange` will be divided by this number to yield the number of trainings necessary to complete the backtest. <br> **Datatype:** Float.
| `save_backtest_models` | Backtesting operates most efficiently by saving the prediction data and reusing them directly for subsequent runs (when users wish to tune entry/exit parameters). If a user wishes to save models to disk when running backtesting, they should activate `save_backtest_models`. A user may wish to do this if they plan to use the same model files for starting a dry/live instance with the same `identifier`. <br> **Datatype:** Boolean. Default: `False`.
| `identifier` | **Required.** <br> A unique name for the current model. This can be reused to reload pre-trained models/data. <br> **Datatype:** String.
| `live_retrain_hours` | Frequency of retraining during dry/live runs. <br> Default set to 0, which means the model will retrain as often as possible. <br> **Datatype:** Float > 0.
| `expiration_hours` | Avoid making predictions if a model is more than `expiration_hours` old. <br> Defaults set to 0, which means models never expire. <br> **Datatype:** Positive integer.
| `fit_live_predictions_candles` | Number of historical candles to use for computing target (label) statistics from prediction data, instead of from the training data set. <br> **Datatype:** Positive integer.
| `follow_mode` | If true, this instance of FreqAI will look for models associated with `identifier` and load those for inferencing. A `follower` will **not** train new models. <br> **Datatype:** Boolean. Default: `False`.
| `continual_learning` | If true, FreqAI will start training new models from the final state of the most recently trained model. <br> **Datatype:** Boolean. Default: `False`.
|  |  **Feature parameters**
| `feature_parameters` | A dictionary containing the parameters used to engineer the feature set. Details and examples are shown [here](#feature-engineering). <br> **Datatype:** Dictionary.
| `include_timeframes` | A list of timeframes that all indicators in `populate_any_indicators` will be created for. The list is added as features to the base asset feature set. <br> **Datatype:** List of timeframes (strings).
| `include_corr_pairlist` | A list of correlated coins that FreqAI will add as additional features to all `pair_whitelist` coins. All indicators set in `populate_any_indicators` during feature engineering (see details [here](#feature-engineering)) will be created for each coin in this list, and that set of features is added to the base asset feature set. <br> **Datatype:** List of assets (strings).
| `label_period_candles` | Number of candles into the future that the labels are created for. This is used in `populate_any_indicators` (see `templates/FreqaiExampleStrategy.py` for detailed usage). The user can create custom labels, making use of this parameter or not. <br> **Datatype:** Positive integer.
| `include_shifted_candles` | Add features from previous candles to subsequent candles to add historical information. FreqAI takes all features from the `include_shifted_candles` previous candles, duplicates and shifts them so that the information is available for the subsequent candle. <br> **Datatype:** Positive integer.
| `weight_factor` | Used to set weights for training data points according to their recency. See details about how it works [here](#controlling-the-model-learning-process). <br> **Datatype:** Positive float (typically < 1).
| `indicator_max_period_candles` | **No longer used**. User must use the strategy set `startup_candle_count` which defines the maximum *period* used in `populate_any_indicators()` for indicator creation (timeframe independent). FreqAI uses this information in combination with the maximum timeframe to calculate how many data points it should download so that the first data point does not have a NaN <br> **Datatype:** positive integer.
| `indicator_periods_candles` | Calculate indicators for `indicator_periods_candles` time periods and add them to the feature set. <br> **Datatype:** List of positive integers.
| `stratify_training_data` | This value is used to indicate the grouping of the data. For example, 2 would set every 2nd data point into a separate dataset to be pulled from during training/testing. See details about how it works [here](#stratifying-the-data-for-training-and-testing-the-model) <br> **Datatype:** Positive integer.
| `principal_component_analysis` | Automatically reduce the dimensionality of the data set using Principal Component Analysis. See details about how it works [here](#reducing-data-dimensionality-with-principal-component-analysis) <br> **Datatype:** Boolean.
| `DI_threshold` | Activates the Dissimilarity Index for outlier detection when > 0. See details about how it works [here](#removing-outliers-with-the-dissimilarity-index). <br> **Datatype:** Positive float (typically < 1).
| `use_SVM_to_remove_outliers` | Train a support vector machine to detect and remove outliers from the training data set, as well as from incoming data points. See details about how it works [here](#removing-outliers-using-a-support-vector-machine-svm). <br> **Datatype:** Boolean.
| `svm_params` | All parameters available in Sklearn's `SGDOneClassSVM()`. See details about some select parameters [here](#removing-outliers-using-a-support-vector-machine-svm). <br> **Datatype:** Dictionary.
| `use_DBSCAN_to_remove_outliers` | Cluster data using DBSCAN to identify and remove outliers from training and prediction data. See details about how it works [here](#removing-outliers-with-dbscan). <br> **Datatype:** Boolean. 
| `inlier_metric_window` | If set, FreqAI will add the `inlier_metric` to the training feature set and set the lookback to be the `inlier_metric_window`. Details of how the `inlier_metric` is computed can be found [here](#using-the-inliermetric) <br> **Datatype:** int. Default: 0
| `noise_standard_deviation` | If > 0, FreqAI adds noise to the training features. FreqAI generates random deviates from a gaussian distribution with a standard deviation of `noise_standard_deviation` and adds them to all data points. Value should be kept relative to the normalized space between -1 and 1). In other words, since data is always normalized between -1 and 1 in FreqAI, the user can expect a `noise_standard_deviation: 0.05` to see 32% of data randomly increased/decreased by more than 2.5% (i.e. the percent of data falling within the first standard deviation). Good for preventing overfitting. <br> **Datatype:** int. Default: 0
| `outlier_protection_percentage` | If more than `outlier_protection_percentage` % of points are detected as outliers by the SVM or DBSCAN, FreqAI will log a warning message and ignore outlier detection while keeping the original dataset intact. If the outlier protection is triggered, no predictions will be made based on the training data. <br> **Datatype:** Float. Default: `30`
| `reverse_train_test_order` | If true, FreqAI will train on the latest data split and test on historical split of the data. This allows the model to be trained up to the most recent data point, while avoiding overfitting. However, users should be careful to understand unorthodox nature of this parameter before employing it. <br> **Datatype:** Boolean. Default: False
|  |  **Data split parameters**
| `data_split_parameters` | Include any additional parameters available from Scikit-learn `test_train_split()`, which are shown [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) (external website). <br> **Datatype:** Dictionary.
| `test_size` | Fraction of data that should be used for testing instead of training. <br> **Datatype:** Positive float < 1.
| `shuffle` | Shuffle the training data points during training. Typically, for time-series forecasting, this is set to `False`. <br> **Datatype:** Boolean.
|  |  **Model training parameters**
| `model_training_parameters` | A flexible dictionary that includes all parameters available by the user selected model library. For example, if the user uses `LightGBMRegressor`, this dictionary can contain any parameter available by the `LightGBMRegressor` [here](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) (external website). If the user selects a different model, this dictionary can contain any parameter from that model.  <br> **Datatype:** Dictionary.
| `n_estimators` | The number of boosted trees to fit in regression. <br> **Datatype:** Integer.
| `learning_rate` | Boosting learning rate during regression. <br> **Datatype:** Float.
| `n_jobs`, `thread_count`, `task_type` | Set the number of threads for parallel processing and the `task_type` (`gpu` or `cpu`). Different model libraries use different parameter names. <br> **Datatype:** Float.
|  |  **Extraneous parameters**
| `keras` | If your model makes use of Keras (typical for Tensorflow-based prediction models), activate this flag so that the model save/loading follows Keras standards. <br> **Datatype:** Boolean. Default: `False`.
| `conv_width` | The width of a convolutional neural network input tensor. This replaces the need for shifting candles (`include_shifted_candles`) by feeding in historical data points as the second dimension of the tensor. Technically, this parameter can also be used for regressors, but it only adds computational overhead and does not change the model training/prediction. <br> **Datatype:** Integer. Default: 2.

## Important dataframe key patterns

Below are the values the user can expect to include/use inside a typical strategy dataframe (`df[]`):

|  DataFrame Key | Description |
|------------|-------------|
| `df['&*']` | Any dataframe column prepended with `&` in `populate_any_indicators()` is treated as a training target (label) inside FreqAI (typically following the naming convention `&-s*`). The names of these dataframe columns are fed back to the user as the predictions. For example, if the user wishes to predict the price change in the next 40 candles (similar to `templates/FreqaiExampleStrategy.py`), they set `df['&-s_close']`. FreqAI makes the predictions and gives them back under the same key (`df['&-s_close']`) to be used in `populate_entry/exit_trend()`. <br> **Datatype:** Depends on the output of the model.
| `df['&*_std/mean']` | Standard deviation and mean values of the user-defined labels during training (or live tracking with `fit_live_predictions_candles`). Commonly used to understand the rarity of a prediction (use the z-score as shown in `templates/FreqaiExampleStrategy.py` to evaluate how often a particular prediction was observed during training or historically with `fit_live_predictions_candles`). <br> **Datatype:** Float.
| `df['do_predict']` | Indication of an outlier data point. The return value is integer between -1 and 2, which lets the user know if the prediction is trustworthy or not. `do_predict==1` means the prediction is trustworthy. If the Dissimilarity Index (DI, see details [here](#removing-outliers-with-the-dissimilarity-index)) of the input data point is above the user-defined threshold, FreqAI will subtract 1 from `do_predict`, resulting in `do_predict==0`. If `use_SVM_to_remove_outliers()` is active, the Support Vector Machine (SVM) may also detect outliers in training and prediction data. In this case, the SVM will also subtract 1 from `do_predict`. If the input data point was considered an outlier by the SVM but not by the DI, the result will be `do_predict==0`. If both the DI and the SVM considers the input data point to be an outlier, the result will be `do_predict==-1`. A particular case is when `do_predict == 2`, which means that the model has expired due to exceeding `expired_hours`. <br> **Datatype:** Integer between -1 and 2.
| `df['DI_values']` | Dissimilarity Index values are proxies to the level of confidence FreqAI has in the prediction. A lower DI means the prediction is close to the training data, i.e., higher prediction confidence. <br> **Datatype:** Float.
| `df['%*']` | Any dataframe column prepended with `%` in `populate_any_indicators()` is treated as a training feature. For example, the user can include the RSI in the training feature set (similar to in `templates/FreqaiExampleStrategy.py`) by setting `df['%-rsi']`. See more details on how this is done [here](#feature-engineering). <br> **Note**: Since the number of features prepended with `%` can multiply very quickly (10s of thousands of features is easily engineered using the multiplictative functionality described in the `feature_parameters` table shown above), these features are removed from the dataframe upon return from FreqAI. If the user wishes to keep a particular type of feature for plotting purposes, they can prepend it with `%%`. <br> **Datatype:** Depends on the output of the model.

## Building a custom prediction model

FreqAI has multiple example prediction model libraries, such as `Catboost` regression (`freqai/prediction_models/CatboostRegressor.py`), `LightGBM`, `XGBoost` regression. However, the user can customize and create their own prediction models using the `IFreqaiModel` class. The user is encouraged to inherit `fit()`, `train()` and `predict()` to let them customize various aspects of their training procedures.

## Setting classifier targets

`FreqAI` includes a variety of classifiers, such as the `CatboostClassifier` via the flag `--freqaimodel CatboostClassifier`. If the user elects to use a classifier, they must ensure the classes are set using strings. For example:

```python
df['&s-up_or_down'] = np.where( df["close"].shift(-100) > df["close"], 'up', 'down')
```

Additionally, the example classifier models do not accommodate multiple labels, but they do allow multi-class classification within a single label column.
