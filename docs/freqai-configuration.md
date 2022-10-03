# Configuration

FreqAI is configured through the typical [Freqtrade config file](configuration.md) and the standard [Freqtrade strategy](strategy-customization.md). Examples of FreqAI config and strategy files can be found in `config_examples/config_freqai.example.json` and `freqtrade/templates/FreqaiExampleStrategy.py`, respectively.

## Setting up the configuration file

 Although there are plenty of additional parameters to choose from, as highlighted in the [parameter table](freqai-parameter-table.md#parameter-table), a FreqAI config must at minimum include the following parameters (the parameter values are only examples):

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

A full example config is available in `config_examples/config_freqai.example.json`.

## Building a FreqAI strategy

The FreqAI strategy requires including the following lines of code in the standard [Freqtrade strategy](strategy-customization.md):

```python
    # user should define the maximum startup candle count (the largest number of candles
    # passed to any single indicator)
    startup_candle_count: int = 20

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

Notice how the `populate_any_indicators()` is where [features](freqai-feature-engineering.md#feature-engineering) and labels/targets are added. A full example strategy is available in `templates/FreqaiExampleStrategy.py`. 

Notice also the location of the labels under `if set_generalized_indicators:` at the bottom of the example. This is where single features and labels/targets should be added to the feature set to avoid duplication of them from various configuration parameters that multiply the feature set, such as `include_timeframes`.

!!! Note
    The `self.freqai.start()` function cannot be called outside the `populate_indicators()`.

!!! Note
    Features **must** be defined in `populate_any_indicators()`. Defining FreqAI features in `populate_indicators()`
    will cause the algorithm to fail in live/dry mode. In order to add generalized features that are not associated with a specific pair or timeframe, the following structure inside `populate_any_indicators()` should be used
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

    Please see the example script located in `freqtrade/templates/FreqaiExampleStrategy.py` for a full example of `populate_any_indicators()`.

## Important dataframe key patterns

Below are the values you can expect to include/use inside a typical strategy dataframe (`df[]`):

|  DataFrame Key | Description |
|------------|-------------|
| `df['&*']` | Any dataframe column prepended with `&` in `populate_any_indicators()` is treated as a training target (label) inside FreqAI (typically following the naming convention `&-s*`). For example, to predict the close price 40 candles into the future, you would set `df['&-s_close'] = df['close'].shift(-self.freqai_info["feature_parameters"]["label_period_candles"])` with `"label_period_candles": 40` in the config. FreqAI makes the predictions and gives them back under the same key (`df['&-s_close']`) to be used in `populate_entry/exit_trend()`. <br> **Datatype:** Depends on the output of the model.
| `df['&*_std/mean']` | Standard deviation and mean values of the defined labels during training (or live tracking with `fit_live_predictions_candles`). Commonly used to understand the rarity of a prediction (use the z-score as shown in `templates/FreqaiExampleStrategy.py` and explained [here](#creating-a-dynamic-target-threshold) to evaluate how often a particular prediction was observed during training or historically with `fit_live_predictions_candles`). <br> **Datatype:** Float.
| `df['do_predict']` | Indication of an outlier data point. The return value is integer between -2 and 2, which lets you know if the prediction is trustworthy or not. `do_predict==1` means that the prediction is trustworthy. If the Dissimilarity Index (DI, see details [here](freqai-feature-engineering.md#identifying-outliers-with-the-dissimilarity-index-di)) of the input data point is above the threshold defined in the config, FreqAI will subtract 1 from `do_predict`, resulting in `do_predict==0`. If `use_SVM_to_remove_outliers()` is active, the Support Vector Machine (SVM, see details [here](freqai-feature-engineering.md#identifying-outliers-using-a-support-vector-machine-svm)) may also detect outliers in training and prediction data. In this case, the SVM will also subtract 1 from `do_predict`. If the input data point was considered an outlier by the SVM but not by the DI, or vice versa, the result will be `do_predict==0`. If both the DI and the SVM considers the input data point to be an outlier, the result will be `do_predict==-1`. As with the SVM, if `use_DBSCAN_to_remove_outliers` is active, DBSCAN (see details [here](freqai-feature-engineering.md#identifying-outliers-with-dbscan)) may also detect outliers and subtract 1 from `do_predict`. Hence, if both the SVM and DBSCAN are active and identify a datapoint that was above the DI threshold as an outlier, the result will be `do_predict==-2`. A particular case is when `do_predict == 2`, which means that the model has expired due to exceeding `expired_hours`. <br> **Datatype:** Integer between -2 and 2.
| `df['DI_values']` | Dissimilarity Index (DI) values are proxies for the level of confidence FreqAI has in the prediction. A lower DI means the prediction is close to the training data, i.e., higher prediction confidence. See details about the DI [here](freqai-feature-engineering.md#identifying-outliers-with-the-dissimilarity-index-di). <br> **Datatype:** Float.
| `df['%*']` | Any dataframe column prepended with `%` in `populate_any_indicators()` is treated as a training feature. For example, you can include the RSI in the training feature set (similar to in `templates/FreqaiExampleStrategy.py`) by setting `df['%-rsi']`. See more details on how this is done [here](freqai-feature-engineering.md). <br> **Note:** Since the number of features prepended with `%` can multiply very quickly (10s of thousands of features are easily engineered using the multiplictative functionality of, e.g., `include_shifted_candles` and `include_timeframes` as described in the [parameter table](freqai-parameter-table.md)), these features are removed from the dataframe that is returned from FreqAI to the strategy. To keep a particular type of feature for plotting purposes, you would prepend it with `%%`. <br> **Datatype:** Depends on the output of the model.

## Setting the `startup_candle_count`

The `startup_candle_count` in the FreqAI strategy needs to be set up in the same way as in the standard Freqtrade strategy (see details [here](strategy-customization.md#strategy-startup-period)). This value is used by Freqtrade to ensure that a sufficient amount of data is provided when calling the `dataprovider`, to avoid any NaNs at the beginning of the first training. You can easily set this value by identifying the longest period (in candle units) which is passed to the indicator creation functions (e.g., Ta-Lib functions). In the presented example, `startup_candle_count` is 20 since this is the maximum value in `indicators_periods_candles`.

!!! Note
    There are instances where the Ta-Lib functions actually require more data than just the passed `period` or else the feature dataset gets populated with NaNs. Anecdotally, multiplying the `startup_candle_count` by 2 always leads to a fully NaN free training dataset. Hence, it is typically safest to multiply the expected `startup_candle_count` by 2. Look out for this log message to confirm that the data is clean:

    ```
    2022-08-31 15:14:04 - freqtrade.freqai.data_kitchen - INFO - dropped 0 training points due to NaNs in populated dataset 4319.
    ```

## Creating a dynamic target threshold

Deciding when to enter or exit a trade can be done in a dynamic way to reflect current market conditions. FreqAI allows you to return additional information from the training of a model (more info [here](freqai-feature-engineering.md#returning-additional-info-from-training)). For example, the `&*_std/mean` return values describe the statistical distribution of the target/label *during the most recent training*. Comparing a given prediction to these values allows you to know the rarity of the prediction. In `templates/FreqaiExampleStrategy.py`, the `target_roi` and  `sell_roi` are defined to be 1.25 z-scores away from the mean which causes predictions that are closer to the mean to be filtered out. 

```python
dataframe["target_roi"] = dataframe["&-s_close_mean"] + dataframe["&-s_close_std"] * 1.25
dataframe["sell_roi"] = dataframe["&-s_close_mean"] - dataframe["&-s_close_std"] * 1.25
```

To consider the population of *historical predictions* for creating the dynamic target instead of information from the training as discussed above, you would set `fit_live_prediction_candles` in the config to the number of historical prediction candles you wish to use to generate target statistics.

```json
    "freqai": {
        "fit_live_prediction_candles": 300,
    }
```

If this value is set, FreqAI will initially use the predictions from the training data and subsequently begin introducing real prediction data as it is generated. FreqAI will save this historical data to be reloaded if you stop and restart a model with the same `identifier`.

## Using different prediction models

FreqAI has multiple example prediction model libraries that are ready to be used as is via the flag `--freqaimodel`. These libraries include `Catboost`, `LightGBM`, and `XGBoost` regression, classification, and multi-target models, and can be found in `freqai/prediction_models/`. However, it is possible to customize and create your own prediction models using the `IFreqaiModel` class. You are encouraged to inherit `fit()`, `train()`, and `predict()` to let these customize various aspects of the training procedures.

### Setting classifier targets

FreqAI includes a variety of classifiers, such as the `CatboostClassifier` via the flag `--freqaimodel CatboostClassifier`. If you elects to use a classifier, the classes need to be set using strings. For example:

```python
df['&s-up_or_down'] = np.where( df["close"].shift(-100) > df["close"], 'up', 'down')
```

Additionally, the example classifier models do not accommodate multiple labels, but they do allow multi-class classification within a single label column.
