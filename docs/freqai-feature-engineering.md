# Feature engineering

Feature engineering is handled within the `FreqAI` config file and the user strategy. The user adds all their `base features` to their strategy, such as `RSI`, `MFI`, `EMA`, `SMA` etc. These can be custom indicators or they can be imported from any technical-analysis library that the user can find. These features are added by the user inside the `populate_any_indicators()` method of the strategy by prepending indicators with `%`, and labels with `&`.

Users should start from an existing `populate_any_indicators()` to ensure they are following some of the conventions that help with feature engineering. Here is an example:

```python
    def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):
        """
        Function designed to automatically generate, name, and merge features
        from user-indicated timeframes in the configuration file. The user controls the indicators
        passed to the training/prediction by prepending indicators with `'%-' + coin `
        (see convention below). I.e., the user should not prepend any supporting metrics
        (e.g., bb_lowerband below) with % unless they explicitly want to pass that metric to the
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

            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(informative), window=t, stds=2.2
            )
            informative[f"{coin}bb_lowerband-period_{t}"] = bollinger["lower"]
            informative[f"{coin}bb_middleband-period_{t}"] = bollinger["mid"]
            informative[f"{coin}bb_upperband-period_{t}"] = bollinger["upper"]

            informative[f"%-{coin}bb_width-period_{t}"] = (
                informative[f"{coin}bb_upperband-period_{t}"]
                - informative[f"{coin}bb_lowerband-period_{t}"]
            ) / informative[f"{coin}bb_middleband-period_{t}"]
            informative[f"%-{coin}close-bb_lower-period_{t}"] = (
                informative["close"] / informative[f"{coin}bb_lowerband-period_{t}"]
            )

            informative[f"%-{coin}relative_volume-period_{t}"] = (
                informative["volume"] / informative["volume"].rolling(t).mean()
            )

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
            df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
            df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25

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

In the presented example strategy, the user does not wish to pass the `bb_lowerband` as a feature to the model,
and has therefore not prepended it with `%`. The user does, however, wish to pass `bb_width` to the
model for training/prediction and has therefore prepended it with `%`.

Now that the user has set their `base features`, they will next expand upon their base features using the powerful `feature_parameters` in their configuration file:

```json
    "freqai": {
        ...
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
        ...
    }
```

The `include_timeframes` in the config above are the timeframes (`tf`) of each call to `populate_any_indicators()` in the strategy. In the present case, the user is asking for the `5m`, `15m`, and `4h` timeframes of the `rsi`, `mfi`, `roc`, and `bb_width` to be included in the feature set.

The user can ask for each of the defined features to be included also from informative pairs using the `include_corr_pairlist`. This means that the feature set will include all the features from `populate_any_indicators` on all the `include_timeframes` for each of the correlated pairs defined in the config (`ETH/USD`, `LINK/USD`, and `BNB/USD`).

`include_shifted_candles` indicates the number of previous candles to include in the feature set. For example, `include_shifted_candles: 2` tells `FreqAI` to include the past 2 candles for each of the features in the feature set.

In total, the number of features the user of the presented example strat has created is: length of `include_timeframes` * no. features in `populate_any_indicators()` * length of `include_corr_pairlist` * no. `include_shifted_candles` * length of `indicator_periods_candles`
 $= 3 * 3 * 3 * 2 * 2 = 108$.


### Feature normalization

`FreqAI` is strict when it comes to data normalization - all data is always automatically normalized to the training feature space according to industry standards. This includes all test data and unseen prediction data (dry/live/backtest). `FreqAI` stores all the metadata required to ensure that prediction features will be properly normalized and that predictions are properly denormalized. For this reason, it is not recommended to eschew industry standards and modify `FreqAI` internals - however - advanced users can do so by inheriting `train()` in their custom `IFreqaiModel` and using their own normalization functions.

### Reducing data dimensionality with Principal Component Analysis

Users can reduce the dimensionality of their features by activating the `principal_component_analysis` in the config:

```json
    "freqai": {
        "feature_parameters" : {
            "principal_component_analysis": true
        }
    }
```

This will perform PCA on the features and reduce the dimensionality of the data so that the explained variance of the data set is >= 0.999.

### Stratifying the data for training and testing the model

The user can stratify (group) the training/testing data using:

```json
    "freqai": {
        "feature_parameters" : {
            "stratify_training_data": 3
        }
    }
```

This will split the data chronologically so that every Xth data point is used to test the model after training. In the
example above, the user is asking for every third data point in the dataframe to be used for
testing; the other points are used for training.

The test data is used to evaluate the performance of the model after training. If the test score is high, the model is able to capture the behavior of the data well. If the test score is low, either the model either does not capture the complexity of the data, the test data is significantly different from the train data, or a different model should be used.

### Using the `inlier_metric`

The `inlier_metric` is a metric aimed at quantifying how different a prediction data point is from the most recent historic data points. 

User can set `inlier_metric_window` to set the look back window. FreqAI will compute the distance between the present prediction point and each of the previous data points (total of `inlier_metric_window` points). 

This function goes one step further - during training, it computes the `inlier_metric` for all training data points and builds weibull distributions for each each lookback point. The cumulative distribution function for the weibull distribution is used to produce a quantile for each of the data points. The quantiles for each lookback point are averaged to create the `inlier_metric`. 

FreqAI adds this `inlier_metric` score to the training features! In other words, your model is trained to recognize how this temporal inlier metric is related to the user set labels. 

This function does **not** remove outliers from the data set.

### Controlling the model learning process

Model training parameters are unique to the machine learning library selected by the user. FreqAI allows the user to set any parameter for any library using the `model_training_parameters` dictionary in the user configuration file. The example configuration file (found in `config_examples/config_freqai.example.json`) show some of the example parameters associated with `Catboost` and `LightGBM`, but the user can add any parameters available in those libraries.

Data split parameters are defined in `data_split_parameters` which can be any parameters associated with `Sklearn`'s `train_test_split()` function.

FreqAI includes some additional parameters such as `weight_factor`, which allows the user to weight more recent data more strongly
than past data via an exponential function:

$$ W_i = \exp(\frac{-i}{\alpha*n}) $$

where $W_i$ is the weight of data point $i$ in a total set of $n$ data points. Below is a figure showing the effect of different weight factors on the data points (candles) in a feature set.

![weight-factor](assets/freqai_weight-factor.jpg)

`train_test_split()` has a parameters called `shuffle` that allows the user to keep the data unshuffled. This is particularly useful to avoid biasing training with temporally auto-correlated data.

Finally, `label_period_candles` defines the offset (number of candles into the future) used for the `labels`. In the presented example config,
the user is asking for `labels` that are 24 candles in the future.

#### Continual learning

Users can choose to adopt a "continual learning" strategy by setting `"continual_learning": true` in their configuration file. This setting will train an initial model from scratch, and subsequent trainings will start from the final model state of the preceding training. By default, this is set to `false` which trains a new model from scratch upon each subsequent training. 

### Outlier removal

#### Removing outliers with the Dissimilarity Index

The user can tell FreqAI to remove outlier data points from the training/test data sets using a Dissimilarity Index by including the following statement in the config:

```json
    "freqai": {
        "feature_parameters" : {
            "DI_threshold": 1
        }
    }
```

Equity and crypto markets suffer from a high level of non-patterned noise in the form of outlier data points. The Dissimilarity Index (DI) aims to quantify the uncertainty associated with each prediction made by the model. The DI allows predictions which are outliers (not existent in the model feature space) to be thrown out due to low levels of certainty.

To do so, FreqAI measures the distance between each training data point (feature vector), $X_{a}$, and all other training data points:

$$ d_{ab} = \sqrt{\sum_{j=1}^p(X_{a,j}-X_{b,j})^2} $$

where $d_{ab}$ is the distance between the normalized points $a$ and $b$. $p$ is the number of features, i.e., the length of the vector $X$. The characteristic distance, $\overline{d}$ for a set of training data points is simply the mean of the average distances:

$$ \overline{d} = \sum_{a=1}^n(\sum_{b=1}^n(d_{ab}/n)/n) $$

$\overline{d}$ quantifies the spread of the training data, which is compared to the distance between a new prediction feature vectors, $X_k$ and all the training data:

$$ d_k = \arg \min d_{k,i} $$

which enables the estimation of the Dissimilarity Index as:

$$ DI_k = d_k/\overline{d} $$

The user can tweak the DI through the `DI_threshold` to increase or decrease the extrapolation of the trained model.

Below is a figure that describes the DI for a 3D data set.

![DI](assets/freqai_DI.jpg)

#### Removing outliers using a Support Vector Machine (SVM)

The user can tell FreqAI to remove outlier data points from the training/test data sets using a SVM by setting:

```json
    "freqai": {
        "feature_parameters" : {
            "use_SVM_to_remove_outliers": true
        }
    }
```

FreqAI will train an SVM on the training data (or components of it if the user activated
`principal_component_analysis`) and remove any data point that the SVM deems to be beyond the feature space.

The parameter `shuffle` is by default set to `False` to ensure consistent results. If it is set to `True`, running the SVM multiple times on the same data set might result in different outcomes due to `max_iter` being to low for the algorithm to reach the demanded `tol`. Increasing `max_iter` solves this issue but causes the procedure to take longer time.

The parameter `nu`, *very* broadly, is the amount of data points that should be considered outliers.

#### Removing outliers with DBSCAN

The user can configure FreqAI to use DBSCAN to cluster and remove outliers from the training/test data set or incoming outliers from predictions, by activating `use_DBSCAN_to_remove_outliers` in the config:

```json
    "freqai": {
        "feature_parameters" : {
            "use_DBSCAN_to_remove_outliers": true
        }
    }
```

DBSCAN is an unsupervised machine learning algorithm that clusters data without needing to know how many clusters there should be.

Given a number of data points $N$, and a distance $\varepsilon$, DBSCAN clusters the data set by setting all data points that have $N-1$ other data points within a distance of $\varepsilon$ as *core points*. A data point that is within a distance of $\varepsilon$ from a *core point* but that does not have $N-1$ other data points within a distance of $\varepsilon$ from itself is considered an *edge point*. A cluster is then the collection of *core points* and *edge points*. Data points that have no other data points at a distance $<\varepsilon$ are considered outliers. The figure below shows a cluster with $N = 3$.

![dbscan](assets/freqai_dbscan.jpg)

FreqAI uses `sklearn.cluster.DBSCAN` (details are available on scikit-learn's webpage [here](#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)) with `min_samples` ($N$) taken as 1/4 of the no. of time points in the feature set, and `eps` ($\varepsilon$) taken as the elbow point in the *k-distance graph* computed from the nearest neighbors in the pairwise distances of all data points in the feature set.
