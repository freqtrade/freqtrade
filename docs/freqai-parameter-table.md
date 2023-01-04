# Parameter table

The table below will list all configuration parameters available for FreqAI. Some of the parameters are exemplified in `config_examples/config_freqai.example.json`.

Mandatory parameters are marked as **Required** and have to be set in one of the suggested ways.

### General configuration parameters

|  Parameter | Description |
|------------|-------------|
|  |  **General configuration parameters within the `config.freqai` tree**
| `freqai` | **Required.** <br> The parent dictionary containing all the parameters for controlling FreqAI. <br> **Datatype:** Dictionary.
| `train_period_days` | **Required.** <br> Number of days to use for the training data (width of the sliding window). <br> **Datatype:** Positive integer.
| `backtest_period_days` | **Required.** <br> Number of days to inference from the trained model before sliding the `train_period_days` window defined above, and retraining the model during backtesting (more info [here](freqai-running.md#backtesting)). This can be fractional days, but beware that the provided `timerange` will be divided by this number to yield the number of trainings necessary to complete the backtest. <br> **Datatype:** Float.
| `identifier` | **Required.** <br> A unique ID for the current model. If models are saved to disk, the `identifier` allows for reloading specific pre-trained models/data. <br> **Datatype:** String.
| `live_retrain_hours` | Frequency of retraining during dry/live runs. <br> **Datatype:** Float > 0. <br> Default: `0` (models retrain as often as possible).
| `expiration_hours` | Avoid making predictions if a model is more than `expiration_hours` old. <br> **Datatype:** Positive integer. <br> Default: `0` (models never expire).
| `purge_old_models` | Delete all unused models during live runs (not relevant to backtesting). If set to false (not default), dry/live runs will accumulate all unused models to disk. If <br> **Datatype:** Boolean. <br> Default: `True`.
| `save_backtest_models` | Save models to disk when running backtesting. Backtesting operates most efficiently by saving the prediction data and reusing them directly for subsequent runs (when you wish to tune entry/exit parameters). Saving backtesting models to disk also allows to use the same model files for starting a dry/live instance with the same model `identifier`. <br> **Datatype:** Boolean. <br> Default: `False` (no models are saved).
| `fit_live_predictions_candles` | Number of historical candles to use for computing target (label) statistics from prediction data, instead of from the training dataset (more information can be found [here](freqai-configuration.md#creating-a-dynamic-target-threshold)). <br> **Datatype:** Positive integer.
| `follow_mode` | Use a `follower` that will look for models associated with a specific `identifier` and load those for inferencing. A `follower` will **not** train new models. <br> **Datatype:** Boolean. <br> Default: `False`.
| `continual_learning` | Use the final state of the most recently trained model as starting point for the new model, allowing for incremental learning (more information can be found [here](freqai-running.md#continual-learning)). <br> **Datatype:** Boolean. <br> Default: `False`.
| `write_metrics_to_disk` | Collect train timings, inference timings and cpu usage in json file. <br> **Datatype:** Boolean. <br> Default: `False`
| `data_kitchen_thread_count` | <br> Designate the number of threads you want to use for data processing (outlier methods, normalization, etc.). This has no impact on the number of threads used for training. If user does not set it (default), FreqAI will use max number of threads - 2 (leaving 1 physical core available for Freqtrade bot and FreqUI) <br> **Datatype:** Positive integer.

### Feature parameters

|  Parameter | Description |
|------------|-------------|
|  |  **Feature parameters within the `freqai.feature_parameters` sub dictionary**
| `feature_parameters` | A dictionary containing the parameters used to engineer the feature set. Details and examples are shown [here](freqai-feature-engineering.md). <br> **Datatype:** Dictionary.
| `include_timeframes` | A list of timeframes that all indicators in `feature_engineering_expand_*()` will be created for. The list is added as features to the base indicators dataset. <br> **Datatype:** List of timeframes (strings).
| `include_corr_pairlist` | A list of correlated coins that FreqAI will add as additional features to all `pair_whitelist` coins. All indicators set in `feature_engineering_expand_*()` during feature engineering (see details [here](freqai-feature-engineering.md)) will be created for each correlated coin. The correlated coins features are added to the base indicators dataset. <br> **Datatype:** List of assets (strings).
| `label_period_candles` | Number of candles into the future that the labels are created for. This is used in `feature_engineering_expand_all()` (see `templates/FreqaiExampleStrategy.py` for detailed usage). You can create custom labels and choose whether to make use of this parameter or not. <br> **Datatype:** Positive integer.
| `include_shifted_candles` | Add features from previous candles to subsequent candles with the intent of adding historical information. If used, FreqAI will duplicate and shift all features from the `include_shifted_candles` previous candles so that the information is available for the subsequent candle. <br> **Datatype:** Positive integer.
| `weight_factor` | Weight training data points according to their recency (see details [here](freqai-feature-engineering.md#weighting-features-for-temporal-importance)). <br> **Datatype:** Positive float (typically < 1).
| `indicator_max_period_candles` | **No longer used (#7325)**. Replaced by `startup_candle_count` which is set in the [strategy](freqai-configuration.md#building-a-freqai-strategy). `startup_candle_count` is timeframe independent and defines the maximum *period* used in `feature_engineering_*()` for indicator creation. FreqAI uses this parameter together with the maximum timeframe in `include_time_frames` to calculate how many data points to download such that the first data point does not include a NaN. <br> **Datatype:** Positive integer.
| `indicator_periods_candles` | Time periods to calculate indicators for. The indicators are added to the base indicator dataset. <br> **Datatype:** List of positive integers.
| `principal_component_analysis` | Automatically reduce the dimensionality of the data set using Principal Component Analysis. See details about how it works [here](#reducing-data-dimensionality-with-principal-component-analysis) <br> **Datatype:** Boolean. <br> Default: `False`.
| `plot_feature_importances` | Create a feature importance plot for each model for the top/bottom `plot_feature_importances` number of features. Plot is stored in `user_data/models/<identifier>/sub-train-<COIN>_<timestamp>.html`. <br> **Datatype:** Integer. <br> Default: `0`.
| `DI_threshold` | Activates the use of the Dissimilarity Index for outlier detection when set to > 0. See details about how it works [here](freqai-feature-engineering.md#identifying-outliers-with-the-dissimilarity-index-di). <br> **Datatype:** Positive float (typically < 1).
| `use_SVM_to_remove_outliers` | Train a support vector machine to detect and remove outliers from the training dataset, as well as from incoming data points. See details about how it works [here](freqai-feature-engineering.md#identifying-outliers-using-a-support-vector-machine-svm). <br> **Datatype:** Boolean.
| `svm_params` | All parameters available in Sklearn's `SGDOneClassSVM()`. See details about some select parameters [here](freqai-feature-engineering.md#identifying-outliers-using-a-support-vector-machine-svm). <br> **Datatype:** Dictionary.
| `use_DBSCAN_to_remove_outliers` | Cluster data using the DBSCAN algorithm to identify and remove outliers from training and prediction data. See details about how it works [here](freqai-feature-engineering.md#identifying-outliers-with-dbscan). <br> **Datatype:** Boolean. 
| `inlier_metric_window` | If set, FreqAI adds an `inlier_metric` to the training feature set and set the lookback to be the `inlier_metric_window`, i.e., the number of previous time points to compare the current candle to. Details of how the `inlier_metric` is computed can be found [here](freqai-feature-engineering.md#inlier-metric). <br> **Datatype:** Integer. <br> Default: `0`.
| `noise_standard_deviation` | If set, FreqAI adds noise to the training features with the aim of preventing overfitting. FreqAI generates random deviates from a gaussian distribution with a standard deviation of `noise_standard_deviation` and adds them to all data points. `noise_standard_deviation` should be kept relative to the normalized space, i.e., between -1 and 1. In other words, since data in FreqAI is always normalized to be between -1 and 1, `noise_standard_deviation: 0.05` would result in 32% of the data being randomly increased/decreased by more than 2.5% (i.e., the percent of data falling within the first standard deviation). <br> **Datatype:** Integer. <br> Default: `0`.
| `outlier_protection_percentage` | Enable to prevent outlier detection methods from discarding too much data. If more than `outlier_protection_percentage` % of points are detected as outliers by the SVM or DBSCAN, FreqAI will log a warning message and ignore outlier detection, i.e., the original dataset will be kept intact. If the outlier protection is triggered, no predictions will be made based on the training dataset. <br> **Datatype:** Float. <br> Default: `30`.
| `reverse_train_test_order` | Split the feature dataset (see below) and use the latest data split for training and test on historical split of the data. This allows the model to be trained up to the most recent data point, while avoiding overfitting. However, you should be careful to understand the unorthodox nature of this parameter before employing it. <br> **Datatype:** Boolean. <br> Default: `False` (no reversal).

### Data split parameters

|  Parameter | Description |
|------------|-------------|
|  |  **Data split parameters within the `freqai.data_split_parameters` sub dictionary**
| `data_split_parameters` | Include any additional parameters available from Scikit-learn `test_train_split()`, which are shown [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) (external website). <br> **Datatype:** Dictionary.
| `test_size` | The fraction of data that should be used for testing instead of training. <br> **Datatype:** Positive float < 1.
| `shuffle` | Shuffle the training data points during training. Typically, to not remove the chronological order of data in time-series forecasting, this is set to `False`. <br> **Datatype:** Boolean. <br> Defaut: `False`.

### Model training parameters

|  Parameter | Description |
|------------|-------------|
|  |  **Model training parameters within the `freqai.model_training_parameters` sub dictionary**
| `model_training_parameters` | A flexible dictionary that includes all parameters available by the selected model library. For example, if you use `LightGBMRegressor`, this dictionary can contain any parameter available by the `LightGBMRegressor` [here](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html) (external website). If you select a different model, this dictionary can contain any parameter from that model. A list of the currently available models can be found [here](freqai-configuration.md#using-different-prediction-models).  <br> **Datatype:** Dictionary.
| `n_estimators` | The number of boosted trees to fit in the training of the model. <br> **Datatype:** Integer.
| `learning_rate` | Boosting learning rate during training of the model. <br> **Datatype:** Float.
| `n_jobs`, `thread_count`, `task_type` | Set the number of threads for parallel processing and the `task_type` (`gpu` or `cpu`). Different model libraries use different parameter names. <br> **Datatype:** Float.

### Reinforcement Learning parameters

|  Parameter | Description |
|------------|-------------|
|  |  **Reinforcement Learning Parameters within the `freqai.rl_config` sub dictionary**
| `rl_config` | A dictionary containing the control parameters for a Reinforcement Learning model. <br> **Datatype:** Dictionary.
| `train_cycles` | Training time steps will be set based on the `train_cycles * number of training data points. <br> **Datatype:** Integer.
| `cpu_count` | Number of processors to dedicate to the Reinforcement Learning training process. <br> **Datatype:** int.
| `max_trade_duration_candles`| Guides the agent training to keep trades below desired length. Example usage shown in `prediction_models/ReinforcementLearner.py` within the customizable `calculate_reward()` function. <br> **Datatype:** int.
| `model_type` | Model string from stable_baselines3 or SBcontrib. Available strings include: `'TRPO', 'ARS', 'RecurrentPPO', 'MaskablePPO', 'PPO', 'A2C', 'DQN'`. User should ensure that `model_training_parameters` match those available to the corresponding stable_baselines3 model by visiting their documentaiton. [PPO doc](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) (external website) <br> **Datatype:** string.
| `policy_type` | One of the available policy types from stable_baselines3 <br> **Datatype:** string.
| `max_training_drawdown_pct` | The maximum drawdown that the agent is allowed to experience during training. <br> **Datatype:** float. <br> Default: 0.8
| `cpu_count` | Number of threads/cpus to dedicate to the Reinforcement Learning training process (depending on if `ReinforcementLearning_multiproc` is selected or not). Recommended to leave this untouched, by default, this value is set to the total number of physical cores minus 1. <br> **Datatype:** int. 
| `model_reward_parameters` | Parameters used inside the customizable `calculate_reward()` function in `ReinforcementLearner.py` <br> **Datatype:** int.
| `add_state_info` | Tell FreqAI to include state information in the feature set for training and inferencing. The current state variables include trade duration, current profit, trade position. This is only available in dry/live runs, and is automatically switched to false for backtesting. <br> **Datatype:** bool. <br> Default: `False`.
| `net_arch` | Network architecture which is well described in [`stable_baselines3` doc](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#examples). In summary: `[<shared layers>, dict(vf=[<non-shared value network layers>], pi=[<non-shared policy network layers>])]`. By default this is set to `[128, 128]`, which defines 2 shared hidden layers with 128 units each.
| `randomize_starting_position` | Randomize the starting point of each episode to avoid overfitting. <br> **Datatype:** bool. <br> Default: `False`.

### Additional parameters

|  Parameter | Description |
|------------|-------------|
|  |  **Extraneous parameters**
| `freqai.keras` | If the selected model makes use of Keras (typical for Tensorflow-based prediction models), this flag needs to be activated so that the model save/loading follows Keras standards. <br> **Datatype:** Boolean. <br> Default: `False`.
| `freqai.conv_width` | The width of a convolutional neural network input tensor. This replaces the need for shifting candles (`include_shifted_candles`) by feeding in historical data points as the second dimension of the tensor. Technically, this parameter can also be used for regressors, but it only adds computational overhead and does not change the model training/prediction. <br> **Datatype:** Integer. <br> Default: `2`.
| `freqai.reduce_df_footprint` | Recast all numeric columns to float32/int32, with the objective of reducing ram/disk usage and decreasing train/inference timing. This parameter is set in the main level of the Freqtrade configuration file (not inside FreqAI). <br> **Datatype:** Boolean. <br> Default: `False`.
