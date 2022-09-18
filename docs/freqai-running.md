# Running FreqAI

There are two ways to train and deploy an adaptive machine learning model - live deployment and deployment for backtesting analysis. FreqAI can be used for both and allows for periodic training of models in live and backtesting, as shown in the following figure.

![freqai-window](assets/freqai_moving-window.jpg)

## Running the model live

FreqAI can be run dry/live using the following command:

```bash
freqtrade trade --strategy FreqaiExampleStrategy --config config_freqai.example.json --freqaimodel LightGBMRegressor
```

When launched, FreqAI will start training a new model based on the user's config settings. Following training, the model will be used to make predictions on incoming candles until a new model is available. New models are typically generated as often as possible, with FreqAI managing an internal queue of the coin pairs to try to keep all models equally up to date. FreqAI will always use the most recently trained model to make predictions on incoming live data. If the user does not want FreqAI to retrain new models as often as possible, they can set `live_retrain_hours` to tell FreqAI to wait at least that number of hours before training a new model. Additionally, the user can set `expired_hours` to tell FreqAI to avoid making predictions on models that are older than that number of hours.

Trained models are by default saved to disk to allow for reuse during backtesting or after a crash. The user can opt to purge old models to save disk space by setting `"purge_old_models": true` in the config.

If the user wishes to start a dry/live run from a saved backtest model (or from a previously crashed dry/live session), the user only needs to specify the `identifier` of the specific model:

```json
    "freqai": {
        "identifier": "example",
        "live_retrain_hours": 0.5
    }
```

In this case, although FreqAI will initiate with a pre-trained model, it will still check to see how much time has elapsed since the model was trained. If a full `live_retrain_hours` has elapsed since the end of the loaded model, FreqAI will start training a new model.

## Backtesting

The FreqAI backtesting module can be executed with the following command:

```bash
freqtrade backtesting --strategy FreqaiExampleStrategy --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --freqaimodel LightGBMRegressor --timerange 20210501-20210701
```

If this command has never been executed with the existing config file, FreqAI will train a new model
for each pair, for each backtesting window within the expanded `--timerange`.

Backtesting mode requires the user [download the necessary data](freqai-data-handling.md#downloading-data-to-cover-the-full-backtest-period) before deployment (unlike in dry/live mode where FreqAI handles the data downloading automatically). The user should be careful to consider that the time range of the downloaded data is more than the backtesting time range. This is because FreqAI needs data prior to the desired backtesting time range in order to train a model to be ready to make predictions on the first candle of the user-set backtesting time range. More details on how to calculate the data to download can be found [here](#deciding-the-size-of-the-sliding-training-window-and-backtesting-duration). 

!!! Note "Model reuse"
    Once the training is completed, the user can execute the backtesting again with the same config file and
    FreqAI will find the trained models and load them instead of spending time training. This is useful
    if the user wants to tweak (or even hyperopt) buy and sell criteria inside the strategy. If the user
    *wants* to retrain a new model with the same config file, then they should simply change the `identifier`.
    This way, the user can return to using any model they wish by simply specifying the `identifier`.

---

## Deciding the size of the sliding training window and backtesting duration

The user defines the backtesting timerange with the typical `--timerange` parameter in the configuration file. The duration of the sliding training window is set by `train_period_days`, whilst `backtest_period_days` is the sliding backtesting window, both in number of days (`backtest_period_days` can be
a float to indicate sub-daily retraining in live/dry mode). In the presented [example config](freqai-configuration.md#setting-up-the-configuration-file) (found in `config_examples/config_freqai.example.json`), the user is asking FreqAI to use a training period of 30 days and backtest on the subsequent 7 days. After the training of the model, FreqAI will backtest the subsequent 7 days. The "sliding window" then moves one week forward (emulating FreqAI retraining once per week in live mode) and the new model uses the previous 30 days (including the 7 days used for backtesting by the previous model) to train. This is repeated until the end of `--timerange`.  This means that if the user sets `--timerange 20210501-20210701`, FreqAI will have trained 8 separate models at the end of `--timerange` (because the full range comprises 8 weeks).

!!! Note
    Although fractional `backtest_period_days` is allowed, the user should be aware that the `--timerange` is divided by this value to determine the number of models that FreqAI will need to train in order to backtest the full range. For example, if the user wants to set a `--timerange` of 10 days, and asks for a `backtest_period_days` of 0.1, FreqAI will need to train 100 models per pair to complete the full backtest. Because of this, a true backtest of FreqAI adaptive training would take a *very* long time. The best way to fully test a model is to run it dry and let it train constantly. In this case, backtesting would take the exact same amount of time as a dry run.

## Defining model expirations

During dry/live mode, FreqAI trains each coin pair sequentially (on separate threads/GPU from the main Freqtrade bot). This means that there is always an age discrepancy between models. If a user is training on 50 pairs, and each pair requires 5 minutes to train, the oldest model will be over 4 hours old. This may be undesirable if the characteristic time scale (the trade duration target) for a strategy is less than 4 hours. The user can decide to only make trade entries if the model is less than
a certain number of hours old by setting the `expiration_hours` in the config file:

```json
    "freqai": {
        "expiration_hours": 0.5,
    }
```

In the presented example config, the user will only allow predictions on models that are less than 1/2 hours old.

## Data stratification for training and testing the model

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

The test data is used to evaluate the performance of the model after training. If the test score is high, the model is able to capture the behavior of the data well. If the test score is low, either the model does not capture the complexity of the data, the test data is significantly different from the train data, or a different type of model should be used.

## Controlling the model learning process

Model training parameters are unique to the machine learning library selected by the user. FreqAI allows the user to set any parameter for any library using the `model_training_parameters` dictionary in the user configuration file. The example config (found in `config_examples/config_freqai.example.json`) shows some of the example parameters associated with `Catboost` and `LightGBM`, but the user can add any parameters available in those libraries or any other machine learning library they choose to implement.

Data split parameters are defined in `data_split_parameters` which can be any parameters associated with Scikit-learn's `train_test_split()` function. `train_test_split()` has a parameters called `shuffle` which allows the user to shuffle the data or keep it unshuffled. This is particularly useful to avoid biasing training with temporally auto-correlated data. More details about these parameters can be found the [Scikit-learn website](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) (external website).

The FreqAI specific parameter `label_period_candles` defines the offset (number of candles into the future) used for the `labels`. In the presented [example config](freqai-configuration.md#setting-up-the-configuration-file), the user is asking for `labels` that are 24 candles in the future.

## Continual learning

The user can choose to adopt a continual learning scheme by setting `"continual_learning": true` in their configuration file. By enabling `continual_learning`, after training an initial model from scratch, subsequent trainings will start from the final model state of the preceding training. This gives the new model a "memory" of the previous state. By default, this is set to `false` which means that all new models are trained from scratch, without input from previous models. 

## Hyperopt

The user can hyperopt using the same command as for [typical Freqtrade hyperopt](hyperopt.md):

```bash
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy FreqaiExampleStrategy --freqaimodel LightGBMRegressor --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --timerange 20220428-20220507
```

`hyperopt` requires the user to have the data pre-downloaded in the same fashion as if they were doing [backtesting](#backtesting). In addition, the user must consider some restrictions when trying to hyperopt FreqAI strategies:

- The `--analyze-per-epoch` hyperopt parameter is not compatible with FreqAI.
- It's not possible to hyperopt indicators in the `populate_any_indicators()` function. This means that the user cannot optimize model parameters using hyperopt. Apart from this exception, it is possible to optimize all other [spaces](hyperopt.md#running-hyperopt-with-smaller-search-space).
- The backtesting instructions also apply to hyperopt.

The best method for combining hyperopt and FreqAI is to focus on hyperopting entry/exit thresholds/criteria. The user needs to focus on hyperopting parameters that are not used in their FreqAI features. For example, the user should not try to hyperopt rolling window lengths in their feature creation, or any part of their FreqAI config which changes predictions. In order to efficiently hyperopt the FreqAI strategy, FreqAI stores predictions as dataframes and reuses them. Hence the requirement to hyperopt entry/exit thresholds/criteria only. 

A good example of a hyperoptable parameter in FreqAI is a threshold for the [Dissimilarity Index (DI)](freqai-outlier-detection.md#identifying-outliers-with-the-dissimilarity-index-di) `DI_values` beyond which we consider data points as outliers:

```python
di_max = IntParameter(low=1, high=20, default=10, space='buy', optimize=True, load=True)
dataframe['outlier'] = np.where(dataframe['DI_values'] > self.di_max.value/10, 1, 0)
```

This specific hyperopt would help the user understand the appropriate `DI_values` for their particular parameter space.

## Setting up a follower

The user can indicate to the bot that it should not train models, but instead should look for models trained by a leader with a specific `identifier` by defining:

```json
    "freqai": {
        "follow_mode": true,
        "identifier": "example"
    }
```

In this example, the user has a leader bot with the `"identifier": "example"`. The leader bot is already running or is launched simultaneously with the follower. The follower will load models created by the leader and inference them to obtain predictions instead of training its own models.
