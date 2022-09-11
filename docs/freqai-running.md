# Running FreqAI

There are two ways to train and deploy an adaptive machine learning model. FreqAI enables live deployment as well as backtesting analyses. In both cases, a model is trained periodically, as shown in the following figure.

![freqai-window](assets/freqai_moving-window.jpg)

### Running the model live

FreqAI can be run dry/live using the following command:

```bash
freqtrade trade --strategy FreqaiExampleStrategy --config config_freqai.example.json --freqaimodel LightGBMRegressor
```

By default, FreqAI will not find any existing models and will start by training a new one
based on the user's configuration settings. Following training, the model will be used to make predictions on incoming candles until a new model is available. New models are typically generated as often as possible, with FreqAI managing an internal queue of the coin pairs to try to keep all models equally up to date. FreqAI will always use the most recently trained model to make predictions on incoming live data. If the user does not want FreqAI to retrain new models as often as possible, they can set `live_retrain_hours` to tell FreqAI to wait at least that number of hours before training a new model. Additionally, the user can set `expired_hours` to tell FreqAI to avoid making predictions on models that are older than that number of hours.

If the user wishes to start a dry/live run from a saved backtest model (or from a previously crashed dry/live session), the user only needs to reuse
the same `identifier` parameter:

```json
    "freqai": {
        "identifier": "example",
        "live_retrain_hours": 0.5
    }
```

In this case, although FreqAI will initiate with a
pre-trained model, it will still check to see how much time has elapsed since the model was trained,
and if a full `live_retrain_hours` has elapsed since the end of the loaded model, FreqAI will retrain.

### Backtesting

The FreqAI backtesting module can be executed with the following command:

```bash
freqtrade backtesting --strategy FreqaiExampleStrategy --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --freqaimodel LightGBMRegressor --timerange 20210501-20210701
```

Backtesting mode requires the user to have the data [pre-downloaded](#downloading-data-for-backtesting) (unlike in dry/live mode where FreqAI automatically downloads the necessary data). The user should be careful to consider that the time range of the downloaded data is more than the backtesting time range. This is because FreqAI needs data prior to the desired backtesting time range in order to train a model to be ready to make predictions on the first candle of the user-set backtesting time range. More details on how to calculate the data to download can be found [here](#deciding-the-sliding-training-window-and-backtesting-duration). 

If this command has never been executed with the existing config file, it will train a new model
for each pair, for each backtesting window within the expanded `--timerange`.

!!! Note "Model reuse"
    Once the training is completed, the user can execute the backtesting again with the same config file and
    FreqAI will find the trained models and load them instead of spending time training. This is useful
    if the user wants to tweak (or even hyperopt) buy and sell criteria inside the strategy. If the user
    *wants* to retrain a new model with the same config file, then they should simply change the `identifier`.
    This way, the user can return to using any model they wish by simply specifying the `identifier`.

---

### Hyperopt

Users can hyperopt using the same command as typical [hyperopt](hyperopt.md):

```bash
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy FreqaiExampleStrategy --freqaimodel LightGBMRegressor --strategy-path freqtrade/templates --config config_examples/config_freqai.example.json --timerange 20220428-20220507
```

Users need to have the data pre-downloaded in the same fashion as if they were doing a FreqAI [backtest](#backtesting). In addition, users must consider some restrictions when trying to [Hyperopt](hyperopt.md)  FreqAI strategies:

- The `--analyze-per-epoch` hyperopt parameter is not compatible with FreqAI.
- It's not possible to hyperopt indicators in `populate_any_indicators()` function. This means that the user cannot optimize model parameters using hyperopt. Apart from this exception, it is possible to optimize all other [spaces](hyperopt.md#running-hyperopt-with-smaller-search-space).
- The [Backtesting](#backtesting) instructions also apply to Hyperopt.

The best method for combining hyperopt and FreqAI is to focus on hyperopting entry/exit thresholds/criteria. Users need to focus on hyperopting parameters that are not used in their FreqAI features. For example, users should not try to hyperopt rolling window lengths in their feature creation, or any of their FreqAI config which changes predictions. In order to efficiently hyperopt the FreqAI strategy, FreqAI stores predictions as dataframes and reuses them. Hence the requirement to hyperopt entry/exit thresholds/criteria only. 

A good example of a hyperoptable parameter in FreqAI is a value for `DI_values` beyond which we consider outliers and below which we consider inliers:

```python
di_max = IntParameter(low=1, high=20, default=10, space='buy', optimize=True, load=True)
dataframe['outlier'] = np.where(dataframe['DI_values'] > self.di_max.value/10, 1, 0)
```

Which would help the user understand the appropriate Dissimilarity Index values for their particular parameter space.

### Deciding the size of the sliding training window and backtesting duration

The user defines the backtesting timerange with the typical `--timerange` parameter in the configuration file. The duration of the sliding training window is set by `train_period_days`, whilst `backtest_period_days` is the sliding backtesting window, both in number of days (`backtest_period_days` can be
a float to indicate sub-daily retraining in live/dry mode). In the presented example config, the user is asking FreqAI to use a training period of 30 days and backtest on the subsequent 7 days. This means that if the user sets `--timerange 20210501-20210701`, FreqAI will train have trained 8 separate models at the end of `--timerange` (because the full range comprises 8 weeks). After the training of the model, FreqAI will backtest the subsequent 7 days. The "sliding window" then moves one week forward (emulating FreqAI retraining once per week in live mode) and the new model uses the previous 30 days (including the 7 days used for backtesting by the previous model) to train. This is repeated until the end of `--timerange`.

!!! Note
    Although fractional `backtest_period_days` is allowed, the user should be aware that the `--timerange` is divided by this value to determine the number of models that FreqAI will need to train in order to backtest the full range. For example, if the user wants to set a `--timerange` of 10 days, and asks for a `backtest_period_days` of 0.1, FreqAI will need to train 100 models per pair to complete the full backtest. Because of this, a true backtest of FreqAI adaptive training would take a *very* long time. The best way to fully test a model is to run it dry and let it constantly train. In this case, backtesting would take the exact same amount of time as a dry run.

### Downloading data for backtesting
Live/dry instances will download the data automatically for the user, but users who wish to use backtesting functionality still need to download the necessary data using `download-data` (details [here](data-download.md#data-downloading)). FreqAI users need to pay careful attention to understanding how much *additional* data needs to be downloaded to ensure that they have a sufficient amount of training data *before* the start of their backtesting timerange. The amount of additional data can be roughly estimated by moving the start date of the timerange backwards by `train_period_days` and the `startup_candle_count` ([details](#setting-the-startupcandlecount)) from the beginning of the desired backtesting timerange. 

As an example, if we wish to backtest the `--timerange` above of `20210501-20210701`, and we use the example config which sets `train_period_days` to 15. The startup candle count is 40 on a maximum `include_timeframes` of 1h. We would need 20210501 - 15 days - 40 * 1h / 24 hours = 20210414 (16.7 days earlier than the start of the desired training timerange).

### Setting up a follower

The user can define:

```json
    "freqai": {
        "follow_mode": true,
        "identifier": "example"
    }
```

to indicate to the bot that it should not train models, but instead should look for models trained by a leader with the same `identifier`. In this example, the user has a leader bot with the `identifier: "example"`. The leader bot is already running or launching simultaneously as the follower.
The follower will load models created by the leader and inference them to obtain predictions.
