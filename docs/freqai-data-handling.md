# Data handling

`FreqAI` aims to organize prediction data, model files, and meta data in a way that automates crash resilient reloading and simplifies post-processing. Data is organized into `user_data_dir/models/` and contains all the data associated with the trainings and backtests. This file structure is heavily controlled and inferenced by the `FreqaiDataKitchen()` and should therefore not be manually modified.

## File structure

The file structure is automatically generated based on the user set `identifier` in the configuration file. The following structure shows where the data is stored for post processing:

* `config_*.json`
  * a copy of the user submitted configuration file
* `historic_predictions.pkl`
  * all historic predictions generated during the lifetime of the `identifier` live deployment. These are also used to reload the model after a crash or a config change. A backup file is always held incase of corruption on the main file - FreqAI automatically detects corruption and replaces the corrupted file with the backup. 
* `pair_dictionary.json`
  * contains the training queue as well as the location of the most recently trained model on disk.
* `sub-train-*_TIMESTAMP`
  * a folder containing all the files associated with a single model, such as:
    * `*_metadata.json`
      * metadata for the model, such as normalization max/mins, expected training feature list, etc.
    * `*_model.*`
      * the model file saved to disk for reloading from crash. Can be `joblib` (typical boosting libs), `zip` (stable_baselines), `hd5` (keras type), etc.
    * `*_pca_object.pkl`
      * the PCA transform (if the user set `principal_component_analysis: true` in their config) which will be used to transform unseen prediction features.
    * `*_svm_model.pkl`
      * the Support Vector Machine model which is used to detect outliers in unseen prediction features.
    * `*_trained_df.pkl`
      * the dataframe containing all the training features used to train the particular model. This is used for computing the Dissimilarity Index and can be used for post-processing.
    * `*_trained_dates.df.pkl`
      * dates associated with the `trained_df.pkl`, useful for post-processing.

The example file structure would look like this:

```
├── models
│   └── unique-id
│       ├── config_freqai.example.json
│       ├── historic_predictions.backup.pkl
│       ├── historic_predictions.pkl
│       ├── pair_dictionary.json
│       ├── sub-train-1INCH_1662821319
│       │   ├── cb_1inch_1662821319_metadata.json
│       │   ├── cb_1inch_1662821319_model.joblib
│       │   ├── cb_1inch_1662821319_pca_object.pkl
│       │   ├── cb_1inch_1662821319_svm_model.joblib
│       │   ├── cb_1inch_1662821319_trained_dates_df.pkl
│       │   └── cb_1inch_1662821319_trained_df.pkl
│       ├── sub-train-1INCH_1662821371
│       │   ├── cb_1inch_1662821371_metadata.json
│       │   ├── cb_1inch_1662821371_model.joblib
│       │   ├── cb_1inch_1662821371_pca_object.pkl
│       │   ├── cb_1inch_1662821371_svm_model.joblib
│       │   ├── cb_1inch_1662821371_trained_dates_df.pkl
│       │   └── cb_1inch_1662821371_trained_df.pkl
│       ├── sub-train-ADA_1662821344
│       │   ├── cb_ada_1662821344_metadata.json
│       │   ├── cb_ada_1662821344_model.joblib
│       │   ├── cb_ada_1662821344_pca_object.pkl
│       │   ├── cb_ada_1662821344_svm_model.joblib
│       │   ├── cb_ada_1662821344_trained_dates_df.pkl
│       │   └── cb_ada_1662821344_trained_df.pkl
│       └── sub-train-ADA_1662821399
│           ├── cb_ada_1662821399_metadata.json
│           ├── cb_ada_1662821399_model.joblib
│           ├── cb_ada_1662821399_pca_object.pkl
│           ├── cb_ada_1662821399_svm_model.joblib
│           ├── cb_ada_1662821399_trained_dates_df.pkl
│           └── cb_ada_1662821399_trained_df.pkl
```

## Backtesting

When users run a backtest, `FreqAI` will automatically save the predictions to be reused for future runs under the same `identifier`. This is a performance enhancement geared towards enabling high-level hyperopting of entry/exit criteria. That means the user will see an additional directory created in their `unique-id` folder called `predictions` which contains all the predictions stored in `hdf` format. 

If users wish to change their features, they **must** use a new identifier which will signal to `FreqAI` to train new models. If users wish to save the models generated during a particular backtest so they can start a live deployment without an initial training, they must set `save_backtest_models` to `True` in their configuration file.

### Downloading data for backtesting
Live/dry instances will download the data automatically for the user, but users who wish to use backtesting functionality still need to download the necessary data using `download-data` (details [here](data-download.md#data-downloading)). FreqAI users need to pay careful attention to understanding how much *additional* data needs to be downloaded to ensure that they have a sufficient amount of training data *before* the start of their backtesting timerange. The amount of additional data can be roughly estimated by moving the start date of the timerange backwards by `train_period_days` and the `startup_candle_count` ([details](freqai-configuration.md#setting-the-startupcandlecount)) from the beginning of the desired backtesting timerange. 

As an example, if we wish to backtest the `--timerange` above of `20210501-20210701`, and we use the example config which sets `train_period_days` to 15. The startup candle count is 40 on a maximum `include_timeframes` of 1h. We would need 20210501 - 15 days - 40 * 1h / 24 hours = 20210414 (16.7 days earlier than the start of the desired training timerange).

## Live deployments

### Auto data download

`FreqAI` automatically downloads and proper amount of data to ensure it can train a model using the user defined `train_period_days` and the strategy defined `startup_candle_count`. 

### Historical predictions

The historical predictions are collected for the life-time of a single `identifier` and stored in `historical_predictions.pkl`

### Defining model expirations

During dry/live mode, FreqAI trains each coin pair sequentially (on separate threads/GPU from the main Freqtrade bot). This means that there is always an age discrepancy between models. If a user is training on 50 pairs, and each pair requires 5 minutes to train, the oldest model will be over 4 hours old. This may be undesirable if the characteristic time scale (the trade duration target) for a strategy is less than 4 hours. The user can decide to only make trade entries if the model is less than
a certain number of hours old by setting the `expiration_hours` in the config file:

```json
    "freqai": {
        "expiration_hours": 0.5,
    }
```

In the presented example config, the user will only allow predictions on models that are less than 1/2 hours old.

### Purging old model data

FreqAI stores new model files each time it retrains. These files become obsolete as new models are trained and FreqAI adapts to new market conditions. Users planning to leave FreqAI running for extended periods of time with high frequency retraining should enable `purge_old_models` in their config:

```json
    "freqai": {
        "purge_old_models": true,
    }
```

This will automatically purge all models older than the two most recently trained ones.

### Returning additional info from training

The user may find that there are some important metrics that they'd like to return to the strategy at the end of each model training.
The user can include these metrics by assigning them to `dk.data['extra_returns_per_train']['my_new_value'] = XYZ` inside their custom prediction model class. FreqAI takes the `my_new_value` assigned in this dictionary and expands it to fit the return dataframe to the strategy.
The user can then use the value in the strategy with `dataframe['my_new_value']`. An example of how this is already used in FreqAI is
the `&*_mean` and `&*_std` values, which indicate the mean and standard deviation of the particular target (label) during the most recent training.
An example, where the user wants to use live metrics from the trade database, is shown below:

```json
    "freqai": {
        "extra_returns_per_train": {"total_profit": 4}
    }
```

The user needs to set the standard dictionary in the config so that FreqAI can return proper dataframe shapes.  These values will likely be overridden by the prediction model, but in the case where the model has yet to set them, or needs a default initial value, this is the value that will be returned.
