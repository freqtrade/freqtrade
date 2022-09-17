# Data handling

`FreqAI` aims to organize model files, prediction data, and meta data in a way that simplifies post-processing and enhances crash recililence by automatic data reloading. The data is saved in a file structure,`user_data_dir/models/`, which contains all the data associated with the trainings and backtests. The `FreqaiDataKitchen()` relies heavily on the file structure for proper training and inferencing and should therefore not be manually modified.

## File structure

The file structure is automatically generated based on the model `identifier` set by the user in the [config](freqai-configuration.md#setting-up-the-configuration-file). The following structure shows where the data is stored for post processing:

| Structure | Description |
|-----------|-------------|
| `config_*.json` | A copy of the model specific configuration file. |
| `historic_predictions.pkl` | A file containing all historic predictions generated during the lifetime of the `identifier` model during live deployment. `historic_predictions.pkl` is used to reload the model after a crash or a config change. A backup file is always held incase of corruption on the main file. **FreqAI automatically detects corruption and replaces the corrupted file with the backup**. |
| `pair_dictionary.json` | A file containing the training queue as well as the on disk location of the most recently trained model. |
| `sub-train-*_TIMESTAMP` | A folder containing all the files associated with a single model, such as: <br>
|| `*_metadata.json` - Metadata for the model, such as normalization max/mins, expected training feature list, etc. <br>
|| `*_model.*` - The model file saved to disk for reloading from a crash. Can be `joblib` (typical boosting libs), `zip` (stable_baselines), `hd5` (keras type), etc. <br>
|| `*_pca_object.pkl` - The [Principal component analysis (PCA)](freqai-feature-engineering.md#data-dimensionality-reduction-with-principal-component-analysis) transform (if the user set `principal_component_analysis: true` in their config) which will be used to transform unseen prediction features. <br>
|| `*_svm_model.pkl` - The [Support Vector Machine (SVM)](freqai-outlier-detection.md#identifying-outliers-using-a-support-vector-machine-svm) model which is used to detect outliers in unseen prediction features. <br>
|| `*_trained_df.pkl` - The dataframe containing all the training features used to train the `identifier` model. This is used for computing the [Dissimilarity Index (DI)](freqai-outlier-detection.md#identifying-outliers-with-the-dissimilarity-index-di) and can also be used for post-processing. <br>
|| `*_trained_dates.df.pkl` - The dates associated with the `trained_df.pkl`, which is useful for post-processing. |

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

## Live deployments

### Automatic data download

FreqAI automatically downloads the proper amount of data needed to ensure training of a model through the user defined `train_period_days` and `startup_candle_count` (see the [parameter table](freqai-parameter-table.md) for detailed descriptions of these parameters). 

### Saving prediction data

All predictions made during the lifetime of a specific `identifier` model are stored in `historical_predictions.pkl` to allow for reloading after a crash or changes made to the config.

### Purging old model data

FreqAI stores new model files after each successful training. These files become obsolete as new models are generated to adapt to new market conditions. The user who is planning to leave FreqAI running for extended periods of time with high frequency retraining should enable `purge_old_models` in their config:

```json
    "freqai": {
        "purge_old_models": true,
    }
```

This will automatically purge all models older than the two most recently trained ones to save disk space.

### Returning additional info from training

The user may find that there are important metrics that they would like to return to their strategy at the end of each model training. Such metrics are returned by assigning them to `dk.data['extra_returns_per_train']['my_new_value'] = XYZ` inside the user's custom prediction model class. 

FreqAI takes the `my_new_value` assigned in this dictionary and expands it to fit the dataframe that is returned to the strategy. The user can then use the returned metrics in their strategy through `dataframe['my_new_value']`. An example of how return values can be used in FreqAI are the `&*_mean` and `&*_std` values that are used to [created a dynamic target threshold](freqai-configuration.md#creating-a-dynamic-target-threshold).

Another example, where the user wants to use live metrics from the trade database, is shown below:

```json
    "freqai": {
        "extra_returns_per_train": {"total_profit": 4}
    }
```

The user needs to set the standard dictionary in the config so that FreqAI can return proper dataframe shapes.  These values will likely be overridden by the prediction model, but in the case where the model has yet to set them, or needs a default initial value, the preset values are what will be returned.

## Backtesting 

### Saving prediction data

To allow the user to tweak their strategy (**not** the features!), FreqAI will automatically save the predictions during backtesting so that they can be reused for future backtests and live runs using the same `identifier` model. This provides a performance enhancement geared towards enabling **high-level hyperopting** of entry/exit criteria. 

An additional directory called `predictions`, which contains all the predictions stored in `hdf` format, will be created in the `unique-id` folder. 

If the user wishes to change their **features**, they **must** set a new `identifier` in the config to signal to `FreqAI` to train new models. 

If the user wishes to save the models generated during a particular backtest so that they can start a live deployment from one of them instead of training a new model, they must set `save_backtest_models` to `True` in their configuration file.

### Downloading data to cover the full backtest period

For live/dry deployments, FreqAI will download the necessary data automatically. However, the user who wishes to use backtesting functionality needs to download the necessary data using `download-data` (details [here](data-download.md#data-downloading)). FreqAI users need to pay careful attention to understanding how much *additional* data needs to be downloaded to ensure that they have a sufficient amount of training data *before* the start of their backtesting timerange. The amount of additional data can be roughly estimated by moving the start date of the timerange backwards by `train_period_days` and the `startup_candle_count` (see the [parameter table](freqai-parameter-table.md) for detailed descriptions of these parameters) from the beginning of the desired backtesting timerange. 

As an example, to backtest the `--timerange 20210501-20210701` using the [example config](freqai-configuration.md#setting-up-the-configuration-file) which sets `train_period_days` to 30, together with `startup_candle_count: 40` on a maximum `include_timeframes` of 1h, the start date for the downloaded data needs to be `20210501` - 30 days - 40 * 1h / 24 hours = 20210330 (31.7 days earlier than the start of the desired training timerange).
