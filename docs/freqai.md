![freqai-logo](assets/freqai_doc_logo.svg)

# FreqAI

FreqAI is a module designed to automate a variety of tasks associated with training a predictive machine learning model to generate market forecasts given a set of input features.

Features include:

* **Self-adaptive retraining**: retrain models during [live deployments](freqai-running.md#running-the-model-live) to self-adapt to the market in an unsupervised manner.
* **Rapid feature engineering**: create large rich [feature sets](freqai-feature-engineering.md#feature-engineering) (10k+ features) based on simple user-created strategies.
* **High performance**: adaptive retraining occurs on a separate thread (or on GPU if available) from inferencing and bot trade operations. Newest models and data are kept in memory for rapid inferencing.
* **Realistic backtesting**: emulate self-adaptive retraining with a [backtesting module](freqai-running.md#backtesting) that automates past retraining.
* **Extensibility**: use the generalized and robust architecture for incorporating any [machine learning library/method](freqai-configuration.md#building-a-custom-prediction-model) available in Python. Eight examples are currently available, including classifiers, regressors, and a convolutional neural network.
* **Smart outlier removal**: remove outliers from training and prediction data sets using a variety of [outlier detection techniques](freqai-feature-engineering.md#outlier-removal).
* **Crash resilience**: store model to disk to make reloading from a crash fast and easy, and [purge obsolete files](freqai-data-handling.md#purging-old-model-data) for sustained dry/live runs.
* **Automatic data normalization**: [normalize the data](freqai-feature-engineering.md#feature-normalization) in a smart and statistically safe way.
* **Automatic data download**: compute the data download timerange and update historic data (in live deployments).
* **Cleaning of incoming data**: handle NaNs safely before training and prediction.
* **Dimensionality reduction**: reduce the size of the training data via [Principal Component Analysis](freqai-feature-engineering.md#reducing-data-dimensionality-with-principal-component-analysis).
* **Deploying bot fleets**: set one bot to train models while a fleet of [follower bots](freqai-running.md#setting-up-a-follower) inference the models and handle trades.

## Quick start

The easiest way to quickly test FreqAI is to run it in dry mode with the following command

```bash
freqtrade trade --config config_examples/config_freqai.example.json --strategy FreqaiExampleStrategy --freqaimodel LightGBMRegressor --strategy-path freqtrade/templates
```

The user will see the boot-up process of automatic data downloading, followed by simultaneous training and trading.

The example strategy, example prediction model, and example config can be found in
`freqtrade/templates/FreqaiExampleStrategy.py`, `freqtrade/freqai/prediction_models/LightGBMRegressor.py`, and
`config_examples/config_freqai.example.json`, respectively.

## General approach

The user provides FreqAI with a set of custom *base* indicators (the same way as in a typical Freqtrade strategy) as well as target values (*labels*). FreqAI trains a model to predict the target values based on the input of custom indicators, for each pair in the whitelist. These models are consistently retrained to adapt to market conditions. FreqAI offers the ability to both backtest strategies (emulating reality with periodic retraining) and deploy dry/live runs. In dry/live conditions, FreqAI can be set to constant retraining in a background thread in an effort to keep models as up to date as possible.

An overview of the algorithm is shown below, explaining the data processing pipeline and the model usage.

![freqai-algo](assets/freqai_algo.jpg)

### Important machine learning vocabulary

**Features** - the quantities with which a model is trained. All features for a single candle is stored as a vector. In FreqAI, the user builds the feature sets from anything they can construct in the strategy.

**Labels** - the target values that a model is trained toward. Each set of features is associated with a single label that is defined by the user within the strategy. These labels intentionally look into the future, and are not available to the model during dry/live/backtesting.

**Training** - the process of feeding individual feature sets, composed of historic data, with associated labels into the model with the goal of matching input feature sets to associated labels.

**Train data** - a subset of the historic data that is fed to the model during training. This data directly influences weight connections in the model.

**Test data** - a subset of the historic data that is used to evaluate the performance of the model after training. This data does not influence nodal weights within the model.

## Install prerequisites

The normal Freqtrade install process will ask the user if they wish to install FreqAI dependencies. The user should reply "yes" to this question if they wish to use FreqAI. If the user did not reply yes, they can manually install these dependencies after the install with:

``` bash
pip install -r requirements-freqai.txt
```

!!! Note
    Catboost will not be installed on arm devices (raspberry, Mac M1, ARM based VPS, ...), since Catboost does not provide wheels for this platform.

### Usage with docker

For docker users, a dedicated tag with freqAI dependencies is available as `:freqai`. As such - you can replace the image line in your docker-compose file with `image: freqtradeorg/freqtrade:develop_freqai`. This image contains the regular freqAI dependencies. Similar to native installs, Catboost will not be available on ARM based devices.

## Credits

FreqAI was developed by a group of individuals who all contributed specific skillsets to the project.

Conception and software development:
Robert Caulk @robcaulk

Theoretical brainstorming, data analysis:
Elin Törnquist @th0rntwig

Code review, software architecture brainstorming:
@xmatthias

Development:
Wagner Costa @wagnercosta

Beta testing and bug reporting:
Stefan Gehring @bloodhunter4rc, @longyu, @paranoidandy, @smidelis, Ryan McMullan @smarmau,
Juha Nykänen @suikula, Johan van der Vlugt @jooopiert, Richárd Józsa @richardjosza
