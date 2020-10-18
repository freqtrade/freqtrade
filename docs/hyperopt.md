# Hyperopt

This page explains how to tune your strategy by finding the optimal
parameters, a process called hyperparameter optimization. The bot uses several
algorithms included in the `scikit-optimize` package to accomplish this. The
search will burn all your CPU cores, make your laptop sound like a fighter jet
and still take a long time.

In general, the search for best parameters starts with a few random combinations (see [below](#reproducible-results) for more details) and then uses Bayesian search with a ML regressor algorithm (currently ExtraTreesRegressor) to quickly find a combination of parameters in the search hyperspace that minimizes the value of the [loss function](#loss-functions).

Hyperopt requires historic data to be available, just as backtesting does.
To learn how to get data for the pairs and exchange you're interested in, head over to the [Data Downloading](data-download.md) section of the documentation.

!!! Bug
    Hyperopt can crash when used with only 1 CPU Core as found out in [Issue #1133](https://github.com/freqtrade/freqtrade/issues/1133)

## Install hyperopt dependencies

Since Hyperopt dependencies are not needed to run the bot itself, are heavy, can not be easily built on some platforms (like Raspberry PI), they are not installed by default. Before you run Hyperopt, you need to install the corresponding dependencies, as described in this section below.

!!! Note
    Since Hyperopt is a resource intensive process, running it on a Raspberry Pi is not recommended nor supported.

### Docker

The docker-image includes hyperopt dependencies, no further action needed.

### Easy installation script (setup.sh) / Manual installation

```bash
source .env/bin/activate
pip install -r requirements-hyperopt.txt
```

## Prepare Hyperopting

Before we start digging into Hyperopt, we recommend you to take a look at
the sample hyperopt file located in [user_data/hyperopts/](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt.py).

Configuring hyperopt is similar to writing your own strategy, and many tasks will be similar.

!!! Tip "About this page"
    For this page, we will be using a fictional strategy called `AwesomeStrategy` - which will be optimized using the `AwesomeHyperopt` class.

The simplest way to get started is to use the following, command, which will create a new hyperopt file from a template, which will be located under `user_data/hyperopts/AwesomeHyperopt.py`.

``` bash
freqtrade new-hyperopt --hyperopt AwesomeHyperopt
```

### Hyperopt checklist

Checklist on all tasks / possibilities in hyperopt

Depending on the space you want to optimize, only some of the below are required:

* fill `buy_strategy_generator` - for buy signal optimization
* fill `indicator_space` - for buy signal optimization
* fill `sell_strategy_generator` - for sell signal optimization
* fill `sell_indicator_space` - for sell signal optimization

!!! Note
    `populate_indicators` needs to create all indicators any of thee spaces may use, otherwise hyperopt will not work.

Optional in hyperopt - can also be loaded from a strategy (recommended):

* copy `populate_indicators` from your strategy - otherwise default-strategy will be used
* copy `populate_buy_trend` from your strategy - otherwise default-strategy will be used
* copy `populate_sell_trend` from your strategy - otherwise default-strategy will be used

!!! Note
    You always have to provide a strategy to Hyperopt, even if your custom Hyperopt class contains all methods.
    Assuming the optional methods are not in your hyperopt file, please use `--strategy AweSomeStrategy` which contains these methods so hyperopt can use these methods instead.

Rarely you may also need to override:

* `roi_space` - for custom ROI optimization (if you need the ranges for the ROI parameters in the optimization hyperspace that differ from default)
* `generate_roi_table` - for custom ROI optimization (if you need the ranges for the values in the ROI table that differ from default or the number of entries (steps) in the ROI table which differs from the default 4 steps)
* `stoploss_space` - for custom stoploss optimization (if you need the range for the stoploss parameter in the optimization hyperspace that differs from default)
* `trailing_space` - for custom trailing stop optimization (if you need the ranges for the trailing stop parameters in the optimization hyperspace that differ from default)

!!! Tip "Quickly optimize ROI, stoploss and trailing stoploss"
    You can quickly optimize the spaces `roi`, `stoploss` and `trailing` without changing anything (i.e. without creation of a "complete" Hyperopt class with dimensions, parameters, triggers and guards, as described in this document) from the default hyperopt template by relying on your strategy to do most of the calculations.

    ```python
    # Have a working strategy at hand.
    freqtrade new-hyperopt --hyperopt EmptyHyperopt

    freqtrade hyperopt --hyperopt EmptyHyperopt --hyperopt-loss SharpeHyperOptLossDaily --spaces roi stoploss trailing --strategy MyWorkingStrategy --config config.json -e 100
    ```

### Create a Custom Hyperopt File

Let assume you want a hyperopt file `AwesomeHyperopt.py`:  

``` bash
freqtrade new-hyperopt --hyperopt AwesomeHyperopt
```

This command will create a new hyperopt file from a template, allowing you to get started quickly.

### Configure your Guards and Triggers

There are two places you need to change in your hyperopt file to add a new buy hyperopt for testing:

* Inside `indicator_space()` - the parameters hyperopt shall be optimizing.
* Inside `populate_buy_trend()` - applying the parameters.

There you have two different types of indicators: 1. `guards` and 2. `triggers`.

1. Guards are conditions like "never buy if ADX < 10", or never buy if current price is over EMA10.
2. Triggers are ones that actually trigger buy in specific moment, like "buy when EMA5 crosses over EMA10" or "buy when close price touches lower Bollinger band".

!!! Hint "Guards and Triggers"
    Technically, there is no difference between Guards and Triggers.  
    However, this guide will make this distinction to make it clear that signals should not be "sticking".
    Sticking signals are signals that are active for multiple candles. This can lead into buying a signal late (right before the signal disappears - which means that the chance of success is a lot lower than right at the beginning).

Hyper-optimization will, for each epoch round, pick one trigger and possibly
multiple guards. The constructed strategy will be something like "*buy exactly when close price touches lower Bollinger band, BUT only if
ADX > 10*".

If you have updated the buy strategy, i.e. changed the contents of `populate_buy_trend()` method, you have to update the `guards` and `triggers` your hyperopt must use correspondingly.

#### Sell optimization

Similar to the buy-signal above, sell-signals can also be optimized.
Place the corresponding settings into the following methods

* Inside `sell_indicator_space()` - the parameters hyperopt shall be optimizing.
* Inside `populate_sell_trend()` - applying the parameters.

The configuration and rules are the same than for buy signals.
To avoid naming collisions in the search-space, please prefix all sell-spaces with `sell-`.

#### Using timeframe as a part of the Strategy

The Strategy class exposes the timeframe value as the `self.timeframe` attribute.
The same value is available as class-attribute `HyperoptName.timeframe`.
In the case of the linked sample-value this would be `AwesomeHyperopt.timeframe`.

## Solving a Mystery

Let's say you are curious: should you use MACD crossings or lower Bollinger
Bands to trigger your buys. And you also wonder should you use RSI or ADX to
help with those buy decisions. If you decide to use RSI or ADX, which values
should I use for them? So let's use hyperparameter optimization to solve this
mystery.

We will start by defining a search space:

```python
    def indicator_space() -> List[Dimension]:
        """
        Define your Hyperopt space for searching strategy parameters
        """
        return [
            Integer(20, 40, name='adx-value'),
            Integer(20, 40, name='rsi-value'),
            Categorical([True, False], name='adx-enabled'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical(['bb_lower', 'macd_cross_signal'], name='trigger')
        ]
```

Above definition says: I have five parameters I want you to randomly combine
to find the best combination. Two of them are integer values (`adx-value`
and `rsi-value`) and I want you test in the range of values 20 to 40.  
Then we have three category variables. First two are either `True` or `False`.
We use these to either enable or disable the ADX and RSI guards. The last
one we call `trigger` and use it to decide which buy trigger we want to use.

So let's write the buy strategy using these values:

```python
        def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
            conditions = []
            # GUARDS AND TRENDS
            if 'adx-enabled' in params and params['adx-enabled']:
                conditions.append(dataframe['adx'] > params['adx-value'])
            if 'rsi-enabled' in params and params['rsi-enabled']:
                conditions.append(dataframe['rsi'] < params['rsi-value'])

            # TRIGGERS
            if 'trigger' in params:
                if params['trigger'] == 'bb_lower':
                    conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
                if params['trigger'] == 'macd_cross_signal':
                    conditions.append(qtpylib.crossed_above(
                        dataframe['macd'], dataframe['macdsignal']
                    ))

            # Check that volume is not 0
            conditions.append(dataframe['volume'] > 0)

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'buy'] = 1

            return dataframe

        return populate_buy_trend
```

Hyperopt will now call `populate_buy_trend()` many times (`epochs`) with different value combinations.  
It will use the given historical data and make buys based on the buy signals generated with the above function.  
Based on the results, hyperopt will tell you which parameter combination produced the best results (based on the configured [loss function](#loss-functions)).

!!! Note
    The above setup expects to find ADX, RSI and Bollinger Bands in the populated indicators.
    When you want to test an indicator that isn't used by the bot currently, remember to
    add it to the `populate_indicators()` method in your strategy or hyperopt file.

## Loss-functions

Each hyperparameter tuning requires a target. This is usually defined as a loss function (sometimes also called objective function), which should decrease for more desirable results, and increase for bad results.

A loss function must be specified via the `--hyperopt-loss <Class-name>` argument (or optionally via the configuration under the `"hyperopt_loss"` key).
This class should be in its own file within the `user_data/hyperopts/` directory.

Currently, the following loss functions are builtin:

* `ShortTradeDurHyperOptLoss` (default legacy Freqtrade hyperoptimization loss function) - Mostly for short trade duration and avoiding losses.
* `OnlyProfitHyperOptLoss` (which takes only amount of profit into consideration)
* `SharpeHyperOptLoss` (optimizes Sharpe Ratio calculated on trade returns relative to standard deviation)
* `SharpeHyperOptLossDaily` (optimizes Sharpe Ratio calculated on **daily** trade returns relative to standard deviation)
* `SortinoHyperOptLoss` (optimizes Sortino Ratio calculated on trade returns relative to **downside** standard deviation)
* `SortinoHyperOptLossDaily` (optimizes Sortino Ratio calculated on **daily** trade returns relative to **downside** standard deviation)

Creation of a custom loss function is covered in the [Advanced Hyperopt](advanced-hyperopt.md) part of the documentation.

## Execute Hyperopt

Once you have updated your hyperopt configuration you can run it.
Because hyperopt tries a lot of combinations to find the best parameters it will take time to get a good result. More time usually results in better results.

We strongly recommend to use `screen` or `tmux` to prevent any connection loss.

```bash
freqtrade hyperopt --config config.json --hyperopt <hyperoptname> --hyperopt-loss <hyperoptlossname> --strategy <strategyname> -e 500 --spaces all
```

Use `<hyperoptname>` as the name of the custom hyperopt used.

The `-e` option will set how many evaluations hyperopt will do. Since hyperopt uses Bayesian search, running too many epochs at once may not produce greater results. Experience has shown that best results are usually not improving much after 500-1000 epochs.  
Doing multiple runs (executions) with a few 1000 epochs and different random state will most likely produce different results.

The `--spaces all` option determines that all possible parameters should be optimized. Possibilities are listed below.

!!! Note
    Hyperopt will store hyperopt results with the timestamp of the hyperopt start time.
    Reading commands (`hyperopt-list`, `hyperopt-show`) can use `--hyperopt-filename <filename>` to read and display older hyperopt results.
    You can find a list of filenames with `ls -l user_data/hyperopt_results/`.

### Execute Hyperopt with different historical data source

If you would like to hyperopt parameters using an alternate historical data set that
you have on-disk, use the `--datadir PATH` option. By default, hyperopt
uses data from directory `user_data/data`.

### Running Hyperopt with a smaller test-set

Use the `--timerange` argument to change how much of the test-set you want to use.
For example, to use one month of data, pass the following parameter to the hyperopt call:

```bash
freqtrade hyperopt --hyperopt <hyperoptname> --strategy <strategyname> --timerange 20180401-20180501
```

### Running Hyperopt using methods from a strategy

Hyperopt can reuse `populate_indicators`, `populate_buy_trend`, `populate_sell_trend` from your strategy, assuming these methods are **not** in your custom hyperopt file, and a strategy is provided.

```bash
freqtrade hyperopt --hyperopt AwesomeHyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy AwesomeStrategy
```

### Running Hyperopt with Smaller Search Space

Use the `--spaces` option to limit the search space used by hyperopt.
Letting Hyperopt optimize everything is a huuuuge search space. 
Often it might make more sense to start by just searching for initial buy algorithm.
Or maybe you just want to optimize your stoploss or roi table for that awesome new buy strategy you have.

Legal values are:

* `all`: optimize everything
* `buy`: just search for a new buy strategy
* `sell`: just search for a new sell strategy
* `roi`: just optimize the minimal profit table for your strategy
* `stoploss`: search for the best stoploss value
* `trailing`: search for the best trailing stop values
* `default`: `all` except `trailing`
* space-separated list of any of the above values for example `--spaces roi stoploss`

The default Hyperopt Search Space, used when no `--space` command line option is specified, does not include the `trailing` hyperspace. We recommend you to run optimization for the `trailing` hyperspace separately, when the best parameters for other hyperspaces were found, validated and pasted into your custom strategy.

### Position stacking and disabling max market positions

In some situations, you may need to run Hyperopt (and Backtesting) with the
`--eps`/`--enable-position-staking` and `--dmmp`/`--disable-max-market-positions` arguments.

By default, hyperopt emulates the behavior of the Freqtrade Live Run/Dry Run, where only one
open trade is allowed for every traded pair. The total number of trades open for all pairs
is also limited by the `max_open_trades` setting. During Hyperopt/Backtesting this may lead to
some potential trades to be hidden (or masked) by previously open trades.

The `--eps`/`--enable-position-stacking` argument allows emulation of buying the same pair multiple times,
while `--dmmp`/`--disable-max-market-positions` disables applying `max_open_trades`
during Hyperopt/Backtesting (which is equal to setting `max_open_trades` to a very high
number).

!!! Note
    Dry/live runs will **NOT** use position stacking - therefore it does make sense to also validate the strategy without this as it's closer to reality.

You can also enable position stacking in the configuration file by explicitly setting
`"position_stacking"=true`.

### Reproducible results

The search for optimal parameters starts with a few (currently 30) random combinations in the hyperspace of parameters, random Hyperopt epochs. These random epochs are marked with an asterisk character (`*`) in the first column in the Hyperopt output.

The initial state for generation of these random values (random state) is controlled by the value of the `--random-state` command line option. You can set it to some arbitrary value of your choice to obtain reproducible results.

If you have not set this value explicitly in the command line options, Hyperopt seeds the random state with some random value for you. The random state value for each Hyperopt run is shown in the log, so you can copy and paste it into the `--random-state` command line option to repeat the set of the initial random epochs used.

If you have not changed anything in the command line options, configuration, timerange, Strategy and Hyperopt classes, historical data and the Loss Function -- you should obtain same hyper-optimization results with same random state value used.

## Understand the Hyperopt Result

Once Hyperopt is completed you can use the result to create a new strategy.
Given the following result from hyperopt:

```
Best result:

    44/100:    135 trades. Avg profit  0.57%. Total profit  0.03871918 BTC (0.7722Σ%). Avg duration 180.4 mins. Objective: 1.94367

Buy hyperspace params:
{    'adx-value': 44,
     'rsi-value': 29,
     'adx-enabled': False,
     'rsi-enabled': True,
     'trigger': 'bb_lower'}
```

You should understand this result like:

- The buy trigger that worked best was `bb_lower`.
- You should not use ADX because `adx-enabled: False`)
- You should **consider** using the RSI indicator (`rsi-enabled: True` and the best value is `29.0` (`rsi-value: 29.0`)

You have to look inside your strategy file into `buy_strategy_generator()`
method, what those values match to.

So for example you had `rsi-value: 29.0` so we would look at `rsi`-block, that translates to the following code block:

```python
(dataframe['rsi'] < 29.0)
```

Translating your whole hyperopt result as the new buy-signal would then look like:

```python
def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
    dataframe.loc[
        (
            (dataframe['rsi'] < 29.0) &  # rsi-value
            dataframe['close'] < dataframe['bb_lowerband']  # trigger
        ),
        'buy'] = 1
    return dataframe
```

By default, hyperopt prints colorized results -- epochs with positive profit are printed in the green color. This highlighting helps you find epochs that can be interesting for later analysis. Epochs with zero total profit or with negative profits (losses) are printed in the normal color. If you do not need colorization of results (for instance, when you are redirecting hyperopt output to a file) you can switch colorization off by specifying the `--no-color` option in the command line.

You can use the `--print-all` command line option if you would like to see all results in the hyperopt output, not only the best ones. When `--print-all` is used, current best results are also colorized by default -- they are printed in bold (bright) style. This can also be switched off with the `--no-color` command line option.

!!! Note "Windows and color output"
    Windows does not support color-output natively, therefore it is automatically disabled. To have color-output for hyperopt running under windows, please consider using WSL.

### Understand Hyperopt ROI results

If you are optimizing ROI (i.e. if optimization search-space contains 'all', 'default' or 'roi'), your result will look as follows and include a ROI table:

```
Best result:

    44/100:    135 trades. Avg profit  0.57%. Total profit  0.03871918 BTC (0.7722Σ%). Avg duration 180.4 mins. Objective: 1.94367

ROI table:
{   0: 0.10674,
    21: 0.09158,
    78: 0.03634,
    118: 0}
```

In order to use this best ROI table found by Hyperopt in backtesting and for live trades/dry-run, copy-paste it as the value of the `minimal_roi` attribute of your custom strategy:

```
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        0: 0.10674,
        21: 0.09158,
        78: 0.03634,
        118: 0
    }
```

As stated in the comment, you can also use it as the value of the `minimal_roi` setting in the configuration file.

#### Default ROI Search Space

If you are optimizing ROI, Freqtrade creates the 'roi' optimization hyperspace for you -- it's the hyperspace of components for the ROI tables. By default, each ROI table generated by the Freqtrade consists of 4 rows (steps). Hyperopt implements adaptive ranges for ROI tables with ranges for values in the ROI steps that depend on the timeframe used. By default the values vary in the following ranges (for some of the most used timeframes, values are rounded to 5 digits after the decimal point):

| # step | 1m     |                   | 5m       |             | 1h         |                   | 1d           |                   |
| ------ | ------ | ----------------- | -------- | ----------- | ---------- | ----------------- | ------------ | ----------------- |
| 1      | 0      | 0.01161...0.11992 | 0        | 0.03...0.31 | 0          | 0.06883...0.71124 | 0            | 0.12178...1.25835 |
| 2      | 2...8  | 0.00774...0.04255 | 10...40  | 0.02...0.11 | 120...480  | 0.04589...0.25238 | 2880...11520 | 0.08118...0.44651 |
| 3      | 4...20 | 0.00387...0.01547 | 20...100 | 0.01...0.04 | 240...1200 | 0.02294...0.09177 | 5760...28800 | 0.04059...0.16237 |
| 4      | 6...44 | 0.0               | 30...220 | 0.0         | 360...2640 | 0.0               | 8640...63360 | 0.0               |

These ranges should be sufficient in most cases. The minutes in the steps (ROI dict keys) are scaled linearly depending on the timeframe used. The ROI values in the steps (ROI dict values) are scaled logarithmically depending on the timeframe used.

If you have the `generate_roi_table()` and `roi_space()` methods in your custom hyperopt file, remove them in order to utilize these adaptive ROI tables and the ROI hyperoptimization space generated by Freqtrade by default.

Override the `roi_space()` method if you need components of the ROI tables to vary in other ranges. Override the `generate_roi_table()` and `roi_space()` methods and implement your own custom approach for generation of the ROI tables during hyperoptimization if you need a different structure of the ROI tables or other amount of rows (steps). 

A sample for these methods can be found in [sample_hyperopt_advanced.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt_advanced.py).

### Understand Hyperopt Stoploss results

If you are optimizing stoploss values (i.e. if optimization search-space contains 'all', 'default' or 'stoploss'), your result will look as follows and include stoploss:

```
Best result:

    44/100:    135 trades. Avg profit  0.57%. Total profit  0.03871918 BTC (0.7722Σ%). Avg duration 180.4 mins. Objective: 1.94367

Buy hyperspace params:
{   'adx-value': 44,
    'rsi-value': 29,
    'adx-enabled': False,
    'rsi-enabled': True,
    'trigger': 'bb_lower'}
Stoploss: -0.27996
```

In order to use this best stoploss value found by Hyperopt in backtesting and for live trades/dry-run, copy-paste it as the value of the `stoploss` attribute of your custom strategy:

``` python
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.27996
```

As stated in the comment, you can also use it as the value of the `stoploss` setting in the configuration file.

#### Default Stoploss Search Space

If you are optimizing stoploss values, Freqtrade creates the 'stoploss' optimization hyperspace for you. By default, the stoploss values in that hyperspace vary in the range -0.35...-0.02, which is sufficient in most cases.

If you have the `stoploss_space()` method in your custom hyperopt file, remove it in order to utilize Stoploss hyperoptimization space generated by Freqtrade by default.

Override the `stoploss_space()` method and define the desired range in it if you need stoploss values to vary in other range during hyperoptimization. A sample for this method can be found in [user_data/hyperopts/sample_hyperopt_advanced.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt_advanced.py).

### Understand Hyperopt Trailing Stop results

If you are optimizing trailing stop values (i.e. if optimization search-space contains 'all' or 'trailing'), your result will look as follows and include trailing stop parameters:

```
Best result:

    45/100:    606 trades. Avg profit  1.04%. Total profit  0.31555614 BTC ( 630.48Σ%). Avg duration 150.3 mins. Objective: -1.10161

Trailing stop:
{   'trailing_only_offset_is_reached': True,
    'trailing_stop': True,
    'trailing_stop_positive': 0.02001,
    'trailing_stop_positive_offset': 0.06038}
```

In order to use these best trailing stop parameters found by Hyperopt in backtesting and for live trades/dry-run, copy-paste them as the values of the corresponding attributes of your custom strategy:

``` python
    # Trailing stop
    # These attributes will be overridden if the config file contains corresponding values.
    trailing_stop = True
    trailing_stop_positive = 0.02001
    trailing_stop_positive_offset = 0.06038
    trailing_only_offset_is_reached = True
```

As stated in the comment, you can also use it as the values of the corresponding settings in the configuration file.

#### Default Trailing Stop Search Space

If you are optimizing trailing stop values, Freqtrade creates the 'trailing' optimization hyperspace for you. By default, the `trailing_stop` parameter is always set to True in that hyperspace, the value of the `trailing_only_offset_is_reached` vary between True and False, the values of the `trailing_stop_positive` and `trailing_stop_positive_offset` parameters vary in the ranges 0.02...0.35 and 0.01...0.1 correspondingly, which is sufficient in most cases.

Override the `trailing_space()` method and define the desired range in it if you need values of the trailing stop parameters to vary in other ranges during hyperoptimization. A sample for this method can be found in [user_data/hyperopts/sample_hyperopt_advanced.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt_advanced.py).

## Show details of Hyperopt results

After you run Hyperopt for the desired amount of epochs, you can later list all results for analysis, select only best or profitable once, and show the details for any of the epochs previously evaluated. This can be done with the `hyperopt-list` and `hyperopt-show` sub-commands. The usage of these sub-commands is described in the [Utils](utils.md#list-hyperopt-results) chapter.

## Validate backtesting results

Once the optimized strategy has been implemented into your strategy, you should backtest this strategy to make sure everything is working as expected.

To achieve same results (number of trades, their durations, profit, etc.) than during Hyperopt, please use same configuration and parameters (timerange, timeframe, ...) used for hyperopt `--dmmp`/`--disable-max-market-positions` and `--eps`/`--enable-position-stacking` for Backtesting.

Should results don't match, please double-check to make sure you transferred all conditions correctly.
Pay special care to the stoploss (and trailing stoploss) parameters, as these are often set in configuration files, which override changes to the strategy.
You should also carefully review the log of your backtest to ensure that there were no parameters inadvertently set by the configuration (like `stoploss` or `trailing_stop`).
