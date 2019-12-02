# Hyperopt

This page explains how to tune your strategy by finding the optimal
parameters, a process called hyperparameter optimization. The bot uses several
algorithms included in the `scikit-optimize` package to accomplish this. The
search will burn all your CPU cores, make your laptop sound like a fighter jet
and still take a long time.

Hyperopt requires historic data to be available, just as backtesting does.
To learn how to get data for the pairs and exchange you're interrested in, head over to the [Data Downloading](data-download.md) section of the documentation.

!!! Bug
    Hyperopt can crash when used with only 1 CPU Core as found out in [Issue #1133](https://github.com/freqtrade/freqtrade/issues/1133)

## Prepare Hyperopting

Before we start digging into Hyperopt, we recommend you to take a look at
the sample hyperopt file located in [user_data/hyperopts/](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt.py).

Configuring hyperopt is similar to writing your own strategy, and many tasks will be similar and a lot of code can be copied across from the strategy.

The simplest way to get started is to use `freqtrade new-hyperopt --hyperopt AwesomeHyperopt`.
This will create a new hyperopt file from a template, which will be located under `user_data/hyperopts/AwesomeHyperopt.py`.

### Checklist on all tasks / possibilities in hyperopt

Depending on the space you want to optimize, only some of the below are required:

* fill `buy_strategy_generator` - for buy signal optimization
* fill `indicator_space` - for buy signal optimzation
* fill `sell_strategy_generator` - for sell signal optimization
* fill `sell_indicator_space` - for sell signal optimzation

!!! Note
    `populate_indicators` needs to create all indicators any of thee spaces may use, otherwise hyperopt will not work.

Optional - can also be loaded from a strategy:

* copy `populate_indicators` from your strategy - otherwise default-strategy will be used
* copy `populate_buy_trend` from your strategy - otherwise default-strategy will be used
* copy `populate_sell_trend` from your strategy - otherwise default-strategy will be used

!!! Note
    Assuming the optional methods are not in your hyperopt file, please use `--strategy AweSomeStrategy` which contains these methods so hyperopt can use these methods instead.

Rarely you may also need to override:

* `roi_space` - for custom ROI optimization (if you need the ranges for the ROI parameters in the optimization hyperspace that differ from default)
* `generate_roi_table` - for custom ROI optimization (if you need the ranges for the values in the ROI table that differ from default or the number of entries (steps) in the ROI table which differs from the default 4 steps)
* `stoploss_space` - for custom stoploss optimization (if you need the range for the stoploss parameter in the optimization hyperspace that differs from default)
* `trailing_space` - for custom trailing stop optimization (if you need the ranges for the trailing stop parameters in the optimization hyperspace that differ from default)

### 1. Install a Custom Hyperopt File

Put your hyperopt file into the directory `user_data/hyperopts`.

Let assume you want a hyperopt file `awesome_hyperopt.py`:  
Copy the file `user_data/hyperopts/sample_hyperopt.py` into `user_data/hyperopts/awesome_hyperopt.py`

### 2. Configure your Guards and Triggers

There are two places you need to change in your hyperopt file to add a new buy hyperopt for testing:

- Inside `indicator_space()` - the parameters hyperopt shall be optimizing.
- Inside `populate_buy_trend()` - applying the parameters.

There you have two different types of indicators: 1. `guards` and 2. `triggers`.

1. Guards are conditions like "never buy if ADX < 10", or never buy if current price is over EMA10.
2. Triggers are ones that actually trigger buy in specific moment, like "buy when EMA5 crosses over EMA10" or "buy when close price touches lower bollinger band".

Hyperoptimization will, for each eval round, pick one trigger and possibly
multiple guards. The constructed strategy will be something like
"*buy exactly when close price touches lower bollinger band, BUT only if
ADX > 10*".

If you have updated the buy strategy, i.e. changed the contents of
`populate_buy_trend()` method, you have to update the `guards` and
`triggers` your hyperopt must use correspondingly.

#### Sell optimization

Similar to the buy-signal above, sell-signals can also be optimized.
Place the corresponding settings into the following methods

* Inside `sell_indicator_space()` - the parameters hyperopt shall be optimizing.
* Inside `populate_sell_trend()` - applying the parameters.

The configuration and rules are the same than for buy signals.
To avoid naming collisions in the search-space, please prefix all sell-spaces with `sell-`.

#### Using ticker-interval as part of the Strategy

The Strategy exposes the ticker-interval as `self.ticker_interval`. The same value is available as class-attribute `HyperoptName.ticker_interval`.
In the case of the linked sample-value this would be `SampleHyperOpt.ticker_interval`.

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

``` python
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

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'buy'] = 1

            return dataframe

        return populate_buy_trend
```

Hyperopting will now call this `populate_buy_trend` as many times you ask it (`epochs`)
with different value combinations. It will then use the given historical data and make
buys based on the buy signals generated with the above function and based on the results
it will end with telling you which paramter combination produced the best profits.

The search for best parameters starts with a few random combinations and then uses a
regressor algorithm (currently ExtraTreesRegressor) to quickly find a parameter combination
that minimizes the value of the [loss function](#loss-functions).

The above setup expects to find ADX, RSI and Bollinger Bands in the populated indicators.
When you want to test an indicator that isn't used by the bot currently, remember to
add it to the `populate_indicators()` method in your custom hyperopt file.

## Loss-functions

Each hyperparameter tuning requires a target. This is usually defined as a loss function (sometimes also called objective function), which should decrease for more desirable results, and increase for bad results.

By default, FreqTrade uses a loss function, which has been with freqtrade since the beginning and optimizes mostly for short trade duration and avoiding losses.

A different loss function can be specified by using the `--hyperopt-loss <Class-name>` argument.
This class should be in its own file within the `user_data/hyperopts/` directory.

Currently, the following loss functions are builtin:

* `DefaultHyperOptLoss` (default legacy Freqtrade hyperoptimization loss function)
* `OnlyProfitHyperOptLoss` (which takes only amount of profit into consideration)
* `SharpeHyperOptLoss` (optimizes Sharpe Ratio calculated on the trade returns)

### Creating and using a custom loss function

To use a custom loss function class, make sure that the function `hyperopt_loss_function` is defined in your custom hyperopt loss class.
For the sample below, you then need to add the command line parameter `--hyperopt-loss SuperDuperHyperOptLoss` to your hyperopt call so this fuction is being used.

A sample of this can be found below, which is identical to the Default Hyperopt loss implementation. A full sample can be found [user_data/hyperopts/](https://github.com/freqtrade/freqtrade/blob/develop/user_data/hyperopts/sample_hyperopt_loss.py)

``` python
from freqtrade.optimize.hyperopt import IHyperOptLoss

TARGET_TRADES = 600
EXPECTED_MAX_PROFIT = 3.0
MAX_ACCEPTED_TRADE_DURATION = 300

class SuperDuperHyperOptLoss(IHyperOptLoss):
    """
    Defines the default loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results
        This is the legacy algorithm (used until now in freqtrade).
        Weights are distributed as follows:
        * 0.4 to trade duration
        * 0.25: Avoiding trade loss
        * 1.0 to total profit, compared to the expected value (`EXPECTED_MAX_PROFIT`) defined above
        """
        total_profit = results.profit_percent.sum()
        trade_duration = results.trade_duration.mean()

        trade_loss = 1 - 0.25 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5.8)
        profit_loss = max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
        duration_loss = 0.4 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1)
        result = trade_loss + profit_loss + duration_loss
        return result
```

Currently, the arguments are:

* `results`: DataFrame containing the result  
    The following columns are available in results (corresponds to the output-file of backtesting when used with `--export trades`):  
    `pair, profit_percent, profit_abs, open_time, close_time, open_index, close_index, trade_duration, open_at_end, open_rate, close_rate, sell_reason`
* `trade_count`: Amount of trades (identical to `len(results)`)
* `min_date`: Start date of the hyperopting TimeFrame
* `min_date`: End date of the hyperopting TimeFrame

This function needs to return a floating point number (`float`). Smaller numbers will be interpreted as better results. The parameters and balancing for this is up to you.

!!! Note
    This function is called once per iteration - so please make sure to have this as optimized as possible to not slow hyperopt down unnecessarily.

!!! Note
    Please keep the arguments `*args` and `**kwargs` in the interface to allow us to extend this interface later.

## Execute Hyperopt

Once you have updated your hyperopt configuration you can run it.
Because hyperopt tries a lot of combinations to find the best parameters it will take time to get a good result. More time usually results in better results.

We strongly recommend to use `screen` or `tmux` to prevent any connection loss.

```bash
freqtrade hyperopt --config config.json --hyperopt <hyperoptname> -e 5000 --spaces all
```

Use  `<hyperoptname>` as the name of the custom hyperopt used.

The `-e` option will set how many evaluations hyperopt will do. We recommend
running at least several thousand evaluations.

The `--spaces all` option determines that all possible parameters should be optimized. Possibilities are listed below.

!!! Note
    By default, hyperopt will erase previous results and start from scratch. Continuation can be archived by using `--continue`.

!!! Warning
    When switching parameters or changing configuration options, make sure to not use the argument `--continue` so temporary results can be removed.

### Execute Hyperopt with Different Ticker-Data Source

If you would like to hyperopt parameters using an alternate ticker data that
you have on-disk, use the `--datadir PATH` option. Default hyperopt will
use data from directory `user_data/data`.

### Running Hyperopt with Smaller Testset

Use the `--timerange` argument to change how much of the testset you want to use.
For example, to use one month of data, pass the following parameter to the hyperopt call:

```bash
freqtrade hyperopt --timerange 20180401-20180501
```

### Running Hyperopt using methods from a strategy

Hyperopt can reuse `populate_indicators`, `populate_buy_trend`, `populate_sell_trend` from your strategy, assuming these methods are **not** in your custom hyperopt file, and a strategy is provided.

```bash
freqtrade hyperopt --strategy SampleStrategy --customhyperopt SampleHyperopt
```

### Running Hyperopt with Smaller Search Space

Use the `--spaces` option to limit the search space used by hyperopt.
Letting Hyperopt optimize everything is a huuuuge search space. Often it
might make more sense to start by just searching for initial buy algorithm.
Or maybe you just want to optimize your stoploss or roi table for that awesome
new buy strategy you have.

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
some potential trades to be hidden (or masked) by previosly open trades.

The `--eps`/`--enable-position-stacking` argument allows emulation of buying the same pair multiple times,
while `--dmmp`/`--disable-max-market-positions` disables applying `max_open_trades` 
during Hyperopt/Backtesting (which is equal to setting `max_open_trades` to a very high
number).

!!! Note
    Dry/live runs will **NOT** use position stacking - therefore it does make sense to also validate the strategy without this as it's closer to reality.

You can also enable position stacking in the configuration file by explicitly setting 
`"position_stacking"=true`.

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

``` python
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

If you are optimizing ROI, Freqtrade creates the 'roi' optimization hyperspace for you -- it's the hyperspace of components for the ROI tables. By default, each ROI table generated by the Freqtrade consists of 4 rows (steps). Hyperopt implements adaptive ranges for ROI tables with ranges for values in the ROI steps that depend on the ticker_interval used. By default the values vary in the following ranges (for some of the most used ticker intervals, values are rounded to 5 digits after the decimal point):

| # step | 1m |  | 5m |  | 1h |  | 1d |  |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.01161...0.11992 | 0 | 0.03...0.31 | 0 | 0.06883...0.71124 | 0 | 0.12178...1.25835 |
| 2 | 2...8 | 0.00774...0.04255 | 10...40 | 0.02...0.11 | 120...480 | 0.04589...0.25238 | 2880...11520 | 0.08118...0.44651 |
| 3 | 4...20 | 0.00387...0.01547 | 20...100 | 0.01...0.04 | 240...1200 | 0.02294...0.09177 | 5760...28800 | 0.04059...0.16237 |
| 4 | 6...44 | 0.0 | 30...220 | 0.0 | 360...2640 | 0.0 | 8640...63360 | 0.0 |

These ranges should be sufficient in most cases. The minutes in the steps (ROI dict keys) are scaled linearly depending on the ticker interval used. The ROI values in the steps (ROI dict values) are scaled logarithmically depending on the ticker interval used.

If you have the `generate_roi_table()` and `roi_space()` methods in your custom hyperopt file, remove them in order to utilize these adaptive ROI tables and the ROI hyperoptimization space generated by Freqtrade by default.

Override the `roi_space()` method if you need components of the ROI tables to vary in other ranges. Override the `generate_roi_table()` and `roi_space()` methods and implement your own custom approach for generation of the ROI tables during hyperoptimization if you need a different structure of the ROI tables or other amount of rows (steps). A sample for these methods can be found in [user_data/hyperopts/sample_hyperopt_advanced.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt_advanced.py).

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

```
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

```
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

Override the `trailing_space()` method and define the desired range in it if you need values of the trailing stop parameters to vary in other ranges during hyperoptimization. A sample for this method can be found in [user_data/hyperopts/sample_hyperopt_advanced.py](https://github.com/freqtrade/freqtrade/blob/develop/user_data/hyperopts/sample_hyperopt_advanced.py).

### Validate backtesting results

Once the optimized strategy has been implemented into your strategy, you should backtest this strategy to make sure everything is working as expected.

To achieve same results (number of trades, their durations, profit, etc.) than during Hyperopt, please use same set of arguments `--dmmp`/`--disable-max-market-positions` and `--eps`/`--enable-position-stacking` for Backtesting.

## Next Step

Now you have a perfect bot and want to control it from Telegram. Your
next step is to learn the [Telegram usage](telegram-usage.md).
