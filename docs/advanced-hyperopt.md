# Advanced Hyperopt

This page explains some advanced Hyperopt topics that may require higher
coding skills and Python knowledge than creation of an ordinal hyperoptimization
class.

## Creating and using a custom loss function

To use a custom loss function class, make sure that the function `hyperopt_loss_function` is defined in your custom hyperopt loss class.
For the sample below, you then need to add the command line parameter `--hyperopt-loss SuperDuperHyperOptLoss` to your hyperopt call so this function is being used.

A sample of this can be found below, which is identical to the Default Hyperopt loss implementation. A full sample can be found in [userdata/hyperopts](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/sample_hyperopt_loss.py).

``` python
from datetime import datetime
from typing import Dict

from pandas import DataFrame

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
                               config: Dict, processed: Dict[str, DataFrame],
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results
        This is the legacy algorithm (used until now in freqtrade).
        Weights are distributed as follows:
        * 0.4 to trade duration
        * 0.25: Avoiding trade loss
        * 1.0 to total profit, compared to the expected value (`EXPECTED_MAX_PROFIT`) defined above
        """
        total_profit = results['profit_ratio'].sum()
        trade_duration = results['trade_duration'].mean()

        trade_loss = 1 - 0.25 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5.8)
        profit_loss = max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
        duration_loss = 0.4 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1)
        result = trade_loss + profit_loss + duration_loss
        return result
```

Currently, the arguments are:

* `results`: DataFrame containing the result  
    The following columns are available in results (corresponds to the output-file of backtesting when used with `--export trades`):  
    `pair, profit_ratio, profit_abs, open_date, open_rate, fee_open, close_date, close_rate, fee_close, amount, trade_duration, is_open, sell_reason, stake_amount, min_rate, max_rate, stop_loss_ratio, stop_loss_abs`
* `trade_count`: Amount of trades (identical to `len(results)`)
* `min_date`: Start date of the timerange used
* `min_date`: End date of the timerange used
* `config`: Config object used (Note: Not all strategy-related parameters will be updated here if they are part of a hyperopt space).
* `processed`: Dict of Dataframes with the pair as keys containing the data used for backtesting.

This function needs to return a floating point number (`float`). Smaller numbers will be interpreted as better results. The parameters and balancing for this is up to you.

!!! Note
    This function is called once per iteration - so please make sure to have this as optimized as possible to not slow hyperopt down unnecessarily.

!!! Note
    Please keep the arguments `*args` and `**kwargs` in the interface to allow us to extend this interface later.

## Overriding pre-defined spaces

To override a pre-defined space (`roi_space`, `generate_roi_table`, `stoploss_space`, `trailing_space`), define a nested class called Hyperopt and define the required spaces as follows:

```python
class MyAwesomeStrategy(IStrategy):
    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space(self):
            return [SKDecimal(-0.05, -0.01, decimals=3, name='stoploss')]
```

## Space options

For the additional spaces, scikit-optimize (in combination with Freqtrade) provides the following space types:

* `Categorical` - Pick from a list of categories (e.g. `Categorical(['a', 'b', 'c'], name="cat")`)
* `Integer` - Pick from a range of whole numbers (e.g. `Integer(1, 10, name='rsi')`)
* `SKDecimal` - Pick from a range of decimal numbers with limited precision (e.g. `SKDecimal(0.1, 0.5, decimals=3, name='adx')`). *Available only with freqtrade*.
* `Real` - Pick from a range of decimal numbers with full precision (e.g. `Real(0.1, 0.5, name='adx')`

You can import all of these from `freqtrade.optimize.space`, although `Categorical`, `Integer` and `Real` are only aliases for their corresponding scikit-optimize Spaces. `SKDecimal` is provided by freqtrade for faster optimizations.

``` python
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real  # noqa
```

!!! Hint "SKDecimal vs. Real"
    We recommend to use `SKDecimal` instead of the `Real` space in almost all cases. While the Real space provides full accuracy (up to ~16 decimal places) - this precision is rarely needed, and leads to unnecessary long hyperopt times.

    Assuming the definition of a rather small space (`SKDecimal(0.10, 0.15, decimals=2, name='xxx')`) - SKDecimal will have 5 possibilities (`[0.10, 0.11, 0.12, 0.13, 0.14, 0.15]`).

    A corresponding real space `Real(0.10, 0.15 name='xxx')`  on the other hand has an almost unlimited number of possibilities (`[0.10, 0.010000000001, 0.010000000002, ... 0.014999999999, 0.01500000000]`).

---

## Legacy Hyperopt

This Section explains the configuration of an explicit Hyperopt file (separate to the strategy).

!!! Warning "Deprecated / legacy mode"
    Since the 2021.4 release you no longer have to write a separate hyperopt class, but all strategies can be hyperopted.
    Please read the [main hyperopt page](hyperopt.md) for more details.

### Prepare hyperopt file

Configuring an explicit hyperopt file is similar to writing your own strategy, and many tasks will be similar.

!!! Tip "About this page"
    For this page, we will be using a fictional strategy called `AwesomeStrategy` - which will be optimized using the `AwesomeHyperopt` class.

#### Create a Custom Hyperopt File

The simplest way to get started is to use the following command, which will create a new hyperopt file from a template, which will be located under `user_data/hyperopts/AwesomeHyperopt.py`.

Let assume you want a hyperopt file `AwesomeHyperopt.py`:

``` bash
freqtrade new-hyperopt --hyperopt AwesomeHyperopt
```

#### Legacy Hyperopt checklist

Checklist on all tasks / possibilities in hyperopt

Depending on the space you want to optimize, only some of the below are required:

* fill `buy_strategy_generator` - for buy signal optimization
* fill `indicator_space` - for buy signal optimization
* fill `sell_strategy_generator` - for sell signal optimization
* fill `sell_indicator_space` - for sell signal optimization

!!! Note
    `populate_indicators` needs to create all indicators any of thee spaces may use, otherwise hyperopt will not work.

Optional in hyperopt - can also be loaded from a strategy (recommended):

* `populate_indicators` - fallback to create indicators
* `populate_buy_trend` - fallback if not optimizing for buy space. should come from strategy
* `populate_sell_trend` - fallback if not optimizing for sell space. should come from strategy

!!! Note
    You always have to provide a strategy to Hyperopt, even if your custom Hyperopt class contains all methods.
    Assuming the optional methods are not in your hyperopt file, please use `--strategy AweSomeStrategy` which contains these methods so hyperopt can use these methods instead.

Rarely you may also need to override:

* `roi_space` - for custom ROI optimization (if you need the ranges for the ROI parameters in the optimization hyperspace that differ from default)
* `generate_roi_table` - for custom ROI optimization (if you need the ranges for the values in the ROI table that differ from default or the number of entries (steps) in the ROI table which differs from the default 4 steps)
* `stoploss_space` - for custom stoploss optimization (if you need the range for the stoploss parameter in the optimization hyperspace that differs from default)
* `trailing_space` - for custom trailing stop optimization (if you need the ranges for the trailing stop parameters in the optimization hyperspace that differ from default)

#### Defining a buy signal optimization

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
to find the best combination. Two of them are integer values (`adx-value` and `rsi-value`) and I want you test in the range of values 20 to 40.  
Then we have three category variables. First two are either `True` or `False`.
We use these to either enable or disable the ADX and RSI guards.
The last one we call `trigger` and use it to decide which buy trigger we want to use.

So let's write the buy strategy generator using these values:

```python
    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by Hyperopt.
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
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

#### Sell optimization

Similar to the buy-signal above, sell-signals can also be optimized.
Place the corresponding settings into the following methods

* Inside `sell_indicator_space()` - the parameters hyperopt shall be optimizing.
* Within `sell_strategy_generator()` - populate the nested method `populate_sell_trend()` to apply the parameters.

The configuration and rules are the same than for buy signals.
To avoid naming collisions in the search-space, please prefix all sell-spaces with `sell-`.

### Execute Hyperopt

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

#### Running Hyperopt using methods from a strategy

Hyperopt can reuse `populate_indicators`, `populate_buy_trend`, `populate_sell_trend` from your strategy, assuming these methods are **not** in your custom hyperopt file, and a strategy is provided.

```bash
freqtrade hyperopt --hyperopt AwesomeHyperopt --hyperopt-loss SharpeHyperOptLossDaily --strategy AwesomeStrategy
```

### Understand the Hyperopt Result

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

* The buy trigger that worked best was `bb_lower`.
* You should not use ADX because `adx-enabled: False`)
* You should **consider** using the RSI indicator (`rsi-enabled: True` and the best value is `29.0` (`rsi-value: 29.0`)

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

### Validate backtesting results

Once the optimized parameters and conditions have been implemented into your strategy, you should backtest the strategy to make sure everything is working as expected.

To achieve same results (number of trades, their durations, profit, etc.) than during Hyperopt, please use same configuration and parameters (timerange, timeframe, ...) used for hyperopt `--dmmp`/`--disable-max-market-positions` and `--eps`/`--enable-position-stacking` for Backtesting.

Should results don't match, please double-check to make sure you transferred all conditions correctly.
Pay special care to the stoploss (and trailing stoploss) parameters, as these are often set in configuration files, which override changes to the strategy.
You should also carefully review the log of your backtest to ensure that there were no parameters inadvertently set by the configuration (like `stoploss` or `trailing_stop`).

### Sharing methods with your strategy

Hyperopt classes provide access to the Strategy via the `strategy` class attribute.
This can be a great way to reduce code duplication if used correctly, but will also complicate usage for inexperienced users.

``` python
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy
import freqtrade.vendor.qtpylib.indicators as qtpylib

class MyAwesomeStrategy(IStrategy):

    buy_params = {
        'rsi-value': 30,
        'adx-value': 35,
    }

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.buy_strategy_generator(self.buy_params, dataframe, metadata)

    @staticmethod
    def buy_strategy_generator(params, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['rsi'], params['rsi-value']) &
                dataframe['adx'] > params['adx-value']) &
                dataframe['volume'] > 0
            )
            , 'buy'] = 1
        return dataframe

class MyAwesomeHyperOpt(IHyperOpt):
    ...
    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by Hyperopt.
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            # Call strategy's buy strategy generator
            return self.StrategyClass.buy_strategy_generator(params, dataframe, metadata)

        return populate_buy_trend
```
