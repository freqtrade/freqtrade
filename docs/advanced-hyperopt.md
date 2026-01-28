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
from typing import Any, Dict

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.optimize.hyperopt import IHyperOptLoss

TARGET_TRADES = 600
EXPECTED_MAX_PROFIT = 3.0
MAX_ACCEPTED_TRADE_DURATION = 300

class SuperDuperHyperOptLoss(IHyperOptLoss):
    """
    Defines the default loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(
        *,
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        backtest_stats: dict[str, Any],
        starting_balance: float,
        **kwargs,
    ) -> float:
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

* `results`: DataFrame containing the resulting trades.
    The following columns are available in results (corresponds to the output-file of backtesting when used with `--export trades`):  
    `pair, profit_ratio, profit_abs, open_date, open_rate, fee_open, close_date, close_rate, fee_close, amount, trade_duration, is_open, exit_reason, stake_amount, min_rate, max_rate, stop_loss_ratio, stop_loss_abs`
* `trade_count`: Amount of trades (identical to `len(results)`)
* `min_date`: Start date of the timerange used
* `min_date`: End date of the timerange used
* `config`: Config object used (Note: Not all strategy-related parameters will be updated here if they are part of a hyperopt space).
* `processed`: Dict of Dataframes with the pair as keys containing the data used for backtesting.
* `backtest_stats`: Backtesting statistics using the same format as the backtesting file "strategy" substructure. Available fields can be seen in `generate_strategy_stats()` in `optimize_reports.py`.
* `starting_balance`: Starting balance used for backtesting.

This function needs to return a floating point number (`float`). Smaller numbers will be interpreted as better results. The parameters and balancing for this is up to you.

!!! Note
    This function is called once per epoch - so please make sure to have this as optimized as possible to not slow hyperopt down unnecessarily.

!!! Note "`*args` and `**kwargs`"
    Please keep the arguments `*args` and `**kwargs` in the interface to allow us to extend this interface in the future.

## Overriding pre-defined spaces

To override a pre-defined space (`roi_space`, `generate_roi_table`, `stoploss_space`, `trailing_space`, `max_open_trades_space`), define a nested class called Hyperopt and define the required spaces as follows:

```python
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

class MyAwesomeStrategy(IStrategy):
    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.05, -0.01, decimals=3, name='stoploss')]

        # Define custom ROI space
        def roi_space() -> List[Dimension]:
            return [
                Integer(10, 120, name='roi_t1'),
                Integer(10, 60, name='roi_t2'),
                Integer(10, 40, name='roi_t3'),
                SKDecimal(0.01, 0.04, decimals=3, name='roi_p1'),
                SKDecimal(0.01, 0.07, decimals=3, name='roi_p2'),
                SKDecimal(0.01, 0.20, decimals=3, name='roi_p3'),
            ]

        def generate_roi_table(params: Dict) -> dict[int, float]:

            roi_table = {}
            roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
            roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
            roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
            roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

            return roi_table

        def trailing_space() -> List[Dimension]:
            # All parameters here are mandatory, you can only modify their type or the range.
            return [
                # Fixed to true, if optimizing trailing_stop we assume to use trailing stop at all times.
                Categorical([True], name='trailing_stop'),

                SKDecimal(0.01, 0.35, decimals=3, name='trailing_stop_positive'),
                # 'trailing_stop_positive_offset' should be greater than 'trailing_stop_positive',
                # so this intermediate parameter is used as the value of the difference between
                # them. The value of the 'trailing_stop_positive_offset' is constructed in the
                # generate_trailing_params() method.
                # This is similar to the hyperspace dimensions used for constructing the ROI tables.
                SKDecimal(0.001, 0.1, decimals=3, name='trailing_stop_positive_offset_p1'),

                Categorical([True, False], name='trailing_only_offset_is_reached'),
        ]

        # Define a custom max_open_trades space
        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(-1, 10, name='max_open_trades'),
            ]
```

!!! Note
    All overrides are optional and can be mixed/matched as necessary.

## Dynamic parameters

Parameters can also be defined dynamically, but must be available to the instance once the [`bot_start()` callback](strategy-callbacks.md#bot-start) has been called.

``` python

class MyAwesomeStrategy(IStrategy):

    def bot_start(self, **kwargs) -> None:
        self.buy_adx = IntParameter(20, 30, default=30, optimize=True)

    # ...
```

!!! Warning
    Parameters created this way will not show up in the `list-strategies` parameter count.

## Overriding Base estimator

You can define your own optuna sampler for Hyperopt by implementing `generate_estimator()` in the Hyperopt subclass.

```python
class MyAwesomeStrategy(IStrategy):
    class HyperOpt:
        def generate_estimator(dimensions: List['Dimension'], **kwargs):
            return "NSGAIIISampler"

```

Possible values are either one of "NSGAIISampler", "TPESampler", "GPSampler", "CmaEsSampler", "NSGAIIISampler", "QMCSampler" (Details can be found in the [optuna-samplers documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)), or "an instance of a class that inherits from `optuna.samplers.BaseSampler`".

Some research will be necessary to find additional Samplers (from optunahub) for example.

!!! Note
    While custom estimators can be provided, it's up to you as User to do research on possible parameters and analyze / understand which ones should be used.
    If you're unsure about this, best use one of the Defaults (`"NSGAIIISampler"` has proven to be the most versatile) without further parameters.

??? Example "Using `AutoSampler` from Optunahub"

    [AutoSampler docs](https://hub.optuna.org/samplers/auto_sampler/)
    
    Install the necessary dependencies 
    ``` bash
    pip install optunahub cmaes torch scipy
    ```
    Implement `generate_estimator()`  in your strategy

    ``` python
    # ...
    from freqtrade.strategy.interface import IStrategy
    from typing import List
    import optunahub
    # ... 

    class my_strategy(IStrategy):
        class HyperOpt:
            def generate_estimator(dimensions: List["Dimension"], **kwargs):
                if "random_state" in kwargs.keys():
                    return optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=kwargs["random_state"])
                else:
                    return optunahub.load_module("samplers/auto_sampler").AutoSampler()

    ```

    Obviously the same approach will work for all other Samplers optuna supports.

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
