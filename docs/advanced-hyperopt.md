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
                               backtest_stats: Dict[str, Any],
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

* `results`: DataFrame containing the resulting trades.
    The following columns are available in results (corresponds to the output-file of backtesting when used with `--export trades`):  
    `pair, profit_ratio, profit_abs, open_date, open_rate, fee_open, close_date, close_rate, fee_close, amount, trade_duration, is_open, sell_reason, stake_amount, min_rate, max_rate, stop_loss_ratio, stop_loss_abs`
* `trade_count`: Amount of trades (identical to `len(results)`)
* `min_date`: Start date of the timerange used
* `min_date`: End date of the timerange used
* `config`: Config object used (Note: Not all strategy-related parameters will be updated here if they are part of a hyperopt space).
* `processed`: Dict of Dataframes with the pair as keys containing the data used for backtesting.
* `backtest_stats`: Backtesting statistics using the same format as the backtesting file "strategy" substructure. Available fields can be seen in `generate_strategy_stats()` in `optimize_reports.py`.

This function needs to return a floating point number (`float`). Smaller numbers will be interpreted as better results. The parameters and balancing for this is up to you.

!!! Note
    This function is called once per epoch - so please make sure to have this as optimized as possible to not slow hyperopt down unnecessarily.

!!! Note "`*args` and `**kwargs`"
    Please keep the arguments `*args` and `**kwargs` in the interface to allow us to extend this interface in the future.

## Overriding pre-defined spaces

To override a pre-defined space (`roi_space`, `generate_roi_table`, `stoploss_space`, `trailing_space`), define a nested class called Hyperopt and define the required spaces as follows:

```python
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
```

!!! Note
    All overrides are optional and can be mixed/matched as necessary.

### Overriding Base estimator

You can define your own estimator for Hyperopt by implementing `generate_estimator()` in the Hyperopt subclass.

```python
class MyAwesomeStrategy(IStrategy):
    class HyperOpt:
        def generate_estimator():
            return "RF"

```

Possible values are either one of "GP", "RF", "ET", "GBRT" (Details can be found in the [scikit-optimize documentation](https://scikit-optimize.github.io/)), or "an instance of a class that inherits from `RegressorMixin` (from sklearn) and where the `predict` method has an optional `return_std` argument, which returns `std(Y | x)` along with `E[Y | x]`".

Some research will be necessary to find additional Regressors.

Example for `ExtraTreesRegressor` ("ET") with additional parameters:

```python
class MyAwesomeStrategy(IStrategy):
    class HyperOpt:
        def generate_estimator():
            from skopt.learning import ExtraTreesRegressor
            # Corresponds to "ET" - but allows additional parameters.
            return ExtraTreesRegressor(n_estimators=100)

```

!!! Note
    While custom estimators can be provided, it's up to you as User to do research on possible parameters and analyze / understand which ones should be used.
    If you're unsure about this, best use one of the Defaults (`"ET"` has proven to be the most versatile) without further parameters.

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
