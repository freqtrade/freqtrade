# Advanced Hyperopt

This page explains some advanced Hyperopt topics that may require higher
coding skills and Python knowledge than creation of an ordinal hyperoptimization
class.

## Derived hyperopt classes

Custom hyperop classes can be derived in the same way [it can be done for strategies](strategy-customization.md#derived-strategies).

Applying to hyperoptimization, as an example, you may override how dimensions are defined in your optimization hyperspace:

```python
class MyAwesomeHyperOpt(IHyperOpt):
    ...
    # Uses default stoploss dimension

class MyAwesomeHyperOpt2(MyAwesomeHyperOpt):
    @staticmethod
    def stoploss_space() -> List[Dimension]:
        # Override boundaries for stoploss
        return [
            Real(-0.33, -0.01, name='stoploss'),
        ]
```

and then quickly switch between hyperopt classes, running optimization process with hyperopt class you need in each particular case:

```
$ freqtrade hyperopt --hyperopt MyAwesomeHyperOpt --hyperopt-loss SharpeHyperOptLossDaily --strategy MyAwesomeStrategy ...
or
$ freqtrade hyperopt --hyperopt MyAwesomeHyperOpt2 --hyperopt-loss SharpeHyperOptLossDaily --strategy MyAwesomeStrategy ...
```

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
