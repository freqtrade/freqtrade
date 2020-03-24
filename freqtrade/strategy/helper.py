# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import logging
import warnings
import numpy as np  # noqa
from pandas import DataFrame
from typing import Dict, Callable
from functools import partial

from joblib import cpu_count, wrap_non_picklable_objects
from multiprocessing import Pool, Manager, Queue

from freqtrade.strategy.interface import IStrategy

# import talib.abstract as ta
# import freqtrade.vendor.qtpylib.indicators as qtpylib

logger = logging.getLogger(__name__)


def error_callback(e, q: Queue):
    print(e)
    q.put((None, None))


def get_all_signals(target: Callable, pairs_args: Dict, jobs=(cpu_count() // 2 or 1)) -> Dict:
    """ Apply function over a dict where the values are the args of the function, parallelly """

    results = {}
    queue = Manager().Queue()
    err = partial(error_callback, q=queue)

    def func_queue(func: Callable, queue: Queue, pair: str, *args) -> DataFrame:
        res = func(*args)
        queue.put((pair, res))
        return res

    target = wrap_non_picklable_objects(target)
    func_queue = wrap_non_picklable_objects(func_queue)

    try:
        with Pool(jobs) as p:
            p.starmap_async(
                func_queue,
                [(target, queue, pair, *v) for pair, v in pairs_args.items()],
                error_callback=err,
            )
            for pair in pairs_args:
                proc_pair, res = queue.get()
                if proc_pair:
                    results[proc_pair] = res
                else:
                    break
        # preserve the dict order
        return {pair: results[pair] for pair in pairs_args}
    except KeyError:
        return {pair: target(*args) for pair, args in pairs_args.items()}


class HelperStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    """

    time_weighted_roi = False

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def ohlcvdata_to_dataframe(self, tickerdata: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Creates a dataframe and populates indicators for given ticker data
        Used by optimize operations only, not during dry / live runs.
        """
        return get_all_signals(
            self.advise_indicators,
            {pair: (pair_data, {"pair": pair}) for pair, pair_data in tickerdata.items()},
        )
