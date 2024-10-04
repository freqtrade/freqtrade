"""
HyperOptAuto class.
This module implements a convenience auto-hyperopt class, which can be used together with strategies
 that implement IHyperStrategy interface.
"""

import logging
from contextlib import suppress
from typing import Callable

from freqtrade.exceptions import OperationalException


with suppress(ImportError):
    from skopt.space import Dimension

from freqtrade.optimize.hyperopt_interface import EstimatorType, IHyperOpt


logger = logging.getLogger(__name__)


def _format_exception_message(space: str, ignore_missing_space: bool) -> None:
    msg = (
        f"The '{space}' space is included into the hyperoptimization "
        f"but no parameter for this space was found in your Strategy. "
    )
    if ignore_missing_space:
        logger.warning(msg + "This space will be ignored.")
    else:
        raise OperationalException(
            msg + f"Please make sure to have parameters for this space enabled for optimization "
            f"or remove the '{space}' space from hyperoptimization."
        )


class HyperOptAuto(IHyperOpt):
    """
    This class delegates functionality to Strategy(IHyperStrategy) and Strategy.HyperOpt classes.
     Most of the time Strategy.HyperOpt class would only implement indicator_space and
     sell_indicator_space methods, but other hyperopt methods can be overridden as well.
    """

    def _get_func(self, name) -> Callable:
        """
        Return a function defined in Strategy.HyperOpt class, or one defined in super() class.
        :param name: function name.
        :return: a requested function.
        """
        hyperopt_cls = getattr(self.strategy, "HyperOpt", None)
        default_func = getattr(super(), name)
        if hyperopt_cls:
            return getattr(hyperopt_cls, name, default_func)
        else:
            return default_func

    def _generate_indicator_space(self, category):
        for attr_name, attr in self.strategy.enumerate_parameters(category):
            if attr.optimize:
                yield attr.get_space(attr_name)

    def _get_indicator_space(self, category) -> list:
        # TODO: is this necessary, or can we call "generate_space" directly?
        indicator_space = list(self._generate_indicator_space(category))
        if len(indicator_space) > 0:
            return indicator_space
        else:
            _format_exception_message(
                category, self.config.get("hyperopt_ignore_missing_space", False)
            )
            return []

    def buy_indicator_space(self) -> list["Dimension"]:
        return self._get_indicator_space("buy")

    def sell_indicator_space(self) -> list["Dimension"]:
        return self._get_indicator_space("sell")

    def protection_space(self) -> list["Dimension"]:
        return self._get_indicator_space("protection")

    def generate_roi_table(self, params: dict) -> dict[int, float]:
        return self._get_func("generate_roi_table")(params)

    def roi_space(self) -> list["Dimension"]:
        return self._get_func("roi_space")()

    def stoploss_space(self) -> list["Dimension"]:
        return self._get_func("stoploss_space")()

    def generate_trailing_params(self, params: dict) -> dict:
        return self._get_func("generate_trailing_params")(params)

    def trailing_space(self) -> list["Dimension"]:
        return self._get_func("trailing_space")()

    def max_open_trades_space(self) -> list["Dimension"]:
        return self._get_func("max_open_trades_space")()

    def generate_estimator(self, dimensions: list["Dimension"], **kwargs) -> EstimatorType:
        return self._get_func("generate_estimator")(dimensions=dimensions, **kwargs)
