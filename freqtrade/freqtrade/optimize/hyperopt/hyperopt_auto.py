"""
HyperOptAuto class.
This module implements a convenience auto-hyperopt class, which can be used together with strategies
 that implement IHyperStrategy interface.
"""

import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Literal

from freqtrade.exceptions import OperationalException


with suppress(ImportError):
    from freqtrade.optimize.space import Dimension

from freqtrade.optimize.hyperopt.hyperopt_interface import EstimatorType, IHyperOpt


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

    def get_available_spaces(self) -> list[str]:
        """
        Get list of available spaces defined in strategy.
        :return: list of available spaces.
        """
        return list(self.strategy._ft_hyper_params)

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

    def get_indicator_space(
        self, space: Literal["buy", "sell", "enter", "exit", "protection"] | str
    ) -> list:
        """
        Get indicator space for a given space.
        :param space: parameter space to get.
        """
        indicator_space = [
            attr.get_space(attr_name)
            for attr_name, attr in self.strategy.enumerate_parameters(space)
            if attr.optimize
        ]
        if len(indicator_space) > 0:
            return indicator_space
        else:
            _format_exception_message(
                space, self.config.get("hyperopt_ignore_missing_space", False)
            )
            return []

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
