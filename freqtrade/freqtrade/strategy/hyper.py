"""
IHyperStrategy interface, hyperoptable Parameter class.
This module defines a base class for auto-hyperoptable strategies.
"""

import logging
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.misc import deep_merge_dicts
from freqtrade.optimize.hyperopt_tools import HyperoptTools
from freqtrade.strategy.parameters import BaseParameter


logger = logging.getLogger(__name__)


# Type aliases
SpaceParams = dict[str, BaseParameter]
AllSpaceParams = dict[str, SpaceParams]


class HyperStrategyMixin:
    """
    A helper base class which allows HyperOptAuto class to reuse implementations of buy/sell
     strategy logic.
    """

    def __init__(self, config: Config, *args, **kwargs):
        """
        Initialize hyperoptable strategy mixin.
        """
        self.config = config
        self._ft_hyper_params: AllSpaceParams = {}

        params = self.load_params_from_file()
        params = params.get("params", {})
        self._ft_params_from_file = params
        # Init/loading of parameters is done as part of ft_bot_start().

    def enumerate_parameters(self, space: str | None = None) -> Iterator[tuple[str, BaseParameter]]:
        """
        Find all optimizable parameters and return (name, attr) iterator.
        :param space: parameter space to filter for, or None for all spaces.
        :return:
        """
        for space in [c for c in self._ft_hyper_params if space is None or c == space]:
            for par in self._ft_hyper_params[space].values():
                yield par.name, par

    def ft_load_params_from_file(self) -> None:
        """
        Load Parameters from parameter file
        Should/must run before config values are loaded in strategy_resolver.
        """
        if self._ft_params_from_file:
            # Set parameters from Hyperopt results file
            params = self._ft_params_from_file
            self.minimal_roi = params.get("roi", getattr(self, "minimal_roi", {}))

            self.stoploss = params.get("stoploss", {}).get(
                "stoploss", getattr(self, "stoploss", -0.1)
            )
            self.max_open_trades = params.get("max_open_trades", {}).get(
                "max_open_trades", getattr(self, "max_open_trades", -1)
            )
            trailing = params.get("trailing", {})
            self.trailing_stop = trailing.get(
                "trailing_stop", getattr(self, "trailing_stop", False)
            )
            self.trailing_stop_positive = trailing.get(
                "trailing_stop_positive", getattr(self, "trailing_stop_positive", None)
            )
            self.trailing_stop_positive_offset = trailing.get(
                "trailing_stop_positive_offset", getattr(self, "trailing_stop_positive_offset", 0)
            )
            self.trailing_only_offset_is_reached = trailing.get(
                "trailing_only_offset_is_reached",
                getattr(self, "trailing_only_offset_is_reached", 0.0),
            )

    def ft_load_hyper_params(self, hyperopt: bool = False) -> None:
        """
        Load Hyperoptable parameters
        Prevalence:
        * Parameters from parameter file
        * Parameters defined in parameters objects (buy_params, sell_params, ...)
        * Parameter defaults
        """
        self._ft_hyper_params = detect_all_parameters(self)

        for space in self._ft_hyper_params.keys():
            params_values = deep_merge_dicts(
                self._ft_params_from_file.get(space, {}), getattr(self, f"{space}_params", {})
            )
            self._ft_load_params(self._ft_hyper_params[space], params_values, space, hyperopt)

    def load_params_from_file(self) -> dict:
        filename_str = getattr(self, "__file__", "")
        if not filename_str:
            return {}
        filename = Path(filename_str).with_suffix(".json")

        if filename.is_file():
            logger.info(f"Loading parameters from file {filename}")
            try:
                params = HyperoptTools.load_params(filename)
                if params.get("strategy_name") != self.__class__.__name__:
                    raise OperationalException("Invalid parameter file provided.")
                return params
            except ValueError:
                logger.warning("Invalid parameter file format.")
                return {}
        logger.info("Found no parameter file.")

        return {}

    def _ft_load_params(
        self, params: SpaceParams, param_values: dict, space: str, hyperopt: bool = False
    ) -> None:
        """
        Set optimizable parameter values.
        :param params: Dictionary with new parameter values.
        """
        if not param_values:
            logger.info(f"No params for {space} found, using default values.")

        for param_name, param in params.items():
            param.in_space = hyperopt and HyperoptTools.has_space(self.config, space)
            if not param.space:
                param.space = space

            if param_values and param_name in param_values:
                if param.load:
                    param.value = param_values[param_name]
                    logger.info(f"Strategy Parameter: {param_name} = {param.value}")
                else:
                    logger.warning(
                        f'Parameter "{param_name}" exists, but is disabled. '
                        f'Default value "{param.value}" used.'
                    )
            else:
                logger.info(f"Strategy Parameter(default): {param_name} = {param.value}")

    def get_no_optimize_params(self) -> dict[str, dict]:
        """
        Returns list of Parameters that are not part of the current optimize job
        """
        params: dict[str, dict] = defaultdict(dict)
        for name, p in self.enumerate_parameters():
            if p.space and (not p.optimize or not p.in_space):
                params[p.space][name] = p.value
        return params


def detect_all_parameters(
    obj: HyperStrategyMixin | type[HyperStrategyMixin],
) -> AllSpaceParams:
    """
    Detect all hyperoptable parameters for this object.
    :param obj: Strategy object or class
    :return: Dictionary of detected parameters by space
    """
    auto_categories = ["buy", "sell", "enter", "exit", "protection"]
    result: AllSpaceParams = defaultdict(dict)
    for attr_name in dir(obj):
        if attr_name.startswith("__"):  # Ignore internals
            continue
        attr = getattr(obj, attr_name)
        if not issubclass(attr.__class__, BaseParameter):
            continue
        if not attr.space:
            # space auto detection
            for space in auto_categories:
                if attr_name.startswith(space + "_"):
                    attr.space = space
                    break
        if attr.space is None:
            raise OperationalException(f"Cannot determine parameter space for {attr_name}.")

        if attr.space in ("all", "default") or attr.space.isidentifier() is False:
            raise OperationalException(
                f"'{attr.space}' is not a valid space. Parameter: {attr_name}."
            )
        attr.name = attr_name
        result[attr.space][attr_name] = attr
    return result
