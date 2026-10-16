"""
IHyperStrategy interface, hyperoptable Parameter class.
This module defines a base class for auto-hyperoptable strategies.
"""

import logging
from collections import defaultdict
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

from freqtrade.constants import Config
from freqtrade.exceptions import ConfigurationError, DependencyException, OperationalException
from freqtrade.misc import deep_merge_dicts
from freqtrade.optimize.hyperopt_tools import HyperoptTools
from freqtrade.strategy.parameters import (
    BaseParameter,
    CategoricalParameter,
    DecimalParameter,
    NumericParameter,
)


logger = logging.getLogger(__name__)


# Type aliases
SpaceParams = dict[str, BaseParameter]
AllSpaceParams = dict[str, SpaceParams]

# Parameter values, grouped by space - as loaded from a file or provided via the API
SpaceParamValues = dict[str, dict[str, Any]]


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
        self._ft_copy_parameters()

        params = self.load_params_from_file()
        params = params.get("params", {})
        self._ft_params_from_file: SpaceParamValues = params
        self.__ft_params_from_api: SpaceParamValues = {}
        # Init/loading of parameters is done as part of ft_bot_start().

    def _ft_copy_parameters(self) -> None:
        """
        Give this instance its own copy of all class-level parameters.
        Parameters of a base class imported from another module are shared by every strategy
        that inherits from it - so values set on one strategy would leak into all others.
        """
        for attr_name in dir(type(self)):
            if attr_name.startswith("__"):
                continue
            attr = getattr(type(self), attr_name, None)
            if isinstance(attr, BaseParameter):
                setattr(self, attr_name, deepcopy(attr))

    def enumerate_parameters(self, space: str | None = None) -> Iterator[tuple[str, BaseParameter]]:
        """
        Find all optimizable parameters and return (name, attr) iterator.
        :param space: parameter space to filter for, or None for all spaces.
        :return:
        """
        for space_ in [c for c in self._ft_hyper_params if space is None or c == space]:
            for par in self._ft_hyper_params[space_].values():
                yield par.name, par

    def ft_set_special_params_from_file(self) -> None:
        """
        Sets special parameters (stoploss, roi, trailing, max_open_trades) from the
        previously loaded file.
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
        * Parameters from API (Only applicable to webserver mode)
        * Parameters from parameter file
        * Parameters defined in parameters objects (buy_params, sell_params, ...)
        * Parameter defaults
        """
        self._ft_hyper_params = detect_all_parameters(self)
        used_params = self.ft_used_params

        for space in self._ft_hyper_params:
            params_values = deep_merge_dicts(
                used_params.get(space, {}), deepcopy(getattr(self, f"{space}_params", {}))
            )
            self._ft_set_param(self._ft_hyper_params[space], params_values, space, hyperopt)

    @property
    def ft_used_params(self) -> SpaceParamValues:
        """
        Parameters used by this strategy, by space.
        Parameters from the parameter file, with API overrides applied on top.
        Does not contain parameter defaults or values from the `<space>_params` attributes.
        """
        return deep_merge_dicts(self.__ft_params_from_api, deepcopy(self._ft_params_from_file))

    def ft_set_api_params(self, params: list[dict[str, Any]] | None) -> None:
        """
        Store API-provided parameter overrides for the next ft_load_hyper_params() call.
        :raises: ConfigurationError if an override does not match a loadable parameter.
        """
        self.__ft_params_from_api = {}
        if not params:
            return

        if not self._ft_hyper_params:
            self._ft_hyper_params = detect_all_parameters(self)

        strategy_parameters = {name: param for name, param in self.enumerate_parameters()}
        grouped_values: dict[str, dict[str, Any]] = defaultdict(dict)

        for param_data in params:
            name = param_data.get("name", None)
            param_type = param_data.get("param_type", None)
            value = param_data.get("value", None)
            base_warning = "Invalid strategy parameter override:"

            if not name or not param_type:
                raise ConfigurationError(f"{base_warning} missing name or param_type.")

            strategy_param = strategy_parameters.get(name)
            if strategy_param is None:
                raise ConfigurationError(f"{base_warning} '{name}' is not a strategy parameter.")
            if strategy_param.param_type != param_type:
                raise ConfigurationError(
                    f"{base_warning} '{name}' is a {strategy_param.param_type}, not a {param_type}."
                )
            if not strategy_param.load:
                raise ConfigurationError(f"{base_warning} '{name}' is not loaded (load=False).")
            if not strategy_param.space or strategy_param.space not in self._ft_hyper_params:
                raise ConfigurationError(f"{base_warning} '{name}' has no valid space.")
            if (
                isinstance(strategy_param, CategoricalParameter)
                and value not in strategy_param.opt_range
            ):
                raise ConfigurationError(
                    f"{base_warning} '{name}' must be one of {strategy_param.opt_range}."
                )
            if isinstance(strategy_param, DecimalParameter) and value is not None:
                # Store the value the parameter will use - keeps ft_used_params accurate
                value = round(value, strategy_param.decimals)
            if (
                isinstance(strategy_param, NumericParameter)
                and value is not None
                and not strategy_param.low <= value <= strategy_param.high
            ):
                logger.warning(
                    f"Strategy parameter override '{name}' ({value}) is outside of its range "
                    f"({strategy_param.low} - {strategy_param.high})."
                )

            grouped_values[strategy_param.space][name] = value

        self.__ft_params_from_api = dict(grouped_values)

    def load_params_from_file(self) -> dict:
        filename_str = getattr(self, "__file__", "")
        if not filename_str:
            return {}
        filename = Path(filename_str).with_suffix(".json")

        if filename.is_file():
            logger.info(f"Loading parameters from file {filename}")
            try:
                params = HyperoptTools.load_params_from_file(filename)
                if params.get("strategy_name") != self.__class__.__name__:
                    raise OperationalException("Invalid parameter file provided.")
                return params
            except ValueError:
                logger.warning("Invalid parameter file format.")
                return {}
        logger.info("Found no parameter file.")

        return {}

    def _ft_set_param(
        self, params: SpaceParams, param_values: dict, space: str, hyperopt: bool = False
    ) -> None:
        """
        Set optimizable parameter values.
        :param params: Dictionary with new parameter values.
        :param param_values: Dictionary with values to set.
        :param space: The space to which the parameters belong.
        :param hyperopt: Flag indicating if we are in hyperopt mode.
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
            raise DependencyException(f"Cannot determine parameter space for {attr_name}.")

        if attr.space in ("all", "default") or attr.space.isidentifier() is False:
            raise DependencyException(
                f"'{attr.space}' is not a valid space. Parameter: {attr_name}."
            )
        attr.name = attr_name
        result[attr.space][attr_name] = attr
    return result
