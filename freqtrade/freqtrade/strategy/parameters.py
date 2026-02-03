"""
IHyperStrategy interface, hyperoptable Parameter class.
This module defines a base class for auto-hyperoptable strategies.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Union

from freqtrade.enums import HyperoptState
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer


with suppress(ImportError):
    from freqtrade.optimize.space import (
        Categorical,
        Integer,
        Real,
        SKDecimal,
    )

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


class BaseParameter(ABC):
    """
    Defines a parameter that can be optimized by hyperopt.
    """

    space: str | None
    default: Any
    value: Any
    in_space: bool = False
    name: str

    def __init__(
        self,
        *,
        default: Any,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        Initialize hyperopt-optimizable parameter.
        :param space: The parameter space. Can be 'buy', 'sell', or a string that's also a
                valid python identifier.
                This parameter is optional if parameter name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to optuna.distributions.
                (IntDistribution|FloatDistribution|CategoricalDistribution).
        """
        if "name" in kwargs:
            raise OperationalException(
                "Name is determined by parameter field name and can not be specified manually."
            )
        self.space = space
        self._space_params = kwargs
        self.value = default
        self.optimize = optimize
        self.load = load

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    @abstractmethod
    def get_space(self, name: str) -> Union["Integer", "Real", "SKDecimal", "Categorical"]:
        """
        Get-space - will be used by Hyperopt to get the hyperopt Space
        """

    def can_optimize(self):
        return (
            self.in_space
            and self.optimize
            and HyperoptStateContainer.state != HyperoptState.OPTIMIZE
        )


class NumericParameter(BaseParameter):
    """Internal parameter used for Numeric purposes"""

    float_or_int = int | float
    default: float_or_int
    value: float_or_int

    def __init__(
        self,
        low: float_or_int | Sequence[float_or_int],
        high: float_or_int | None = None,
        *,
        default: float_or_int,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        Initialize hyperopt-optimizable numeric parameter.
        Cannot be instantiated, but provides the validation for other numeric parameters
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none of entire range is passed first parameter.
        :param default: A default value.
        :param space: The parameter space. Can be 'buy', 'sell', or a string that's also a
                valid python identifier.
                This parameter is optional if parameter name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to optuna.distributions.*.
        """
        if high is not None and isinstance(low, Sequence):
            raise OperationalException(f"{self.__class__.__name__} space invalid.")
        if high is None or isinstance(low, Sequence):
            if not isinstance(low, Sequence) or len(low) != 2:
                raise OperationalException(f"{self.__class__.__name__} space must be [low, high]")
            self.low, self.high = low
        else:
            self.low = low
            self.high = high

        super().__init__(default=default, space=space, optimize=optimize, load=load, **kwargs)


class IntParameter(NumericParameter):
    default: int
    value: int
    low: int
    high: int

    def __init__(
        self,
        low: int | Sequence[int],
        high: int | None = None,
        *,
        default: int,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        Initialize hyperopt-optimizable integer parameter.
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none of entire range is passed first parameter.
        :param default: A default value.
        :param space: The parameter space. Can be 'buy', 'sell', or a string that's also a
                valid python identifier.
                This parameter is optional if parameter name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to optuna.distributions.IntDistribution.
        """

        super().__init__(
            low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs
        )

    def get_space(self, name: str) -> "Integer":
        """
        Create optuna distribution space.
        :param name: A name of parameter field.
        """
        return Integer(low=self.low, high=self.high, name=name, **self._space_params)

    @property
    def range(self):
        """
        Get each value in this space as list.
        Returns a List from low to high (inclusive) in Hyperopt mode.
        Returns a List with 1 item (`value`) in "non-hyperopt" mode, to avoid
        calculating 100ds of indicators.
        """
        if self.can_optimize():
            # optuna distributions ranges are "inclusive", while python's "range" is exclusive
            return range(self.low, self.high + 1)
        else:
            return range(self.value, self.value + 1)


class RealParameter(NumericParameter):
    default: float
    value: float

    def __init__(
        self,
        low: float | Sequence[float],
        high: float | None = None,
        *,
        default: float,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        Initialize hyperopt-optimizable floating point parameter with unlimited precision.
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none if entire range is passed first parameter.
        :param default: A default value.
        :param space: The parameter space. Can be 'buy', 'sell', or a string that's also a
                valid python identifier.
                This parameter is optional if parameter name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to optuna.distributions.FloatDistribution.
        """
        super().__init__(
            low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs
        )

    def get_space(self, name: str) -> "Real":
        """
        Create optimization space.
        :param name: A name of parameter field.
        """
        return Real(low=self.low, high=self.high, name=name, **self._space_params)


class DecimalParameter(NumericParameter):
    default: float

    def __init__(
        self,
        low: float | Sequence[float],
        high: float | None = None,
        *,
        default: float,
        decimals: int = 3,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        Initialize hyperopt-optimizable decimal parameter with a limited precision.
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none if entire range is passed first parameter.
        :param default: A default value.
        :param decimals: A number of decimals after floating point to be included in testing.
        :param space: The parameter space. Can be 'buy', 'sell', or a string that's also a
                valid python identifier.
                This parameter is optional if parameter name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to optuna's NumericParameter.
        """
        self._decimals = decimals
        default = round(default, self._decimals)

        super().__init__(
            low=low, high=high, default=default, space=space, optimize=optimize, load=load, **kwargs
        )

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_value: float):
        self._value = round(new_value, self._decimals)

    def get_space(self, name: str) -> "SKDecimal":
        """
        Create optimization space.
        :param name: A name of parameter field.
        """
        return SKDecimal(
            low=self.low, high=self.high, decimals=self._decimals, name=name, **self._space_params
        )

    @property
    def range(self):
        """
        Get each value in this space as list.
        Returns a List from low to high (inclusive) in Hyperopt mode.
        Returns a List with 1 item (`value`) in "non-hyperopt" mode, to avoid
        calculating 100ds of indicators.
        """
        if self.can_optimize():
            low = int(self.low * pow(10, self._decimals))
            high = int(self.high * pow(10, self._decimals)) + 1
            return [round(n * pow(0.1, self._decimals), self._decimals) for n in range(low, high)]
        else:
            return [self.value]


class CategoricalParameter(BaseParameter):
    default: Any
    value: Any
    opt_range: Sequence[Any]

    def __init__(
        self,
        categories: Sequence[Any],
        *,
        default: Any | None = None,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        Initialize hyperopt-optimizable parameter.
        :param categories: Optimization space, [a, b, ...].
        :param default: A default value. If not specified, first item from specified space will be
                used.
        :param space: The parameter space. Can be 'buy', 'sell', or a string that's also a
                valid python identifier.
                This parameter is optional if parameter name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Compatibility. Optuna's CategoricalDistribution does not
                       accept extra parameters
        """
        if len(categories) < 2:
            raise OperationalException(
                "CategoricalParameter space must be [a, b, ...] (at least two parameters)"
            )
        self.opt_range = categories
        super().__init__(default=default, space=space, optimize=optimize, load=load, **kwargs)

    def get_space(self, name: str) -> "Categorical":
        """
        Create optuna distribution space.
        :param name: A name of parameter field.
        """
        return Categorical(self.opt_range, name=name)

    @property
    def range(self):
        """
        Get each value in this space as list.
        Returns a List of categories in Hyperopt mode.
        Returns a List with 1 item (`value`) in "non-hyperopt" mode, to avoid
        calculating 100ds of indicators.
        """
        if self.can_optimize():
            return self.opt_range
        else:
            return [self.value]


class BooleanParameter(CategoricalParameter):
    def __init__(
        self,
        *,
        default: Any | None = None,
        space: str | None = None,
        optimize: bool = True,
        load: bool = True,
        **kwargs,
    ):
        """
        Initialize hyperopt-optimizable Boolean Parameter.
        It's a shortcut to `CategoricalParameter([True, False])`.
        :param default: A default value. If not specified, first item from specified space will be
                used.
        :param space: The parameter space. Can be 'buy', 'sell', or a string that's also a
                valid python identifier.
                This parameter is optional if parameter name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to optuna.distributions.CategoricalDistribution.
        """

        categories = [True, False]
        super().__init__(
            categories=categories,
            default=default,
            space=space,
            optimize=optimize,
            load=load,
            **kwargs,
        )
