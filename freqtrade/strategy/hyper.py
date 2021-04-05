"""
IHyperStrategy interface, hyperoptable Parameter class.
This module defines a base class for auto-hyperoptable strategies.
"""
import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any, Iterator, Optional, Sequence, Tuple, Union


with suppress(ImportError):
    from skopt.space import Integer, Real, Categorical

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


class BaseParameter(ABC):
    """
    Defines a parameter that can be optimized by hyperopt.
    """
    category: Optional[str]
    default: Any
    value: Any
    opt_range: Sequence[Any]

    def __init__(self, *, opt_range: Sequence[Any], default: Any, space: Optional[str] = None,
                 optimize: bool = True, load: bool = True, **kwargs):
        """
        Initialize hyperopt-optimizable parameter.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
         parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.(Integer|Real|Categorical).
        """
        if 'name' in kwargs:
            raise OperationalException(
                'Name is determined by parameter field name and can not be specified manually.')
        self.category = space
        self._space_params = kwargs
        self.value = default
        self.opt_range = opt_range
        self.optimize = optimize
        self.load = load

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value})'

    @abstractmethod
    def get_space(self, name: str) -> Union['Integer', 'Real', 'Categorical']:
        """
        Get-space - will be used by Hyperopt to get the hyperopt Space
        """

    def _set_value(self, value: Any):
        """
        Update current value. Used by hyperopt functions for the purpose where optimization and
         value spaces differ.
        :param value: A numerical value.
        """
        self.value = value


class IntParameter(BaseParameter):
    default: int
    value: int
    opt_range: Sequence[int]

    def __init__(self, low: Union[int, Sequence[int]], high: Optional[int] = None, *, default: int,
                 space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs):
        """
        Initialize hyperopt-optimizable integer parameter.
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none of entire range is passed first parameter.
        :param default: A default value.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
                      parameter fieldname is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.Integer.
        """
        if high is not None and isinstance(low, Sequence):
            raise OperationalException('IntParameter space invalid.')
        if high is None or isinstance(low, Sequence):
            if not isinstance(low, Sequence) or len(low) != 2:
                raise OperationalException('IntParameter space must be [low, high]')
            opt_range = low
        else:
            opt_range = [low, high]
        super().__init__(opt_range=opt_range, default=default, space=space, optimize=optimize,
                         load=load, **kwargs)

    def get_space(self, name: str) -> 'Integer':
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Integer(*self.opt_range, name=name, **self._space_params)


class RealParameter(BaseParameter):
    default: float
    value: float
    opt_range: Sequence[float]

    def __init__(self, low: Union[float, Sequence[float]], high: Optional[float] = None, *,
                 default: float, space: Optional[str] = None, optimize: bool = True,
                 load: bool = True, **kwargs):
        """
        Initialize hyperopt-optimizable floating point parameter with unlimited precision.
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none if entire range is passed first parameter.
        :param default: A default value.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
                      parameter fieldname is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.Real.
        """
        if high is not None and isinstance(low, Sequence):
            raise OperationalException(f'{self.__class__.__name__} space invalid.')
        if high is None or isinstance(low, Sequence):
            if not isinstance(low, Sequence) or len(low) != 2:
                raise OperationalException(f'{self.__class__.__name__} space must be [low, high]')
            opt_range = low
        else:
            opt_range = [low, high]
        super().__init__(opt_range=opt_range, default=default, space=space, optimize=optimize,
                         load=load, **kwargs)

    def get_space(self, name: str) -> 'Real':
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Real(*self.opt_range, name=name, **self._space_params)


class DecimalParameter(RealParameter):
    default: float
    value: float
    opt_range: Sequence[float]

    def __init__(self, low: Union[float, Sequence[float]], high: Optional[float] = None, *,
                 default: float, decimals: int = 3, space: Optional[str] = None,
                 optimize: bool = True, load: bool = True, **kwargs):
        """
        Initialize hyperopt-optimizable decimal parameter with a limited precision.
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none if entire range is passed first parameter.
        :param default: A default value.
        :param decimals: A number of decimals after floating point to be included in testing.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
                      parameter fieldname is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.Real.
        """
        self._decimals = decimals
        default = round(default, self._decimals)
        super().__init__(low=low, high=high, default=default, space=space, optimize=optimize,
                         load=load, **kwargs)

    def get_space(self, name: str) -> 'Integer':
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        low = int(self.opt_range[0] * pow(10, self._decimals))
        high = int(self.opt_range[1] * pow(10, self._decimals))
        return Integer(low, high, name=name, **self._space_params)

    def _set_value(self, value: int):
        """
        Update current value. Used by hyperopt functions for the purpose where optimization and
         value spaces differ.
        :param value: An integer value.
        """
        self.value = round(value * pow(0.1, self._decimals), self._decimals)


class CategoricalParameter(BaseParameter):
    default: Any
    value: Any
    opt_range: Sequence[Any]

    def __init__(self, categories: Sequence[Any], *, default: Optional[Any] = None,
                 space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs):
        """
        Initialize hyperopt-optimizable parameter.
        :param categories: Optimization space, [a, b, ...].
        :param default: A default value. If not specified, first item from specified space will be
         used.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
         parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.Categorical.
        """
        if len(categories) < 2:
            raise OperationalException(
                'CategoricalParameter space must be [a, b, ...] (at least two parameters)')
        super().__init__(opt_range=categories, default=default, space=space, optimize=optimize,
                         load=load, **kwargs)

    def get_space(self, name: str) -> 'Categorical':
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Categorical(self.opt_range, name=name, **self._space_params)


class HyperStrategyMixin(object):
    """
    A helper base class which allows HyperOptAuto class to reuse implementations of of buy/sell
     strategy logic.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize hyperoptable strategy mixin.
        """
        self._load_params(getattr(self, 'buy_params', None))
        self._load_params(getattr(self, 'sell_params', None))

    def enumerate_parameters(self, category: str = None) -> Iterator[Tuple[str, BaseParameter]]:
        """
        Find all optimizeable parameters and return (name, attr) iterator.
        :param category:
        :return:
        """
        if category not in ('buy', 'sell', None):
            raise OperationalException('Category must be one of: "buy", "sell", None.')
        for attr_name in dir(self):
            if not attr_name.startswith('__'):  # Ignore internals, not strictly necessary.
                attr = getattr(self, attr_name)
                if issubclass(attr.__class__, BaseParameter):
                    if (category and attr_name.startswith(category + '_')
                            and attr.category is not None and attr.category != category):
                        raise OperationalException(
                            f'Inconclusive parameter name {attr_name}, category: {attr.category}.')
                    if (category is None or category == attr.category or
                            (attr_name.startswith(category + '_') and attr.category is None)):
                        yield attr_name, attr

    def _load_params(self, params: dict) -> None:
        """
        Set optimizeable parameter values.
        :param params: Dictionary with new parameter values.
        """
        if not params:
            return
        for attr_name, attr in self.enumerate_parameters():
            if attr_name in params:
                if attr.load:
                    attr.value = params[attr_name]
                    logger.info(f'{attr_name} = {attr.value}')
                else:
                    logger.warning(f'Parameter "{attr_name}" exists, but is disabled. '
                                   f'Default value "{attr.value}" used.')
