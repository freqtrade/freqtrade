"""
IHyperStrategy interface, hyperoptable Parameter class.
This module defines a base class for auto-hyperoptable strategies.
"""
from typing import Iterator, Tuple, Any, Optional, Sequence, Union

from skopt.space import Integer, Real, Categorical

from freqtrade.exceptions import OperationalException


class BaseParameter(object):
    """
    Defines a parameter that can be optimized by hyperopt.
    """
    category: Optional[str]
    default: Any
    value: Any
    space: Sequence[Any]

    def __init__(self, *, space: Sequence[Any], default: Any, category: Optional[str] = None,
                 **kwargs):
        """
        Initialize hyperopt-optimizable parameter.
        :param category: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
         parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param kwargs: Extra parameters to skopt.space.(Integer|Real|Categorical).
        """
        if 'name' in kwargs:
            raise OperationalException(
                'Name is determined by parameter field name and can not be specified manually.')
        self.category = category
        self._space_params = kwargs
        self.value = default
        self.space = space

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value})'

    def get_space(self, name: str) -> Union[Integer, Real, Categorical]:
        raise NotImplementedError()


class IntParameter(BaseParameter):
    default: int
    value: int
    space: Sequence[int]

    def __init__(self, *, space: Sequence[int], default: int, category: Optional[str] = None,
                 **kwargs):
        """
        Initialize hyperopt-optimizable parameter.
        :param space: Optimization space, [min, max].
        :param default: A default value.
        :param category: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
         parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param kwargs: Extra parameters to skopt.space.Integer.
        """
        if len(space) != 2:
            raise OperationalException('IntParameter space must be [min, max]')
        super().__init__(space=space, default=default, category=category, **kwargs)

    def get_space(self, name: str) -> Integer:
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Integer(*self.space, name=name, **self._space_params)


class FloatParameter(BaseParameter):
    default: float
    value: float
    space: Sequence[float]

    def __init__(self, *, space: Sequence[float], default: float, category: Optional[str] = None,
                 **kwargs):
        """
        Initialize hyperopt-optimizable parameter.
        :param space: Optimization space, [min, max].
        :param default: A default value.
        :param category: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
         parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param kwargs: Extra parameters to skopt.space.Real.
        """
        if len(space) != 2:
            raise OperationalException('IntParameter space must be [min, max]')
        super().__init__(space=space, default=default, category=category, **kwargs)

    def get_space(self, name: str) -> Real:
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Real(*self.space, name=name, **self._space_params)


class CategoricalParameter(BaseParameter):
    default: Any
    value: Any
    space: Sequence[Any]

    def __init__(self, *, space: Sequence[Any], default: Optional[Any] = None,
                 category: Optional[str] = None,
                 **kwargs):
        """
        Initialize hyperopt-optimizable parameter.
        :param space: Optimization space, [a, b, ...].
        :param default: A default value. If not specified, first item from specified space will be
         used.
        :param category: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
         parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param kwargs: Extra parameters to skopt.space.Categorical.
        """
        if len(space) < 2:
            raise OperationalException(
                'IntParameter space must be [a, b, ...] (at least two parameters)')
        super().__init__(space=space, default=default, category=category, **kwargs)

    def get_space(self, name: str) -> Categorical:
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Categorical(self.space, name=name, **self._space_params)


class HyperStrategyMixin(object):
    """
    A helper base class which allows HyperOptAuto class to reuse implementations of of buy/sell
     strategy logic.
    """

    def __init__(self):
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
                    if category is None or category == attr.category or \
                       attr_name.startswith(category + '_'):
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
                attr.value = params[attr_name]
