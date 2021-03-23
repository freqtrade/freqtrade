"""
IHyperStrategy interface, hyperoptable Parameter class.
This module defines a base class for auto-hyperoptable strategies.
"""
from abc import ABC
from typing import Union, List, Iterator, Tuple

from skopt.space import Integer, Real, Categorical

from freqtrade.strategy.interface import IStrategy


class Parameter(object):
    """
    Defines a parameter that can be optimized by hyperopt.
    """
    default: Union[int, float, str, bool]
    space: List[Union[int, float, str, bool]]
    category: str

    def __init__(self, *, space: List[Union[int, float, str, bool]], default: Union[int, float, str, bool] = None,
                 category: str = None, **kwargs):
        """
        Initialize hyperopt-optimizable parameter.
        :param space: Optimization space. [min, max] for ints and floats or a list of strings for categorial parameters.
        :param default: A default value. Required for ints and floats, optional for categorial parameters (first item
         from the space will be used). Type of default value determines skopt space used for optimization.
        :param category: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param kwargs: Extra parameters to skopt.space.(Integer|Real|Categorical).
        """
        assert 'name' not in kwargs, 'Name is determined by parameter field name and can not be specified manually.'
        self.value = default
        self.space = space
        self.category = category
        self._space_params = kwargs
        if default is None:
            assert len(space) > 0
            self.value = space[0]

    def get_space(self, name: str) -> Union[Integer, Real, Categorical, None]:
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        :return: skopt space of this parameter, or None if parameter is not optimizable (i.e. space is set to None)
        """
        if not self.space:
            return None
        if isinstance(self.value, int):
            assert len(self.space) == 2
            return Integer(*self.space, name=name, **self._space_params)
        if isinstance(self.value, float):
            assert len(self.space) == 2
            return Real(*self.space, name=name, **self._space_params)

        assert len(self.space) > 0
        return Categorical(self.space, name=name, **self._space_params)


class IHyperStrategy(IStrategy, ABC):
    """
    A helper base class which allows HyperOptAuto class to reuse implementations of of buy/sell strategy logic.
    """

    def __init__(self, config):
        super().__init__(config)
        self._load_params(getattr(self, 'buy_params', None))
        self._load_params(getattr(self, 'sell_params', None))

    def enumerate_parameters(self, category: str = None) -> Iterator[Tuple[str, Parameter]]:
        """
        Find all optimizeable parameters and return (name, attr) iterator.
        :param category:
        :return:
        """
        assert category in ('buy', 'sell', None)
        for attr_name in dir(self):
            if not attr_name.startswith('__'):  # Ignore internals, not strictly necessary.
                attr = getattr(self, attr_name)
                if isinstance(attr, Parameter):
                    if category is None or category == attr.category or attr_name.startswith(category + '_'):
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
