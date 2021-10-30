"""
IHyperStrategy interface, hyperoptable Parameter class.
This module defines a base class for auto-hyperoptable strategies.
"""
import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from freqtrade.misc import deep_merge_dicts, json_load
from freqtrade.optimize.hyperopt_tools import HyperoptTools


with suppress(ImportError):
    from skopt.space import Integer, Real, Categorical
    from freqtrade.optimize.space import SKDecimal

from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


class BaseParameter(ABC):
    """
    Defines a parameter that can be optimized by hyperopt.
    """
    category: Optional[str]
    default: Any
    value: Any
    in_space: bool = False
    name: str

    def __init__(self, *, default: Any, space: Optional[str] = None,
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
        self.optimize = optimize
        self.load = load

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value})'

    @abstractmethod
    def get_space(self, name: str) -> Union['Integer', 'Real', 'SKDecimal', 'Categorical']:
        """
        Get-space - will be used by Hyperopt to get the hyperopt Space
        """


class NumericParameter(BaseParameter):
    """ Internal parameter used for Numeric purposes """
    float_or_int = Union[int, float]
    default: float_or_int
    value: float_or_int

    def __init__(self, low: Union[float_or_int, Sequence[float_or_int]],
                 high: Optional[float_or_int] = None, *, default: float_or_int,
                 space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs):
        """
        Initialize hyperopt-optimizable numeric parameter.
        Cannot be instantiated, but provides the validation for other numeric parameters
        :param low: Lower end (inclusive) of optimization space or [low, high].
        :param high: Upper end (inclusive) of optimization space.
                     Must be none of entire range is passed first parameter.
        :param default: A default value.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
                      parameter fieldname is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.*.
        """
        if high is not None and isinstance(low, Sequence):
            raise OperationalException(f'{self.__class__.__name__} space invalid.')
        if high is None or isinstance(low, Sequence):
            if not isinstance(low, Sequence) or len(low) != 2:
                raise OperationalException(f'{self.__class__.__name__} space must be [low, high]')
            self.low, self.high = low
        else:
            self.low = low
            self.high = high

        super().__init__(default=default, space=space, optimize=optimize,
                         load=load, **kwargs)


class IntParameter(NumericParameter):
    default: int
    value: int

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

        super().__init__(low=low, high=high, default=default, space=space, optimize=optimize,
                         load=load, **kwargs)

    def get_space(self, name: str) -> 'Integer':
        """
        Create skopt optimization space.
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
        if self.in_space and self.optimize:
            # Scikit-optimize ranges are "inclusive", while python's "range" is exclusive
            return range(self.low, self.high + 1)
        else:
            return range(self.value, self.value + 1)


class RealParameter(NumericParameter):
    default: float
    value: float

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
        super().__init__(low=low, high=high, default=default, space=space, optimize=optimize,
                         load=load, **kwargs)

    def get_space(self, name: str) -> 'Real':
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Real(low=self.low, high=self.high, name=name, **self._space_params)


class DecimalParameter(NumericParameter):
    default: float
    value: float

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
        :param kwargs: Extra parameters to skopt.space.Integer.
        """
        self._decimals = decimals
        default = round(default, self._decimals)

        super().__init__(low=low, high=high, default=default, space=space, optimize=optimize,
                         load=load, **kwargs)

    def get_space(self, name: str) -> 'SKDecimal':
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return SKDecimal(low=self.low, high=self.high, decimals=self._decimals, name=name,
                         **self._space_params)

    @property
    def range(self):
        """
        Get each value in this space as list.
        Returns a List from low to high (inclusive) in Hyperopt mode.
        Returns a List with 1 item (`value`) in "non-hyperopt" mode, to avoid
        calculating 100ds of indicators.
        """
        if self.in_space and self.optimize:
            low = int(self.low * pow(10, self._decimals))
            high = int(self.high * pow(10, self._decimals)) + 1
            return [round(n * pow(0.1, self._decimals), self._decimals) for n in range(low, high)]
        else:
            return [self.value]


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
        self.opt_range = categories
        super().__init__(default=default, space=space, optimize=optimize,
                         load=load, **kwargs)

    def get_space(self, name: str) -> 'Categorical':
        """
        Create skopt optimization space.
        :param name: A name of parameter field.
        """
        return Categorical(self.opt_range, name=name, **self._space_params)

    @property
    def range(self):
        """
        Get each value in this space as list.
        Returns a List of categories in Hyperopt mode.
        Returns a List with 1 item (`value`) in "non-hyperopt" mode, to avoid
        calculating 100ds of indicators.
        """
        if self.in_space and self.optimize:
            return self.opt_range
        else:
            return [self.value]


class BooleanParameter(CategoricalParameter):

    def __init__(self, *, default: Optional[Any] = None,
                 space: Optional[str] = None, optimize: bool = True, load: bool = True, **kwargs):
        """
        Initialize hyperopt-optimizable Boolean Parameter.
        It's a shortcut to `CategoricalParameter([True, False])`.
        :param default: A default value. If not specified, first item from specified space will be
         used.
        :param space: A parameter category. Can be 'buy' or 'sell'. This parameter is optional if
         parameter field
         name is prefixed with 'buy_' or 'sell_'.
        :param optimize: Include parameter in hyperopt optimizations.
        :param load: Load parameter value from {space}_params.
        :param kwargs: Extra parameters to skopt.space.Categorical.
        """

        categories = [True, False]
        super().__init__(categories=categories, default=default, space=space, optimize=optimize,
                         load=load, **kwargs)


class HyperStrategyMixin(object):
    """
    A helper base class which allows HyperOptAuto class to reuse implementations of buy/sell
     strategy logic.
    """

    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        """
        Initialize hyperoptable strategy mixin.
        """
        self.config = config
        self.ft_buy_params: List[BaseParameter] = []
        self.ft_sell_params: List[BaseParameter] = []
        self.ft_protection_params: List[BaseParameter] = []

        self._load_hyper_params(config.get('runmode') == RunMode.HYPEROPT)

    def enumerate_parameters(self, category: str = None) -> Iterator[Tuple[str, BaseParameter]]:
        """
        Find all optimizable parameters and return (name, attr) iterator.
        :param category:
        :return:
        """
        if category not in ('buy', 'sell', 'protection', None):
            raise OperationalException(
                'Category must be one of: "buy", "sell", "protection", None.')

        if category is None:
            params = self.ft_buy_params + self.ft_sell_params + self.ft_protection_params
        else:
            params = getattr(self, f"ft_{category}_params")

        for par in params:
            yield par.name, par

    @classmethod
    def detect_parameters(cls, category: str) -> Iterator[Tuple[str, BaseParameter]]:
        """ Detect all parameters for 'category' """
        for attr_name in dir(cls):
            if not attr_name.startswith('__'):  # Ignore internals, not strictly necessary.
                attr = getattr(cls, attr_name)
                if issubclass(attr.__class__, BaseParameter):
                    if (attr_name.startswith(category + '_')
                            and attr.category is not None and attr.category != category):
                        raise OperationalException(
                            f'Inconclusive parameter name {attr_name}, category: {attr.category}.')
                    if (category == attr.category or
                            (attr_name.startswith(category + '_') and attr.category is None)):
                        yield attr_name, attr

    @classmethod
    def detect_all_parameters(cls) -> Dict:
        """ Detect all parameters and return them as a list"""
        params: Dict = {
            'buy': list(cls.detect_parameters('buy')),
            'sell': list(cls.detect_parameters('sell')),
            'protection': list(cls.detect_parameters('protection')),
        }
        params.update({
            'count': len(params['buy'] + params['sell'] + params['protection'])
        })

        return params

    def _load_hyper_params(self, hyperopt: bool = False) -> None:
        """
        Load Hyperoptable parameters
        """
        params = self.load_params_from_file()
        params = params.get('params', {})
        self._ft_params_from_file = params
        buy_params = deep_merge_dicts(params.get('buy', {}), getattr(self, 'buy_params', {}))
        sell_params = deep_merge_dicts(params.get('sell', {}), getattr(self, 'sell_params', {}))
        protection_params = deep_merge_dicts(params.get('protection', {}),
                                             getattr(self, 'protection_params', {}))

        self._load_params(buy_params, 'buy', hyperopt)
        self._load_params(sell_params, 'sell', hyperopt)
        self._load_params(protection_params, 'protection', hyperopt)

    def load_params_from_file(self) -> Dict:
        filename_str = getattr(self, '__file__', '')
        if not filename_str:
            return {}
        filename = Path(filename_str).with_suffix('.json')

        if filename.is_file():
            logger.info(f"Loading parameters from file {filename}")
            try:
                with filename.open('r') as f:
                    params = json_load(f)
                if params.get('strategy_name') != self.__class__.__name__:
                    raise OperationalException('Invalid parameter file provided.')
                return params
            except ValueError:
                logger.warning("Invalid parameter file format.")
                return {}
        logger.info("Found no parameter file.")

        return {}

    def _load_params(self, params: Dict, space: str, hyperopt: bool = False) -> None:
        """
        Set optimizable parameter values.
        :param params: Dictionary with new parameter values.
        """
        if not params:
            logger.info(f"No params for {space} found, using default values.")
        param_container: List[BaseParameter] = getattr(self, f"ft_{space}_params")

        for attr_name, attr in self.detect_parameters(space):
            attr.name = attr_name
            attr.in_space = hyperopt and HyperoptTools.has_space(self.config, space)
            if not attr.category:
                attr.category = space

            param_container.append(attr)

            if params and attr_name in params:
                if attr.load:
                    attr.value = params[attr_name]
                    logger.info(f'Strategy Parameter: {attr_name} = {attr.value}')
                else:
                    logger.warning(f'Parameter "{attr_name}" exists, but is disabled. '
                                   f'Default value "{attr.value}" used.')
            else:
                logger.info(f'Strategy Parameter(default): {attr_name} = {attr.value}')

    def get_no_optimize_params(self):
        """
        Returns list of Parameters that are not part of the current optimize job
        """
        params = {
            'buy': {},
            'sell': {},
            'protection': {},
        }
        for name, p in self.enumerate_parameters():
            if not p.optimize or not p.in_space:
                params[p.category][name] = p.value
        return params
