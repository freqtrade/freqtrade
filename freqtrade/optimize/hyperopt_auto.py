"""
HyperOptAuto class.
This module implements a convenience auto-hyperopt class, which can be used together with strategies
 that implement IHyperStrategy interface.
"""
from contextlib import suppress
from typing import Any, Callable, Dict, List

from pandas import DataFrame


with suppress(ImportError):
    from skopt.space import Dimension

from freqtrade.optimize.hyperopt_interface import IHyperOpt


class HyperOptAuto(IHyperOpt):
    """
    This class delegates functionality to Strategy(IHyperStrategy) and Strategy.HyperOpt classes.
     Most of the time Strategy.HyperOpt class would only implement indicator_space and
     sell_indicator_space methods, but other hyperopt methods can be overridden as well.
    """

    def buy_strategy_generator(self, params: Dict[str, Any]) -> Callable:
        def populate_buy_trend(dataframe: DataFrame, metadata: dict):
            for attr_name, attr in self.strategy.enumerate_parameters('buy'):
                if attr.optimize:
                    # noinspection PyProtectedMember
                    attr.value = params[attr_name]
            return self.strategy.populate_buy_trend(dataframe, metadata)

        return populate_buy_trend

    def sell_strategy_generator(self, params: Dict[str, Any]) -> Callable:
        def populate_sell_trend(dataframe: DataFrame, metadata: dict):
            for attr_name, attr in self.strategy.enumerate_parameters('sell'):
                if attr.optimize:
                    # noinspection PyProtectedMember
                    attr.value = params[attr_name]
            return self.strategy.populate_sell_trend(dataframe, metadata)

        return populate_sell_trend

    def _get_func(self, name) -> Callable:
        """
        Return a function defined in Strategy.HyperOpt class, or one defined in super() class.
        :param name: function name.
        :return: a requested function.
        """
        hyperopt_cls = getattr(self.strategy, 'HyperOpt', None)
        default_func = getattr(super(), name)
        if hyperopt_cls:
            return getattr(hyperopt_cls, name, default_func)
        else:
            return default_func

    def _generate_indicator_space(self, category):
        for attr_name, attr in self.strategy.enumerate_parameters(category):
            if attr.optimize:
                yield attr.get_space(attr_name)

    def _get_indicator_space(self, category, fallback_method_name):
        indicator_space = list(self._generate_indicator_space(category))
        if len(indicator_space) > 0:
            return indicator_space
        else:
            return self._get_func(fallback_method_name)()

    def indicator_space(self) -> List['Dimension']:
        return self._get_indicator_space('buy', 'indicator_space')

    def sell_indicator_space(self) -> List['Dimension']:
        return self._get_indicator_space('sell', 'sell_indicator_space')

    def generate_roi_table(self, params: Dict) -> Dict[int, float]:
        return self._get_func('generate_roi_table')(params)

    def roi_space(self) -> List['Dimension']:
        return self._get_func('roi_space')()

    def stoploss_space(self) -> List['Dimension']:
        return self._get_func('stoploss_space')()

    def generate_trailing_params(self, params: Dict) -> Dict:
        return self._get_func('generate_trailing_params')(params)

    def trailing_space(self) -> List['Dimension']:
        return self._get_func('trailing_space')()
