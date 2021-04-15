"""
IHyperOpt interface
This module defines the interface to apply for hyperopt
"""
import logging
import math
from abc import ABC
from typing import Any, Callable, Dict, List

from skopt.space import Categorical, Dimension, Integer

from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import round_dict
from freqtrade.optimize.space import SKDecimal
from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


def _format_exception_message(method: str, space: str) -> str:
    return (f"The '{space}' space is included into the hyperoptimization "
            f"but {method}() method is not found in your "
            f"custom Hyperopt class. You should either implement this "
            f"method or remove the '{space}' space from hyperoptimization.")


class IHyperOpt(ABC):
    """
    Interface for freqtrade hyperopt
    Defines the mandatory structure must follow any custom hyperopt

    Class attributes you can use:
        timeframe -> int: value of the timeframe to use for the strategy
    """
    ticker_interval: str  # DEPRECATED
    timeframe: str
    strategy: IStrategy

    def __init__(self, config: dict) -> None:
        self.config = config

        # Assign ticker_interval to be used in hyperopt
        IHyperOpt.ticker_interval = str(config['timeframe'])  # DEPRECATED
        IHyperOpt.timeframe = str(config['timeframe'])

    def buy_strategy_generator(self, params: Dict[str, Any]) -> Callable:
        """
        Create a buy strategy generator.
        """
        raise OperationalException(_format_exception_message('buy_strategy_generator', 'buy'))

    def sell_strategy_generator(self, params: Dict[str, Any]) -> Callable:
        """
        Create a sell strategy generator.
        """
        raise OperationalException(_format_exception_message('sell_strategy_generator', 'sell'))

    def indicator_space(self) -> List[Dimension]:
        """
        Create an indicator space.
        """
        raise OperationalException(_format_exception_message('indicator_space', 'buy'))

    def sell_indicator_space(self) -> List[Dimension]:
        """
        Create a sell indicator space.
        """
        raise OperationalException(_format_exception_message('sell_indicator_space', 'sell'))

    def generate_roi_table(self, params: Dict) -> Dict[int, float]:
        """
        Create a ROI table.

        Generates the ROI table that will be used by Hyperopt.
        You may override it in your custom Hyperopt class.
        """
        roi_table = {}
        roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
        roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
        roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
        roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

        return roi_table

    def roi_space(self) -> List[Dimension]:
        """
        Create a ROI space.

        Defines values to search for each ROI steps.

        This method implements adaptive roi hyperspace with varied
        ranges for parameters which automatically adapts to the
        timeframe used.

        It's used by Freqtrade by default, if no custom roi_space method is defined.
        """

        # Default scaling coefficients for the roi hyperspace. Can be changed
        # to adjust resulting ranges of the ROI tables.
        # Increase if you need wider ranges in the roi hyperspace, decrease if shorter
        # ranges are needed.
        roi_t_alpha = 1.0
        roi_p_alpha = 1.0

        timeframe_min = timeframe_to_minutes(self.timeframe)

        # We define here limits for the ROI space parameters automagically adapted to the
        # timeframe used by the bot:
        #
        # * 'roi_t' (limits for the time intervals in the ROI tables) components
        #   are scaled linearly.
        # * 'roi_p' (limits for the ROI value steps) components are scaled logarithmically.
        #
        # The scaling is designed so that it maps exactly to the legacy Freqtrade roi_space()
        # method for the 5m timeframe.
        roi_t_scale = timeframe_min / 5
        roi_p_scale = math.log1p(timeframe_min) / math.log1p(5)
        roi_limits = {
            'roi_t1_min': int(10 * roi_t_scale * roi_t_alpha),
            'roi_t1_max': int(120 * roi_t_scale * roi_t_alpha),
            'roi_t2_min': int(10 * roi_t_scale * roi_t_alpha),
            'roi_t2_max': int(60 * roi_t_scale * roi_t_alpha),
            'roi_t3_min': int(10 * roi_t_scale * roi_t_alpha),
            'roi_t3_max': int(40 * roi_t_scale * roi_t_alpha),
            'roi_p1_min': 0.01 * roi_p_scale * roi_p_alpha,
            'roi_p1_max': 0.04 * roi_p_scale * roi_p_alpha,
            'roi_p2_min': 0.01 * roi_p_scale * roi_p_alpha,
            'roi_p2_max': 0.07 * roi_p_scale * roi_p_alpha,
            'roi_p3_min': 0.01 * roi_p_scale * roi_p_alpha,
            'roi_p3_max': 0.20 * roi_p_scale * roi_p_alpha,
        }
        logger.debug(f"Using roi space limits: {roi_limits}")
        p = {
            'roi_t1': roi_limits['roi_t1_min'],
            'roi_t2': roi_limits['roi_t2_min'],
            'roi_t3': roi_limits['roi_t3_min'],
            'roi_p1': roi_limits['roi_p1_min'],
            'roi_p2': roi_limits['roi_p2_min'],
            'roi_p3': roi_limits['roi_p3_min'],
        }
        logger.info(f"Min roi table: {round_dict(self.generate_roi_table(p), 3)}")
        p = {
            'roi_t1': roi_limits['roi_t1_max'],
            'roi_t2': roi_limits['roi_t2_max'],
            'roi_t3': roi_limits['roi_t3_max'],
            'roi_p1': roi_limits['roi_p1_max'],
            'roi_p2': roi_limits['roi_p2_max'],
            'roi_p3': roi_limits['roi_p3_max'],
        }
        logger.info(f"Max roi table: {round_dict(self.generate_roi_table(p), 3)}")

        return [
            Integer(roi_limits['roi_t1_min'], roi_limits['roi_t1_max'], name='roi_t1'),
            Integer(roi_limits['roi_t2_min'], roi_limits['roi_t2_max'], name='roi_t2'),
            Integer(roi_limits['roi_t3_min'], roi_limits['roi_t3_max'], name='roi_t3'),
            SKDecimal(roi_limits['roi_p1_min'], roi_limits['roi_p1_max'], decimals=3,
                      name='roi_p1'),
            SKDecimal(roi_limits['roi_p2_min'], roi_limits['roi_p2_max'], decimals=3,
                      name='roi_p2'),
            SKDecimal(roi_limits['roi_p3_min'], roi_limits['roi_p3_max'], decimals=3,
                      name='roi_p3'),
        ]

    def stoploss_space(self) -> List[Dimension]:
        """
        Create a stoploss space.

        Defines range of stoploss values to search.
        You may override it in your custom Hyperopt class.
        """
        return [
            SKDecimal(-0.35, -0.02, decimals=3, name='stoploss'),
        ]

    def generate_trailing_params(self, params: Dict) -> Dict:
        """
        Create dict with trailing stop parameters.
        """
        return {
            'trailing_stop': params['trailing_stop'],
            'trailing_stop_positive': params['trailing_stop_positive'],
            'trailing_stop_positive_offset': (params['trailing_stop_positive'] +
                                              params['trailing_stop_positive_offset_p1']),
            'trailing_only_offset_is_reached': params['trailing_only_offset_is_reached'],
        }

    def trailing_space(self) -> List[Dimension]:
        """
        Create a trailing stoploss space.

        You may override it in your custom Hyperopt class.
        """
        return [
            # It was decided to always set trailing_stop is to True if the 'trailing' hyperspace
            # is used. Otherwise hyperopt will vary other parameters that won't have effect if
            # trailing_stop is set False.
            # This parameter is included into the hyperspace dimensions rather than assigning
            # it explicitly in the code in order to have it printed in the results along with
            # other 'trailing' hyperspace parameters.
            Categorical([True], name='trailing_stop'),

            SKDecimal(0.01, 0.35, decimals=3, name='trailing_stop_positive'),

            # 'trailing_stop_positive_offset' should be greater than 'trailing_stop_positive',
            # so this intermediate parameter is used as the value of the difference between
            # them. The value of the 'trailing_stop_positive_offset' is constructed in the
            # generate_trailing_params() method.
            # This is similar to the hyperspace dimensions used for constructing the ROI tables.
            SKDecimal(0.001, 0.1, decimals=3, name='trailing_stop_positive_offset_p1'),

            Categorical([True, False], name='trailing_only_offset_is_reached'),
        ]

    # This is needed for proper unpickling the class attribute ticker_interval
    # which is set to the actual value by the resolver.
    # Why do I still need such shamanic mantras in modern python?
    def __getstate__(self):
        state = self.__dict__.copy()
        state['timeframe'] = self.timeframe
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        IHyperOpt.ticker_interval = state['timeframe']
        IHyperOpt.timeframe = state['timeframe']
