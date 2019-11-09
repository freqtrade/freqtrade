# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom hyperopts
"""
import logging
from pathlib import Path
from typing import Optional, Dict

from freqtrade import OperationalException
from freqtrade.constants import DEFAULT_HYPEROPT, DEFAULT_HYPEROPT_LOSS
from freqtrade.optimize.hyperopt_interface import IHyperOpt
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.resolvers import IResolver

logger = logging.getLogger(__name__)


class HyperOptResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt class
    """

    __slots__ = ['hyperopt']

    def __init__(self, config: Dict) -> None:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        """

        # Verify the hyperopt is in the configuration, otherwise fallback to the default hyperopt
        hyperopt_name = config.get('hyperopt') or DEFAULT_HYPEROPT
        self.hyperopt = self._load_hyperopt(hyperopt_name, config,
                                            extra_dir=config.get('hyperopt_path'))

        if not hasattr(self.hyperopt, 'populate_indicators'):
            logger.warning("Hyperopt class does not provide populate_indicators() method. "
                           "Using populate_indicators from the strategy.")
        if not hasattr(self.hyperopt, 'populate_buy_trend'):
            logger.warning("Hyperopt class does not provide populate_buy_trend() method. "
                           "Using populate_buy_trend from the strategy.")
        if not hasattr(self.hyperopt, 'populate_sell_trend'):
            logger.warning("Hyperopt class does not provide populate_sell_trend() method. "
                           "Using populate_sell_trend from the strategy.")

    def _load_hyperopt(
            self, hyperopt_name: str, config: Dict, extra_dir: Optional[str] = None) -> IHyperOpt:
        """
        Search and loads the specified hyperopt.
        :param hyperopt_name: name of the module to import
        :param config: configuration dictionary
        :param extra_dir: additional directory to search for the given hyperopt
        :return: HyperOpt instance or None
        """
        current_path = Path(__file__).parent.parent.joinpath('optimize').resolve()

        abs_paths = self.build_search_paths(config, current_path=current_path,
                                            user_subdir='hyperopts', extra_dir=extra_dir)

        hyperopt = self._load_object(paths=abs_paths, object_type=IHyperOpt,
                                     object_name=hyperopt_name, kwargs={'config': config})
        if hyperopt:
            return hyperopt
        raise OperationalException(
            f"Impossible to load Hyperopt '{hyperopt_name}'. This class does not exist "
            "or contains Python code errors."
        )


class HyperOptLossResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt loss class
    """

    __slots__ = ['hyperoptloss']

    def __init__(self, config: Dict = None) -> None:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary or None
        """
        config = config or {}

        # Verify the hyperopt is in the configuration, otherwise fallback to the default hyperopt
        hyperopt_name = config.get('hyperopt_loss') or DEFAULT_HYPEROPT_LOSS
        self.hyperoptloss = self._load_hyperoptloss(
            hyperopt_name, config, extra_dir=config.get('hyperopt_path'))

        # Assign ticker_interval to be used in hyperopt
        self.hyperoptloss.__class__.ticker_interval = str(config['ticker_interval'])

        if not hasattr(self.hyperoptloss, 'hyperopt_loss_function'):
            raise OperationalException(
                f"Found hyperopt {hyperopt_name} does not implement `hyperopt_loss_function`.")

    def _load_hyperoptloss(
            self, hyper_loss_name: str, config: Dict,
            extra_dir: Optional[str] = None) -> IHyperOptLoss:
        """
        Search and loads the specified hyperopt loss class.
        :param hyper_loss_name: name of the module to import
        :param config: configuration dictionary
        :param extra_dir: additional directory to search for the given hyperopt
        :return: HyperOptLoss instance or None
        """
        current_path = Path(__file__).parent.parent.joinpath('optimize').resolve()

        abs_paths = self.build_search_paths(config, current_path=current_path,
                                            user_subdir='hyperopts', extra_dir=extra_dir)

        hyperoptloss = self._load_object(paths=abs_paths, object_type=IHyperOptLoss,
                                         object_name=hyper_loss_name)
        if hyperoptloss:
            return hyperoptloss

        raise OperationalException(
            f"Impossible to load HyperoptLoss '{hyper_loss_name}'. This class does not exist "
            "or contains Python code errors."
        )
