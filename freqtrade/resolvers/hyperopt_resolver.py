# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom hyperopts
"""
import logging
from pathlib import Path
from typing import Optional, Dict

from freqtrade.constants import DEFAULT_HYPEROPT
from freqtrade.optimize.hyperopt_interface import IHyperOpt
from freqtrade.resolvers import IResolver

logger = logging.getLogger(__name__)


class HyperOptResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt class
    """

    __slots__ = ['hyperopt']

    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary or None
        """
        config = config or {}

        # Verify the hyperopt is in the configuration, otherwise fallback to the default hyperopt
        hyperopt_name = config.get('hyperopt') or DEFAULT_HYPEROPT
        self.hyperopt = self._load_hyperopt(hyperopt_name, extra_dir=config.get('hyperopt_path'))

        # Assign ticker_interval to be used in hyperopt
        self.hyperopt.__class__.ticker_interval = config.get('ticker_interval')

        if not hasattr(self.hyperopt, 'populate_buy_trend'):
            logger.warning("Custom Hyperopt does not provide populate_buy_trend. "
                           "Using populate_buy_trend from DefaultStrategy.")
        if not hasattr(self.hyperopt, 'populate_sell_trend'):
            logger.warning("Custom Hyperopt does not provide populate_sell_trend. "
                           "Using populate_sell_trend from DefaultStrategy.")

    def _load_hyperopt(
            self, hyperopt_name: str, extra_dir: Optional[str] = None) -> IHyperOpt:
        """
        Search and loads the specified hyperopt.
        :param hyperopt_name: name of the module to import
        :param extra_dir: additional directory to search for the given hyperopt
        :return: HyperOpt instance or None
        """
        current_path = Path(__file__).parent.parent.joinpath('optimize').resolve()

        abs_paths = [
            current_path.parent.parent.joinpath('user_data/hyperopts'),
            current_path,
        ]

        if extra_dir:
            # Add extra hyperopt directory on top of search paths
            abs_paths.insert(0, Path(extra_dir))

        for _path in abs_paths:
            try:
                hyperopt = self._search_object(directory=_path, object_type=IHyperOpt,
                                               object_name=hyperopt_name)
                if hyperopt:
                    logger.info("Using resolved hyperopt %s from '%s'", hyperopt_name, _path)
                    return hyperopt
            except FileNotFoundError:
                logger.warning('Path "%s" does not exist', _path.relative_to(Path.cwd()))

        raise ImportError(
            "Impossible to load Hyperopt '{}'. This class does not exist"
            " or contains Python code errors".format(hyperopt_name)
        )
