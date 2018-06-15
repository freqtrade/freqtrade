# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom strategies
"""
import importlib.util
import inspect
import logging
import os
from collections import OrderedDict
from typing import Dict, Optional, Type

from freqtrade import constants
from freqtrade.strategy import import_strategy
from freqtrade.strategy.interface import IStrategy

logger = logging.getLogger(__name__)


class StrategyResolver(object):
    """
    This class contains all the logic to load custom strategy class
    """

    __slots__ = ['strategy']

    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary or None
        """
        config = config or {}

        # Verify the strategy is in the configuration, otherwise fallback to the default strategy
        strategy_name = config.get('strategy') or constants.DEFAULT_STRATEGY
        self.strategy: IStrategy = self._load_strategy(strategy_name,
                                                       config=config,
                                                       extra_dir=config.get('strategy_path'))


        # Set attributes
        # Check if we need to override configuration
        if 'minimal_roi' in config:
            self.strategy.minimal_roi = config['minimal_roi']
            logger.info("Override strategy \'minimal_roi\' with value in config file.")
        else:
            config['minimal_roi'] = self.strategy.minimal_roi

        if 'stoploss' in config:
            self.strategy.stoploss = config['stoploss']
            logger.info(
                "Override strategy \'stoploss\' with value in config file: %s.", config['stoploss']
            )
        else:
            config['stoploss'] = self.strategy.stoploss

        if 'ticker_interval' in config:
            self.strategy.ticker_interval = config['ticker_interval']
            logger.info(
                "Override strategy \'ticker_interval\' with value in config file: %s.",
                config['ticker_interval']
            )
        else:
            config['ticker_interval'] = self.strategy.ticker_interval

        # Sort and apply type conversions
        self.strategy.minimal_roi = OrderedDict(sorted(
            {int(key): value for (key, value) in self.strategy.minimal_roi.items()}.items(),
            key=lambda t: t[0]))
        self.strategy.stoploss = float(self.strategy.stoploss)

    def _load_strategy(
            self, strategy_name: str, config: dict, extra_dir: Optional[str] = None) -> IStrategy:
        """
        Search and loads the specified strategy.
        :param strategy_name: name of the module to import
        :param config: configuration for the strategy
        :param extra_dir: additional directory to search for the given strategy
        :return: Strategy instance or None
        """
        current_path = os.path.dirname(os.path.realpath(__file__))
        abs_paths = [
            os.path.join(os.getcwd(), 'user_data', 'strategies'),
            current_path,
        ]

        if extra_dir:
            # Add extra strategy directory on top of search paths
            abs_paths.insert(0, extra_dir)

        for path in abs_paths:
            try:
                strategy = self._search_strategy(path, strategy_name=strategy_name, config=config)
                if strategy:
                    logger.info('Using resolved strategy %s from \'%s\'', strategy_name, path)
                    return import_strategy(strategy, config=config)
            except FileNotFoundError:
                logger.warning('Path "%s" does not exist', path)

        raise ImportError(
            "Impossible to load Strategy '{}'. This class does not exist"
            " or contains Python code errors".format(strategy_name)
        )

    @staticmethod
    def _get_valid_strategies(module_path: str, strategy_name: str) -> Optional[Type[IStrategy]]:
        """
        Returns a list of all possible strategies for the given module_path
        :param module_path: absolute path to the module
        :param strategy_name: Class name of the strategy
        :return: Tuple with (name, class) or None
        """

        # Generate spec based on absolute path
        spec = importlib.util.spec_from_file_location('unknown', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore # importlib does not use typehints

        valid_strategies_gen = (
            obj for name, obj in inspect.getmembers(module, inspect.isclass)
            if strategy_name == name and IStrategy in obj.__bases__
        )
        return next(valid_strategies_gen, None)

    @staticmethod
    def _search_strategy(directory: str, strategy_name: str, config: dict) -> Optional[IStrategy]:
        """
        Search for the strategy_name in the given directory
        :param directory: relative or absolute directory path
        :return: name of the strategy class
        """
        logger.debug('Searching for strategy %s in \'%s\'', strategy_name, directory)
        for entry in os.listdir(directory):
            # Only consider python files
            if not entry.endswith('.py'):
                logger.debug('Ignoring %s', entry)
                continue
            strategy = StrategyResolver._get_valid_strategies(
                os.path.abspath(os.path.join(directory, entry)), strategy_name
            )
            if strategy:
                return strategy(config)
        return None
