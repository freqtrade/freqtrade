# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom strategies
"""
import importlib.util
import inspect
import logging
import os
from collections import OrderedDict
from typing import Optional, Dict, Type

from pandas import DataFrame

from freqtrade.constants import Constants
from freqtrade.strategy.interface import IStrategy


logger = logging.getLogger(__name__)


class StrategyResolver(object):
    """
    This class contains all the logic to load custom strategy class
    """
    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Load the custom class from config parameter
        :param config:
        :return:
        """
        config = config or {}

        # Verify the strategy is in the configuration, otherwise fallback to the default strategy
        if 'strategy' in config:
            strategy = config['strategy']
        else:
            strategy = Constants.DEFAULT_STRATEGY

        # Try to load the strategy
        self._load_strategy(strategy)

        # Set attributes
        # Check if we need to override configuration
        if 'minimal_roi' in config:
            self.custom_strategy.minimal_roi = config['minimal_roi']
            logger.info("Override strategy \'minimal_roi\' with value in config file.")

        if 'stoploss' in config:
            self.custom_strategy.stoploss = config['stoploss']
            logger.info(
                "Override strategy \'stoploss\' with value in config file: %s.", config['stoploss']
            )

        if 'ticker_interval' in config:
            self.custom_strategy.ticker_interval = config['ticker_interval']
            logger.info(
                "Override strategy \'ticker_interval\' with value in config file: %s.",
                config['ticker_interval']
            )

        # Minimal ROI designed for the strategy
        self.minimal_roi = OrderedDict(sorted(
            {int(key): value for (key, value) in self.custom_strategy.minimal_roi.items()}.items(),
            key=lambda t: t[0]))  # sort after converting to number

        # Optimal stoploss designed for the strategy
        self.stoploss = float(self.custom_strategy.stoploss)

        self.ticker_interval = int(self.custom_strategy.ticker_interval)

    def _load_strategy(self, strategy_name: str) -> None:
        """
        Search and loads the specified strategy.
        :param strategy_name: name of the module to import
        :return: None
        """
        try:
            current_path = os.path.dirname(os.path.realpath(__file__))
            abs_paths = [
                os.path.join(current_path, '..', '..', 'user_data', 'strategies'),
                current_path,
            ]
            for path in abs_paths:
                self.custom_strategy = self._search_strategy(path, strategy_name)
                if self.custom_strategy:
                    logger.info('Using resolved strategy %s from \'%s\'', strategy_name, path)
                    return None

            raise ImportError('not found')
        # Fallback to the default strategy
        except (ImportError, TypeError) as error:
            logger.error(
                "Impossible to load Strategy '%s'. This class does not exist"
                " or contains Python code errors",
                strategy_name
            )
            logger.error(
                "The error is:\n%s.",
                error
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
        spec = importlib.util.spec_from_file_location('user_data.strategies', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        valid_strategies_gen = (
            obj for name, obj in inspect.getmembers(module, inspect.isclass)
            if strategy_name == name and IStrategy in obj.__bases__
        )
        return next(valid_strategies_gen, None)

    @staticmethod
    def _search_strategy(directory: str, strategy_name: str) -> Optional[IStrategy]:
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
                return strategy()
        return None

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        return self.custom_strategy.populate_indicators(dataframe)

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        :return:
        """
        return self.custom_strategy.populate_buy_trend(dataframe)

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        return self.custom_strategy.populate_sell_trend(dataframe)
