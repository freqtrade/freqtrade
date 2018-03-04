# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom strategies
"""
import importlib
import os
import sys
from collections import OrderedDict
from pandas import DataFrame
from freqtrade.logger import Logger
from freqtrade.constants import Constants
from freqtrade.strategy.interface import IStrategy


sys.path.insert(0, r'../../user_data/strategies')


class Strategy(object):
    """
    This class contains all the logic to load custom strategy class
    """
    def __init__(self, config: dict = {}) -> None:
        """
        Load the custom class from config parameter
        :param config:
        :return:
        """
        self.logger = Logger(name=__name__).get_logger()

        # Verify the strategy is in the configuration, otherwise fallback to the default strategy
        if 'strategy' in config:
            strategy = config['strategy']
        else:
            strategy = Constants.DEFAULT_STRATEGY

        # Load the strategy
        self._load_strategy(strategy)

        # Set attributes
        # Check if we need to override configuration
        if 'minimal_roi' in config:
            self.custom_strategy.minimal_roi = config['minimal_roi']
            self.logger.info("Override strategy \'minimal_roi\' with value in config file.")

        if 'stoploss' in config:
            self.custom_strategy.stoploss = config['stoploss']
            self.logger.info(
                "Override strategy \'stoploss\' with value in config file: %s.", config['stoploss']
            )

        if 'ticker_interval' in config:
            self.custom_strategy.ticker_interval = config['ticker_interval']
            self.logger.info(
                "Override strategy \'ticker_interval\' with value in config file: %s.",
                config['ticker_interval']
            )

        # Minimal ROI designed for the strategy
        self.minimal_roi = OrderedDict(sorted(
            {int(key): value for (key, value) in self.custom_strategy.minimal_roi.items()}.items(),
            key=lambda tuple: tuple[0]))  # sort after converting to number

        # Optimal stoploss designed for the strategy
        self.stoploss = float(self.custom_strategy.stoploss)

        self.ticker_interval = self.custom_strategy.ticker_interval

    def _load_strategy(self, strategy_name: str) -> None:
        """
        Search and load the custom strategy. If no strategy found, fallback on the default strategy
        Set the object into self.custom_strategy
        :param strategy_name: name of the module to import
        :return: None
        """

        try:
            # Start by sanitizing the file name (remove any extensions)
            strategy_name = self._sanitize_module_name(filename=strategy_name)

            # Search where can be the strategy file
            path = self._search_strategy(filename=strategy_name)

            # Load the strategy
            self.custom_strategy = self._load_class(path + strategy_name)

        # Fallback to the default strategy
        except (ImportError, TypeError) as error:
            self.logger.error(
                "Impossible to load Strategy 'user_data/strategies/%s.py'. This file does not exist"
                " or contains Python code errors",
                strategy_name
            )
            self.logger.error(
                "The error is:\n%s.",
                error
            )

    def _load_class(self, filename: str) -> IStrategy:
        """
        Import a strategy as a module
        :param filename: path to the strategy (path from freqtrade/strategy/)
        :return: return the strategy class
        """
        module = importlib.import_module(filename, __package__)
        custom_strategy = getattr(module, module.class_name)

        self.logger.info("Load strategy class: %s (%s.py)", module.class_name, filename)
        return custom_strategy()

    @staticmethod
    def _sanitize_module_name(filename: str) -> str:
        """
        Remove any extension from filename
        :param filename: filename to sanatize
        :return: return the filename without extensions
        """
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        return filename

    @staticmethod
    def _search_strategy(filename: str) -> str:
        """
        Search for the Strategy file in different folder
        1. search into the user_data/strategies folder
        2. search into the freqtrade/strategy folder
        3. if nothing found, return None
        :param strategy_name: module name to search
        :return: module path where is the strategy
        """
        pwd = os.path.dirname(os.path.realpath(__file__)) + '/'
        user_data = os.path.join(pwd, '..', '..', 'user_data', 'strategies', filename + '.py')
        strategy_folder = os.path.join(pwd, filename + '.py')

        path = None
        if os.path.isfile(user_data):
            path = 'user_data.strategies.'
        elif os.path.isfile(strategy_folder):
            path = '.'

        return path

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
