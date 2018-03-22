# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom hyperopts
"""
import importlib
import os
import sys
from typing import Dict, Any, Callable

from pandas import DataFrame

from freqtrade.constants import Constants
from freqtrade.logger import Logger
from freqtrade.optimize.interface import IHyperOpt

sys.path.insert(0, r'../../user_data/hyperopts')


class CustomHyperOpt(object):
    """
    This class contains all the logic to load custom hyperopt class
    """
    def __init__(self, config: dict = {}) -> None:
        """
        Load the custom class from config parameter
        :param config:
        :return:
        """
        self.logger = Logger(name=__name__).get_logger()

        # Verify the hyperopt is in the configuration, otherwise fallback to the default hyperopt
        if 'hyperopt' in config:
            hyperopt = config['hyperopt']
        else:
            hyperopt = Constants.DEFAULT_HYPEROPT

        # Load the hyperopt
        self._load_hyperopt(hyperopt)

    def _load_hyperopt(self, hyperopt_name: str) -> None:
        """
        Search and load the custom hyperopt. If no hyperopt found, fallback on the default hyperopt
        Set the object into self.custom_hyperopt
        :param hyperopt_name: name of the module to import
        :return: None
        """

        try:
            # Start by sanitizing the file name (remove any extensions)
            hyperopt_name = self._sanitize_module_name(filename=hyperopt_name)

            # Search where can be the hyperopt file
            path = self._search_hyperopt(filename=hyperopt_name)

            # Load the hyperopt
            self.custom_hyperopt = self._load_class(path + hyperopt_name)

        # Fallback to the default hyperopt
        except (ImportError, TypeError) as error:
            self.logger.error(
                "Impossible to load Hyperopt 'user_data/hyperopts/%s.py'. This file does not exist"
                " or contains Python code errors",
                hyperopt_name
            )
            self.logger.error(
                "The error is:\n%s.",
                error
            )

    def _load_class(self, filename: str) -> IHyperOpt:
        """
        Import a hyperopt as a module
        :param filename: path to the hyperopt (path from freqtrade/optimize/)
        :return: return the hyperopt class
        """
        module = importlib.import_module(filename, __package__)
        custom_hyperopt = getattr(module, module.class_name)

        self.logger.info("Load hyperopt class: %s (%s.py)", module.class_name, filename)
        return custom_hyperopt()

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
    def _search_hyperopt(filename: str) -> str:
        """
        Search for the hyperopt file in different folder
        1. search into the user_data/hyperopts folder
        2. search into the freqtrade/optimize folder
        3. if nothing found, return None
        :param hyperopt_name: module name to search
        :return: module path where is the hyperopt
        """
        pwd = os.path.dirname(os.path.realpath(__file__)) + '/'
        user_data = os.path.join(pwd, '..', '..', 'user_data', 'hyperopts', filename + '.py')
        hyperopt_folder = os.path.join(pwd, filename + '.py')

        path = None
        if os.path.isfile(user_data):
            path = 'user_data.hyperopts.'
        elif os.path.isfile(hyperopt_folder):
            path = '.'

        return path

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell hyperopt
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        return self.custom_hyperopt.populate_indicators(dataframe)

    def buy_strategy_generator(self, params: Dict[str, Any]) -> Callable:
        """
        Create a buy strategy generator
        """
        return self.custom_hyperopt.buy_strategy_generator(params)

    def indicator_space(self) -> Dict[str, Any]:
        """
        Create an indicator space
        """
        return self.custom_hyperopt.indicator_space()

    def generate_roi_table(self, params: Dict) -> Dict[int, float]:
        """
        Create an roi table
        """
        return self.custom_hyperopt.generate_roi_table(params)

    def stoploss_space(self) -> Dict[str, Any]:
        """
        Create a stoploss space
        """
        return self.custom_hyperopt.stoploss_space()

    def roi_space(self) -> Dict[str, Any]:
        """
        Create a roi space
        """
        return self.custom_hyperopt.roi_space()
