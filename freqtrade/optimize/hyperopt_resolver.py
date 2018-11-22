# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom hyperopts
"""
import importlib.util
import inspect
import logging
import os
from typing import Optional, Dict, Type

from freqtrade.constants import DEFAULT_HYPEROPT
from freqtrade.optimize.hyperopt_interface import IHyperOpt


logger = logging.getLogger(__name__)


class HyperOptResolver(object):
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

    def _load_hyperopt(
            self, hyperopt_name: str, extra_dir: Optional[str] = None) -> IHyperOpt:
        """
        Search and loads the specified hyperopt.
        :param hyperopt_name: name of the module to import
        :param extra_dir: additional directory to search for the given hyperopt
        :return: HyperOpt instance or None
        """
        current_path = os.path.dirname(os.path.realpath(__file__))
        abs_paths = [
            os.path.join(current_path, '..', '..', 'user_data', 'hyperopts'),
            current_path,
        ]

        if extra_dir:
            # Add extra hyperopt directory on top of search paths
            abs_paths.insert(0, extra_dir)

        for path in abs_paths:
            hyperopt = self._search_hyperopt(path, hyperopt_name)
            if hyperopt:
                logger.info('Using resolved hyperopt %s from \'%s\'', hyperopt_name, path)
                return hyperopt

        raise ImportError(
            "Impossible to load Hyperopt '{}'. This class does not exist"
            " or contains Python code errors".format(hyperopt_name)
        )

    @staticmethod
    def _get_valid_hyperopts(module_path: str, hyperopt_name: str) -> Optional[Type[IHyperOpt]]:
        """
        Returns a list of all possible hyperopts for the given module_path
        :param module_path: absolute path to the module
        :param hyperopt_name: Class name of the hyperopt
        :return: Tuple with (name, class) or None
        """

        # Generate spec based on absolute path
        spec = importlib.util.spec_from_file_location('user_data.hyperopts', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore # importlib does not use typehints

        valid_hyperopts_gen = (
            obj for name, obj in inspect.getmembers(module, inspect.isclass)
            if hyperopt_name == name and IHyperOpt in obj.__bases__
        )
        return next(valid_hyperopts_gen, None)

    @staticmethod
    def _search_hyperopt(directory: str, hyperopt_name: str) -> Optional[IHyperOpt]:
        """
        Search for the hyperopt_name in the given directory
        :param directory: relative or absolute directory path
        :return: name of the hyperopt class
        """
        logger.debug('Searching for hyperopt %s in \'%s\'', hyperopt_name, directory)
        for entry in os.listdir(directory):
            # Only consider python files
            if not entry.endswith('.py'):
                logger.debug('Ignoring %s', entry)
                continue
            hyperopt = HyperOptResolver._get_valid_hyperopts(
                os.path.abspath(os.path.join(directory, entry)), hyperopt_name
            )
            if hyperopt:
                return hyperopt()
        return None
