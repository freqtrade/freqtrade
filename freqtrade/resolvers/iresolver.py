# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom hyperopts
"""
import importlib.util
import inspect
import logging
import os
from typing import Optional, Dict, Type, Any

from freqtrade.constants import DEFAULT_HYPEROPT
from freqtrade.optimize.hyperopt_interface import IHyperOpt


logger = logging.getLogger(__name__)


class IResolver(object):
    """
    This class contains all the logic to load custom hyperopt class
    """

    def __init__(self, object_type, config: Optional[Dict] = None) -> None:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary or None
        """
        config = config or {}

    @staticmethod
    def _get_valid_objects(object_type, module_path: str,
                           object_name: str) -> Optional[Type[Any]]:
        """
        Returns a list of all possible objects for the given module_path of type oject_type
        :param object_type: object_type (class)
        :param module_path: absolute path to the module
        :param object_name: Class name of the object
        :return: Tuple with (name, class) or None
        """

        # Generate spec based on absolute path
        spec = importlib.util.spec_from_file_location('unknown', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore # importlib does not use typehints

        valid_objects_gen = (
            obj for name, obj in inspect.getmembers(module, inspect.isclass)
            if object_name == name and object_type in obj.__bases__
        )
        return next(valid_objects_gen, None)

    @staticmethod
    def _search_object(directory: str, object_type, object_name: str, 
                       kwargs: dict) -> Optional[Any]:
        """
        Search for the objectname in the given directory
        :param directory: relative or absolute directory path
        :return: object instance
        """
        logger.debug('Searching for %s %s in \'%s\'', object_type.__name__,  object_name, directory)
        for entry in os.listdir(directory):
            # Only consider python files
            if not entry.endswith('.py'):
                logger.debug('Ignoring %s', entry)
                continue
            obj = IResolver._get_valid_objects(
                object_type, os.path.abspath(os.path.join(directory, entry)), object_name
            )
            if obj:
                return obj(**kwargs)
        return None
