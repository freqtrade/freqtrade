# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom objects
"""
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, Generator

logger = logging.getLogger(__name__)


class IResolver:
    """
    This class contains all the logic to load custom classes
    """

    def build_search_paths(self, config, current_path: Path, user_subdir: str,
                           extra_dir: Optional[str] = None) -> List[Path]:

        abs_paths = [
            config['user_data_dir'].joinpath(user_subdir),
            current_path,
        ]

        if extra_dir:
            # Add extra directory to the top of the search paths
            abs_paths.insert(0, Path(extra_dir).resolve())

        return abs_paths

    @staticmethod
    def _get_valid_object(object_type, module_path: Path,
                          object_name: str) -> Generator[Any, None, None]:
        """
        Generator returning objects with matching object_type and object_name in the path given.
        :param object_type: object_type (class)
        :param module_path: absolute path to the module
        :param object_name: Class name of the object
        :return: generator containing matching objects
        """

        # Generate spec based on absolute path
        # Pass object_name as first argument to have logging print a reasonable name.
        spec = importlib.util.spec_from_file_location(object_name, str(module_path))
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore # importlib does not use typehints
        except (ModuleNotFoundError, SyntaxError) as err:
            # Catch errors in case a specific module is not installed
            logger.warning(f"Could not import {module_path} due to '{err}'")

        valid_objects_gen = (
            obj for name, obj in inspect.getmembers(module, inspect.isclass)
            if object_name == name and object_type in obj.__bases__
        )
        return valid_objects_gen

    @staticmethod
    def _search_object(directory: Path, object_type, object_name: str,
                       kwargs: dict = {}) -> Union[Tuple[Any, Path], Tuple[None, None]]:
        """
        Search for the objectname in the given directory
        :param directory: relative or absolute directory path
        :return: object instance
        """
        logger.debug("Searching for %s %s in '%s'", object_type.__name__, object_name, directory)
        for entry in directory.iterdir():
            # Only consider python files
            if not str(entry).endswith('.py'):
                logger.debug('Ignoring %s', entry)
                continue
            module_path = entry.resolve()

            obj = next(IResolver._get_valid_object(object_type, module_path, object_name), None)

            if obj:
                return (obj(**kwargs), module_path)
        return (None, None)

    @staticmethod
    def _load_object(paths: List[Path], object_type, object_name: str,
                     kwargs: dict = {}) -> Optional[Any]:
        """
        Try to load object from path list.
        """

        for _path in paths:
            try:
                (module, module_path) = IResolver._search_object(directory=_path,
                                                                 object_type=object_type,
                                                                 object_name=object_name,
                                                                 kwargs=kwargs)
                if module:
                    logger.info(
                        f"Using resolved {object_type.__name__.lower()[1:]} {object_name} "
                        f"from '{module_path}'...")
                    return module
            except FileNotFoundError:
                logger.warning('Path "%s" does not exist.', _path.resolve())

        return None
