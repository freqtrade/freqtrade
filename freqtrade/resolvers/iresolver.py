# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom objects
"""
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


class IResolver:
    """
    This class contains all the logic to load custom classes
    """
    # Childclasses need to override this
    object_type: Type[Any]

    @staticmethod
    def build_search_paths(config, current_path: Path, user_subdir: Optional[str] = None,
                           extra_dir: Optional[str] = None) -> List[Path]:

        abs_paths: List[Path] = [current_path]

        if user_subdir:
            abs_paths.insert(0, config['user_data_dir'].joinpath(user_subdir))

        if extra_dir:
            # Add extra directory to the top of the search paths
            abs_paths.insert(0, Path(extra_dir).resolve())

        return abs_paths

    @classmethod
    def _get_valid_object(cls, module_path: Path,
                          object_name: str) -> Generator[Any, None, None]:
        """
        Generator returning objects with matching object_type and object_name in the path given.
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
            if object_name == name and cls.object_type in obj.__bases__
        )
        return valid_objects_gen

    @classmethod
    def _search_object(cls, directory: Path, object_name: str,
                       kwargs: dict = {}) -> Union[Tuple[Any, Path], Tuple[None, None]]:
        """
        Search for the objectname in the given directory
        :param directory: relative or absolute directory path
        :param object_name: ClassName of the object to load
        :return: object instance
        """
        logger.debug("Searching for %s %s in '%s'",
                     cls.object_type.__name__, object_name, directory)
        for entry in directory.iterdir():
            # Only consider python files
            if not str(entry).endswith('.py'):
                logger.debug('Ignoring %s', entry)
                continue
            module_path = entry.resolve()

            obj = next(cls._get_valid_object(module_path, object_name), None)

            if obj:
                return (obj(**kwargs), module_path)
        return (None, None)

    @classmethod
    def _load_object(cls, paths: List[Path], object_name: str,
                     kwargs: dict = {}) -> Optional[Any]:
        """
        Try to load object from path list.
        """

        for _path in paths:
            try:
                (module, module_path) = cls._search_object(directory=_path,
                                                           object_name=object_name,
                                                           kwargs=kwargs)
                if module:
                    logger.info(
                        f"Using resolved {cls.object_type.__name__.lower()[1:]} {object_name} "
                        f"from '{module_path}'...")
                    return module
            except FileNotFoundError:
                logger.warning('Path "%s" does not exist.', _path.resolve())

        return None
