# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom objects
"""
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


class IResolver:
    """
    This class contains all the logic to load custom classes
    """
    # Childclasses need to override this
    object_type: Type[Any]
    object_type_str: str
    user_subdir: Optional[str] = None
    initial_search_path: Optional[Path]

    @classmethod
    def build_search_paths(cls, config: Dict[str, Any], user_subdir: Optional[str] = None,
                           extra_dir: Optional[str] = None) -> List[Path]:

        abs_paths: List[Path] = []
        if cls.initial_search_path:
            abs_paths.append(cls.initial_search_path)

        if user_subdir:
            abs_paths.insert(0, config['user_data_dir'].joinpath(user_subdir))

        if extra_dir:
            # Add extra directory to the top of the search paths
            abs_paths.insert(0, Path(extra_dir).resolve())

        return abs_paths

    @classmethod
    def _get_valid_object(cls, module_path: Path, object_name: Optional[str],
                          enum_failed: bool = False) -> Iterator[Any]:
        """
        Generator returning objects with matching object_type and object_name in the path given.
        :param module_path: absolute path to the module
        :param object_name: Class name of the object
        :param enum_failed: If True, will return None for modules which fail.
            Otherwise, failing modules are skipped.
        :return: generator containing tuple of matching objects
             Tuple format: [Object, source]
        """

        # Generate spec based on absolute path
        # Pass object_name as first argument to have logging print a reasonable name.
        spec = importlib.util.spec_from_file_location(object_name or "", str(module_path))
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore # importlib does not use typehints
        except (ModuleNotFoundError, SyntaxError, ImportError, NameError) as err:
            # Catch errors in case a specific module is not installed
            logger.warning(f"Could not import {module_path} due to '{err}'")
            if enum_failed:
                return iter([None])

        valid_objects_gen = (
            (obj, inspect.getsource(module)) for
            name, obj in inspect.getmembers(
                module, inspect.isclass) if ((object_name is None or object_name == name)
                                             and issubclass(obj, cls.object_type)
                                             and obj is not cls.object_type)
        )
        return valid_objects_gen

    @classmethod
    def _search_object(cls, directory: Path, *, object_name: str, add_source: bool = False
                       ) -> Union[Tuple[Any, Path], Tuple[None, None]]:
        """
        Search for the objectname in the given directory
        :param directory: relative or absolute directory path
        :param object_name: ClassName of the object to load
        :return: object class
        """
        logger.debug(f"Searching for {cls.object_type.__name__} {object_name} in '{directory}'")
        for entry in directory.iterdir():
            # Only consider python files
            if not str(entry).endswith('.py'):
                logger.debug('Ignoring %s', entry)
                continue
            module_path = entry.resolve()

            obj = next(cls._get_valid_object(module_path, object_name), None)

            if obj:
                obj[0].__file__ = str(entry)
                if add_source:
                    obj[0].__source__ = obj[1]
                return (obj[0], module_path)
        return (None, None)

    @classmethod
    def _load_object(cls, paths: List[Path], *, object_name: str, add_source: bool = False,
                     kwargs: dict = {}) -> Optional[Any]:
        """
        Try to load object from path list.
        """

        for _path in paths:
            try:
                (module, module_path) = cls._search_object(directory=_path,
                                                           object_name=object_name,
                                                           add_source=add_source)
                if module:
                    logger.info(
                        f"Using resolved {cls.object_type.__name__.lower()[1:]} {object_name} "
                        f"from '{module_path}'...")
                    return module(**kwargs)
            except FileNotFoundError:
                logger.warning('Path "%s" does not exist.', _path.resolve())

        return None

    @classmethod
    def load_object(cls, object_name: str, config: dict, *, kwargs: dict,
                    extra_dir: Optional[str] = None) -> Any:
        """
        Search and loads the specified object as configured in hte child class.
        :param objectname: name of the module to import
        :param config: configuration dictionary
        :param extra_dir: additional directory to search for the given pairlist
        :raises: OperationalException if the class is invalid or does not exist.
        :return: Object instance or None
        """

        abs_paths = cls.build_search_paths(config,
                                           user_subdir=cls.user_subdir,
                                           extra_dir=extra_dir)

        found_object = cls._load_object(paths=abs_paths, object_name=object_name,
                                        kwargs=kwargs)
        if found_object:
            return found_object
        raise OperationalException(
            f"Impossible to load {cls.object_type_str} '{object_name}'. This class does not exist "
            "or contains Python code errors."
        )

    @classmethod
    def search_all_objects(cls, directory: Path,
                           enum_failed: bool) -> List[Dict[str, Any]]:
        """
        Searches a directory for valid objects
        :param directory: Path to search
        :param enum_failed: If True, will return None for modules which fail.
            Otherwise, failing modules are skipped.
        :return: List of dicts containing 'name', 'class' and 'location' entires
        """
        logger.debug(f"Searching for {cls.object_type.__name__} '{directory}'")
        objects = []
        for entry in directory.iterdir():
            # Only consider python files
            if not str(entry).endswith('.py'):
                logger.debug('Ignoring %s', entry)
                continue
            module_path = entry.resolve()
            logger.debug(f"Path {module_path}")
            for obj in cls._get_valid_object(module_path, object_name=None,
                                             enum_failed=enum_failed):
                objects.append(
                    {'name': obj[0].__name__ if obj is not None else '',
                     'class': obj[0] if obj is not None else None,
                     'location': entry,
                     })
        return objects
