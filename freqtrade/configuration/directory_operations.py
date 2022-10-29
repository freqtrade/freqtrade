import logging
import shutil
from pathlib import Path
from typing import Optional

from freqtrade.constants import (USER_DATA_FILES, USERPATH_FREQAIMODELS, USERPATH_HYPEROPTS,
                                 USERPATH_NOTEBOOKS, USERPATH_STRATEGIES, Config)
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def create_datadir(config: Config, datadir: Optional[str] = None) -> Path:

    folder = Path(datadir) if datadir else Path(f"{config['user_data_dir']}/data")
    if not datadir:
        # set datadir
        exchange_name = config.get('exchange', {}).get('name', '').lower()
        folder = folder.joinpath(exchange_name)

    if not folder.is_dir():
        folder.mkdir(parents=True)
        logger.info(f'Created data directory: {datadir}')
    return folder


def chown_user_directory(directory: Path) -> None:
    """
    Use Sudo to change permissions of the home-directory if necessary
    Only applies when running in docker!
    """
    import os
    if os.environ.get('FT_APP_ENV') == 'docker':
        try:
            import subprocess
            subprocess.check_output(
                ['sudo', 'chown', '-R', 'ftuser:', str(directory.resolve())])
        except Exception:
            logger.warning(f"Could not chown {directory}")


def create_userdata_dir(directory: str, create_dir: bool = False) -> Path:
    """
    Create userdata directory structure.
    if create_dir is True, then the parent-directory will be created if it does not exist.
    Sub-directories will always be created if the parent directory exists.
    Raises OperationalException if given a non-existing directory.
    :param directory: Directory to check
    :param create_dir: Create directory if it does not exist.
    :return: Path object containing the directory
    """
    sub_dirs = ["backtest_results", "data", USERPATH_HYPEROPTS, "hyperopt_results", "logs",
                USERPATH_NOTEBOOKS, "plot", USERPATH_STRATEGIES, USERPATH_FREQAIMODELS]
    folder = Path(directory)
    chown_user_directory(folder)
    if not folder.is_dir():
        if create_dir:
            folder.mkdir(parents=True)
            logger.info(f'Created user-data directory: {folder}')
        else:
            raise OperationalException(
                f"Directory `{folder}` does not exist. "
                "Please use `freqtrade create-userdir` to create a user directory")

    # Create required subdirectories
    for f in sub_dirs:
        subfolder = folder / f
        if not subfolder.is_dir():
            subfolder.mkdir(parents=False)
    return folder


def copy_sample_files(directory: Path, overwrite: bool = False) -> None:
    """
    Copy files from templates to User data directory.
    :param directory: Directory to copy data to
    :param overwrite: Overwrite existing sample files
    """
    if not directory.is_dir():
        raise OperationalException(f"Directory `{directory}` does not exist.")
    sourcedir = Path(__file__).parents[1] / "templates"
    for source, target in USER_DATA_FILES.items():
        targetdir = directory / target
        if not targetdir.is_dir():
            raise OperationalException(f"Directory `{targetdir}` does not exist.")
        targetfile = targetdir / source
        if targetfile.exists():
            if not overwrite:
                logger.warning(f"File `{targetfile}` exists already, not deploying sample file.")
                continue
            logger.warning(f"File `{targetfile}` exists already, overwriting.")
        shutil.copy(str(sourcedir / source), str(targetfile))
