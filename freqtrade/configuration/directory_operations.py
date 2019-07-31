import logging
from typing import Any, Dict, Optional
from pathlib import Path

from freqtrade import OperationalException

logger = logging.getLogger(__name__)


def create_datadir(config: Dict[str, Any], datadir: Optional[str] = None) -> str:

    folder = Path(datadir) if datadir else Path(f"{config['user_data_dir']}/data")
    if not datadir:
        # set datadir
        exchange_name = config.get('exchange', {}).get('name').lower()
        folder = folder.joinpath(exchange_name)

    if not folder.is_dir():
        folder.mkdir(parents=True)
        logger.info(f'Created data directory: {datadir}')
    return str(folder)


def create_userdata_dir(directory: str, create_dir=False) -> Path:
    """
    Create userdata directory structure.
    if create_dir is True, then the parent-directory will be created if it does not exist.
    Sub-directories will always be created if the parent directory exists.
    Raises OperationalException if given a non-existing directory.
    :param directory: Directory to check
    :param create_dir: Create directory if it does not exist.
    :return: Path object containing the directory
    """
    sub_dirs = ["backtest_results", "data", "hyperopts", "hyperopt_results", "plot", "strategies", ]
    folder = Path(directory)
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
