import logging
from typing import Any, Dict, Optional
from pathlib import Path


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


def create_userdata_dir(directory: str) -> str:
    sub_dirs = ["backtest_results", "data", "hyperopts", "plots", "strategies", ]
    folder = Path(directory)
    if not folder.is_dir():
        folder.mkdir(parents=True)
        logger.info(f'Created user-data directory: {folder}')

    # Create required subdirectories
    for f in sub_dirs:
        subfolder = folder / f
        if not subfolder.is_dir():
            subfolder.mkdir(parents=False)
    # TODO: convert this to return Path
    return folder
