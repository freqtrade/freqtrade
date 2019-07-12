import logging
import os
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def create_datadir(config: Dict[str, Any], datadir: Optional[str] = None) -> str:
    if not datadir:
        # set datadir
        exchange_name = config.get('exchange', {}).get('name').lower()
        datadir = os.path.join('user_data', 'data', exchange_name)

    if not os.path.isdir(datadir):
        os.makedirs(datadir)
        logger.info(f'Created data directory: {datadir}')
    return datadir
