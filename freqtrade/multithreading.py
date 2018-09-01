"""
This module contains the class to use the dask module
"""

from typing import Dict
from dask.distributed import Client


def init(config: Dict) -> object:
    """
    Initialise Dask Distributed Client
    """
    workers_number = config['multithreading']['workers_number']
    client = Client(processes=False, threads_per_worker=1, n_workers=workers_number)

    return client


