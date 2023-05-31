
from typing import Any, Dict, List, Literal, Optional, TypedDict
from uuid import uuid4


class JobsContainer(TypedDict):
    category: Literal['pairlist']
    is_running: bool
    status: str
    progress: Optional[float]
    result: Any
    error: Optional[str]


class ApiBG():
    # Backtesting type: Backtesting
    bt: Dict[str, Any] = {
        'bt': None,
        'data': None,
        'timerange': None,
        'last_config': {},
        'bt_error': None,
    }
    bgtask_running: bool = False
    # Exchange - only available in webserver mode.
    exchange = None

    # Generic background jobs
    running_jobs: List[str] = []
    # TODO: Change this to TTLCache
    jobs: Dict[str, JobsContainer] = {}
    # Pairlist evaluate things
    pairlist_running: bool = False

    @staticmethod
    def get_job_id() -> str:
        return str(uuid4())
