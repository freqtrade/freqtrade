
from typing import Any, Dict, Optional


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
    # Pairlist evaluate things
    pairlist_error: Optional[str] = None
    pairlist_running: bool = False
    pairlist_result: Dict[str, Any] = {}
