
from typing import Any, Dict


class ApiBG():
    # Backtesting type: Backtesting
    _bt: Dict[str, Any] = {
        'bt': None,
        'data': None,
        'timerange': None,
        'last_config': {},
        'bt_error': None,
    }
    _bgtask_running: bool = False
    # Exchange - only available in webserver mode.
    _exchange = None
