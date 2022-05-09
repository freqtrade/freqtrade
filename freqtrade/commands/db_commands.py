from typing import Any, Dict

from freqtrade.configuration.config_setup import setup_utils_configuration
from freqtrade.enums.runmode import RunMode


def start_db_convert(args: Dict[str, Any]) -> None:
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    pass
