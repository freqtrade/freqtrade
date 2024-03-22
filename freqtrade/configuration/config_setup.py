import logging
from typing import Any, Dict

from freqtrade.enums import RunMode

from .config_validation import validate_config_consistency
from .configuration import Configuration


logger = logging.getLogger(__name__)


def setup_utils_configuration(
        args: Dict[str, Any], method: RunMode, *, set_dry: bool = True) -> Dict[str, Any]:
    """
    Prepare the configuration for utils subcommands
    :param args: Cli args from Arguments()
    :param method: Bot running mode
    :return: Configuration
    """
    configuration = Configuration(args, method)
    config = configuration.get_config()

    # Ensure these modes are using Dry-run
    if set_dry:
        config['dry_run'] = True
    validate_config_consistency(config, preliminary=True)

    return config
