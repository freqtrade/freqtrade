import logging
import os
from typing import Any

from freqtrade.constants import ENV_VAR_PREFIX
from freqtrade.misc import deep_merge_dicts


logger = logging.getLogger(__name__)


def _get_var_typed(val):
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            if val.lower() in ("t", "true"):
                return True
            elif val.lower() in ("f", "false"):
                return False
    # keep as string
    return val


def _flat_vars_to_nested_dict(env_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    """
    Environment variables must be prefixed with FREQTRADE.
    FREQTRADE__{section}__{key}
    :param env_dict: Dictionary to validate - usually os.environ
    :param prefix: Prefix to consider (usually FREQTRADE__)
    :return: Nested dict based on available and relevant variables.
    """
    no_convert = ["CHAT_ID", "PASSWORD"]
    relevant_vars: dict[str, Any] = {}

    for env_var, val in sorted(env_dict.items()):
        if env_var.startswith(prefix):
            logger.info(f"Loading variable '{env_var}'")
            key = env_var.replace(prefix, "")
            for k in reversed(key.split("__")):
                val = {
                    k.lower(): (
                        _get_var_typed(val)
                        if not isinstance(val, dict) and k not in no_convert
                        else val
                    )
                }
            relevant_vars = deep_merge_dicts(val, relevant_vars)
    return relevant_vars


def enironment_vars_to_dict() -> dict[str, Any]:
    """
    Read environment variables and return a nested dict for relevant variables
    Relevant variables must follow the FREQTRADE__{section}__{key} pattern
    :return: Nested dict based on available and relevant variables.
    """
    return _flat_vars_to_nested_dict(os.environ.copy(), ENV_VAR_PREFIX)
