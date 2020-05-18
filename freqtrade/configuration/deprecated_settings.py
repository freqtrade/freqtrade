"""
Functions to handle deprecated settings
"""

import logging
from typing import Any, Dict

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def check_conflicting_settings(config: Dict[str, Any],
                               section1: str, name1: str,
                               section2: str, name2: str) -> None:
    section1_config = config.get(section1, {})
    section2_config = config.get(section2, {})
    if name1 in section1_config and name2 in section2_config:
        raise OperationalException(
            f"Conflicting settings `{section1}.{name1}` and `{section2}.{name2}` "
            "(DEPRECATED) detected in the configuration file. "
            "This deprecated setting will be removed in the next versions of Freqtrade. "
            f"Please delete it from your configuration and use the `{section1}.{name1}` "
            "setting instead."
        )


def process_deprecated_setting(config: Dict[str, Any],
                               section1: str, name1: str,
                               section2: str, name2: str) -> None:
    section2_config = config.get(section2, {})

    if name2 in section2_config:
        logger.warning(
            "DEPRECATED: "
            f"The `{section2}.{name2}` setting is deprecated and "
            "will be removed in the next versions of Freqtrade. "
            f"Please use the `{section1}.{name1}` setting in your configuration instead."
        )
        section1_config = config.get(section1, {})
        section1_config[name1] = section2_config[name2]


def process_temporary_deprecated_settings(config: Dict[str, Any]) -> None:

    check_conflicting_settings(config, 'ask_strategy', 'use_sell_signal',
                               'experimental', 'use_sell_signal')
    check_conflicting_settings(config, 'ask_strategy', 'sell_profit_only',
                               'experimental', 'sell_profit_only')
    check_conflicting_settings(config, 'ask_strategy', 'ignore_roi_if_buy_signal',
                               'experimental', 'ignore_roi_if_buy_signal')

    process_deprecated_setting(config, 'ask_strategy', 'use_sell_signal',
                               'experimental', 'use_sell_signal')
    process_deprecated_setting(config, 'ask_strategy', 'sell_profit_only',
                               'experimental', 'sell_profit_only')
    process_deprecated_setting(config, 'ask_strategy', 'ignore_roi_if_buy_signal',
                               'experimental', 'ignore_roi_if_buy_signal')

    if (config.get('edge', {}).get('enabled', False)
       and 'capital_available_percentage' in config.get('edge', {})):
        logger.warning(
            "DEPRECATED: "
            "Using 'edge.capital_available_percentage' has been deprecated in favor of "
            "'tradable_balance_ratio'. Please migrate your configuration to "
            "'tradable_balance_ratio' and remove 'capital_available_percentage' "
            "from the edge configuration."
        )
