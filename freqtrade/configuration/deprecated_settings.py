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


def process_removed_setting(config: Dict[str, Any],
                            section1: str, name1: str,
                            section2: str, name2: str) -> None:
    """
    :param section1: Removed section
    :param name1: Removed setting name
    :param section2: new section for this key
    :param name2: new setting name
    """
    section1_config = config.get(section1, {})
    if name1 in section1_config:
        raise OperationalException(
            f"Setting `{section1}.{name1}` has been moved to `{section2}.{name2}. "
            f"Please delete it from your configuration and use the `{section2}.{name2}` "
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

    # Kept for future deprecated / moved settings
    # check_conflicting_settings(config, 'ask_strategy', 'use_sell_signal',
    #                            'experimental', 'use_sell_signal')
    # process_deprecated_setting(config, 'ask_strategy', 'use_sell_signal',
    #                            'experimental', 'use_sell_signal')

    process_removed_setting(config, 'experimental', 'use_sell_signal',
                            'ask_strategy', 'use_sell_signal')
    process_removed_setting(config, 'experimental', 'sell_profit_only',
                            'ask_strategy', 'sell_profit_only')
    process_removed_setting(config, 'experimental', 'ignore_roi_if_buy_signal',
                            'ask_strategy', 'ignore_roi_if_buy_signal')

    if (config.get('edge', {}).get('enabled', False)
       and 'capital_available_percentage' in config.get('edge', {})):
        raise OperationalException(
            "DEPRECATED: "
            "Using 'edge.capital_available_percentage' has been deprecated in favor of "
            "'tradable_balance_ratio'. Please migrate your configuration to "
            "'tradable_balance_ratio' and remove 'capital_available_percentage' "
            "from the edge configuration."
        )
    if 'ticker_interval' in config:
        logger.warning(
            "DEPRECATED: "
            "Please use 'timeframe' instead of 'ticker_interval."
        )
        if 'timeframe' in config:
            raise OperationalException(
                "Both 'timeframe' and 'ticker_interval' detected."
                "Please remove 'ticker_interval' from your configuration to continue operating."
                )
        config['timeframe'] = config['ticker_interval']
