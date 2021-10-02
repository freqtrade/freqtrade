"""
Functions to handle deprecated settings
"""

import logging
from typing import Any, Dict, Optional

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def check_conflicting_settings(config: Dict[str, Any],
                               section_old: str, name_old: str,
                               section_new: Optional[str], name_new: str) -> None:
    section_new_config = config.get(section_new, {}) if section_new else config
    section_old_config = config.get(section_old, {})
    if name_new in section_new_config and name_old in section_old_config:
        new_name = f"{section_new}.{name_new}" if section_new else f"{name_new}"
        raise OperationalException(
            f"Conflicting settings `{new_name}` and `{section_old}.{name_old}` "
            "(DEPRECATED) detected in the configuration file. "
            "This deprecated setting will be removed in the next versions of Freqtrade. "
            f"Please delete it from your configuration and use the `{new_name}` "
            "setting instead."
        )


def process_removed_setting(config: Dict[str, Any],
                            section1: str, name1: str,
                            section2: Optional[str], name2: str) -> None:
    """
    :param section1: Removed section
    :param name1: Removed setting name
    :param section2: new section for this key
    :param name2: new setting name
    """
    section1_config = config.get(section1, {})
    if name1 in section1_config:
        section_2 = f"{section2}.{name2}" if section2 else f"{name2}"
        raise OperationalException(
            f"Setting `{section1}.{name1}` has been moved to `{section_2}. "
            f"Please delete it from your configuration and use the `{section_2}` "
            "setting instead."
        )


def process_deprecated_setting(config: Dict[str, Any],
                               section_old: str, name_old: str,
                               section_new: Optional[str], name_new: str
                               ) -> None:
    check_conflicting_settings(config, section_old, name_old, section_new, name_new)
    section_old_config = config.get(section_old, {})

    if name_old in section_old_config:
        section_2 = f"{section_new}.{name_new}" if section_new else f"{name_new}"
        logger.warning(
            "DEPRECATED: "
            f"The `{section_old}.{name_old}` setting is deprecated and "
            "will be removed in the next versions of Freqtrade. "
            f"Please use the `{section_2}` setting in your configuration instead."
        )

        section_new_config = config.get(section_new, {}) if section_new else config
        section_new_config[name_new] = section_old_config[name_old]


def process_temporary_deprecated_settings(config: Dict[str, Any]) -> None:

    # Kept for future deprecated / moved settings
    # check_conflicting_settings(config, 'ask_strategy', 'use_sell_signal',
    #                            'experimental', 'use_sell_signal')
    process_deprecated_setting(config, 'ask_strategy', 'use_sell_signal',
                               None, 'use_sell_signal')
    process_deprecated_setting(config, 'ask_strategy', 'sell_profit_only',
                               None, 'sell_profit_only')
    process_deprecated_setting(config, 'ask_strategy', 'sell_profit_offset',
                               None, 'sell_profit_offset')
    process_deprecated_setting(config, 'ask_strategy', 'ignore_roi_if_buy_signal',
                               None, 'ignore_roi_if_buy_signal')
    process_deprecated_setting(config, 'ask_strategy', 'ignore_buying_expired_candle_after',
                               None, 'ignore_buying_expired_candle_after')

    # Legacy way - having them in experimental ...
    process_removed_setting(config, 'experimental', 'use_sell_signal',
                            None, 'use_sell_signal')
    process_removed_setting(config, 'experimental', 'sell_profit_only',
                            None, 'sell_profit_only')
    process_removed_setting(config, 'experimental', 'ignore_roi_if_buy_signal',
                            None, 'ignore_roi_if_buy_signal')

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

    if 'protections' in config:
        logger.warning("DEPRECATED: Setting 'protections' in the configuration is deprecated.")
