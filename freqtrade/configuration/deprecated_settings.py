"""
Functions to handle deprecated settings
"""

import logging
from typing import Any, Dict

from freqtrade import OperationalException


logger = logging.getLogger(__name__)


def check_conflicting_settings(config: Dict[str, Any],
                               section1: str, name1: str,
                               section2: str, name2: str):
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
                               section2: str, name2: str):
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

    if config.get('pairlist', {}).get("method") == 'VolumePairList':
        logger.warning(
            "DEPRECATED: "
            f"Using VolumePairList in pairlist is deprecated and must be moved to pairlists. "
            "Please refer to the docs on configuration details")
        config['pairlists'].append({'method': 'VolumePairList',
                                    'config': config.get('pairlist', {}).get('config')
                                    })

    if config.get('pairlist', {}).get('config', {}).get('precision_filter'):
        logger.warning(
            "DEPRECATED: "
            f"Using precision_filter setting is deprecated and has been replaced by"
            "PrecisionFilter. Please refer to the docs on configuration details")
        config['pairlists'].append({'method': 'PrecisionFilter'})
