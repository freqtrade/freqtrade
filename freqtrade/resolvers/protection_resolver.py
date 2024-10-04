"""
This module load custom pairlists
"""

import logging
from pathlib import Path

from freqtrade.constants import Config
from freqtrade.plugins.protections import IProtection
from freqtrade.resolvers import IResolver


logger = logging.getLogger(__name__)


class ProtectionResolver(IResolver):
    """
    This class contains all the logic to load custom PairList class
    """

    object_type = IProtection
    object_type_str = "Protection"
    user_subdir = None
    initial_search_path = Path(__file__).parent.parent.joinpath("plugins/protections").resolve()

    @staticmethod
    def load_protection(
        protection_name: str, config: Config, protection_config: dict
    ) -> IProtection:
        """
        Load the protection with protection_name
        :param protection_name: Classname of the pairlist
        :param config: configuration dictionary
        :param protection_config: Configuration dedicated to this pairlist
        :return: initialized Protection class
        """
        return ProtectionResolver.load_object(
            protection_name,
            config,
            kwargs={
                "config": config,
                "protection_config": protection_config,
            },
        )
