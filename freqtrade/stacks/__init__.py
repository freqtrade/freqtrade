"""
Stacks blockchain integration module for ApexTrader-Stacks.
Provides client interfaces for Hiro API and escrow contract management.
"""

from freqtrade.stacks.client import StacksClient
from freqtrade.stacks.escrow import EscrowManager
from freqtrade.stacks.hiro_api import HiroAPI

__all__ = ["StacksClient", "EscrowManager", "HiroAPI"]
