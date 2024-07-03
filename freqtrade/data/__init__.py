"""
Module to handle data operations for freqtrade
"""

from freqtrade.data import converter


# limit what's imported when using `from freqtrade.data import *`
__all__ = ["converter"]
