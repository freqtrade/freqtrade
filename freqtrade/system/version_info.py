import logging
from freqtrade import __version__

logger = logging.getLogger(__name__)


def print_version_info():
    """Print version information for freqtrade and its key dependencies."""
    import platform
    import sys

    import ccxt

    logger.info(f"Operating System:\t{platform.platform()}")
    logger.info(f"Python Version:\t\tPython {sys.version.split(' ')[0]}")
    logger.info(f"CCXT Version:\t\t{ccxt.__version__}")
    logger.info("")
    logger.info(f"Freqtrade Version:\tfreqtrade {__version__}")
