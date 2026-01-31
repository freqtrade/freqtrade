"""
freqtrade_client package exports.
Keep __version__ aligned with freqtrade.
"""
__version__ = "2025.12"

# Public API
try:
    from .ft_rest_client import FtRestClient  # common layout
except Exception:
    from .rest_client import FtRestClient  # alternate layout

__all__ = ["FtRestClient", "__version__"]
