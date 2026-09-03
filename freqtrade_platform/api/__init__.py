"""API boundaries for the platform layer.

This namespace exists so future REST endpoints can be added without modifying the existing
Freqtrade FastAPI server implementation. The platform API is intentionally separated from the
Freqtrade core API surface.
"""

from freqtrade_platform.core.context import PlatformContext

__all__ = ["PlatformContext"]
