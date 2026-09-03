"""Platform-layer exceptions and validation errors."""

from __future__ import annotations


class PlatformError(Exception):
    """Base exception for platform layer errors."""


class PlatformValidationError(PlatformError, ValueError):
    """Raised when platform-domain validation fails."""


class PlatformNotFoundError(PlatformError, KeyError):
    """Raised when a requested platform object cannot be found."""


class PlatformConfigurationError(PlatformError):
    """Raised when configuration is invalid for the platform layer."""
