"""
Web dashboard configuration.

Centralises all web server settings with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WebConfig:
    """Configuration for the web dashboard server."""

    host: str = "127.0.0.1"
    port: int = 8501
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"])
    log_level: str = "info"
    open_browser: bool = True

    # Generation snapshot persistence (enables drill-down into past generations)
    save_generation_snapshots: bool = True
    max_snapshot_generations: int = 500  # Keep at most N generation snapshots per run

    # WebSocket
    ws_heartbeat_interval: float = 30.0  # seconds

    # Backtesting
    max_concurrent_backtests: int = 2  # Max simultaneous on-demand backtests

    @classmethod
    def from_dict(cls, data: dict) -> "WebConfig":
        """Create from a config dict (e.g. from ga_config.yaml 'web_dashboard' section)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
