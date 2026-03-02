"""
Terminal Monitor — Live dashboard for GA evolution progress.

Provides a rich terminal UI that replaces scrolling log output with a
static, live-updating dashboard showing evolution progress, metrics,
and phase information.

Three view modes (switchable at runtime via keyboard):
  [S] Simple   — Key metrics only (best/avg fitness, profit, diversity)
  [D] Detailed — Phase timing, population composition, convergence info
  [L] Logs     — Live scrolling log view with pinned header

Dependencies:
  pip install rich          (or: pip install -r requirements-monitor.txt)
  Falls back to NullMonitor (no-op) if rich is not installed.

Usage:
  from genetic_algorithm.monitor import create_monitor
  monitor = create_monitor(config)
  monitor.start(config)
  ...
  monitor.stop()
"""

from genetic_algorithm.monitor.null_monitor import NullMonitor

# View mode enum-like constants
VIEW_SIMPLE = "simple"
VIEW_DETAILED = "detailed"
VIEW_LOGS = "logs"


def create_monitor(config: dict, enabled: bool = True, default_mode: str | None = None):
    """
    Factory: create the appropriate monitor based on config and availability.

    Args:
        config: Full GA configuration dict.
        enabled: Whether terminal monitor is enabled (can be overridden by
                 --no-monitor CLI flag).
        default_mode: Override default view mode ('simple', 'detailed', 'logs').

    Returns:
        TerminalMonitor if rich is available and monitor is enabled,
        NullMonitor otherwise.
    """
    monitor_config = config.get("terminal_monitor", {})
    if not enabled or not monitor_config.get("enabled", True):
        return NullMonitor()

    # Web dashboard monitor mode
    if default_mode == "web":
        try:
            from genetic_algorithm.web.ws_monitor import WebSocketMonitor
            run_id = config.get("_web_run_id", "default")
            return WebSocketMonitor(run_id=run_id, config=config)
        except ImportError:
            return NullMonitor()

    try:
        from genetic_algorithm.monitor.terminal_monitor import TerminalMonitor  # noqa: F811

        mode = default_mode or monitor_config.get("default_mode", VIEW_SIMPLE)
        return TerminalMonitor(config, default_mode=mode)
    except ImportError:
        # rich not installed — fall back silently
        return NullMonitor()
