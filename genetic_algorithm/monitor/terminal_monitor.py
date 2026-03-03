"""
Terminal Monitor — Rich-based live dashboard for GA evolution.

Provides a static, live-updating terminal interface with three view modes:
  [S] Simple   — Compact overview: best/avg fitness, profit, diversity, trends
  [D] Detailed — Phase timing, population composition, convergence, holdout
  [L] Logs     — Live scrolling log output with pinned header

The header bar (elapsed time, ETA, generation progress) is always visible.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "The 'rich' package is required for the terminal monitor. "
        "Install it with: pip install rich>=13.0"
    ) from _exc

from genetic_algorithm.monitor.key_listener import KeyListener
from genetic_algorithm.monitor.log_capture import MonitorLogHandler

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
VIEW_SIMPLE = "simple"
VIEW_DETAILED = "detailed"
VIEW_LOGS = "logs"

PHASE_LABELS = {
    "eval": "Evaluating strategies",
    "selection": "Selection / Crossover / Mutation",
    "holdout": "Holdout monitoring",
    "overhead": "Overhead / Stats",
}

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float], width: int = 20) -> str:
    """Render a sparkline string from a sequence of floats."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    indices = [int((v - mn) / rng * (len(SPARKLINE_CHARS) - 1)) for v in values]
    return "".join(SPARKLINE_CHARS[i] for i in indices[-width:])


def _fmt_time(seconds: float) -> str:
    """Format seconds into human-readable H:MM:SS or M:SS."""
    if seconds < 0:
        return "--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _delta_str(current: float, previous: float) -> str:
    """Return a colored delta string like '+0.12' or '-0.05'."""
    diff = current - previous
    if diff > 0:
        return f"[green]+{diff:.4f}[/green]"
    elif diff < 0:
        return f"[red]{diff:.4f}[/red]"
    return f"[dim]{diff:.4f}[/dim]"


def _pct_delta_str(current: float, previous: float) -> str:
    """Return a colored percentage delta like '+2.3%' or '-1.1%'."""
    diff = current - previous
    if diff > 0:
        return f"[green]+{diff:.2f}%[/green]"
    elif diff < 0:
        return f"[red]{diff:.2f}%[/red]"
    return f"[dim]{diff:.2f}%[/dim]"


# ═══════════════════════════════════════════════════════════════════════════
class TerminalMonitor:
    """
    Rich-based live terminal dashboard for GA evolution.

    Hooks into the evolution loop via ``on_*`` callbacks and renders a
    static, periodically-refreshed display using ``rich.live.Live``.
    """

    active: bool = True  # Flag checked by evolution.py (True = suppress tqdm)

    def __init__(self, config: dict, default_mode: str = VIEW_SIMPLE):
        self._config = config
        monitor_cfg = config.get("terminal_monitor", {})
        self._view_mode: str = default_mode
        self._show_keys: bool = monitor_cfg.get("show_keybindings", True)
        self._history_window: int = monitor_cfg.get("history_window", 20)
        self._refresh_rate: int = monitor_cfg.get("refresh_rate", 4)

        # Runtime state
        self._start_time: float = 0.0
        self._current_gen: int = 0
        self._total_gens: int = 1
        self._current_phase: str = ""
        self._phase_timings: Dict[str, float] = {}  # latest generation
        self._eval_completed: int = 0
        self._eval_total: int = 0

        # Generation history (rolling window)
        self._gen_history: deque = deque(maxlen=self._history_window)
        # Best fitness history for sparkline
        self._best_fitness_history: List[float] = []
        self._avg_fitness_history: List[float] = []

        # Current generation stats (populated by on_generation_end)
        self._stats: Optional[Any] = None  # PopulationStats
        self._timing: Optional[Any] = None  # GenerationTiming
        self._best_individual: Optional[Any] = None
        self._extras: Dict[str, Any] = {}
        self._new_best_flash: bool = False  # Flash indicator for new best
        self._convergence_warning: Optional[str] = None

        # Log capture
        self._log_handler = MonitorLogHandler(max_lines=200)

        # Key listener
        self._key_listener = KeyListener(on_key=self._on_key)

        # Rich objects
        self._console = Console()
        self._live: Optional[Live] = None

        # GA config info (set in start())
        self._pop_size: int = 0
        self._mutation_rate: float = 0.0
        self._mode: str = "single_objective"

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────
    def start(self, config: dict) -> None:
        """Start the live display and key listener."""
        ga_cfg = config.get("genetic_algorithm", {})
        self._pop_size = ga_cfg.get("population_size", 0)
        self._total_gens = ga_cfg.get("generations", 1)
        self._mutation_rate = ga_cfg.get("mutation_rate", 0.0)
        self._mode = ga_cfg.get("mode", "single_objective")
        self._start_time = time.monotonic()

        # Attach log handler to 'GeneticAlgorithm' logger
        ga_logger = logging.getLogger("GeneticAlgorithm")
        ga_logger.addHandler(self._log_handler)
        # Also capture root logger messages
        root_logger = logging.getLogger()
        root_logger.addHandler(self._log_handler)

        # Start live display
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=self._refresh_rate,
            screen=False,
            transient=False,
        )
        self._live.start()

        # Start key listener
        self._key_listener.start()

    def stop(self) -> None:
        """Stop the live display and key listener."""
        self._key_listener.stop()
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

        # Remove log handler
        for logger_name in ("GeneticAlgorithm", None):
            lg = logging.getLogger(logger_name)
            if self._log_handler in lg.handlers:
                lg.removeHandler(self._log_handler)

    # ──────────────────────────────────────────────────────────────────────
    # Callbacks (called from evolution.py)
    # ──────────────────────────────────────────────────────────────────────
    def on_generation_start(self, gen: int, total: int) -> None:
        self._current_gen = gen
        self._total_gens = total
        self._current_phase = "initializing"
        self._phase_timings = {}
        self._eval_completed = 0
        self._eval_total = 0
        self._new_best_flash = False
        self._convergence_warning = None
        self._refresh()

    def on_phase_start(self, phase: str) -> None:
        self._current_phase = phase
        self._refresh()

    def on_phase_end(self, phase: str, elapsed: float = 0.0) -> None:
        self._phase_timings[phase] = elapsed
        self._current_phase = ""
        self._refresh()

    def on_eval_progress(self, completed: int, total: int) -> None:
        self._eval_completed = completed
        self._eval_total = total
        self._refresh()

    def on_generation_end(
        self,
        gen: int,
        stats,
        timing,
        best_individual,
        extras: dict | None = None,
    ) -> None:
        self._stats = stats
        self._timing = timing
        self._best_individual = best_individual
        self._extras = extras or {}

        # Track history
        record = {
            "gen": gen,
            "best_fitness": getattr(stats, "best_fitness", None),
            "avg_fitness": getattr(stats, "avg_fitness", None),
            "diversity": getattr(stats, "genetic_diversity", None),
            "best_profit": (
                best_individual.metrics.get("profit", 0)
                if best_individual and best_individual.metrics
                else 0
            ),
            "wall_seconds": getattr(timing, "wall_seconds", 0) if timing else 0,
        }
        self._gen_history.append(record)

        if stats and stats.best_fitness is not None:
            self._best_fitness_history.append(stats.best_fitness)
        if stats and stats.avg_fitness is not None:
            self._avg_fitness_history.append(stats.avg_fitness)

        # Update mutation rate from extras
        if "mutation_rate" in self._extras:
            self._mutation_rate = self._extras["mutation_rate"]

        self._current_phase = ""
        self._refresh()

    def on_new_best(self, individual) -> None:
        self._new_best_flash = True
        self._best_individual = individual
        self._refresh()

    def on_convergence_warning(self, no_improvement: int, patience: int) -> None:
        self._convergence_warning = f"No improvement for {no_improvement}/{patience} generations"
        self._refresh()

    def on_evolution_complete(self, summary: dict | None = None) -> None:
        """Show final summary and stop."""
        self._current_phase = "COMPLETE"
        self._refresh()
        # Give user a moment to see the final state
        time.sleep(1.0)
        self.stop()
        # Print a final static summary to the terminal
        self._print_final_summary(summary)

    def on_checkpoint_saved(self, generation: int, path: str = "") -> None:
        """Flash checkpoint indicator on the dashboard."""
        self._extras['last_checkpoint'] = f"Gen {generation}"
        if path:
            self._extras['checkpoint_path'] = path
        self._refresh()

    def on_log(self, message: str, level: str = "info") -> None:
        """Forward an explicit log message to the monitor's log buffer."""
        self._log_handler.emit_text(message, level)
        self._refresh()

    def on_error(self, message: str, details: dict | None = None) -> None:
        """Show error prominently and log it."""
        detail_str = f" | {details}" if details else ""
        self._log_handler.emit_text(f"ERROR: {message}{detail_str}", "error")
        self._extras['last_error'] = message
        self._refresh()

    # ──────────────────────────────────────────────────────────────────────
    # Key handling
    # ──────────────────────────────────────────────────────────────────────
    def _on_key(self, key: str) -> None:
        if key == "s":
            self._view_mode = VIEW_SIMPLE
        elif key == "d":
            self._view_mode = VIEW_DETAILED
        elif key == "l":
            self._view_mode = VIEW_LOGS
        elif key == "q":
            # q doesn't stop evolution — just noted
            pass
        self._refresh()

    # ──────────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────────
    def _refresh(self) -> None:
        if self._live is not None:
            try:
                self._live.update(self._render())
            except Exception:
                pass  # Don't crash evolution on render failure

    def _render(self) -> Panel:
        """Build the full dashboard layout."""
        parts: List = []

        # Header (always visible)
        parts.append(self._render_header())

        # Body (view-mode dependent)
        if self._view_mode == VIEW_LOGS:
            parts.append(self._render_logs())
        elif self._view_mode == VIEW_DETAILED:
            parts.append(self._render_detailed())
        else:
            parts.append(self._render_simple())

        # Footer (keybinding hints)
        if self._show_keys:
            parts.append(self._render_footer())

        # Stack everything vertically using a group
        from rich.console import Group

        panel = Panel(
            Group(*parts),
            title="[bold cyan]GA EVOLUTION MONITOR[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
        return panel

    # ── Header ──────────────────────────────────────────────────────────
    def _render_header(self) -> Table:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        gens_done = self._current_gen + 1 if self._stats else self._current_gen
        gens_total = self._total_gens

        # ETA calculation
        if gens_done > 0 and self._gen_history:
            avg_gen_time = sum(r["wall_seconds"] for r in self._gen_history) / len(self._gen_history)
            remaining_gens = gens_total - gens_done
            eta = avg_gen_time * remaining_gens
        else:
            eta = -1

        # Progress fraction
        progress_pct = gens_done / max(gens_total, 1) * 100

        # Build header table (single row, multiple columns)
        header = Table.grid(padding=(0, 2))
        header.add_column("elapsed", justify="left", min_width=14)
        header.add_column("progress", justify="center", min_width=30)
        header.add_column("eta", justify="right", min_width=14)
        header.add_column("phase", justify="right", min_width=24)

        # Progress bar text
        bar_filled = int(progress_pct / 100 * 20)
        bar_empty = 20 - bar_filled
        bar = f"[green]{'█' * bar_filled}[/green][dim]{'░' * bar_empty}[/dim]"
        progress_text = f"Gen {gens_done}/{gens_total}  {bar}  {progress_pct:.0f}%"

        # Phase indicator
        phase_text = ""
        if self._current_phase:
            phase_label = PHASE_LABELS.get(self._current_phase, self._current_phase)
            if self._current_phase == "eval" and self._eval_total > 0:
                eval_pct = self._eval_completed / self._eval_total * 100
                phase_text = f"[yellow]⟳ {phase_label} ({self._eval_completed}/{self._eval_total} — {eval_pct:.0f}%)[/yellow]"
            elif self._current_phase == "COMPLETE":
                phase_text = "[bold green]✓ COMPLETE[/bold green]"
            else:
                phase_text = f"[yellow]⟳ {phase_label}[/yellow]"

        header.add_row(
            f"[bold]⏱ {_fmt_time(elapsed)}[/bold]",
            progress_text,
            f"ETA [bold]{_fmt_time(eta)}[/bold]" if eta >= 0 else "[dim]ETA --:--[/dim]",
            phase_text,
        )

        return header

    # ── Simple view ─────────────────────────────────────────────────────
    def _render_simple(self) -> Table:
        table = Table(
            title="[bold]Evolution Overview[/bold]",
            show_header=True,
            header_style="bold",
            border_style="blue",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Metric", style="cyan", min_width=18)
        table.add_column("Current", justify="right", min_width=12)
        table.add_column("Δ Prev", justify="right", min_width=12)
        table.add_column("Trend", justify="left", min_width=22)

        stats = self._stats
        prev = self._gen_history[-2] if len(self._gen_history) >= 2 else None

        # -- Best fitness
        best_fit = stats.best_fitness if stats and stats.best_fitness is not None else 0
        best_fit_str = f"[bold green]{best_fit:.4f}[/bold green]" if best_fit > 0 else "[dim]—[/dim]"
        delta_best = _delta_str(best_fit, prev["best_fitness"]) if prev and prev["best_fitness"] is not None else "[dim]—[/dim]"
        spark_best = _sparkline(self._best_fitness_history)

        # -- Avg fitness
        avg_fit = stats.avg_fitness if stats and stats.avg_fitness is not None else 0
        avg_fit_str = f"{avg_fit:.4f}"
        delta_avg = _delta_str(avg_fit, prev["avg_fitness"]) if prev and prev["avg_fitness"] is not None else "[dim]—[/dim]"
        spark_avg = _sparkline(self._avg_fitness_history)

        # -- Best profit
        best_profit = 0.0
        if self._best_individual and self._best_individual.metrics:
            best_profit = self._best_individual.metrics.get("profit", 0)
        bp_str = f"[bold]{best_profit:.2f}%[/bold]"
        if best_profit > 0:
            bp_str = f"[bold green]{best_profit:.2f}%[/bold green]"
        elif best_profit < 0:
            bp_str = f"[bold red]{best_profit:.2f}%[/bold red]"
        delta_bp = _pct_delta_str(best_profit, prev["best_profit"]) if prev else "[dim]—[/dim]"

        # -- Diversity
        diversity = stats.genetic_diversity if stats and stats.genetic_diversity is not None else None
        div_str = f"{diversity:.4f}" if diversity is not None else "[dim]—[/dim]"
        delta_div = ""
        if prev and prev["diversity"] is not None and diversity is not None:
            delta_div = _delta_str(diversity, prev["diversity"])
        else:
            delta_div = "[dim]—[/dim]"

        # -- Mutation rate
        mr_str = f"{self._mutation_rate:.2%}"

        # -- Generation time
        gen_time = self._timing.wall_seconds if self._timing else 0
        gt_str = f"{gen_time:.1f}s"

        table.add_row("Best Fitness", best_fit_str, delta_best, spark_best)
        table.add_row("Avg Fitness", avg_fit_str, delta_avg, spark_avg)
        table.add_row("Best Profit", bp_str, delta_bp, "")
        table.add_row("Diversity", div_str, delta_div, "")
        table.add_row("Mutation Rate", mr_str, "", "")
        table.add_row("Gen Time", gt_str, "", "")

        # New best flash
        if self._new_best_flash and self._best_individual:
            ind = self._best_individual
            table.add_row(
                "",
                f"[bold yellow]★ NEW BEST: {ind.id}[/bold yellow]",
                "",
                "",
            )

        # Convergence warning
        if self._convergence_warning:
            table.add_row(
                "",
                f"[bold red]⚠ {self._convergence_warning}[/bold red]",
                "",
                "",
            )

        return table

    # ── Detailed view ───────────────────────────────────────────────────
    def _render_detailed(self) -> Table:
        from rich.console import Group

        parts = []

        # 1. Include the simple overview table
        parts.append(self._render_simple())

        # 2. Phase timing breakdown
        timing_table = Table(
            title="[bold]Phase Timing (last generation)[/bold]",
            show_header=True,
            header_style="bold",
            border_style="magenta",
            padding=(0, 1),
            expand=True,
        )
        timing_table.add_column("Phase", style="cyan", min_width=14)
        timing_table.add_column("Time", justify="right", min_width=10)
        timing_table.add_column("% of Gen", justify="right", min_width=10)
        timing_table.add_column("Bar", min_width=20)

        if self._timing:
            total = max(self._timing.wall_seconds, 0.001)
            phases = [
                ("Evaluation", self._timing.eval_seconds),
                ("Selection", self._timing.selection_seconds),
                ("Holdout", self._timing.holdout_seconds),
                ("Overhead", self._timing.overhead_seconds),
            ]
            for name, secs in phases:
                pct = secs / total * 100
                bar_len = int(pct / 100 * 20)
                bar = f"[magenta]{'█' * bar_len}[/magenta][dim]{'░' * (20 - bar_len)}[/dim]"
                timing_table.add_row(name, f"{secs:.1f}s", f"{pct:.1f}%", bar)
            timing_table.add_row(
                "[bold]Total[/bold]", f"[bold]{total:.1f}s[/bold]", "[bold]100%[/bold]", ""
            )
        else:
            timing_table.add_row("[dim]No timing data yet[/dim]", "", "", "")

        parts.append(timing_table)

        # 3. Population composition & advanced stats
        detail_table = Table(
            title="[bold]Population & Evolution Details[/bold]",
            show_header=True,
            header_style="bold",
            border_style="green",
            padding=(0, 1),
            expand=True,
        )
        detail_table.add_column("Metric", style="cyan", min_width=24)
        detail_table.add_column("Value", justify="right", min_width=16)

        stats = self._stats

        # Population stats
        detail_table.add_row("Population Size", str(self._pop_size))
        detail_table.add_row("Mode", self._mode.replace("_", " ").title())

        if stats:
            detail_table.add_row(
                "Median Fitness",
                f"{stats.median_fitness:.4f}" if stats.median_fitness is not None else "—",
            )
            detail_table.add_row(
                "Worst Fitness",
                f"{stats.worst_fitness:.4f}" if stats.worst_fitness is not None else "—",
            )
            detail_table.add_row(
                "Raw Best Fitness",
                f"{stats.best_raw_fitness:.4f}" if stats.best_raw_fitness is not None else "—",
            )
            detail_table.add_row(
                "Diversity (σ)",
                f"{stats.diversity_score:.4f}" if stats.diversity_score is not None else "—",
            )

        # Holdout monitoring
        if stats and stats.holdout_num_evaluated:
            detail_table.add_row("", "")
            detail_table.add_row("[bold]Holdout Monitoring[/bold]", "")
            detail_table.add_row(
                "  Evaluated",
                str(stats.holdout_num_evaluated or 0),
            )
            detail_table.add_row(
                "  Profitable",
                str(stats.holdout_num_profitable or 0),
            )
            deg = stats.holdout_avg_degradation
            if deg is not None:
                color = "green" if deg < 0.3 else ("yellow" if deg < 0.5 else "red")
                detail_table.add_row(
                    "  Avg Degradation", f"[{color}]{deg:.1%}[/{color}]"
                )

        # Extras
        extras = self._extras
        if extras:
            if extras.get("holdout_penalties_applied", 0) > 0:
                detail_table.add_row(
                    "  Penalties Applied", str(extras["holdout_penalties_applied"])
                )
            if extras.get("llm_seeds_count", 0) or extras.get("llm_immigrants_count", 0):
                detail_table.add_row("", "")
                detail_table.add_row("[bold]LLM Strategies[/bold]", "")
                detail_table.add_row("  Seeds", str(extras.get("llm_seeds_count", 0)))
                detail_table.add_row("  Immigrants", str(extras.get("llm_immigrants_count", 0)))

        parts.append(detail_table)

        # 4. Generation history (last 5)
        if len(self._gen_history) > 1:
            hist_table = Table(
                title="[bold]Recent Generations[/bold]",
                show_header=True,
                header_style="bold",
                border_style="yellow",
                padding=(0, 1),
                expand=True,
            )
            hist_table.add_column("Gen", justify="center", min_width=6)
            hist_table.add_column("Best Fit", justify="right", min_width=10)
            hist_table.add_column("Avg Fit", justify="right", min_width=10)
            hist_table.add_column("Diversity", justify="right", min_width=10)
            hist_table.add_column("Time", justify="right", min_width=8)

            for rec in list(self._gen_history)[-5:]:
                gen_str = str(rec["gen"] + 1)
                bf = f"{rec['best_fitness']:.4f}" if rec["best_fitness"] is not None else "—"
                af = f"{rec['avg_fitness']:.4f}" if rec["avg_fitness"] is not None else "—"
                dv = f"{rec['diversity']:.4f}" if rec["diversity"] is not None else "—"
                wt = f"{rec['wall_seconds']:.1f}s"
                hist_table.add_row(gen_str, bf, af, dv, wt)

            parts.append(hist_table)

        # Assemble with Group
        return Group(*parts)

    # ── Logs view ───────────────────────────────────────────────────────
    def _render_logs(self) -> Panel:
        lines = self._log_handler.get_lines(last_n=40)
        if not lines:
            content = Text("No log output yet...", style="dim")
        else:
            content = Text()
            for level, line in lines:
                if level >= logging.ERROR:
                    content.append(line + "\n", style="bold red")
                elif level >= logging.WARNING:
                    content.append(line + "\n", style="yellow")
                elif level >= logging.INFO:
                    content.append(line + "\n")
                else:
                    content.append(line + "\n", style="dim")

        return Panel(
            content,
            title="[bold]Live Logs[/bold]",
            border_style="white",
            height=42,
        )

    # ── Footer ──────────────────────────────────────────────────────────
    def _render_footer(self) -> Text:
        mode_indicators = {
            VIEW_SIMPLE: "[bold reverse green] S [/bold reverse green]",
            VIEW_DETAILED: "[bold reverse blue] D [/bold reverse blue]",
            VIEW_LOGS: "[bold reverse white] L [/bold reverse white]",
        }
        active = mode_indicators.get(self._view_mode, "")

        parts = []
        for mode, label, key in [
            (VIEW_SIMPLE, "Simple", "S"),
            (VIEW_DETAILED, "Detailed", "D"),
            (VIEW_LOGS, "Logs", "L"),
        ]:
            if mode == self._view_mode:
                parts.append(f"[bold reverse] {key} {label} [/bold reverse]")
            else:
                parts.append(f"[dim] {key} {label} [/dim]")

        footer_text = Text.from_markup(
            "  ".join(parts) + "    [dim]Press key to switch view[/dim]"
        )
        return footer_text

    # ── Final summary (printed after Live stops) ────────────────────────
    def _print_final_summary(self, summary: dict | None = None) -> None:
        self._console.print()
        self._console.rule("[bold green]EVOLUTION COMPLETE[/bold green]")
        elapsed = time.monotonic() - self._start_time if self._start_time else 0

        summary_table = Table(show_header=False, border_style="green", padding=(0, 2))
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Total Time", _fmt_time(elapsed))
        summary_table.add_row("Generations", str(self._current_gen + 1))

        if self._best_individual:
            ind = self._best_individual
            summary_table.add_row("Best Strategy", str(ind.id))
            summary_table.add_row(
                "Best Fitness",
                f"{ind.fitness:.4f}" if ind.fitness is not None else "—",
            )
            if ind.metrics:
                summary_table.add_row(
                    "Best Profit", f"{ind.metrics.get('profit', 0):.2f}%"
                )
                summary_table.add_row(
                    "Win Rate", f"{ind.metrics.get('win_rate', 0):.1%}"
                )
                summary_table.add_row(
                    "Sharpe", f"{ind.metrics.get('sharpe_ratio', 0):.2f}"
                )

        if self._gen_history:
            avg_gen_time = sum(r["wall_seconds"] for r in self._gen_history) / len(
                self._gen_history
            )
            summary_table.add_row("Avg Gen Time", f"{avg_gen_time:.1f}s")

        self._console.print(summary_table)
        self._console.print()
