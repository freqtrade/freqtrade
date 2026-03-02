"""
WebSocket Monitor — Bridges the GA engine's Monitor interface to the EventBus.

Drop-in replacement for TerminalMonitor / NullMonitor.
All monitor callbacks translate to EventBus events so connected WebSocket
clients receive real-time updates.

Also stores per-generation population snapshots to disk for the drill-down UI.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from genetic_algorithm.web.event_bus import Event, EventType, get_event_bus

logger = logging.getLogger(__name__)


class WebSocketMonitor:
    """
    Monitor implementation that publishes events to the global EventBus.

    Each instance is associated with a single evolution run (identified by *run_id*).
    The web server subscribes to the EventBus and fans events out over WebSocket.

    Implements the same interface as NullMonitor / TerminalMonitor so it can be
    used as a drop-in replacement.
    """

    active: bool = True  # Suppresses tqdm in evolution.py

    def __init__(
        self,
        run_id: str,
        config: dict,
        snapshot_dir: Optional[Path] = None,
    ) -> None:
        self.run_id = run_id
        self.config = config
        self.bus = get_event_bus()

        # Where to persist per-generation population snapshots
        self.snapshot_dir = snapshot_dir or Path(
            f"genetic_algorithm/data/runs/{run_id}"
        )
        self._save_snapshots = config.get("web_dashboard", {}).get(
            "save_generation_snapshots", True
        )

        # In-memory generation stats cache (for late-joining clients)
        self.generation_stats: List[Dict[str, Any]] = []
        self._current_gen: int = 0
        self._total_gens: int = 0
        self._start_time: float = 0.0

        # Latest population snapshot (held until persisted at gen end)
        self._latest_population: Optional[List[dict]] = None

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self, config: dict) -> None:
        self._start_time = time.time()
        ga = config.get("genetic_algorithm", {})
        self.bus.publish(Event(
            type=EventType.RUN_STARTED,
            run_id=self.run_id,
            data={
                "population_size": ga.get("population_size"),
                "generations": ga.get("generations"),
                "mutation_rate": ga.get("mutation_rate"),
                "mode": ga.get("mode", "single_objective"),
                "pairs": config.get("backtesting", {}).get("pairs", []),
            },
        ))

    def stop(self) -> None:
        pass  # stop is called inside on_evolution_complete

    # ── Generation lifecycle ───────────────────────────────────────

    def on_generation_start(self, gen: int, total: int) -> None:
        self._current_gen = gen
        self._total_gens = total
        self.bus.publish(Event(
            type=EventType.GENERATION_START,
            run_id=self.run_id,
            data={"generation": gen, "total": total},
        ))

    def on_generation_end(
        self,
        gen: int,
        stats,
        timing,
        best_individual,
        extras: dict | None = None,
    ) -> None:
        # Build serialisable stats dict
        stats_dict: Dict[str, Any] = {}
        if stats is not None:
            for attr in (
                "generation", "size", "best_fitness", "avg_fitness",
                "worst_fitness", "median_fitness", "best_raw_fitness",
                "avg_raw_fitness", "genetic_diversity",
                "holdout_avg_degradation", "holdout_best_degradation",
                "holdout_num_evaluated", "holdout_num_profitable",
            ):
                val = getattr(stats, attr, None)
                if val is not None:
                    stats_dict[attr] = val

        timing_dict: Dict[str, Any] = {}
        if timing is not None:
            for attr in (
                "generation", "wall_seconds", "eval_seconds",
                "selection_seconds", "holdout_seconds", "overhead_seconds",
            ):
                val = getattr(timing, attr, None)
                if val is not None:
                    timing_dict[attr] = val

        best_dict: Dict[str, Any] = {}
        if best_individual is not None:
            try:
                best_dict = best_individual.to_dict()
            except Exception:
                best_dict = {"id": getattr(best_individual, "id", "?")}

        if extras:
            stats_dict.update(extras)

        self.generation_stats.append(stats_dict)

        # Flatten data so RunManager._on_event and the frontend store
        # can access fields like data.best_fitness directly
        flat_data: Dict[str, Any] = {"generation": gen}
        flat_data.update(stats_dict)   # best_fitness, avg_fitness, etc.
        flat_data.update(timing_dict)  # wall_seconds, eval_seconds, etc.
        if best_dict:
            flat_data["best_individual"] = best_dict
        # Keep nested copies for backward compat / drill-down
        flat_data["_stats"] = stats_dict
        flat_data["_timing"] = timing_dict

        self.bus.publish(Event(
            type=EventType.GENERATION_END,
            run_id=self.run_id,
            data=flat_data,
        ))

        # Persist population snapshot to disk
        self._persist_generation_snapshot(gen, stats_dict)

    # ── Phase tracking ─────────────────────────────────────────────

    def on_phase_start(self, phase: str) -> None:
        self.bus.publish(Event(
            type=EventType.PHASE_START,
            run_id=self.run_id,
            data={"phase": phase, "generation": self._current_gen},
        ))

    def on_phase_end(self, phase: str, elapsed: float = 0.0) -> None:
        self.bus.publish(Event(
            type=EventType.PHASE_END,
            run_id=self.run_id,
            data={"phase": phase, "elapsed": elapsed, "generation": self._current_gen},
        ))

    # ── Evaluation progress ────────────────────────────────────────

    def on_eval_progress(self, completed: int, total: int) -> None:
        self.bus.publish(Event(
            type=EventType.EVAL_PROGRESS,
            run_id=self.run_id,
            data={
                "completed": completed,
                "total": total,
                "generation": self._current_gen,
            },
        ))

    # ── Events ─────────────────────────────────────────────────────

    def on_new_best(self, individual) -> None:
        try:
            ind_dict = individual.to_dict()
        except Exception:
            ind_dict = {"id": getattr(individual, "id", "?"), "fitness": getattr(individual, "fitness", None)}
        self.bus.publish(Event(
            type=EventType.NEW_BEST,
            run_id=self.run_id,
            data={"individual": ind_dict},
        ))

    def on_convergence_warning(self, no_improvement: int, patience: int) -> None:
        self.bus.publish(Event(
            type=EventType.CONVERGENCE_WARNING,
            run_id=self.run_id,
            data={
                "no_improvement": no_improvement,
                "patience": patience,
                "generation": self._current_gen,
            },
        ))

    def on_evolution_complete(self, summary: dict | None = None) -> None:
        self.bus.publish(Event(
            type=EventType.EVOLUTION_COMPLETE,
            run_id=self.run_id,
            data=summary or {},
        ))

    # ── Population snapshot persistence ────────────────────────────

    def store_population_snapshot(self, individuals_dicts: List[dict]) -> None:
        """
        Called by the evolution engine (after evaluation) to hand over the
        current population for snapshot storage.

        The snapshot is actually written to disk in :meth:`on_generation_end`
        so it includes the generation number.
        """
        self._latest_population = individuals_dicts

    def _persist_generation_snapshot(
        self, gen: int, stats_dict: dict
    ) -> None:
        """Write gen_{N}.json with individuals + stats to the snapshot dir."""
        if not self._save_snapshots:
            return

        population_data = self._latest_population
        if population_data is None:
            return

        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = self.snapshot_dir / f"gen_{gen:04d}.json"
            snapshot = {
                "generation": gen,
                "run_id": self.run_id,
                "timestamp": time.time(),
                "stats": stats_dict,
                "individuals": population_data,
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(snapshot, f, default=str)
            tmp.rename(path)

            # Cleanup: keep at most max_snapshot_generations files
            self._cleanup_old_snapshots()

        except Exception:
            logger.exception("Failed to persist generation %d snapshot", gen)
        finally:
            self._latest_population = None

    def _cleanup_old_snapshots(self) -> None:
        """Delete oldest generation snapshots if count exceeds the limit."""
        max_snapshots = self.config.get("web_dashboard", {}).get(
            "max_snapshot_generations", 500
        )
        try:
            snapshots = sorted(self.snapshot_dir.glob("gen_*.json"))
            if len(snapshots) > max_snapshots:
                to_delete = snapshots[: len(snapshots) - max_snapshots]
                for f in to_delete:
                    f.unlink(missing_ok=True)
                logger.debug(
                    "Cleaned up %d old snapshots (keeping %d)",
                    len(to_delete), max_snapshots,
                )
        except Exception:
            logger.exception("Failed to cleanup old snapshots")
