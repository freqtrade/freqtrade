"""
Data Service — Unified data access layer.

Serves data from:
  1. Active runs (via RunManager, in-memory state)
  2. Past runs (from disk — generation snapshots, checkpoints, outputs)
  3. Hall of Fame (from hall_of_fame.json)
  4. Config templates (from config/*.yaml)

File-first approach: JSON/YAML/CSV files are the source of truth.
A SQLite index can be layered on top later for cross-run queries.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from genetic_algorithm.web.models.generation import GenerationDetail, IndividualSummary
from genetic_algorithm.web.models.run import GenerationStatsModel, RunDetail, RunStatus, RunSummary
from genetic_algorithm.web.models.strategy import (
    ConditionModel,
    IndicatorModel,
    QualityAssessment,
    StrategyDetail,
    StrategyGeneModel,
)

logger = logging.getLogger(__name__)

# Default paths (relative to project root)
RUNS_DIR = Path("genetic_algorithm/data/runs")
CHECKPOINTS_DIR = Path("genetic_algorithm/data/checkpoints")
HOF_DIR = Path("genetic_algorithm/data/hall_of_fame")
CONFIG_DIR = Path("genetic_algorithm/config")
OUTPUT_DIR = Path("genetic_algorithm/output")


class DataService:
    """
    Provides read access to all GA data for the REST API.

    For active runs, delegates to RunManager.
    For historical data, reads from the file system.
    """

    def __init__(self, run_manager=None) -> None:
        from genetic_algorithm.web.run_manager import RunManager
        self.run_manager: RunManager = run_manager or RunManager()

    # ── Runs ───────────────────────────────────────────────────────

    def list_runs(self) -> List[RunSummary]:
        """List all runs: active (from RunManager) + past (from disk)."""
        active = self.run_manager.list_runs()
        active_ids = {r.run_id for r in active}

        # Scan disk for past runs
        past: List[RunSummary] = []
        if RUNS_DIR.exists():
            for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                rid = run_dir.name
                if rid in active_ids:
                    continue
                summary = self._load_run_summary_from_disk(rid, run_dir)
                if summary:
                    past.append(summary)

        return active + past

    def get_run_detail(self, run_id: str) -> Optional[RunDetail]:
        """Get full detail for a run (active or past)."""
        handle = self.run_manager.get_run(run_id)
        if handle:
            return self._run_detail_from_handle(handle)
        return self._load_run_detail_from_disk(run_id)

    # ── Generations ────────────────────────────────────────────────

    def get_generation(self, run_id: str, gen: int) -> Optional[GenerationDetail]:
        """Load generation snapshot (all individuals)."""
        path = RUNS_DIR / run_id / f"gen_{gen:04d}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            individuals = [
                IndividualSummary.from_individual_dict(d)
                for d in data.get("individuals", [])
            ]
            return GenerationDetail(
                run_id=run_id,
                generation=gen,
                individuals=individuals,
                stats=data.get("stats"),
            )
        except Exception:
            logger.exception("Failed to load generation %d for run %s", gen, run_id)
            return None

    # ── Strategies ─────────────────────────────────────────────────

    def get_strategy(self, run_id: str, strategy_id: str) -> Optional[StrategyDetail]:
        """
        Find a strategy across all generation snapshots for *run_id*.

        *strategy_id* has the format ``Gen{N}_Ind{M}``.
        Falls back to Hall of Fame if the run directory doesn't exist
        (e.g. HoF entries that reference a timestamp-based run_id).
        """
        import re

        run_dir = RUNS_DIR / run_id
        if run_dir.exists():
            # Try optimized lookup: parse generation number from ID
            m = re.match(r"Gen(\d+)_Ind(\d+)", strategy_id)
            if m:
                target_gen = int(m.group(1))
                target_path = run_dir / f"gen_{target_gen:04d}.json"
                result = self._find_in_gen_file(target_path, strategy_id)
                if result:
                    return result

            # Fallback: scan all generation files
            for path in sorted(run_dir.glob("gen_*.json")):
                result = self._find_in_gen_file(path, strategy_id)
                if result:
                    return result

        # Also check HoF (works even when run_dir doesn't exist)
        return self._search_hof(run_id, strategy_id)

    def _find_in_gen_file(self, path, strategy_id: str) -> Optional[StrategyDetail]:
        """Search a single generation JSON file for a strategy by ID."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            run_id = path.parent.name
            for ind_dict in data.get("individuals", []):
                if ind_dict.get("id") == strategy_id:
                    return self._build_strategy_detail(run_id, ind_dict)
        except Exception:
            pass
        return None

    def get_lineage(self, run_id: str, strategy_id: str) -> List[dict]:
        """
        Trace the ancestral chain of a strategy back through generations.

        Follows the first parent at each generation until we reach gen 0
        or a strategy with no parents.  Returns a list of LineageNode dicts
        sorted in ascending generation order.
        """
        import re

        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            return []

        chain: List[dict] = []
        visited: set = set()
        current_id = strategy_id

        while current_id and current_id not in visited:
            visited.add(current_id)

            m = re.match(r"Gen(\d+)_Ind(\d+)", current_id)
            if not m:
                break

            gen_num = int(m.group(1))
            gen_path = run_dir / f"gen_{gen_num:04d}.json"
            if not gen_path.exists():
                break

            try:
                with open(gen_path) as f:
                    data = json.load(f)
            except Exception:
                break

            ind_dict = None
            for ind in data.get("individuals", []):
                if ind.get("id") == current_id:
                    ind_dict = ind
                    break

            if ind_dict is None:
                break

            metrics = ind_dict.get("metrics", {})
            chain.append(
                {
                    "id": current_id,
                    "generation": gen_num,
                    "fitness": ind_dict.get("fitness"),
                    "raw_fitness": ind_dict.get("raw_fitness"),
                    "profit": metrics.get("total_profit_pct")
                    or metrics.get("profit_total"),
                    "parent_ids": ind_dict.get("parent_ids", []),
                    "mutations": ind_dict.get("mutations", []),
                }
            )

            parent_ids = ind_dict.get("parent_ids", [])
            current_id = parent_ids[0] if parent_ids else None

        # Return ascending by generation
        chain.sort(key=lambda n: n["generation"])
        return chain

    def get_strategy_code(self, run_id: str, strategy_id: str) -> Optional[str]:
        """Generate Python code for a strategy."""
        detail = self.get_strategy(run_id, strategy_id)
        if not detail or not detail.gene:
            return None
        try:
            from genetic_algorithm.core.strategy_gene import StrategyGene
            from genetic_algorithm.strategies.generator import StrategyGenerator

            # Reconstruct StrategyGene from model
            gene_dict = self._gene_model_to_dict(detail.gene, detail)
            gene = StrategyGene.from_dict(gene_dict)

            # Load config for generator
            config = self._load_run_config(run_id)
            if not config:
                # Fall back to default config
                default = CONFIG_DIR / "ga_config.yaml"
                if default.exists():
                    with open(default) as f:
                        config = yaml.safe_load(f)
                else:
                    return None
            generator = StrategyGenerator(config)
            return generator.generate_strategy_code(gene)
        except Exception:
            logger.exception("Failed to generate code for %s/%s", run_id, strategy_id)
            return None

    # ── Hall of Fame ───────────────────────────────────────────────

    def get_hall_of_fame(self) -> List[Dict[str, Any]]:
        """Return all Hall of Fame entries with flattened fields for the frontend."""
        hof_path = HOF_DIR / "hall_of_fame.json"
        if not hof_path.exists():
            return []
        try:
            with open(hof_path) as f:
                data = json.load(f)
            raw_entries = data.get("entries", [])
            return [self._flatten_hof_entry(e) for e in raw_entries]
        except Exception:
            logger.exception("Failed to load Hall of Fame")
            return []

    def _flatten_hof_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a raw HoF entry into the flat shape the frontend expects."""
        gene = entry.get("strategy_gene", entry.get("strategy_gene_dict", {}))
        metrics = entry.get("metrics", {})
        gen_num = gene.get("generation", 0)
        ind_id = gene.get("individual_id", 0)

        # Build added_at from run_timestamp
        ts = entry.get("run_timestamp", 0)
        try:
            from datetime import datetime, timezone
            added_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""
        except Exception:
            added_at = ""

        return {
            "id": f"Gen{gen_num}_Ind{ind_id}",
            "fitness": entry.get("fitness", 0),
            "profit": metrics.get("profit", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "num_trades": metrics.get("num_trades", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "win_rate": metrics.get("win_rate", 0),
            "complexity": metrics.get("complexity", 0),
            "timeframe": gene.get("timeframe", "5m"),
            "added_at": added_at,
            "config_name": entry.get("config_name", ""),
            "run_id": entry.get("run_id", ""),
            "generation_found": entry.get("generation_found", gen_num),
            # Keep the full strategy_gene for injection
            "strategy_gene": gene,
        }

    # ── Config templates ───────────────────────────────────────────

    def get_config_templates(self) -> List[Dict[str, Any]]:
        """List available config templates."""
        templates = []
        for path in sorted(CONFIG_DIR.glob("ga_config*.yaml")):
            try:
                with open(path) as f:
                    config = yaml.safe_load(f)
                ga = config.get("genetic_algorithm", {})
                templates.append({
                    "name": path.stem,
                    "path": str(path),
                    "population_size": ga.get("population_size", "?"),
                    "generations": ga.get("generations", "?"),
                    "pairs": config.get("backtesting", {}).get("pairs", []),
                })
            except Exception:
                continue
        return templates

    def load_config_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a config template by name (stem)."""
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            return None
        with open(path) as f:
            return yaml.safe_load(f)

    # ── Private helpers ────────────────────────────────────────────

    def _load_run_summary_from_disk(self, run_id: str, run_dir: Path) -> Optional[RunSummary]:
        """Build a RunSummary from a past run's files."""
        config_path = run_dir / "config.yaml"
        config: Dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
            except Exception:
                pass

        # Count generation snapshots
        gen_files = sorted(run_dir.glob("gen_*.json"))
        current_gen = 0
        best_fitness = None
        if gen_files:
            try:
                last = gen_files[-1]
                with open(last) as f:
                    data = json.load(f)
                current_gen = data.get("generation", 0)
                stats = data.get("stats", {})
                best_fitness = stats.get("best_fitness")
            except Exception:
                pass

        ga_cfg = config.get("genetic_algorithm", {})
        return RunSummary(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            config_name=config.get("_config_name", run_id),
            current_generation=current_gen,
            total_generations=ga_cfg.get("generations", 0),
            best_fitness=best_fitness,
            population_size=ga_cfg.get("population_size", 0),
            pairs=config.get("backtesting", {}).get("pairs", []),
        )

    def _run_detail_from_handle(self, handle) -> RunDetail:
        """Build RunDetail from an active RunHandle."""
        gen_stats = [
            GenerationStatsModel(**s) for s in handle.generation_stats
            if isinstance(s, dict)
        ]
        ga = handle.config.get("genetic_algorithm", {})
        return RunDetail(
            run_id=handle.run_id,
            status=handle.status,
            config_name=handle.config_name,
            current_generation=handle.current_generation,
            total_generations=handle.total_generations,
            best_fitness=handle.best_fitness,
            best_profit=handle.best_profit,
            population_size=ga.get("population_size", 0),
            started_at=handle.started_at,
            elapsed_seconds=(time.time() - handle.started_at) if handle.started_at else None,
            pairs=handle.config.get("backtesting", {}).get("pairs", []),
            config=handle.config,
            generation_stats=gen_stats,
            best_individual_id=handle.best_individual_id,
            mode=ga.get("mode", "single_objective"),
        )

    def _load_run_detail_from_disk(self, run_id: str) -> Optional[RunDetail]:
        """Build RunDetail from disk files."""
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            return None

        summary = self._load_run_summary_from_disk(run_id, run_dir)
        if not summary:
            return None

        # Load config
        config = self._load_run_config(run_id) or {}

        # Load all generation stats
        gen_stats: List[GenerationStatsModel] = []
        for path in sorted(run_dir.glob("gen_*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                s = data.get("stats", {})
                s["generation"] = data.get("generation", 0)
                gen_stats.append(GenerationStatsModel(**{
                    k: v for k, v in s.items()
                    if k in GenerationStatsModel.model_fields
                }))
            except Exception:
                continue

        ga = config.get("genetic_algorithm", {})
        return RunDetail(
            run_id=summary.run_id,
            status=summary.status,
            config_name=summary.config_name,
            current_generation=summary.current_generation,
            total_generations=summary.total_generations,
            best_fitness=summary.best_fitness,
            population_size=summary.population_size,
            pairs=summary.pairs,
            config=config,
            generation_stats=gen_stats,
            mode=ga.get("mode", "single_objective"),
        )

    def _load_run_config(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = RUNS_DIR / run_id / "config.yaml"
        if not path.exists():
            return None
        with open(path) as f:
            return yaml.safe_load(f)

    def _build_strategy_detail(self, run_id: str, ind_dict: dict) -> StrategyDetail:
        """Build a StrategyDetail from an Individual.to_dict() output."""
        gene_dict = ind_dict.get("strategy_gene", {})
        metrics = ind_dict.get("metrics", {})

        # Build gene model
        gene = StrategyGeneModel(
            generation=gene_dict.get("generation", 0),
            individual_id=gene_dict.get("individual_id", 0),
            indicators=[
                IndicatorModel(**i) for i in gene_dict.get("indicators", [])
            ],
            entry_conditions=[
                ConditionModel(**c) for c in gene_dict.get("entry_conditions", [])
            ],
            exit_conditions=[
                ConditionModel(**c) for c in gene_dict.get("exit_conditions", [])
            ],
            timeframe=gene_dict.get("timeframe", "5m"),
            stoploss=gene_dict.get("stoploss", -0.1),
            minimal_roi=gene_dict.get("minimal_roi", {}),
            max_open_trades=gene_dict.get("max_open_trades", 3),
            informative_timeframes=gene_dict.get("informative_timeframes", []),
            trailing_stop=gene_dict.get("trailing_stop", False),
            trailing_stop_positive=gene_dict.get("trailing_stop_positive"),
            trailing_stop_positive_offset=gene_dict.get("trailing_stop_positive_offset"),
            can_short=gene_dict.get("can_short", False),
        )

        # Build quality assessment from metrics
        quality = self._compute_quality(metrics)

        # Extract walk-forward and Monte Carlo data if available
        walk_forward_windows = ind_dict.get("walk_forward_windows") or metrics.get("walk_forward_windows")
        monte_carlo = ind_dict.get("monte_carlo") or metrics.get("monte_carlo")

        return StrategyDetail(
            id=ind_dict.get("id", ""),
            run_id=run_id,
            generation=gene_dict.get("generation", 0),
            fitness=ind_dict.get("fitness"),
            raw_fitness=ind_dict.get("raw_fitness"),
            metrics=metrics,
            gene=gene,
            quality=quality,
            parent_ids=ind_dict.get("parent_ids", []),
            mutations=ind_dict.get("mutations", []),
            walk_forward_windows=walk_forward_windows if isinstance(walk_forward_windows, list) else None,
            monte_carlo=monte_carlo if isinstance(monte_carlo, dict) else None,
        )

    @staticmethod
    def _compute_quality(metrics: dict) -> QualityAssessment:
        """Compute quality labels from numeric metric values."""
        hd = metrics.get("holdout_degradation")
        mc = metrics.get("mc_robustness")
        wf = metrics.get("train_val_gap")

        def _degradation_label(v):
            if v is None:
                return "UNKNOWN"
            v = abs(v)
            if v < 0.15:
                return "EXCELLENT"
            if v < 0.30:
                return "GOOD"
            if v < 0.50:
                return "MODERATE"
            return "POOR"

        def _robustness_label(v):
            if v is None:
                return "UNKNOWN"
            if v > 0.80:
                return "EXCELLENT"
            if v > 0.60:
                return "GOOD"
            if v > 0.40:
                return "MODERATE"
            return "POOR"

        h_label = _degradation_label(hd)
        wf_label = _degradation_label(wf)
        mc_label = _robustness_label(mc)

        # Composite score: average of available normalised scores (0-1, higher=better)
        scores = []
        if hd is not None:
            scores.append(max(0.0, 1.0 - abs(hd)))
        if wf is not None:
            scores.append(max(0.0, 1.0 - abs(wf)))
        if mc is not None:
            scores.append(mc)
        composite = sum(scores) / len(scores) if scores else None

        def _overall(c):
            if c is None:
                return "UNKNOWN"
            if c > 0.80:
                return "EXCELLENT"
            if c > 0.60:
                return "GOOD"
            if c > 0.40:
                return "MODERATE"
            return "POOR"

        return QualityAssessment(
            holdout_degradation=hd,
            holdout_label=h_label,
            wf_gap=wf,
            wf_label=wf_label,
            mc_robustness=mc,
            mc_label=mc_label,
            composite_score=composite,
            overall_label=_overall(composite),
        )

    def _gene_model_to_dict(self, gene: StrategyGeneModel, detail: StrategyDetail) -> dict:
        """Convert StrategyGeneModel back to a dict for StrategyGene.from_dict()."""
        return {
            "generation": gene.generation,
            "individual_id": gene.individual_id,
            "indicators": [i.model_dump() for i in gene.indicators],
            "entry_conditions": [c.model_dump() for c in gene.entry_conditions],
            "exit_conditions": [c.model_dump() for c in gene.exit_conditions],
            "timeframe": gene.timeframe,
            "stoploss": gene.stoploss,
            "minimal_roi": gene.minimal_roi,
            "max_open_trades": gene.max_open_trades,
            "informative_timeframes": gene.informative_timeframes,
            "trailing_stop": gene.trailing_stop,
            "trailing_stop_positive": gene.trailing_stop_positive,
            "trailing_stop_positive_offset": gene.trailing_stop_positive_offset,
            "can_short": gene.can_short,
        }

    def _search_hof(self, run_id: str, strategy_id: str) -> Optional[StrategyDetail]:
        """Search for a strategy in the Hall of Fame.

        Matches by *strategy_id* and *run_id*.  If no exact run_id match is
        found, falls back to matching by strategy_id alone so that legacy
        HoF entries with timestamp-based run_ids are still accessible.
        """
        entries = self.get_hall_of_fame()
        fallback = None
        for entry in entries:
            gene = entry.get("strategy_gene_dict", entry.get("strategy_gene", {}))
            gen = gene.get("generation", 0)
            ind_id = gene.get("individual_id", 0)
            hof_id = f"Gen{gen}_Ind{ind_id}"
            if hof_id == strategy_id:
                detail = self._build_strategy_detail(run_id, {
                    "id": hof_id,
                    "strategy_gene": gene,
                    "fitness": entry.get("fitness"),
                    "metrics": entry.get("metrics", {}),
                })
                # Prefer exact run_id match
                if entry.get("run_id") == run_id:
                    return detail
                # Keep first match as fallback
                if fallback is None:
                    fallback = detail
        return fallback
