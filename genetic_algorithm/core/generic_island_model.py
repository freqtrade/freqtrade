"""
Generic Island Model Evolution

A configurable N-island system where independent GA populations evolve
strategies with different specializations (indicator pools, pair subsets,
seeds) and periodically exchange individuals through multiple migration
topologies.

Unlike the regime-locked ``IslandModelEvolution``, every island receives
the **full** dataset (or a configured pair subset) and can optionally
enable walk-forward validation.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  Island 0          Island 1         ...  Island N-1  │
    │  (momentum)        (trend)               (mixed)     │
    │  ┌──────────┐     ┌──────────┐     ┌──────────┐     │
    │  │ pop=10   │◄───►│ pop=10   │◄───►│ pop=10   │     │
    │  │ seed=42  │     │ seed=43  │     │ seed=56  │     │
    │  └──────────┘     └──────────┘     └──────────┘     │
    │       ↕ ring / fully_connected / tournament / merge  │
    └──────────────────────────────────────────────────────┘

Usage:
    from genetic_algorithm.core.generic_island_model import (
        GenericIslandModelEvolution,
    )
    evo = GenericIslandModelEvolution("config/ga_config_generic_island.yaml")
    results = evo.evolve()
"""

import copy
import glob
import hashlib
import json
import logging
import os
import random
import signal
import tempfile
import time
import threading
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import (
    Population,
    apply_fitness_sharing,
    calculate_pairwise_distances,
)
from genetic_algorithm.core.hall_of_fame import HallOfFame

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Indicator families for auto-generation
# ══════════════════════════════════════════════════════════════════════

INDICATOR_FAMILIES: Dict[str, List[str]] = {
    'momentum': ['RSI', 'MACD', 'STOCH', 'CCI', 'MFI', 'ROC', 'WILLR'],
    'trend': ['EMA', 'SMA', 'TEMA', 'KAMA', 'SUPERTREND', 'AROON', 'ICHIMOKU', 'SAR'],
    'volatility': ['BBANDS', 'ATR', 'DONCHIAN'],
    'volume': ['OBV', 'CMF', 'VROC', 'VWAP'],
    'candlestick': [
        'CDL_ENGULFING', 'CDL_HAMMER', 'CDL_DOJI', 'CDL_MORNINGSTAR',
        'CDL_EVENINGSTAR', 'CDL_SHOOTINGSTAR', 'CDL_HARAMI', 'CDL_PIERCING',
        'CDL_DARKCLOUD', 'CDL_3WHITESOLDIERS', 'CDL_3BLACKCROWS',
    ],
}

ALL_INDICATORS = [ind for family in INDICATOR_FAMILIES.values() for ind in family]


# ══════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GenericIslandConfig:
    """Configuration for a single generic island."""
    name: str
    population_size: int = 10
    generations: Optional[int] = None  # None = inherit from top-level
    seed: int = 42
    indicator_pool: Optional[List[str]] = None  # None = all indicators
    pairs: Optional[List[str]] = None  # None = inherit from base config
    walk_forward_enabled: Optional[bool] = None  # None = inherit
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericMigrationConfig:
    """Configuration for migration between generic islands."""
    topology: str = 'ring'  # ring, fully_connected, tournament, hierarchical
    interval: int = 3  # every N generations
    count: int = 2  # top-N individuals to migrate
    merge_rounds: bool = False  # periodic global pool-and-redistribute
    merge_interval: int = 5  # if merge_rounds=True, every N gens
    tournament_size: int = 3  # if topology=tournament


@dataclass
class GenericIslandStats:
    """Statistics for one generic island across evolution."""
    name: str
    best_fitness: float = 0.0
    best_profit: float = 0.0
    avg_fitness: float = 0.0
    generations_completed: int = 0
    migrants_sent: int = 0
    migrants_received: int = 0


@dataclass
class GenericMigrationEvent:
    """Record of a migration event."""
    generation: int
    source: str
    target: str
    count: int
    fitnesses: List[float]


@dataclass
class _AggregateStats:
    """Lightweight aggregate stats for the terminal monitor."""
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    genetic_diversity: Optional[float] = None
    generation: int = 0
    best_raw_fitness: Optional[float] = None
    median_fitness: Optional[float] = None
    diversity_score: Optional[float] = None
    holdout_avg_degradation: Optional[float] = None
    holdout_best_degradation: Optional[float] = None
    holdout_num_evaluated: Optional[int] = None
    holdout_num_profitable: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════
# Generic Island Model Evolution
# ══════════════════════════════════════════════════════════════════════

class GenericIslandModelEvolution:
    """
    Orchestrates N independent GeneticAlgorithm instances as peer islands
    with configurable migration topologies.

    Unlike the regime-locked ``IslandModelEvolution``, islands are NOT
    tied to regime segments.  Each island gets the full dataset (or an
    optional pair subset) and evolves independently with its own seed,
    optional indicator pool restriction, and optional walk-forward.
    """

    def __init__(
        self,
        config_path: str,
        visualize: bool = False,
        interactive: bool = True,
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.visualize = visualize
        self.interactive = interactive
        self.logger = logging.getLogger(f"{__name__}.GenericIslandModel")

        # Parse generic_island_model config
        gim_cfg = self.config.get('generic_island_model', {})
        self.num_islands: int = gim_cfg.get('num_islands', 15)
        self.parallel_islands: bool = gim_cfg.get('parallel_islands', False)
        self.population_per_island: int = gim_cfg.get('population_per_island', 10)
        self.generations: int = gim_cfg.get(
            'generations',
            self.config.get('genetic_algorithm', {}).get('generations', 20),
        )

        # Specialization config
        spec_cfg = gim_cfg.get('specialization', {})
        self.rotate_seeds: bool = spec_cfg.get('rotate_seeds', True)
        self.use_indicator_pools: bool = spec_cfg.get('indicator_pools', True)
        self.indicator_overlap: float = spec_cfg.get('indicator_overlap', 0.5)
        self.pair_rotation: bool = spec_cfg.get('pair_rotation', False)
        self.pair_subset_size: int = spec_cfg.get('pair_subset_size', 3)

        # Migration config
        mig_cfg = gim_cfg.get('migration', {})
        self.migration = GenericMigrationConfig(
            topology=mig_cfg.get('topology', 'ring'),
            interval=mig_cfg.get('interval', 3),
            count=mig_cfg.get('count', 2),
            merge_rounds=mig_cfg.get('merge_rounds', False),
            merge_interval=mig_cfg.get('merge_interval', 5),
            tournament_size=mig_cfg.get('tournament_size', 3),
        )

        # Walk-forward config (per-island override)
        wf_cfg = gim_cfg.get('walk_forward', {})
        self.island_walk_forward: Optional[bool] = (
            wf_cfg.get('enabled') if 'enabled' in wf_cfg else None
        )

        # Base seed (from GA config or default)
        self.base_seed: int = (
            self.config.get('genetic_algorithm', {}).get('seed', 42)
        )

        # Build island configs
        explicit_islands = gim_cfg.get('islands', [])
        if explicit_islands:
            self.island_configs = self._parse_explicit_islands(explicit_islands)
        else:
            self.island_configs = self._auto_generate_islands()

        # Runtime state
        self.islands: Dict[str, GeneticAlgorithm] = {}
        self.island_populations: Dict[str, Population] = {}
        self.island_stats: Dict[str, GenericIslandStats] = {}
        self.generation_stats: Dict[str, list] = {}
        self.migration_history: List[GenericMigrationEvent] = []
        self.hall_of_fame = HallOfFame(
            max_size=self.config.get('hall_of_fame', {}).get('max_size', 50)
        )

        # External migration (cross-machine strategy exchange)
        ext_cfg = gim_cfg.get('external_migration', {})
        self.external_migration_enabled: bool = ext_cfg.get('enabled', False)
        self.external_migration_dir: Path = Path(
            ext_cfg.get('directory', 'genetic_algorithm/data/incoming_migrants')
        )
        self.external_export_dir: Path = Path(
            ext_cfg.get('export_directory', 'genetic_algorithm/data/outgoing_migrants')
        )
        self.external_migration_interval: int = ext_cfg.get(
            'interval', self.migration.interval
        )
        self.external_migration_count: int = ext_cfg.get('count', 3)

        # Thread safety
        self._hof_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._migration_lock = threading.Lock()
        self._shutdown_requested = False

    # ------------------------------------------------------------------
    # Island configuration building
    # ------------------------------------------------------------------

    def _parse_explicit_islands(
        self, island_defs: List[Dict[str, Any]],
    ) -> List[GenericIslandConfig]:
        """Parse explicitly defined island configs from YAML."""
        configs = []
        for i, idef in enumerate(island_defs):
            configs.append(GenericIslandConfig(
                name=idef.get('name', f'island_{i}'),
                population_size=idef.get(
                    'population_size', self.population_per_island,
                ),
                generations=idef.get('generations'),
                seed=idef.get('seed', self.base_seed + i),
                indicator_pool=idef.get('indicator_pool'),
                pairs=idef.get('pairs'),
                walk_forward_enabled=idef.get('walk_forward_enabled'),
                extra_config=idef.get('extra_config', {}),
            ))
        return configs

    def _auto_generate_islands(self) -> List[GenericIslandConfig]:
        """
        Auto-generate island configs with rotated seeds, optionally
        split indicator pools and rotated pair subsets.
        """
        indicator_pools = self._split_indicator_pools() if self.use_indicator_pools else None
        pair_subsets = self._rotate_pairs() if self.pair_rotation else None

        configs = []
        for i in range(self.num_islands):
            seed = (self.base_seed + i) if self.rotate_seeds else self.base_seed

            pool = None
            if indicator_pools:
                pool = indicator_pools[i % len(indicator_pools)]

            pairs = None
            if pair_subsets:
                pairs = pair_subsets[i % len(pair_subsets)]

            name_suffix = ''
            if pool:
                # Name after the primary family
                primary_family = self._identify_primary_family(pool)
                name_suffix = f'_{primary_family}'

            configs.append(GenericIslandConfig(
                name=f'island_{i}{name_suffix}',
                population_size=self.population_per_island,
                seed=seed,
                indicator_pool=pool,
                pairs=pairs,
            ))

        self.logger.info(
            "Auto-generated %d island configs (seeds=%s, indicator_pools=%s, pair_rotation=%s)",
            len(configs), self.rotate_seeds, self.use_indicator_pools, self.pair_rotation,
        )
        return configs

    def _split_indicator_pools(self) -> List[List[str]]:
        """
        Split indicators into overlapping pools by family.

        Each pool gets 2-3 families as its core, plus a random selection
        from other families for overlap.
        """
        families = list(INDICATOR_FAMILIES.keys())
        pools = []

        # Create one pool per family pair (sliding window with overlap)
        for i in range(len(families)):
            # Core: 2 adjacent families
            core_families = [families[i], families[(i + 1) % len(families)]]
            core_indicators = []
            for fam in core_families:
                core_indicators.extend(INDICATOR_FAMILIES[fam])

            # Overlap: random picks from other families
            other_indicators = [
                ind for fam, inds in INDICATOR_FAMILIES.items()
                if fam not in core_families
                for ind in inds
            ]
            overlap_count = max(1, int(len(other_indicators) * self.indicator_overlap))
            rng = random.Random(self.base_seed + i)
            overlap_picks = rng.sample(
                other_indicators, min(overlap_count, len(other_indicators)),
            )

            pool = sorted(set(core_indicators + overlap_picks))
            pools.append(pool)

        return pools

    def _rotate_pairs(self) -> List[List[str]]:
        """
        Create rotated pair subsets from the base config's pair list.
        """
        base_pairs = self.config.get('backtesting', {}).get('pairs', ['BTC/USDT'])
        if len(base_pairs) <= self.pair_subset_size:
            return [base_pairs]

        subsets = []
        for i in range(len(base_pairs)):
            subset = []
            for j in range(self.pair_subset_size):
                subset.append(base_pairs[(i + j) % len(base_pairs)])
            subsets.append(subset)
        return subsets

    @staticmethod
    def _identify_primary_family(indicators: List[str]) -> str:
        """Determine which indicator family dominates a pool."""
        best_family = 'mixed'
        best_count = 0
        for family, members in INDICATOR_FAMILIES.items():
            count = sum(1 for ind in indicators if ind in members)
            if count > best_count:
                best_count = count
                best_family = family
        return best_family

    # ------------------------------------------------------------------
    # Build sub-GA for an island
    # ------------------------------------------------------------------

    def _build_island_config(self, ic: GenericIslandConfig) -> Dict[str, Any]:
        """
        Build a complete config dict for one island's GA, derived
        from the base config with island-specific overrides.
        """
        cfg = copy.deepcopy(self.config)

        # Override population size and related params
        cfg['genetic_algorithm']['population_size'] = ic.population_size
        cfg['genetic_algorithm']['elite_size'] = max(2, ic.population_size // 10)
        cfg['genetic_algorithm']['random_immigrants'] = max(
            2, ic.population_size // 10,
        )

        # Set seed
        cfg['genetic_algorithm']['seed'] = ic.seed

        # Set generations (per-island override or global)
        island_gens = ic.generations if ic.generations is not None else self.generations
        cfg['genetic_algorithm']['generations'] = island_gens

        # Walk-forward: configurable per island (unlike old model)
        wf_enabled = ic.walk_forward_enabled
        if wf_enabled is None:
            wf_enabled = self.island_walk_forward
        if wf_enabled is not None:
            cfg.setdefault('walk_forward', {})['enabled'] = wf_enabled

        # Indicator pool restriction (if specified)
        if ic.indicator_pool is not None:
            cfg.setdefault('indicators', {})['available'] = list(
                ic.indicator_pool
            )

        # Pair subset (if specified)
        if ic.pairs is not None:
            cfg.setdefault('backtesting', {})['pairs'] = list(ic.pairs)

        # Disable nested island models to prevent recursion
        cfg['island_model'] = {'enabled': False}
        cfg['generic_island_model'] = {'enabled': False}

        # Disable terminal monitor for sub-islands
        cfg['terminal_monitor'] = {'enabled': False}

        # Tag island name for logging
        cfg['_island_name'] = ic.name

        # Apply extra config overrides
        for key, value in ic.extra_config.items():
            cfg[key] = value

        # Parallel evaluation: split workers across islands
        if self.parallel_islands:
            par_cfg = cfg.get('parallel_evaluation', {})
            if par_cfg.get('enabled', False):
                total_workers = par_cfg.get('num_workers') or max(
                    1, os.cpu_count() - 1,
                )
                workers_per_island = max(
                    1, total_workers // len(self.island_configs),
                )
                par_cfg['num_workers'] = workers_per_island

        return cfg

    def _create_island_ga(self, ic: GenericIslandConfig) -> GeneticAlgorithm:
        """
        Create a GeneticAlgorithm instance for one island.

        Writes a temporary YAML config and instantiates a GA from it.
        Unlike the regime model, no regime segments are injected —
        each island gets the full dataset.
        """
        island_cfg = self._build_island_config(ic)

        # Write temp config
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False,
            prefix=f'generic_island_{ic.name}_',
        ) as tmp:
            yaml.dump(island_cfg, tmp)
            tmp_path = tmp.name

        try:
            ga = GeneticAlgorithm(
                config_path=tmp_path,
                visualize=False,
                interactive=False,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Share the hall of fame
        ga.hall_of_fame = self.hall_of_fame

        return ga

    # ------------------------------------------------------------------
    # Main evolution interface
    # ------------------------------------------------------------------

    def evolve(self) -> Dict[str, List[Individual]]:
        """
        Run the full generic island model evolution.

        Returns:
            Dict mapping island name -> list of top individuals.
        """
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _shutdown(signum, frame):
            if self._shutdown_requested:
                signal.signal(signal.SIGINT, original_sigint)
                raise KeyboardInterrupt
            self._shutdown_requested = True
            self.logger.warning("[SHUTDOWN] Graceful shutdown requested")

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        try:
            return self._evolve_inner()
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _evolve_inner(self) -> Dict[str, List[Individual]]:
        start_time = time.time()

        # Terminal monitor
        from genetic_algorithm.monitor import create_monitor
        monitor_cfg = copy.deepcopy(self.config)
        monitor_cfg.setdefault('terminal_monitor', {})['enabled'] = (
            self.config.get('terminal_monitor', {}).get('enabled', True)
        )
        self.monitor = create_monitor(monitor_cfg)
        self.monitor.start(monitor_cfg)

        self.logger.info("=" * 70)
        self.logger.info("GENERIC ISLAND MODEL EVOLUTION STARTING")
        self.logger.info("=" * 70)
        self.logger.info("  Islands: %d", len(self.island_configs))
        self.logger.info("  Pop/island: %d", self.population_per_island)
        self.logger.info("  Generations: %d", self.generations)
        self.logger.info("  Migration: topology=%s interval=%d count=%d",
                         self.migration.topology, self.migration.interval,
                         self.migration.count)
        if self.migration.merge_rounds:
            self.logger.info("  Merge rounds: every %d gens", self.migration.merge_interval)
        self.logger.info("  Parallel: %s", self.parallel_islands)
        self.logger.info("=" * 70)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: CREATE ISLANDS
        # ═══════════════════════════════════════════════════════════════
        self._phase1_create_islands()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: EVOLUTION
        # ═══════════════════════════════════════════════════════════════
        results = self._phase2_evolve()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: REPORTING
        # ═══════════════════════════════════════════════════════════════
        total_elapsed = time.time() - start_time
        self._phase3_report(results, total_elapsed)

        self.monitor.on_evolution_complete({
            'total_time': total_elapsed,
            'generations': self.generations,
            'islands': len(self.islands),
            'migrations': len(self.migration_history),
        })

        return results

    # ------------------------------------------------------------------
    # Phase 1: Create Islands
    # ------------------------------------------------------------------

    def _phase1_create_islands(self):
        """Create GA instances and initialize populations for all islands."""
        phase_start = time.time()
        self.logger.info("")
        self.logger.info("═" * 70)
        self.logger.info("  PHASE 1: CREATING %d ISLANDS", len(self.island_configs))
        self.logger.info("═" * 70)

        for ic in self.island_configs:
            ga = self._create_island_ga(ic)
            self.islands[ic.name] = ga
            self.island_stats[ic.name] = GenericIslandStats(name=ic.name)
            self.generation_stats[ic.name] = []

            pop = ga.initialize_population()
            self.island_populations[ic.name] = pop

            pool_desc = (
                f"pool={len(ic.indicator_pool)} indicators"
                if ic.indicator_pool else "all indicators"
            )
            pairs_desc = (
                f"pairs={ic.pairs}" if ic.pairs else "all pairs"
            )
            self.logger.info(
                "  Island %-25s: pop=%d, seed=%d, %s, %s",
                ic.name, len(pop.individuals), ic.seed, pool_desc, pairs_desc,
            )

        phase_elapsed = time.time() - phase_start
        self.logger.info(
            "  Phase 1 complete: %.1f seconds. %d islands created.",
            phase_elapsed, len(self.islands),
        )

    # ------------------------------------------------------------------
    # Phase 2: Evolution
    # ------------------------------------------------------------------

    def _phase2_evolve(self) -> Dict[str, List[Individual]]:
        """
        Run the generation loop with periodic migration across all islands.
        """
        phase_start = time.time()
        self.logger.info("")
        self.logger.info("═" * 70)
        self.logger.info("  PHASE 2: EVOLVING %d ISLANDS × %d GENERATIONS%s",
                         len(self.islands), self.generations,
                         " (PARALLEL)" if self.parallel_islands else "")
        self.logger.info("═" * 70)

        overall_best_individual = None

        for gen in range(self.generations):
            if self._shutdown_requested:
                self.logger.info("[SHUTDOWN] Stopping at generation %d", gen)
                break

            gen_start = time.time()
            self.logger.info("")
            self.logger.info("─" * 70)
            self.logger.info("GENERATION %d/%d", gen + 1, self.generations)
            self.logger.info("─" * 70)

            self.monitor.on_generation_start(gen, self.generations)

            # Evolve each island for one generation
            if self.parallel_islands and len(self.island_configs) > 1:
                self._evolve_all_islands_parallel(gen)
            else:
                for ic in self.island_configs:
                    self._evolve_island_one_generation(
                        self.islands[ic.name],
                        self.island_populations[ic.name],
                        ic.name,
                        gen,
                    )

            # Migration (skip generation 0 — populations not yet evaluated)
            if (
                gen > 0
                and self.migration.interval > 0
                and (gen + 1) % self.migration.interval == 0
            ):
                self._migrate(gen)

            # External migration (cross-machine strategy exchange)
            if (
                self.external_migration_enabled
                and gen > 0
                and self.external_migration_interval > 0
                and (gen + 1) % self.external_migration_interval == 0
            ):
                self._load_external_migrants(gen)
                self._export_for_external_migration(gen)

            # Merge rounds (global pool-and-redistribute)
            if (
                gen > 0
                and self.migration.merge_rounds
                and self.migration.merge_interval > 0
                and (gen + 1) % self.migration.merge_interval == 0
            ):
                self._merge_round(gen)

            # Log generation summary
            gen_elapsed = time.time() - gen_start
            self._log_generation_summary(gen, gen_elapsed)

            # Update overall best and notify monitor
            for ic in self.island_configs:
                pop = self.island_populations.get(ic.name)
                if pop:
                    best_list = pop.get_best(1)
                    if best_list:
                        cand = best_list[0]
                        if (
                            cand.raw_fitness
                            and (
                                overall_best_individual is None
                                or cand.raw_fitness > (overall_best_individual.raw_fitness or 0)
                            )
                        ):
                            overall_best_individual = cand
                            self.monitor.on_new_best(cand)

            agg_best = max(
                (ist.best_fitness for ist in self.island_stats.values()),
                default=0,
            )
            agg_avg = (
                sum(ist.avg_fitness for ist in self.island_stats.values())
                / max(len(self.island_stats), 1)
            )
            _agg_stats = _AggregateStats(
                best_fitness=agg_best,
                avg_fitness=agg_avg,
                worst_fitness=0,
                generation=gen,
            )
            self.monitor.on_generation_end(
                gen=gen,
                stats=_agg_stats,
                timing=None,
                best_individual=overall_best_individual,
                extras={
                    'island_count': len(self.islands),
                    'migrations': len(self.migration_history),
                },
            )

        # Collect results: pool top-5 from every island, deduplicate
        results = self._collect_final_results()

        phase_elapsed = time.time() - phase_start
        self.logger.info("")
        self.logger.info(
            "  Phase 2 complete: %.1f seconds (%.1f minutes). "
            "%d migrations performed.",
            phase_elapsed, phase_elapsed / 60, len(self.migration_history),
        )

        return results

    # ------------------------------------------------------------------
    # Single-island evolution (one generation)
    # ------------------------------------------------------------------

    def _evolve_island_one_generation(
        self,
        ga: GeneticAlgorithm,
        population: Population,
        island_name: str,
        generation: int,
    ):
        """
        Run one generation of evolution on a single island.

        Manually executes the core GA steps (evaluate -> select ->
        crossover/mutate -> next gen) without calling ga.evolve().
        """
        ga.current_generation = generation

        # Reset LLM generation budget if applicable
        if ga.llm_enabled and ga.strategy_designer:
            ga.strategy_designer.reset_generation_budget()

        # Step 1: Evaluate fitness
        ga.evaluate_population(population)

        # Step 2: Fitness sharing
        if ga.fitness_sharing and len(population.individuals) >= 2:
            distance_matrix = calculate_pairwise_distances(
                list(population.individuals),
            )
            apply_fitness_sharing(
                population, sigma_share=ga.sharing_radius,
                distance_matrix=distance_matrix,
            )

        # Step 3: Get stats
        stats = population.get_stats()

        # Update best
        best = population.get_best(1)
        if best:
            best_ind = best[0]
            with self._stats_lock:
                ist = self.island_stats[island_name]
                if best_ind.raw_fitness and best_ind.raw_fitness > ist.best_fitness:
                    ist.best_fitness = best_ind.raw_fitness
                    profit = best_ind.metrics.get('profit', 0)
                    ist.best_profit = profit
                    self.logger.info(
                        "  [%s] NEW BEST: fitness=%.4f profit=%.2f%%",
                        island_name, ist.best_fitness, profit,
                    )
                ist.avg_fitness = stats.avg_fitness
                ist.generations_completed = generation + 1

        # Step 4: Update hall of fame
        try:
            with self._hof_lock:
                ga.hall_of_fame.update(population, generation)
        except Exception as e:
            self.logger.warning(
                "Hall of fame update failed for %s: %s", island_name, e,
            )

        # Step 5: Log island stats
        self.logger.info(
            "  [%-25s] best=%.4f avg=%.4f diversity=%.4f",
            island_name,
            stats.best_fitness,
            stats.avg_fitness,
            stats.genetic_diversity or 0,
        )

        # Step 6: Record generation stats
        stats.generation = generation
        with self._stats_lock:
            self.generation_stats[island_name].append(stats)

        # Step 6b: Record LLM strategy performance
        if ga.llm_enabled and ga.strategy_designer and ga.strategy_designer.enabled:
            try:
                ga.strategy_designer.record_llm_performance(generation, population)
            except Exception as e:
                self.logger.warning(
                    "LLM performance recording failed for %s gen %d: %s",
                    island_name, generation, e,
                )

        # Step 7: Create next generation
        if generation < self.generations - 1:
            next_pop = ga.create_next_generation(population)
            self.island_populations[island_name] = next_pop

    def _evolve_all_islands_parallel(self, generation: int):
        """Evolve all islands in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(len(self.island_configs), os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for ic in self.island_configs:
                future = executor.submit(
                    self._evolve_island_one_generation,
                    self.islands[ic.name],
                    self.island_populations[ic.name],
                    ic.name,
                    generation,
                )
                futures[future] = ic.name

            for future in as_completed(futures):
                island_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(
                        "Island %s failed at generation %d: %s",
                        island_name, generation, e,
                    )

    # ------------------------------------------------------------------
    # Migration dispatch
    # ------------------------------------------------------------------

    def _migrate(self, generation: int):
        """Dispatch migration based on configured topology."""
        topology = self.migration.topology
        self.logger.info(
            "[MIGRATION] Gen %d — topology=%s, count=%d",
            generation + 1, topology, self.migration.count,
        )

        with self._migration_lock:
            if topology == 'ring':
                self._migrate_ring(generation)
            elif topology == 'fully_connected':
                self._migrate_fully_connected(generation)
            elif topology == 'tournament':
                self._migrate_tournament(generation)
            elif topology == 'hierarchical':
                self._migrate_hierarchical(generation)
            else:
                self.logger.warning(
                    "Unknown migration topology '%s', falling back to ring",
                    topology,
                )
                self._migrate_ring(generation)

    # ------------------------------------------------------------------
    # Migration topologies
    # ------------------------------------------------------------------

    def _migrate_ring(self, generation: int):
        """
        Ring topology: island[i] sends top-N to island[(i+1) % N].
        """
        names = [ic.name for ic in self.island_configs]
        n = len(names)

        for i in range(n):
            source = names[i]
            target = names[(i + 1) % n]

            top = self._get_top_individuals(source, self.migration.count)
            if not top:
                continue

            replaced = self._inject_migrants(target, top, generation)
            fitnesses = [
                ind.raw_fitness for ind in top if ind.raw_fitness is not None
            ]

            self.migration_history.append(GenericMigrationEvent(
                generation=generation,
                source=source,
                target=target,
                count=replaced,
                fitnesses=fitnesses,
            ))

            with self._stats_lock:
                self.island_stats[source].migrants_sent += len(top)
                self.island_stats[target].migrants_received += replaced

            self.logger.info(
                "  %s → %s: %d migrants (fitnesses: %s)",
                source, target, replaced,
                [f"{f:.4f}" for f in fitnesses],
            )

    def _migrate_fully_connected(self, generation: int):
        """
        Fully connected: every island sends top-N to every other island.
        """
        names = [ic.name for ic in self.island_configs]

        for source in names:
            top = self._get_top_individuals(source, self.migration.count)
            if not top:
                continue

            for target in names:
                if target == source:
                    continue

                replaced = self._inject_migrants(target, top, generation)
                fitnesses = [
                    ind.raw_fitness for ind in top if ind.raw_fitness is not None
                ]

                self.migration_history.append(GenericMigrationEvent(
                    generation=generation,
                    source=source,
                    target=target,
                    count=replaced,
                    fitnesses=fitnesses,
                ))

                with self._stats_lock:
                    self.island_stats[source].migrants_sent += len(top)
                    self.island_stats[target].migrants_received += replaced

            self.logger.info(
                "  %s → all: %d migrants (best=%.4f)",
                source, len(top),
                max((ind.raw_fitness for ind in top if ind.raw_fitness), default=0),
            )

    def _migrate_tournament(self, generation: int):
        """
        Tournament topology: pick random pairs, compare best individuals,
        inject winners into losers.
        """
        names = [ic.name for ic in self.island_configs]
        rng = random.Random(self.base_seed + generation)

        # Shuffle and pair up
        shuffled = list(names)
        rng.shuffle(shuffled)

        # Process pairs (drop last island if odd count)
        for i in range(0, len(shuffled) - 1, 2):
            island_a = shuffled[i]
            island_b = shuffled[i + 1]

            top_a = self._get_top_individuals(island_a, self.migration.count)
            top_b = self._get_top_individuals(island_b, self.migration.count)

            best_a = max(
                (ind.raw_fitness for ind in top_a if ind.raw_fitness),
                default=0,
            )
            best_b = max(
                (ind.raw_fitness for ind in top_b if ind.raw_fitness),
                default=0,
            )

            if best_a >= best_b and top_a:
                winner, loser = island_a, island_b
                migrants = top_a
            elif top_b:
                winner, loser = island_b, island_a
                migrants = top_b
            else:
                continue

            replaced = self._inject_migrants(loser, migrants, generation)
            fitnesses = [
                ind.raw_fitness for ind in migrants if ind.raw_fitness is not None
            ]

            self.migration_history.append(GenericMigrationEvent(
                generation=generation,
                source=winner,
                target=loser,
                count=replaced,
                fitnesses=fitnesses,
            ))

            with self._stats_lock:
                self.island_stats[winner].migrants_sent += len(migrants)
                self.island_stats[loser].migrants_received += replaced

            self.logger.info(
                "  TOURNAMENT: %s (%.4f) beats %s (%.4f) → %d migrants",
                winner, best_a if winner == island_a else best_b,
                loser, best_b if loser == island_b else best_a,
                replaced,
            )

    def _migrate_hierarchical(self, generation: int):
        """
        Hierarchical merge: pair islands, merge populations of each pair
        (top individuals from both replace worst in both).
        """
        names = [ic.name for ic in self.island_configs]
        rng = random.Random(self.base_seed + generation)

        shuffled = list(names)
        rng.shuffle(shuffled)

        for i in range(0, len(shuffled) - 1, 2):
            island_a = shuffled[i]
            island_b = shuffled[i + 1]

            top_a = self._get_top_individuals(island_a, self.migration.count)
            top_b = self._get_top_individuals(island_b, self.migration.count)

            # Bidirectional: A's best → B, B's best → A
            if top_a:
                replaced_b = self._inject_migrants(island_b, top_a, generation)
                fitnesses_a = [
                    ind.raw_fitness for ind in top_a if ind.raw_fitness is not None
                ]
                self.migration_history.append(GenericMigrationEvent(
                    generation=generation,
                    source=island_a,
                    target=island_b,
                    count=replaced_b,
                    fitnesses=fitnesses_a,
                ))
                with self._stats_lock:
                    self.island_stats[island_a].migrants_sent += len(top_a)
                    self.island_stats[island_b].migrants_received += replaced_b

            if top_b:
                replaced_a = self._inject_migrants(island_a, top_b, generation)
                fitnesses_b = [
                    ind.raw_fitness for ind in top_b if ind.raw_fitness is not None
                ]
                self.migration_history.append(GenericMigrationEvent(
                    generation=generation,
                    source=island_b,
                    target=island_a,
                    count=replaced_a,
                    fitnesses=fitnesses_b,
                ))
                with self._stats_lock:
                    self.island_stats[island_b].migrants_sent += len(top_b)
                    self.island_stats[island_a].migrants_received += replaced_a

            self.logger.info(
                "  HIERARCHICAL: %s ↔ %s (bidirectional exchange)",
                island_a, island_b,
            )

    # ------------------------------------------------------------------
    # Merge round (global pool-and-redistribute)
    # ------------------------------------------------------------------

    def _merge_round(self, generation: int):
        """
        Pool top-K from ALL islands, rank globally, redistribute
        top individuals back to all islands (replacing worst).
        """
        self.logger.info(
            "[MERGE] Global merge round at generation %d", generation + 1,
        )

        # Pool top individuals from every island
        global_pool: List[Individual] = []
        for ic in self.island_configs:
            top = self._get_top_individuals(ic.name, self.migration.count * 2)
            global_pool.extend(top)

        if not global_pool:
            self.logger.info("  MERGE: No eligible individuals to merge")
            return

        # Deduplicate by gene hash
        seen_hashes = set()
        unique_pool: List[Individual] = []
        for ind in global_pool:
            gene_hash = self._gene_hash(ind)
            if gene_hash not in seen_hashes:
                seen_hashes.add(gene_hash)
                unique_pool.append(ind)

        # Rank globally by raw fitness
        unique_pool.sort(
            key=lambda x: x.raw_fitness if x.raw_fitness is not None else -1,
            reverse=True,
        )

        # Take top-N to redistribute
        top_global = unique_pool[:self.migration.count]

        self.logger.info(
            "  MERGE: Pooled %d unique individuals, redistributing top %d",
            len(unique_pool), len(top_global),
        )

        # Inject into every island
        for ic in self.island_configs:
            replaced = self._inject_migrants(ic.name, top_global, generation)
            if replaced > 0:
                self.logger.info(
                    "    → %s: injected %d global elites", ic.name, replaced,
                )

    # ------------------------------------------------------------------
    # Migration helpers
    # ------------------------------------------------------------------

    def _get_top_individuals(
        self, island_name: str, count: int,
    ) -> List[Individual]:
        """Get top-N individuals from an island's population by raw_fitness."""
        pop = self.island_populations.get(island_name)
        if pop is None:
            return []

        ranked = sorted(
            [
                ind for ind in pop.individuals
                if ind.raw_fitness is not None and ind.raw_fitness > 0
            ],
            key=lambda x: x.raw_fitness,
            reverse=True,
        )
        return ranked[:count]

    def _inject_migrants(
        self,
        target_island: str,
        migrants: List[Individual],
        generation: int,
    ) -> int:
        """
        Inject migrant individuals into a target island's population,
        replacing the worst individuals.

        Returns the number of individuals replaced.
        """
        pop = self.island_populations.get(target_island)
        if pop is None or not migrants:
            return 0

        # Sort population worst-first
        sorted_inds = sorted(
            pop.individuals,
            key=lambda x: x.raw_fitness if x.raw_fitness is not None else -1,
        )

        replaced = 0
        for migrant in migrants:
            if replaced >= len(sorted_inds):
                break

            # Deep-copy the migrant's gene so it's independent
            gene_copy = migrant.strategy_gene.copy()
            gene_copy.generation = generation
            gene_copy.individual_id = sorted_inds[replaced].strategy_gene.individual_id

            new_ind = Individual(strategy_gene=gene_copy)
            new_ind.evaluated = False  # Force re-evaluation
            new_ind.metrics = {'origin': f'migrant_from_{target_island}'}

            # Replace worst individual in-place
            idx = pop.individuals.index(sorted_inds[replaced])
            pop.individuals[idx] = new_ind
            replaced += 1

        return replaced

    # ------------------------------------------------------------------
    # External migration (cross-machine)
    # ------------------------------------------------------------------

    def _load_external_migrants(self, generation: int) -> int:
        """
        Load strategy individuals from the incoming_migrants directory.

        Other machines (or scripts) drop JSON files here containing
        serialized individuals. Each file is loaded, injected into a
        randomly chosen island, and then deleted to prevent re-processing.

        Returns the total number of migrants injected.
        """
        if not self.external_migration_dir.exists():
            return 0

        pattern = str(self.external_migration_dir / '*.json')
        files = sorted(glob.glob(pattern))
        if not files:
            return 0

        self.logger.info(
            "[EXT-MIGRATION] Gen %d — found %d incoming migrant files",
            generation + 1, len(files),
        )

        total_injected = 0
        island_names = [ic.name for ic in self.island_configs]

        for fpath in files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)

                individuals_data = data if isinstance(data, list) else data.get('individuals', [data])

                migrants = []
                for ind_data in individuals_data:
                    try:
                        ind = Individual.from_dict(ind_data)
                        ind.evaluated = False  # Force re-evaluation
                        migrants.append(ind)
                    except Exception as e:
                        self.logger.warning(
                            "[EXT-MIGRATION] Failed to parse individual from %s: %s",
                            fpath, e,
                        )

                if migrants:
                    # Distribute migrants across random islands
                    for migrant in migrants:
                        target = random.choice(island_names)
                        replaced = self._inject_migrants(target, [migrant], generation)
                        if replaced > 0:
                            total_injected += replaced
                            self.logger.info(
                                "[EXT-MIGRATION]   → injected into %s (fitness=%.4f)",
                                target, migrant.raw_fitness or 0,
                            )

                # Remove processed file
                os.remove(fpath)
                self.logger.debug("[EXT-MIGRATION] Processed and removed %s", fpath)

            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(
                    "[EXT-MIGRATION] Failed to read %s: %s", fpath, e,
                )

        if total_injected > 0:
            self.logger.info(
                "[EXT-MIGRATION] Gen %d — injected %d external migrants total",
                generation + 1, total_injected,
            )

        return total_injected

    def _export_for_external_migration(self, generation: int) -> int:
        """
        Export top individuals to the outgoing_migrants directory.

        A separate script (distribute_migrate.sh) picks these up and
        SCPs them to other machines' incoming_migrants directories.

        Returns the number of individuals exported.
        """
        self.external_export_dir.mkdir(parents=True, exist_ok=True)

        # Collect top-N globally across all islands (deduplicated)
        seen_hashes = set()
        top_individuals = []

        for ic in self.island_configs:
            for ind in self._get_top_individuals(ic.name, self.external_migration_count):
                h = self._gene_hash(ind)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    top_individuals.append(ind)

        # Sort by raw fitness and take top-N
        top_individuals.sort(
            key=lambda x: x.raw_fitness if x.raw_fitness else 0,
            reverse=True,
        )
        top_individuals = top_individuals[:self.external_migration_count]

        if not top_individuals:
            return 0

        # Write as a single JSON file with timestamp
        export_data = {
            'source': os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown')),
            'generation': generation,
            'timestamp': time.time(),
            'individuals': [ind.to_dict() for ind in top_individuals],
        }

        filename = f"migrants_gen{generation:04d}_{int(time.time())}.json"
        export_path = self.external_export_dir / filename

        try:
            tmp_path = export_path.with_suffix('.tmp')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            os.replace(tmp_path, export_path)

            self.logger.info(
                "[EXT-MIGRATION] Gen %d — exported %d individuals to %s",
                generation + 1, len(top_individuals), export_path,
            )
        except IOError as e:
            self.logger.error("[EXT-MIGRATION] Failed to export: %s", e)
            return 0

        return len(top_individuals)

    @staticmethod
    def _gene_hash(ind: Individual) -> str:
        """Compute a hash of an individual's strategy gene for deduplication."""
        try:
            gene_dict = ind.strategy_gene.to_dict()
            # Remove volatile fields
            gene_dict.pop('generation', None)
            gene_dict.pop('individual_id', None)
            serialized = json.dumps(gene_dict, sort_keys=True, default=str)
            return hashlib.md5(serialized.encode()).hexdigest()
        except Exception:
            return str(id(ind))

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def _collect_final_results(self) -> Dict[str, List[Individual]]:
        """
        Pool top-5 from every island, deduplicate by gene hash,
        rank by raw fitness, return per-island results plus a
        '__global__' key with the top-N overall.
        """
        results: Dict[str, List[Individual]] = {}

        # Per-island top-5
        global_pool: List[Individual] = []
        for ic in self.island_configs:
            pop = self.island_populations.get(ic.name)
            if pop:
                top5 = sorted(
                    [ind for ind in pop.individuals if ind.raw_fitness is not None],
                    key=lambda x: x.raw_fitness,
                    reverse=True,
                )[:5]
                results[ic.name] = top5
                global_pool.extend(top5)

        # Global deduplication and ranking
        seen_hashes = set()
        unique_global: List[Individual] = []
        for ind in sorted(
            global_pool,
            key=lambda x: x.raw_fitness if x.raw_fitness is not None else -1,
            reverse=True,
        ):
            gene_hash = self._gene_hash(ind)
            if gene_hash not in seen_hashes:
                seen_hashes.add(gene_hash)
                unique_global.append(ind)

        results['__global__'] = unique_global[:20]  # Top-20 globally

        self.logger.info(
            "Final results: %d unique strategies across %d islands (top-20 global)",
            len(unique_global), len(self.island_configs),
        )

        return results

    # ------------------------------------------------------------------
    # Phase 3: Reporting
    # ------------------------------------------------------------------

    def _phase3_report(
        self,
        results: Dict[str, List[Individual]],
        total_elapsed: float,
    ):
        """Generate final summary report."""
        self.logger.info("")
        self.logger.info("═" * 70)
        self.logger.info("  RESULTS SUMMARY")
        self.logger.info("═" * 70)
        self.logger.info("  Total time: %.1f seconds (%.1f minutes)",
                         total_elapsed, total_elapsed / 60)
        self.logger.info("  Islands: %d", len(self.islands))
        self.logger.info("  Generations: %d", self.generations)
        self.logger.info("  Migrations: %d events", len(self.migration_history))
        self.logger.info("")

        # Per-island summary
        for ic in self.island_configs:
            ist = self.island_stats[ic.name]
            self.logger.info(
                "── Island: %s ──", ic.name,
            )
            self.logger.info("  Best fitness:  %.4f", ist.best_fitness)
            self.logger.info("  Best profit:   %.2f%%", ist.best_profit)
            self.logger.info("  Avg fitness:   %.4f", ist.avg_fitness)
            self.logger.info("  Migrants sent: %d  received: %d",
                             ist.migrants_sent, ist.migrants_received)

            top5 = results.get(ic.name, [])
            for rank, ind in enumerate(top5[:3], 1):
                profit = ind.metrics.get('profit', 0)
                sharpe = ind.metrics.get('sharpe_ratio', 0)
                trades = ind.metrics.get('num_trades', 0)
                self.logger.info(
                    "    #%d: fitness=%.4f profit=%.2f%% sharpe=%.2f trades=%d",
                    rank, ind.raw_fitness or 0, profit, sharpe, trades,
                )
            self.logger.info("")

        # Global top-10
        global_top = results.get('__global__', [])
        if global_top:
            self.logger.info("── Global Top-10 (deduplicated) ──")
            for rank, ind in enumerate(global_top[:10], 1):
                profit = ind.metrics.get('profit', 0)
                sharpe = ind.metrics.get('sharpe_ratio', 0)
                trades = ind.metrics.get('num_trades', 0)
                self.logger.info(
                    "  #%d: fitness=%.4f profit=%.2f%% sharpe=%.2f trades=%d",
                    rank, ind.raw_fitness or 0, profit, sharpe, trades,
                )

        # Hall of fame
        if self.hall_of_fame.entries:
            self.logger.info("")
            self.logger.info("── Hall of Fame: %d entries ──",
                             len(self.hall_of_fame.entries))
            for i, entry in enumerate(self.hall_of_fame.entries[:5]):
                self.logger.info(
                    "  #%d: fitness=%.4f (gen %d)",
                    i + 1, entry.fitness, entry.generation_found,
                )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_generation_summary(self, gen: int, elapsed: float):
        """Log a compact summary of all islands for this generation."""
        parts = []
        for ic in self.island_configs:
            ist = self.island_stats[ic.name]
            parts.append(f"{ic.name}={ist.best_fitness:.4f}")

        # Truncate if too many islands
        if len(parts) > 8:
            summary = ', '.join(parts[:4]) + f' ... ({len(parts)} islands)'
        else:
            summary = ', '.join(parts)

        self.logger.info(
            "[SUMMARY] Gen %d/%d (%.1fs): %s | migrations=%d",
            gen + 1, self.generations, elapsed,
            summary, len(self.migration_history),
        )
