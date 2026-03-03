"""
Main Evolution Engine

Coordinates the genetic algorithm evolution process.
Supports both single-objective and multi-objective (NSGA-II) optimization.
"""

import os
import random
import yaml
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from genetic_algorithm.core.population import (
    Population, PopulationStats, apply_fitness_sharing, calculate_pairwise_distances
)
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.selection import select_parents
from genetic_algorithm.core.crossover import crossover, _enforce_min_entry_conditions
from genetic_algorithm.core.mutation import mutate
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.evaluation.regime_aware import RegimeAwareEvaluator, create_regime_aware_evaluator
from genetic_algorithm.core.nsga2 import (
    fast_non_dominated_sort,
    crowding_distance_assignment,
    extract_objectives_from_metrics,
    get_pareto_front,
    nsga2_crowded_comparison_sort,
    DEFAULT_OBJECTIVES
)
from genetic_algorithm.evaluation.parallel import ParallelEvaluator, is_parallel_available
from genetic_algorithm.core.feature_importance import FeatureImportanceTracker
from genetic_algorithm.core.hall_of_fame import HallOfFame
from genetic_algorithm.utils.run_diagnostics import RunDiagnostics
from genetic_algorithm.llm.designer import StrategyDesigner
from genetic_algorithm.monitor import create_monitor


class GeneticAlgorithm:
    """
    Main genetic algorithm engine for evolving trading strategies.
    
    Coordinates the entire evolution process:
    1. Initialize population
    2. Evaluate fitness
    3. Select parents
    4. Apply crossover and mutation
    5. Create next generation
    6. Repeat
    """
    
    def __init__(self, config_path: str = "genetic_algorithm/config/ga_config.yaml", 
                 visualize: bool = False, interactive: bool = True):
        """
        Initialize the genetic algorithm.
        
        Args:
            config_path: Path to configuration file
            visualize: Whether to enable live visualization
            interactive: Whether to use interactive plotting (only applies if visualize=True)
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Extract configuration
        ga_config = self.config['genetic_algorithm']
        
        # Set random seed for reproducibility if specified
        self.random_seed = ga_config.get('random_seed')
        if self.random_seed is not None:
            random.seed(self.random_seed)
            try:
                import numpy as np
                np.random.seed(self.random_seed)
            except ImportError:
                pass  # NumPy not available, skip
            self.logger.info(f"Random seed set to {self.random_seed} for reproducibility")
        
        self.population_size = ga_config['population_size']
        self.generations = ga_config['generations']
        self.mutation_rate = ga_config['mutation_rate']
        self.crossover_rate = ga_config['crossover_rate']
        self.elite_size = ga_config['elite_size']
        self.tournament_size = ga_config.get('tournament_size', 3)
        self.selection_method = ga_config.get('selection_method', 'tournament')
        self.crossover_method = ga_config.get('crossover_method', 'single_point')
        self.convergence_patience = ga_config.get('convergence_patience', 10)
        
        # NSGA-II multi-objective settings
        self.mode = ga_config.get('mode', 'single_objective')  # 'single_objective' or 'nsga2'
        self.nsga2_config = self.config.get('nsga2', {})
        self.objectives_config = self.nsga2_config.get('objectives', DEFAULT_OBJECTIVES)
        self.pareto_front_size = self.nsga2_config.get('pareto_front_size', 20)
        
        # Diversity preservation settings
        self.fitness_sharing = ga_config.get('fitness_sharing', True)
        self.sharing_radius = ga_config.get('sharing_radius', 0.3)
        self.diversity_threshold = ga_config.get('diversity_threshold', 0.15)
        self.allow_self_crossover = ga_config.get('allow_self_crossover', True)
        self.random_immigrants = ga_config.get('random_immigrants', 3)
        
        # Initialize components
        self.strategy_generator = StrategyGenerator(self.config)
        
        # Initialize fitness evaluator - use regime-aware if enabled
        regime_config = self.config.get('regime_aware') or {}
        self.regime_aware_enabled = regime_config.get('enabled', False)
        
        if self.regime_aware_enabled:
            self.fitness_evaluator = create_regime_aware_evaluator(
                self.config,
                auto_detect=True
            )
            self.logger.info("=" * 70)
            self.logger.info("REGIME-AWARE EVALUATION ENABLED")
            self.logger.info(f"  Detection method: {regime_config.get('method', 'sma_adx')}")
            self.logger.info(f"  Aggregation: {regime_config.get('aggregation', 'harmonic_mean')}")
            self.logger.info(f"  Holdout ratio: {regime_config.get('holdout_ratio', 0.20):.0%}")
            self.logger.info("  Strategies evaluated across multiple market regimes")
            self.logger.info("=" * 70)
        else:
            self.fitness_evaluator = FitnessEvaluator(self.config)
        
        # Log walk-forward status
        wf_config = self.config.get('walk_forward', {})
        if wf_config.get('enabled', False):
            self.logger.info("="*80)
            self.logger.info("WALK-FORWARD OPTIMIZATION ENABLED")
            self.logger.info(f"  Train days: {wf_config.get('train_days')}")
            self.logger.info(f"  Validation days: {wf_config.get('validation_days')}")
            self.logger.info(f"  Step days: {wf_config.get('step_days')}")
            self.logger.info(f"  Mode: {wf_config.get('mode', 'rolling')}")
            self.logger.info(f"  Aggregation: {wf_config.get('aggregation', 'mean')}")
            self.logger.info("  Fitness = aggregated validation score (NOT training score)")
            self.logger.info("="*80)
        else:
            self.logger.info("Using standard single-period backtesting (walk-forward disabled)")
        
        # Log NSGA-II mode status
        if self.mode == 'nsga2':
            self.logger.info("=" * 70)
            self.logger.info("NSGA-II MULTI-OBJECTIVE OPTIMIZATION ENABLED")
            self.logger.info(f"  Objectives: {[obj['name'] for obj in self.objectives_config]}")
            self.logger.info(f"  Pareto front size: {self.pareto_front_size}")
            self.logger.info("  Selection method will be overridden to 'nsga2'")
            self.logger.info("=" * 70)
            # Override selection method for NSGA-II
            self.selection_method = 'nsga2'
        
        # Initialize parallel evaluator if enabled
        parallel_config = self.config.get('parallel_evaluation', {})
        self.parallel_enabled = parallel_config.get('enabled', False)
        self.parallel_evaluator = None
        
        if self.parallel_enabled:
            if is_parallel_available():
                self.parallel_evaluator = ParallelEvaluator(
                    self.config,
                    num_workers=parallel_config.get('num_workers')
                )
                self.logger.info("=" * 70)
                self.logger.info("PARALLEL EVALUATION ENABLED")
                self.logger.info(f"  Workers: {self.parallel_evaluator.num_workers}")
                self.logger.info("  Expect 3-6x speedup on multi-core systems")
                self.logger.info("=" * 70)
            else:
                self.logger.warning("Parallel evaluation requested but multiprocessing not available")
                self.parallel_enabled = False
        
        # Initialize visualizer (only if enabled and matplotlib is available)
        self.visualizer = None
        if visualize:
            try:
                from genetic_algorithm.visualization import GAVisualizer
                self.visualizer = GAVisualizer(
                    enabled=True,
                    interactive=interactive,
                    save_plots=True
                )
            except ImportError as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Visualization disabled: {e}. Install matplotlib to enable visualization.")
        
        # Initialize trade visualizer (for trade charts)
        self.trade_visualizer = None
        trade_vis_config = self.config.get('trade_visualization', {})
        if trade_vis_config.get('enabled', False):
            try:
                from genetic_algorithm.visualization.trade_visualizer import TradeVisualizer
                # TradeVisualizer now accepts either a config dict or Path
                self.trade_visualizer = TradeVisualizer(
                    self.config,  # Pass full config, it extracts trade_visualization settings
                    enabled=True
                )
                self.trade_vis_mode = trade_vis_config.get('mode', 'final')
                self.trade_vis_top_n = trade_vis_config.get('top_n_strategies', 3)
                self.logger.info(f"TradeVisualizer initialized (mode={self.trade_vis_mode}, top_n={self.trade_vis_top_n})")
            except ImportError as e:
                self.logger.warning(f"Trade visualization disabled: {e}")
                self.trade_visualizer = None
        
        # Track evolution
        self.current_generation = 0
        self.best_individual: Optional[Individual] = None
        self.generation_stats: List[PopulationStats] = []
        self.no_improvement_count = 0
        self.best_fitness_ever = 0.0
        self._new_best_this_gen = False

        # External control hooks (set by web RunManager when run via dashboard)
        self._web_stop_event = None    # threading/mp.Event — checked each generation
        self._web_pause_event = None   # threading/mp.Event — blocks when set
        self._web_injection_queue = None  # queue.Queue — drained each generation
        self._web_run_id: Optional[str] = None
        self._web_monitor = None       # WebSocketMonitor reference
        
        # Feature importance tracking
        self.feature_tracker = FeatureImportanceTracker()
        self.logger.info("Feature importance tracking enabled")
        
        # Hall of Fame
        hof_config = self.config.get('hall_of_fame', {})
        hof_dir = hof_config.get('directory', 'genetic_algorithm/data/hall_of_fame')
        hof_max = hof_config.get('max_size', 50)
        hof_min_fitness = hof_config.get('min_fitness', 0.0)
        self.hall_of_fame = HallOfFame(
            directory=hof_dir,
            max_size=hof_max,
            min_fitness=hof_min_fitness,
        )
        self.hof_inject_count = hof_config.get('inject_count', 3)
        if self.hall_of_fame.entries:
            self.logger.info(f"Hall of Fame loaded: {len(self.hall_of_fame.entries)} entries (best fitness: {self.hall_of_fame.entries[0].fitness:.4f})")
        
        # Adaptive parameters
        self.base_mutation_rate = self.mutation_rate
        self.adaptive_mutation = ga_config.get('adaptive_mutation', True)
        self.max_adaptation_factor = ga_config.get('max_adaptation_factor', 2.0)
        self.adaptation_step = ga_config.get('adaptation_step', 0.1)
        self.max_mutation_rate = ga_config.get('max_mutation_rate', 0.5)  # Configurable hard cap
        
        # Holdout monitoring (optionally penalizes overfit elites) 
        holdout_mon_config = self.config.get('holdout_monitoring', {})
        self.holdout_monitoring_enabled = holdout_mon_config.get('enabled', False)
        self.holdout_monitoring_interval = holdout_mon_config.get('interval', 5)
        self.holdout_monitoring_top_n = holdout_mon_config.get('top_n', 3)
        
        # Holdout fitness penalty: when enabled, applies a soft multiplicative
        # penalty to overfit elites so they're less likely to survive selection.
        # penalty_factor controls severity: adjusted = fitness * max(0.5, 1.0 - degradation * penalty_factor)
        self.holdout_fitness_penalty = holdout_mon_config.get('fitness_penalty', False)
        self.holdout_penalty_factor = holdout_mon_config.get('penalty_factor', 0.5)
        
        # Holdout-aware early stopping
        self.holdout_early_stop = holdout_mon_config.get('early_stop', False)
        self.holdout_early_stop_threshold = holdout_mon_config.get('early_stop_threshold', 0.60)
        self.holdout_early_stop_checks = holdout_mon_config.get('early_stop_checks', 2)

        # Cached holdout evaluator (created lazily in _run_holdout_monitoring)
        self._holdout_evaluator = None
        self._holdout_range = None
        self._holdout_consecutive_bad = 0  # Counter for consecutive bad checks
        self._holdout_degradation_history: list = []  # Track avg_degrad per check for trend detection
        self.holdout_trend_early_stop = holdout_mon_config.get('trend_early_stop', True)
        self.holdout_trend_checks = holdout_mon_config.get('trend_checks', 3)  # Consecutive worsening checks
        
        # Generation-level holdout history for reporting
        self.generation_holdout_history: list = []
        
        # LLM-based strategy designer
        llm_config = self.config.get('advanced', {}).get('llm', {})
        self.llm_enabled = llm_config.get('enabled', False)
        self.strategy_designer = StrategyDesigner(self.config)
        if self.llm_enabled and self.strategy_designer.enabled:
            self.logger.info("=" * 70)
            self.logger.info("LLM STRATEGY DESIGNER ENABLED")
            self.logger.info(f"  Provider: {llm_config.get('provider', 'N/A')}")
            self.logger.info(f"  Model: {llm_config.get('model', 'default')}")
            self.logger.info(f"  Seed ratio: {self.strategy_designer.seed_ratio:.0%}")
            self.logger.info(f"  Immigrant ratio: {self.strategy_designer.immigrant_ratio:.0%}")
            self.logger.info("=" * 70)
        
        # Progress bar settings
        progress_config = self.config.get('progress', {})
        self.progress_enabled = progress_config.get('enabled', False) and TQDM_AVAILABLE
        self.progress_show_fitness = progress_config.get('show_fitness', True)
        self.progress_show_profit = progress_config.get('show_profit', True)
        self.progress_update_every = progress_config.get('update_every', 1)
        
        if self.progress_enabled:
            self.logger.info("Progress bar enabled")
        elif progress_config.get('enabled', False) and not TQDM_AVAILABLE:
            self.logger.warning("Progress bar requested but tqdm not installed. Run: pip install tqdm")
        
        # Checkpoint settings
        storage_config = self.config.get('storage', {})
        self.checkpoint_dir = Path(storage_config.get('checkpoint_dir', 'genetic_algorithm/data/checkpoints'))
        self.checkpoint_interval = storage_config.get('checkpoint_interval', 5)
        
        # Run diagnostics (CSV, timing, metadata)
        output_config = self.config.get('output', {})
        output_dir = Path(output_config.get('directory', 'genetic_algorithm/output'))
        self.diagnostics = RunDiagnostics(output_dir)

        # Terminal monitor (live dashboard)
        self.monitor = create_monitor(self.config)
    
    def save_checkpoint(self, population: Population, generation: int):
        """
        Save a checkpoint of the current evolution state.
        
        Serializes the population, generation stats, best individual, and 
        adaptive parameters so evolution can be resumed after interruption.
        
        Args:
            population: Current population
            generation: Current generation number
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'generation': generation,
            'population': [ind.to_dict() for ind in population.individuals],
            'population_size': self.population_size,
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'best_fitness_ever': self.best_fitness_ever,
            'no_improvement_count': self.no_improvement_count,
            'mutation_rate': self.mutation_rate,
            'generation_stats': [
                {
                    'generation': s.generation,
                    'size': s.size,
                    'best_fitness': s.best_fitness,
                    'avg_fitness': s.avg_fitness,
                    'worst_fitness': s.worst_fitness,
                    'best_raw_fitness': s.best_raw_fitness,
                    'avg_raw_fitness': s.avg_raw_fitness,
                    'genetic_diversity': s.genetic_diversity,
                    'holdout_avg_degradation': s.holdout_avg_degradation,
                    'holdout_best_degradation': s.holdout_best_degradation,
                    'holdout_num_evaluated': s.holdout_num_evaluated,
                    'holdout_num_profitable': s.holdout_num_profitable,
                }
                for s in self.generation_stats
            ],
            'random_seed': self.random_seed,
            'timestamp': time.time(),
        }
        
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.json'
        # Write to temp file first, then rename for atomicity
        temp_path = self.checkpoint_dir / 'latest_checkpoint.tmp'
        
        try:
            with open(temp_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            temp_path.rename(checkpoint_path)
            self.logger.info(f"[CHECKPOINT] Saved generation {generation} to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"[CHECKPOINT] Failed to save: {e}")
            if temp_path.exists():
                temp_path.unlink()
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint if available.
        
        Returns:
            Checkpoint dictionary if found, None otherwise
        """
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.json'
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            self.logger.info(f"[CHECKPOINT] Found checkpoint at generation {checkpoint['generation']} "
                           f"(saved at {checkpoint.get('timestamp', 'unknown')})")
            return checkpoint
        except Exception as e:
            self.logger.error(f"[CHECKPOINT] Failed to load: {e}")
            return None
    
    def restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> Population:
        """
        Restore evolution state from a checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary from load_checkpoint()
            
        Returns:
            Restored population
        """
        self.current_generation = checkpoint['generation']
        self.best_fitness_ever = checkpoint.get('best_fitness_ever', 0.0)
        self.no_improvement_count = checkpoint.get('no_improvement_count', 0)
        self.mutation_rate = checkpoint.get('mutation_rate', self.base_mutation_rate)
        
        # Restore best individual
        if checkpoint.get('best_individual'):
            self.best_individual = Individual.from_dict(checkpoint['best_individual'])
        
        # Restore population
        population = Population(
            size=checkpoint.get('population_size', self.population_size),
            generation=self.current_generation
        )
        for ind_dict in checkpoint['population']:
            individual = Individual.from_dict(ind_dict)
            population.add_individual(individual)
        
        # Restore generation stats (partial — only serializable fields)
        self.generation_stats = []
        for s in checkpoint.get('generation_stats', []):
            stats = PopulationStats(
                generation=s.get('generation', 0),
                size=s.get('size', self.population_size),
                best_fitness=s.get('best_fitness', 0),
                avg_fitness=s.get('avg_fitness', 0),
                worst_fitness=s.get('worst_fitness', 0),
                best_raw_fitness=s.get('best_raw_fitness'),
                avg_raw_fitness=s.get('avg_raw_fitness'),
                genetic_diversity=s.get('genetic_diversity'),
                holdout_avg_degradation=s.get('holdout_avg_degradation'),
                holdout_best_degradation=s.get('holdout_best_degradation'),
                holdout_num_evaluated=s.get('holdout_num_evaluated'),
                holdout_num_profitable=s.get('holdout_num_profitable'),
            )
            self.generation_stats.append(stats)
        
        self.logger.info(f"[CHECKPOINT] Restored: generation={self.current_generation}, "
                        f"population={len(population.individuals)}, "
                        f"best_fitness={self.best_fitness_ever:.4f}")
        
        return population
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        log_config = self.config.get('logging', {})
        logger = logging.getLogger('GeneticAlgorithm')
        logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        
        # Prevent duplicate logs by not propagating to root logger
        # The root logger is configured by run_ga.py setup_logging()
        logger.propagate = False
        
        # Determine if terminal monitor is active
        # Default to True (matching create_monitor) so log suppression
        # activates even when the terminal_monitor section is missing.
        monitor_cfg = self.config.get('terminal_monitor', {})
        monitor_active = monitor_cfg.get('enabled', True)
        try:
            import rich  # noqa: F401
        except ImportError:
            monitor_active = False
        
        # When monitor is active, strip any existing console StreamHandlers
        # (they may have been added by a previous init or library code)
        if monitor_active:
            for h in list(logger.handlers):
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                    logger.removeHandler(h)
        
        # Only add handlers if not already added (avoid duplicate handlers on re-init)
        if not logger.handlers:
            # Create formatter
            log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            formatter = logging.Formatter(log_format)
            
            # When terminal monitor is active, suppress console StreamHandler
            # (the monitor captures logs via its own handler for the Logs view).
            if not monitor_active and log_config.get('console', True):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # Add file handler if log file path is specified
            log_file = log_config.get('file')
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def initialize_population(self) -> Population:
        """
        Create initial population with a mix of seeded archetypes and random strategies.
        
        Seeded strategies (10-20% of population) provide known-good building blocks
        for crossover, accelerating convergence. The rest are random for diversity.
        
        Returns:
            Initial population
        """
        self.logger.info(f"Initializing population with {self.population_size} individuals")
        
        population = Population(size=self.population_size, generation=0)
        
        # Seed 15% of population with known-good archetype strategies
        seed_count = max(1, int(self.population_size * 0.15))
        seed_start_id = 0
        
        # Re-inject hall of fame members (up to inject_count)
        hof_injected = 0
        try:
            hof_individuals = self.hall_of_fame.get_individuals(self.hof_inject_count)
            for ind in hof_individuals:
                # Enforce min_entry_conditions on HoF strategies
                _enforce_min_entry_conditions(ind.strategy_gene, self.config)
                population.add_individual(ind)
                hof_injected += 1
            if hof_injected > 0:
                self.logger.info(f"Injected {hof_injected} hall-of-fame strategies")
        except Exception as e:
            self.logger.warning(f"Hall of fame injection failed: {e}")
        
        try:
            from genetic_algorithm.core.seed_strategies import create_seed_population
            seed_genes = create_seed_population(
                generation=0,
                count=seed_count,
                config=self.config,
                start_id=seed_start_id
            )
            for gene in seed_genes:
                individual = Individual(strategy_gene=gene)
                population.add_individual(individual)
                seed_start_id += 1
            self.logger.info(f"Seeded {len(seed_genes)} strategies from known archetypes")
        except Exception as e:
            self.logger.warning(f"Failed to seed population: {e}. Using all random strategies.")
            seed_start_id = 0
        
        # Fill remaining slots: LLM-generated + random strategies
        remaining = self.population_size - len(population)
        
        # LLM seed generation (configurable ratio of remaining slots)
        llm_count = 0
        if self.llm_enabled and self.strategy_designer.enabled and remaining > 0:
            llm_count = int(remaining * self.strategy_designer.seed_ratio)
            if llm_count > 0:
                self.logger.info(f"[LLM] Generating {llm_count} seed strategies via LLM...")
                llm_genes = self.strategy_designer.generate_seed_strategies(
                    count=llm_count,
                    generation=0,
                    start_id=seed_start_id,
                )
                for gene in llm_genes:
                    individual = Individual(strategy_gene=gene)
                    individual.metrics['origin'] = 'llm_seed'
                    population.add_individual(individual)
                llm_count = len(llm_genes)
                self.logger.info(f"[LLM] Added {llm_count} LLM-generated seeds")
        
        # Fill rest with random strategies
        random_remaining = self.population_size - len(population)
        for i in range(random_remaining):
            strategy_gene = self.strategy_generator.generate_random_strategy(
                generation=0,
                individual_id=seed_start_id + llm_count + i
            )
            individual = Individual(strategy_gene=strategy_gene)
            population.add_individual(individual)
        
        self.logger.info(f"Population initialized: {hof_injected} hall-of-fame + {seed_count} seeded + {llm_count} LLM + {random_remaining} random")
        return population
    
    def evaluate_population(self, population: Population):
        """
        Evaluate fitness for all unevaluated individuals.
        
        Uses parallel evaluation if enabled in config, otherwise
        evaluates sequentially.
        
        Args:
            population: Population to evaluate
        """
        unevaluated = [ind for ind in population if not ind.evaluated]
        
        if not unevaluated:
            self.logger.info("[EVAL] All individuals already evaluated (using cache)")
            return
        
        # Use parallel evaluation if enabled
        if self.parallel_enabled and self.parallel_evaluator:
            self._evaluate_population_parallel(unevaluated)
        else:
            self._evaluate_population_sequential(unevaluated)

    # ── External-control helpers (web dashboard) ────────────────

    def _drain_injection_queue(self, population: 'Population', gen: int):
        """
        Drain the injection queue and add any injected strategies to the population.

        Called at the start of each generation when the web injection queue is set.
        Handles both strategy gene dicts and command sentinels (e.g. checkpoint requests).
        """
        import queue as _queue_mod
        injected = 0
        while True:
            try:
                item = self._web_injection_queue.get_nowait()
            except (_queue_mod.Empty, Exception):
                break

            # Handle command sentinels
            if isinstance(item, dict) and item.get("_command") == "checkpoint":
                self.logger.info("[WEB] Checkpoint requested via injection queue")
                self.save_checkpoint(population, gen)
                continue

            # Treat as strategy gene dict
            try:
                from genetic_algorithm.core.strategy_gene import StrategyGene
                gene = StrategyGene.from_dict(item)
                gene.generation = gen
                gene.individual_id = self.population_size + injected
                individual = Individual(strategy_gene=gene)
                population.add_individual(individual)
                injected += 1
                self.logger.info(f"[WEB] Injected strategy {individual.id} into population")
            except Exception as e:
                self.logger.warning(f"[WEB] Failed to inject strategy: {e}")

        if injected:
            self.logger.info(f"[WEB] Injected {injected} strategies this generation")

    def get_state_snapshot(self) -> dict:
        """
        Return a lightweight snapshot of current evolution state.

        Used by the web dashboard for real-time status without deep-copying
        the entire population.
        """
        return {
            "current_generation": self.current_generation,
            "total_generations": self.generations,
            "best_fitness_ever": self.best_fitness_ever,
            "no_improvement_count": self.no_improvement_count,
            "mutation_rate": self.mutation_rate,
            "best_individual_id": self.best_individual.id if self.best_individual else None,
            "best_profit": self.best_individual.metrics.get("profit") if self.best_individual and self.best_individual.metrics else None,
            "generation_stats_count": len(self.generation_stats),
        }

    def _post_hoc_walk_forward_validation(self, population: 'Population'):
        """
        Run walk-forward validation on elite candidates after parallel evaluation.
        
        When parallel evaluation is enabled, walk-forward is disabled inside workers
        to avoid the N×W backtest explosion. Instead, we validate the top candidates
        here — in parallel if possible, sequential otherwise.
        
        Only runs if walk_forward.enabled=True and parallel_evaluation.enabled=True.
        Re-evaluates the top `elite_size * 2` individuals with walk-forward and
        replaces their fitness scores with the walk-forward-validated scores.
        """
        wf_config = self.config.get('walk_forward', {})
        if not wf_config.get('enabled', False):
            return
        if not self.parallel_enabled:
            return  # WF already ran inside sequential evaluation
        
        # Get top candidates to validate
        n_validate = min(self.elite_size * 2, len(population.individuals))
        candidates = population.get_best(n_validate)
        
        self.logger.info(f"[WF-POSTHOC] Validating top {len(candidates)} strategies with walk-forward...")
        
        # Use parallel WF validation if parallel evaluator is available
        if self.parallel_evaluator and len(candidates) > 1:
            from genetic_algorithm.evaluation.parallel import parallel_walk_forward_validation
            
            timeout = self.config.get('parallel_evaluation', {}).get('backtest_timeout', 120) * 4
            validated = parallel_walk_forward_validation(
                candidates=candidates,
                config=self.config,
                num_workers=self.parallel_evaluator.num_workers,
                backtest_timeout=timeout,
            )
            self.logger.info(f"[WF-POSTHOC] Parallel validation complete: {validated}/{len(candidates)}")
        else:
            # Fallback: sequential validation
            wf_evaluator = FitnessEvaluator(self.config)
            validated = 0
            for ind in candidates:
                try:
                    wf_fitness, wf_metrics = wf_evaluator.evaluate(ind.strategy_gene)
                    original_fitness = ind.fitness
                    ind.set_fitness(wf_fitness, wf_metrics)
                    validated += 1
                    self.logger.debug(
                        f"[WF-POSTHOC] {ind.id}: {original_fitness:.4f} -> {wf_fitness:.4f} "
                        f"(gap={wf_metrics.get('train_val_gap', 0):.4f})"
                    )
                except Exception as e:
                    self.logger.warning(f"[WF-POSTHOC] Failed for {ind.id}: {e}")
            self.logger.info(f"[WF-POSTHOC] Validated {validated}/{len(candidates)} strategies")
    
    def _run_holdout_test(self, population: 'Population', holdout_config: Dict):
        """
        Run holdout/out-of-sample test on top strategies after evolution completes.
        
        Tests the best strategies on a separate time period that was NOT used during
        evolution, to check for overfitting.
        
        Config:
            holdout_test:
              enabled: true
              timerange: "20250301-20250401"  # Must be outside training range
              top_n: 5  # Number of strategies to test
        """
        holdout_timerange = holdout_config.get('timerange', '')
        top_n = holdout_config.get('top_n', 5)
        
        if not holdout_timerange:
            self.logger.warning("[HOLDOUT] No holdout timerange configured, skipping")
            return
        
        candidates = population.get_best(top_n)
        if not candidates:
            return
        
        self.logger.info("")
        self.logger.info(f"{'─'*70}")
        self.logger.info(f"HOLDOUT TEST - Out-of-sample validation on {holdout_timerange}")
        self.logger.info(f"{'─'*70}")
        
        # Create a modified config with the holdout timerange
        import copy
        holdout_eval_config = copy.deepcopy(self.config)
        holdout_eval_config['backtesting']['timerange'] = holdout_timerange
        # Disable walk-forward for holdout (it's a straight backtest)
        if 'walk_forward' in holdout_eval_config:
            holdout_eval_config['walk_forward']['enabled'] = False
        
        holdout_evaluator = FitnessEvaluator(holdout_eval_config)
        
        results = []
        for ind in candidates:
            try:
                holdout_fitness, holdout_metrics = holdout_evaluator.evaluate(ind.strategy_gene)
                train_fitness = ind.fitness
                overfit_ratio = (train_fitness - holdout_fitness) / max(abs(train_fitness), 0.001)
                
                results.append({
                    'id': ind.id,
                    'train_fitness': train_fitness,
                    'holdout_fitness': holdout_fitness,
                    'overfit_ratio': overfit_ratio,
                    'holdout_profit': holdout_metrics.get('profit', 0),
                    'holdout_trades': holdout_metrics.get('total_trades', 0),
                })
                
                status = "✓" if holdout_fitness > 0 else "✗"
                self.logger.info(
                    f"  {status} {ind.id}: train={train_fitness:.4f} -> holdout={holdout_fitness:.4f} "
                    f"(overfit={overfit_ratio:+.1%}, profit={holdout_metrics.get('profit', 0):.2f}%, "
                    f"trades={holdout_metrics.get('total_trades', 0)})"
                )
            except Exception as e:
                self.logger.warning(f"  ✗ {ind.id}: holdout test failed: {e}")
        
        if results:
            avg_overfit = sum(r['overfit_ratio'] for r in results) / len(results)
            profitable = sum(1 for r in results if r['holdout_fitness'] > 0)
            self.logger.info(f"\n  Summary: {profitable}/{len(results)} profitable on holdout, "
                           f"avg overfit ratio: {avg_overfit:+.1%}")
    
    def _run_holdout_monitoring(self, population: 'Population', generation: int):
        """
        Periodic holdout check during evolution.
        
        Evaluates top-N elites on holdout data and logs the results.
        When holdout_fitness_penalty is enabled, applies a soft multiplicative
        fitness adjustment to overfit elites so they're disfavored in selection
        while keeping their genetic material alive.
        """
        if not self.holdout_monitoring_enabled:
            return
        
        # Only run at specified intervals
        if (generation + 1) % self.holdout_monitoring_interval != 0:
            return
        
        # Get holdout split from the validation config
        holdout_config = self.config.get('holdout_validation', {})
        holdout_pct = holdout_config.get('holdout_pct', 0.15)
        original_timerange = self.config.get('backtesting', {}).get('timerange', '')
        
        if not original_timerange or not holdout_config.get('enabled', False):
            return
        
        try:
            evo_range, holdout_range = FitnessEvaluator.split_timerange_for_holdout(
                original_timerange, holdout_pct
            )
        except Exception as e:
            self.logger.debug(f"[HOLDOUT-MON] Could not split timerange: {e}")
            return
        
        candidates = population.get_best(self.holdout_monitoring_top_n)
        if not candidates:
            return
        
        self.logger.info(f"[HOLDOUT-MON] Gen {generation + 1}: evaluating top-{len(candidates)} on {holdout_range}...")
        
        # Reuse cached holdout evaluator (same holdout range every call)
        if self._holdout_evaluator is None or self._holdout_range != holdout_range:
            import copy
            holdout_eval_config = copy.deepcopy(self.config)
            holdout_eval_config['backtesting']['timerange'] = holdout_range
            if 'walk_forward' in holdout_eval_config:
                holdout_eval_config['walk_forward']['enabled'] = False
            try:
                self._holdout_evaluator = FitnessEvaluator(holdout_eval_config)
                self._holdout_range = holdout_range
                self.logger.debug(f"[HOLDOUT-MON] Created & cached evaluator for {holdout_range}")
            except Exception as e:
                self.logger.warning(f"[HOLDOUT-MON] Failed to create evaluator: {e}")
                return
        holdout_evaluator = self._holdout_evaluator
        
        degradations = []
        for ind in candidates:
            try:
                holdout_fitness, holdout_metrics = holdout_evaluator.evaluate(ind.strategy_gene)
                train_fitness = ind.raw_fitness if ind.raw_fitness is not None else ind.fitness
                degradation = (train_fitness - holdout_fitness) / max(abs(train_fitness), 0.001) * 100
                degradations.append(degradation)
                
                symbol = "✓" if degradation < 30 else "⚠"
                self.logger.info(
                    f"  {symbol} {ind.id}: train={train_fitness:.4f} hold={holdout_fitness:.4f} "
                    f"(degrad={degradation:.1f}%, hold_profit={holdout_metrics.get('profit', 0):.2f}%)"
                )
                
                # Apply holdout fitness penalty when enabled
                # Penalty MUST hit raw_fitness too — elite selection sorts by
                # raw_fitness, and fitness sharing overwrites .fitness from
                # raw_fitness.  Without touching raw_fitness the penalty is a no-op.
                if self.holdout_fitness_penalty and degradation > 0:
                    degradation_frac = degradation / 100.0  # Convert back to 0-1
                    penalty_mult = max(0.3, 1.0 - degradation_frac * self.holdout_penalty_factor)
                    old_raw = ind.raw_fitness if ind.raw_fitness is not None else ind.fitness
                    old_fitness = ind.fitness
                    ind.raw_fitness = old_raw * penalty_mult
                    ind.fitness = ind.fitness * penalty_mult
                    ind.metrics['holdout_penalty'] = 1.0 - penalty_mult
                    ind.metrics['holdout_degradation_monitored'] = degradation_frac
                    self.logger.info(
                        f"    → Holdout penalty applied: raw {old_raw:.4f}->{ind.raw_fitness:.4f}, "
                        f"fit {old_fitness:.4f}->{ind.fitness:.4f} "
                        f"(x{penalty_mult:.3f}, degrad={degradation:.1f}%)"
                    )
            except Exception as e:
                self.logger.debug(f"  ✗ {ind.id}: holdout monitoring failed: {e}")
        
        if degradations:
            avg_degrad = sum(degradations) / len(degradations)
            best_degrad = min(degradations)
            worst_degrad = max(degradations)
            self.logger.info(f"  [HOLDOUT-MON] Avg degradation: {avg_degrad:.1f}%")
            
            # Store in generation stats (if current gen stats exist)
            if self.generation_stats:
                latest_stats = self.generation_stats[-1]
                latest_stats.holdout_avg_degradation = avg_degrad
                latest_stats.holdout_best_degradation = best_degrad
                latest_stats.holdout_num_evaluated = len(degradations)
                latest_stats.holdout_num_profitable = sum(1 for d in degradations if d < 30)
            
            # Append to holdout history for reporting
            from genetic_algorithm.utils.overfit_analysis import GenerationHoldoutStats
            holdout_stat = GenerationHoldoutStats(
                generation=generation,
                avg_degradation=avg_degrad,
                best_degradation=best_degrad,
                worst_degradation=worst_degrad,
                num_evaluated=len(degradations),
                num_profitable=sum(1 for d in degradations if d < 30),
            )
            self.generation_holdout_history.append(holdout_stat)
            
            # Holdout-aware early stopping check
            if self.holdout_early_stop:
                threshold_pct = self.holdout_early_stop_threshold * 100
                if avg_degrad > threshold_pct:
                    self._holdout_consecutive_bad += 1
                    self.logger.warning(
                        f"  [HOLDOUT-MON] ⚠ Degradation {avg_degrad:.1f}% > {threshold_pct:.0f}% threshold "
                        f"({self._holdout_consecutive_bad}/{self.holdout_early_stop_checks} consecutive)"
                    )
                else:
                    self._holdout_consecutive_bad = 0

            # Trend-based early stopping: detect consecutive worsening
            self._holdout_degradation_history.append(avg_degrad)
            if self.holdout_trend_early_stop and len(self._holdout_degradation_history) >= self.holdout_trend_checks + 1:
                recent = self._holdout_degradation_history[-(self.holdout_trend_checks + 1):]
                consecutive_worse = all(
                    recent[i + 1] > recent[i] for i in range(len(recent) - 1)
                )
                if consecutive_worse:
                    self.logger.warning(
                        f"  [HOLDOUT-TREND] ⚠ Degradation worsened for {self.holdout_trend_checks} "
                        f"consecutive checks: {[f'{d:.1f}%' for d in recent]}. "
                        f"Triggering early stop."
                    )
                    # Force the consecutive_bad counter high enough to trigger early stop
                    self._holdout_consecutive_bad = max(
                        self._holdout_consecutive_bad, self.holdout_early_stop_checks
                    )

            return avg_degrad
        return None

    def _evaluate_population_parallel(self, unevaluated: list):
        """
        Evaluate population using parallel workers.
        
        Args:
            unevaluated: List of unevaluated individuals
        """
        self.logger.info(f"[EVAL] Parallel evaluation of {len(unevaluated)} individuals...")
        
        # Create progress callback for tqdm if enabled
        # When terminal monitor is active, skip tqdm to avoid display conflicts
        pbar = None
        if self.progress_enabled and not self.monitor.active:
            pbar = tqdm(
                total=len(unevaluated),
                desc=f"Gen {self.current_generation + 1}",
                unit="strategy",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
        
        def progress_callback(completed, total):
            if pbar:
                pbar.n = completed
                pbar.refresh()
            self.monitor.on_eval_progress(completed, total)
        
        # Run parallel evaluation
        result = self.parallel_evaluator.evaluate_batch(
            unevaluated,
            progress_callback=progress_callback if self.progress_enabled else None
        )
        
        if pbar:
            pbar.close()
        
        # Calculate summary stats
        total_profit = sum(ind.metrics.get('profit', 0) for ind in unevaluated if ind.evaluated)
        avg_profit = total_profit / result.successful if result.successful > 0 else 0
        
        self.logger.info(
            f"[EVAL] Complete: {result.successful} succeeded, {result.failed} failed, "
            f"avg profit: {avg_profit:.2f}% ({result.total_time:.1f}s, ~{result.speedup_estimate:.1f}x speedup)"
        )
    
    def _evaluate_population_sequential(self, unevaluated: list):
        """
        Evaluate population sequentially (original implementation).
        
        Args:
            unevaluated: List of unevaluated individuals
        """
        self.logger.info(f"[EVAL] Sequential evaluation of {len(unevaluated)} individuals...")
        
        successful = 0
        failed = 0
        total_profit = 0.0
        best_fitness = 0.0
        best_profit = 0.0
        
        # Create progress bar if enabled
        # When terminal monitor is active, skip tqdm to avoid display conflicts
        if self.progress_enabled and not self.monitor.active:
            pbar = tqdm(
                enumerate(unevaluated),
                total=len(unevaluated),
                desc=f"Gen {self.current_generation + 1}",
                unit="strategy",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
            )
            iterator = pbar
        else:
            iterator = enumerate(unevaluated)
            pbar = None
        
        for i, individual in iterator:
            strategy_name = individual.id
            
            # Show progress every 5 individuals or at start/end (when no progress bar)
            if not self.progress_enabled:
                if i == 0 or (i + 1) % 5 == 0 or i == len(unevaluated) - 1:
                    self.logger.info(f"[EVAL] Progress: {i+1}/{len(unevaluated)} ({((i+1)/len(unevaluated)*100):.0f}%)")
            
            try:
                # Evaluate fitness
                fitness, metrics = self.fitness_evaluator.evaluate(
                    individual.strategy_gene,
                    strategy_name=strategy_name
                )
                individual.set_fitness(fitness, metrics)
                
                # For NSGA-II: also set objectives
                if self.mode == 'nsga2':
                    objectives = extract_objectives_from_metrics(metrics, self.objectives_config)
                    individual.set_objectives(objectives, metrics)
                
                successful += 1
                profit = metrics.get('profit', 0)
                total_profit += profit
                
                # Track best for progress bar
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_profit = profit
                
                # Update progress bar postfix
                if pbar and (i + 1) % self.progress_update_every == 0:
                    postfix = {}
                    if self.progress_show_fitness:
                        postfix['best_fit'] = f"{best_fitness:.3f}"
                    if self.progress_show_profit:
                        postfix['best_pft'] = f"{best_profit:.1f}%"
                    pbar.set_postfix(postfix)
                
                self.logger.debug(f"  {strategy_name}: fitness={fitness:.4f}, profit={metrics.get('profit', 0):.2f}%")
                
                # Update terminal monitor eval progress
                self.monitor.on_eval_progress(i + 1, len(unevaluated))
                
            except Exception as e:
                # Handle evaluation errors gracefully
                self.logger.warning(f"[EVAL] Failed {strategy_name}: {e}")
                failed += 1
                # Set zero fitness for failed evaluation
                individual.set_fitness(0.0, {
                    'profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'error': str(e)
                })
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Summary
        avg_profit = total_profit / successful if successful > 0 else 0
        self.logger.info(f"[EVAL] Complete: {successful} succeeded, {failed} failed, avg profit: {avg_profit:.2f}%")
    
    def _should_update_best_individual(self, candidate: Individual) -> bool:
        """
        Determine if candidate should replace current best individual.
        
        Uses raw_fitness (not shared_fitness) for true best strategy comparison.
        
        Handles None fitness values correctly:
        - Candidate must have valid (non-None) raw_fitness
        - Updates if no best individual exists yet
        - Updates if current best has None raw_fitness
        - Updates if candidate raw_fitness is higher than current best
        
        Args:
            candidate: Individual to consider as new best
            
        Returns:
            True if candidate should become new best individual
        """
        # Candidate must have valid raw_fitness
        if candidate.raw_fitness is None:
            return False
        
        # Update if no best exists yet
        if self.best_individual is None:
            return True
        
        # Update if current best has invalid raw_fitness
        if self.best_individual.raw_fitness is None:
            return True
        
        # Update if candidate is better (based on raw_fitness)
        return candidate.raw_fitness > self.best_individual.raw_fitness
    
    def _visualize_strategy_trades(self, individual: Individual, generation: int, individual_idx: int):
        """
        Generate trade visualization for a specific individual.
        
        Args:
            individual: Individual to visualize
            generation: Current generation number
            individual_idx: Index of individual in ranking
        """
        if not self.trade_visualizer:
            return
        
        try:
            from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
            
            # Generate strategy code
            strategy_code = self.strategy_generator.generate_strategy_code(individual.strategy_gene)
            # Use the full strategy name with GAStrategy_ prefix (same as generator.py)
            strategy_name = f"GAStrategy_Gen{individual.strategy_gene.generation}_Ind{individual.strategy_gene.individual_id}"
            
            # Run backtest with trade collection
            backtester = DirectBacktester(self.config)
            result = backtester.backtest_strategy_with_trades(strategy_code, strategy_name)
            
            if not result.success:
                self.logger.warning(f"[TRADE VIS] Backtest failed for {strategy_name}: {result.error_message}")
                return
            
            # Generate trade chart
            saved_files = self.trade_visualizer.visualize_strategy_from_backtest(
                strategy_name=strategy_name,
                backtest_result=result,
                generation=generation,
                individual_idx=individual_idx
            )
            
            if saved_files:
                self.logger.info(f"[TRADE VIS] Generated {len(saved_files)} chart(s) for {strategy_name}")
            
        except Exception as e:
            self.logger.warning(f"[TRADE VIS] Failed to visualize {individual.id}: {e}")
            import traceback
            traceback.print_exc()
    
    def create_next_generation(self, population: Population) -> Population:
        """
        Create next generation through selection, crossover, and mutation.
        
        Args:
            population: Current population
            
        Returns:
            Next generation population
        """
        self.logger.info(f"[STEP] Creating generation {self.current_generation + 1}")
        
        # Track GA operator usage
        crossover_count = 0
        mutation_count = 0
        crossover_failures = 0
        mutation_failures = 0
        
        # Sort by fitness
        population.sort_by_fitness(reverse=True)
        
        # Create next generation
        next_gen = Population(size=self.population_size, generation=self.current_generation + 1)
        
        # Step 1: Elitism - keep top performers
        # Use raw_fitness (not shared fitness) to select elites, because
        # fitness sharing can push strong strategies down artificially.
        # This ensures the truly best strategy is never lost to sharing noise.
        self.logger.debug(f"[ELITISM] Preserving top {self.elite_size} individuals (by raw fitness)")
        
        # Select elites by raw_fitness (the un-shared, un-adjusted fitness)
        ranked_by_raw = sorted(
            [ind for ind in population.individuals if ind.raw_fitness is not None],
            key=lambda x: x.raw_fitness,
            reverse=True,
        )
        elites = ranked_by_raw[:self.elite_size]
        
        for individual in elites:
            gene_copy = individual.strategy_gene.copy()
            gene_copy.generation = self.current_generation + 1
            elite_copy = Individual(strategy_gene=gene_copy)
            # Carry over fitness and metrics to avoid re-evaluation
            elite_copy.raw_fitness = individual.raw_fitness
            elite_copy.fitness = individual.fitness
            elite_copy.metrics = individual.metrics.copy() if individual.metrics else {}
            elite_copy.evaluated = True
            # Enforce min_entry_conditions on elite copies
            _enforce_min_entry_conditions(elite_copy.strategy_gene, self.config)
            next_gen.add_individual(elite_copy)
        self.logger.info(f"[ELITISM] Preserved {self.elite_size} elite individuals")
        
        # Step 1b: Parsimony pressure — try to simplify elites
        parsimony_config = self.config.get('parsimony', {})
        # Pass min_entry_conditions so parsimony respects the configured floor
        indicator_config = self.config.get('indicators', {})
        parsimony_config['min_entry_conditions'] = indicator_config.get('min_entry_conditions', 2)
        if parsimony_config.get('enabled', False):
            elite_list = list(next_gen.individuals)
            
            if self.parallel_enabled:
                # Use parallel parsimony: evaluate all removal candidates
                # concurrently across all elites using ProcessPoolExecutor.
                from genetic_algorithm.evaluation.parallel import parallel_parsimony
                
                parallel_cfg = self.config.get('parallel_evaluation', {})
                num_workers = parallel_cfg.get('num_workers') or (os.cpu_count() - 1)
                bt_timeout = parallel_cfg.get('backtest_timeout', 120)
                
                removed = parallel_parsimony(
                    elite_list, parsimony_config, self.config,
                    num_workers=num_workers,
                    backtest_timeout=bt_timeout,
                )
            else:
                # Sequential fallback
                from genetic_algorithm.core.parsimony import apply_parsimony_to_elites
                
                def _eval_fn(gene):
                    return self.fitness_evaluator.evaluate(gene)
                
                removed = apply_parsimony_to_elites(elite_list, _eval_fn, parsimony_config)
            
            if removed > 0:
                self.logger.info(f"[PARSIMONY] Removed {removed} component(s) from elites")
        
        # Helper to calculate next available individual ID
        def calculate_next_id():
            return len(next_gen)
        
        # Step 2: Inject random immigrants to maintain diversity
        # Get current generation stats to check diversity
        stats = population.get_stats()
        immigrant_count = self.random_immigrants
        
        # Double immigrant count if diversity is low
        if stats.genetic_diversity is not None and stats.genetic_diversity < self.diversity_threshold:
            immigrant_count = self.random_immigrants * 2
            self.logger.warning(f"[DIVERSITY] Low diversity ({stats.genetic_diversity:.4f}), doubling immigrants to {immigrant_count}")
        
        # Inject immigrants: LLM-generated + random
        immigrants_before = len(next_gen)
        
        # LLM immigrants (configurable ratio of immigrant slots)
        llm_immigrant_count = 0
        if self.llm_enabled and self.strategy_designer.enabled:
            llm_immigrant_count = max(1, int(immigrant_count * self.strategy_designer.immigrant_ratio))
            
            # Gather context for guided generation
            top_inds = sorted(population.individuals, 
                            key=lambda x: x.fitness, reverse=True)[:5]
            top_summaries = self.strategy_designer.get_top_performer_summaries(top_inds)
            weaknesses = self.strategy_designer.get_population_weaknesses(top_inds)
            
            llm_genes = self.strategy_designer.generate_immigrants(
                count=llm_immigrant_count,
                generation=self.current_generation + 1,
                start_id=calculate_next_id(),
                top_performers=top_summaries,
                weaknesses=weaknesses,
            )
            for gene in llm_genes:
                if len(next_gen) >= self.population_size:
                    break
                ind = Individual(strategy_gene=gene)
                ind.metrics['origin'] = 'llm_immigrant'
                next_gen.add_individual(ind)
            llm_immigrant_count = len(llm_genes)
        
        # Fill remaining immigrant slots with random strategies
        random_immigrant_target = immigrant_count - llm_immigrant_count
        for _ in range(random_immigrant_target):
            if len(next_gen) >= self.population_size:
                break
            immigrant_gene = self.strategy_generator.generate_random_strategy(
                generation=self.current_generation + 1,
                individual_id=calculate_next_id()
            )
            next_gen.add_individual(Individual(strategy_gene=immigrant_gene))
        
        actual_immigrants_added = len(next_gen) - immigrants_before
        if llm_immigrant_count > 0:
            self.logger.info(f"[IMMIGRANTS] Added {actual_immigrants_added} immigrants "
                           f"({llm_immigrant_count} LLM + {actual_immigrants_added - llm_immigrant_count} random)")
        else:
            self.logger.info(f"[IMMIGRANTS] Added {actual_immigrants_added} random immigrants")
        
        # Helper to create child from parent gene
        def create_child(parent_gene, ind_id):
            gene = parent_gene.copy()
            gene.generation = self.current_generation + 1
            gene.individual_id = ind_id
            return Individual(strategy_gene=gene)
        
        # Step 3: Create offspring through selection, crossover, and mutation
        self.logger.debug(f"[OFFSPRING] Creating offspring to fill remaining {self.population_size - len(next_gen)} slots")
        offspring_count = 0
        offspring_added = 0
        
        while len(next_gen) < self.population_size:
            # Select parents using configured method
            parent1, parent2 = select_parents(
                population, num_parents=2,
                method=self.selection_method,
                tournament_size=self.tournament_size,
                allow_duplicates=self.allow_self_crossover
            )
            
            # Pre-calculate IDs for both children before adding them
            child1_id = len(next_gen)
            child2_id = len(next_gen) + 1
            
            # Crossover or copy
            try:
                if random.random() < self.crossover_rate:
                    child1, child2 = crossover(
                        parent1, parent2,
                        generation=self.current_generation + 1,
                        ind_id=child1_id,
                        config=self.config,
                        method=self.crossover_method
                    )
                    crossover_count += 1
                else:
                    child1 = create_child(parent1.strategy_gene, child1_id)
                    child2 = create_child(parent2.strategy_gene, child2_id)
            except (ValueError, KeyError, AttributeError, TypeError) as e:
                # If crossover fails, use clones of parents instead
                self.logger.debug(f"[CROSSOVER] Failed: {e}")
                crossover_failures += 1
                child1 = create_child(parent1.strategy_gene, child1_id)
                child2 = create_child(parent2.strategy_gene, child2_id)
            
            # Mutation - call unconditionally, mutate() handles internal probability checks
            for child in [child1, child2]:
                if len(next_gen) >= self.population_size:
                    break
                try:
                    child = mutate(child, self.mutation_rate, self.config)
                    mutation_count += 1
                    # Enforce min_entry_conditions after mutation
                    _enforce_min_entry_conditions(child.strategy_gene, self.config)
                    next_gen.add_individual(child)
                    offspring_added += 1
                except (ValueError, KeyError, AttributeError, TypeError) as e:
                    self.logger.debug(f"[MUTATION] Failed: {e}")
                    mutation_failures += 1
                    continue
            
            offspring_count += 2
        
        # Log generation summary
        self.logger.info(f"[OFFSPRING] Added {offspring_added} offspring (crossovers: {crossover_count}, mutations: {mutation_count})")
        if crossover_failures > 0 or mutation_failures > 0:
            self.logger.warning(f"[FAILURES] Crossover: {crossover_failures}, Mutation: {mutation_failures}")
        
        return next_gen
    
    def check_convergence(self, stats: PopulationStats) -> bool:
        """
        Check if evolution has converged.
        
        best_fitness_ever is maintained at the [NEW BEST] detection point
        (before holdout monitoring can corrupt raw_fitness in-place).
        This method only increments no_improvement_count when no new best
        was recorded this generation, and handles adaptive mutation.
        
        Args:
            stats: Current generation statistics
            
        Returns:
            True if converged, False otherwise
        """
        if self.best_individual is None:
            return False
        
        # best_fitness_ever and no_improvement_count are already updated
        # at the [NEW BEST] detection point (before holdout penalty).
        # Here we only need to increment no_improvement_count when there
        # was NO new best this generation.
        if not getattr(self, '_new_best_this_gen', False):
            self.no_improvement_count += 1
        # Reset the flag for next generation
        self._new_best_this_gen = False
        
        # Adaptive mutation: increase mutation rate if stuck
        if self.adaptive_mutation and self.no_improvement_count > 0:
            # Gradually increase mutation rate when stuck
            # adaptation_factor = 1.0 + (generations_stuck * adaptation_step)
            # Capped at max_adaptation_factor (default 2.0 = double the rate)
            adaptation_factor = min(
                self.max_adaptation_factor, 
                1.0 + (self.no_improvement_count * self.adaptation_step)
            )
            self.mutation_rate = min(self.max_mutation_rate, self.base_mutation_rate * adaptation_factor)
            self.logger.info(
                f"Adaptive mutation: rate increased to {self.mutation_rate:.3f} "
                f"(factor={adaptation_factor:.2f}, no improvement for {self.no_improvement_count} gens)"
            )
        else:
            # Reset to base rate when improving
            self.mutation_rate = self.base_mutation_rate
        
        if self.no_improvement_count >= self.convergence_patience:
            self.logger.info(f"Converged: No improvement for {self.convergence_patience} generations")
            return True
        
        # Stagnation warning at half patience
        half_patience = self.convergence_patience // 2
        if self.no_improvement_count == half_patience and half_patience > 0:
            self.logger.warning(
                f"⚠ STAGNATION WARNING: No improvement for {half_patience} generations "
                f"(patience: {self.convergence_patience}). Best ever: {self.best_fitness_ever:.4f}"
            )
        
        return False
    
    def evolve(self, resume: bool = False) -> List[Individual]:
        """
        Run the complete evolution process.
        
        Args:
            resume: If True, attempt to resume from a checkpoint
        
        Returns:
            List of best individuals
        """
        self.logger.info("=" * 70)
        self.logger.info("GENETIC ALGORITHM STARTING")
        self.logger.info("=" * 70)
        self.logger.info(f"  Population: {self.population_size} | Generations: {self.generations}")
        self.logger.info(f"  Mutation: {self.mutation_rate:.2%} | Crossover: {self.crossover_rate:.2%} ({self.crossover_method})")
        self.logger.info(f"  Selection: {self.selection_method} | Elite size: {self.elite_size}")
        self.logger.info("=" * 70)
        
        # Start run diagnostics (CSV, timing, metadata)
        self.diagnostics.start_run(self.config)
        
        # Start terminal monitor (live dashboard)
        self.monitor.start(self.config)
        
        # Try to resume from checkpoint
        start_generation = 0
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                population = self.restore_from_checkpoint(checkpoint)
                start_generation = self.current_generation + 1
                self.logger.info(f"Resuming evolution from generation {start_generation}")
            else:
                self.logger.info("No checkpoint found, starting fresh")
                population = self.initialize_population()
        else:
            # Initialize population
            population = self.initialize_population()
        
        # Evolution loop
        # Initialise Pareto archive (if NSGA-II + archive enabled)
        archive_config = self.config.get('pareto_archive', {})
        pareto_archive = None
        if self.mode == 'nsga2' and archive_config.get('enabled', False):
            from genetic_algorithm.core.pareto_archive import ParetoArchive
            pareto_archive = ParetoArchive(
                max_size=archive_config.get('max_size', 100),
                decay_rate=archive_config.get('decay_rate', 0.95),
            )
            self.logger.info(f"[ARCHIVE] Pareto archive enabled (max_size={pareto_archive.max_size}, decay={pareto_archive.decay_rate})")

        for gen in range(start_generation, self.generations):
            self.current_generation = gen

            # ── External control: stop check ──
            if self._web_stop_event and self._web_stop_event.is_set():
                self.logger.info("[WEB] Stop signal received — saving checkpoint and exiting")
                self.save_checkpoint(population, max(gen - 1, 0))
                break

            # ── External control: pause check ──
            if self._web_pause_event and self._web_pause_event.is_set():
                self.logger.info("[WEB] Paused — waiting for resume signal...")
                while self._web_pause_event.is_set():
                    if self._web_stop_event and self._web_stop_event.is_set():
                        break
                    import time as _time
                    _time.sleep(0.5)
                self.logger.info("[WEB] Resumed")
                # Re-check stop after resume
                if self._web_stop_event and self._web_stop_event.is_set():
                    self.save_checkpoint(population, max(gen - 1, 0))
                    break

            # ── External control: strategy injection ──
            if self._web_injection_queue:
                self._drain_injection_queue(population, gen)

            self.logger.info("")
            self.logger.info(f"{'─'*70}")
            self.logger.info(f"GENERATION {gen + 1}/{self.generations}")
            self.logger.info(f"{'─'*70}")
            
            # Start generation timing
            self.diagnostics.start_generation(gen)
            self.monitor.on_generation_start(gen, self.generations)
            
            # Step 1: Evaluate fitness
            self.diagnostics.start_phase('eval')
            self.monitor.on_phase_start('eval')
            self.evaluate_population(population)
            self.diagnostics.end_phase('eval')
            self.monitor.on_phase_end('eval', self.diagnostics.timing._phases.get('eval', 0.0))
            
            # Step 1b: Post-hoc walk-forward validation on elites (when parallel + WF)
            self._post_hoc_walk_forward_validation(population)
            
            # Step 2: Apply ranking based on mode
            distance_matrix = None
            
            if self.mode == 'nsga2':
                # NSGA-II: Non-dominated sorting + crowding distance
                fronts = fast_non_dominated_sort(list(population.individuals))
                for front in fronts:
                    crowding_distance_assignment(front)
                pareto_front = fronts[0] if fronts else []
                self.logger.info(f"[NSGA-II] {len(fronts)} Pareto fronts, front 1 has {len(pareto_front)} individuals")
                
                # Update external archive if enabled
                if pareto_archive is not None:
                    pareto_archive.update(list(population.individuals), generation=gen)
            else:
                # Single-objective: Compute pairwise distances once for efficiency
                if self.fitness_sharing or len(population.individuals) >= 2:
                    distance_matrix = calculate_pairwise_distances(list(population.individuals))
                
                # Apply fitness sharing to preserve diversity
                if self.fitness_sharing:
                    apply_fitness_sharing(population, sigma_share=self.sharing_radius, 
                                        distance_matrix=distance_matrix)
                    self.logger.debug("[FITNESS SHARING] Applied successfully")
            
            # Get statistics (reuses distance matrix for genetic diversity)
            stats = population.get_stats(distance_matrix=distance_matrix)
            self.generation_stats.append(stats)
            
            # Log generation summary
            summary_parts = [f"Best: {stats.best_fitness:.4f}", f"Avg: {stats.avg_fitness:.4f}"]
            if stats.genetic_diversity is not None:
                summary_parts.append(f"Diversity: {stats.genetic_diversity:.4f}")
            self.logger.info(f"[STATS] {' | '.join(summary_parts)}")
            
            # Update visualization if enabled
            if self.visualizer:
                self.visualizer.update(gen, stats, population)
            
            # Update best individual
            best = population.get_best(1)[0]
            if self._should_update_best_individual(best):
                self.best_individual = best
                # Snapshot best_fitness_ever NOW, before holdout monitoring
                # can modify raw_fitness in-place on this same object reference.
                pre_penalty_raw = best.raw_fitness
                if pre_penalty_raw is not None and pre_penalty_raw > self.best_fitness_ever:
                    self.best_fitness_ever = pre_penalty_raw
                    self.no_improvement_count = 0
                    self._new_best_this_gen = True
                self.logger.info(f"[NEW BEST] {best.id} with fitness {best.fitness:.4f}")
                self.monitor.on_new_best(best)
                
                # Generate trade visualization on improvement
                if self.trade_visualizer and self.trade_vis_mode == 'improvement':
                    self._visualize_strategy_trades(best, gen, 0)
            
            # Generate trade visualization each generation (for top N)
            if self.trade_visualizer and self.trade_vis_mode == 'each_generation':
                top_individuals = population.get_best(self.trade_vis_top_n)
                for idx, ind in enumerate(top_individuals):
                    self._visualize_strategy_trades(ind, gen, idx)
            
            # Update feature importance tracking
            try:
                self.feature_tracker.update(population)
                if (gen + 1) % 5 == 0 or gen == self.generations - 1:
                    self.feature_tracker.log_summary(top_n=5)
                
                # Inject adaptive indicator weights into config for mutation
                indicator_weights = self.feature_tracker.get_indicator_weights()
                if indicator_weights:
                    self.config['_indicator_weights'] = indicator_weights
                    self.logger.debug(f"[FEATURE-IMPORTANCE] Updated indicator weights: "
                                    f"{len(indicator_weights)} indicators")
            except Exception as e:
                self.logger.warning(f"Feature importance update failed: {e}")
            
            # Update hall of fame
            try:
                self.hall_of_fame.update(population, gen)
            except Exception as e:
                self.logger.warning(f"Hall of fame update failed: {e}")
            
            # Holdout monitoring — read-only diagnostic (never affects selection)
            # But CAN trigger early stopping if degradation consistently exceeds threshold
            self.diagnostics.start_phase('holdout')
            self.monitor.on_phase_start('holdout')
            try:
                self._run_holdout_monitoring(population, gen)
                
                # Check holdout-aware early stopping
                if (self.holdout_early_stop and 
                    self._holdout_consecutive_bad >= self.holdout_early_stop_checks):
                    self.logger.info(
                        f"[HOLDOUT EARLY STOP] Stopping: holdout degradation exceeded "
                        f"{self.holdout_early_stop_threshold:.0%} for "
                        f"{self._holdout_consecutive_bad} consecutive checks. "
                        f"Further evolution is likely overfitting."
                    )
                    self.feature_tracker.log_summary()
                    self.save_checkpoint(population, gen)
                    break
            except Exception as e:
                self.logger.warning(f"Holdout monitoring failed: {e}")
            self.diagnostics.end_phase('holdout')
            self.monitor.on_phase_end('holdout', self.diagnostics.timing._phases.get('holdout', 0.0))
            
            # Record generation diagnostics (CSV row + timing)
            # Compute new-feature metrics for CSV tracking
            _extras = {'mutation_rate': self.mutation_rate}
            try:
                all_inds = population.get_all() if hasattr(population, 'get_all') else []
                # Holdout penalty stats
                penalties = [ind.metrics.get('holdout_penalty', 0) for ind in all_inds if ind.metrics]
                penalised = [p for p in penalties if p > 0]
                _extras['holdout_penalties_applied'] = len(penalised)
                _extras['avg_holdout_penalty'] = round(sum(penalised) / len(penalised), 4) if penalised else 0.0
                # Unused indicator stats (from fitness eval, stored in metrics)
                unused_counts = [ind.metrics.get('unused_indicators', 0) for ind in all_inds if ind.metrics]
                _extras['avg_unused_indicators'] = round(sum(unused_counts) / max(len(unused_counts), 1), 2)
                # LLM origin counts
                origins = [ind.metrics.get('origin', '') for ind in all_inds if ind.metrics]
                _extras['llm_seeds_count'] = sum(1 for o in origins if o == 'llm_seed')
                _extras['llm_immigrants_count'] = sum(1 for o in origins if o == 'llm_immigrant')
            except Exception:
                pass  # Non-critical diagnostics
            self.diagnostics.end_generation(
                gen, stats, population,
                extras=_extras,
            )
            
            # Update terminal monitor with generation results
            gen_timing = self.diagnostics.timing.history[-1] if self.diagnostics.timing.history else None
            # Store population snapshot BEFORE on_generation_end so it's
            # available when _persist_generation_snapshot runs inside the callback
            if self._web_monitor and hasattr(self._web_monitor, 'store_population_snapshot'):
                try:
                    pop_dicts = [ind.to_dict() for ind in population.individuals]
                    self._web_monitor.store_population_snapshot(pop_dicts)
                except Exception as _snap_err:
                    self.logger.debug(f"Population snapshot failed: {_snap_err}")

            self.monitor.on_generation_end(
                gen, stats, gen_timing, self.best_individual, extras=_extras
            )
            
            # Log walk-forward cache stats periodically
            try:
                if hasattr(self.fitness_evaluator, 'log_wf_cache_stats'):
                    self.fitness_evaluator.log_wf_cache_stats()
                elif hasattr(self.fitness_evaluator, 'base_evaluator'):
                    # RegimeAwareEvaluator wraps a FitnessEvaluator
                    base = self.fitness_evaluator.base_evaluator
                    if hasattr(base, 'log_wf_cache_stats'):
                        base.log_wf_cache_stats()
            except Exception:
                pass  # Non-critical
            
            # Check convergence
            if self.check_convergence(stats):
                self.logger.info("[CONVERGENCE] Evolution converged early")
                self.monitor.on_convergence_warning(self.no_improvement_count, self.convergence_patience)
                self.feature_tracker.log_summary()
                self.save_checkpoint(population, gen)
                break
            elif self.no_improvement_count >= self.convergence_patience // 2:
                self.monitor.on_convergence_warning(self.no_improvement_count, self.convergence_patience)
            
            # Save checkpoint periodically
            if self.checkpoint_interval > 0 and (gen + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(population, gen)
            
            # Create next generation
            if gen < self.generations - 1:  # Don't create next gen on last iteration
                self.diagnostics.start_phase('selection')
                self.monitor.on_phase_start('selection')
                population = self.create_next_generation(population)
                self.diagnostics.end_phase('selection')
                self.monitor.on_phase_end('selection', self.diagnostics.timing._phases.get('selection', 0.0))
        
        # Final feature importance report
        try:
            self.feature_tracker.log_summary(top_n=10)
        except Exception as e:
            self.logger.warning(f"Final feature importance report failed: {e}")
        
        # Final hall of fame summary
        try:
            hof_summary = self.hall_of_fame.get_summary()
            if hof_summary['size'] > 0:
                self.logger.info("")
                self.logger.info(f"[HALL OF FAME] {hof_summary['size']} strategies archived")
                self.logger.info(f"  Best: {hof_summary['best_fitness']:.4f}  Avg: {hof_summary['avg_fitness']:.4f}")
                for i, entry in enumerate(hof_summary['top_5']):
                    self.logger.info(f"  #{i+1}: fitness={entry['fitness']:.4f}  profit={entry['profit']:.2f}%  sharpe={entry['sharpe']:.2f}")
        except Exception as e:
            self.logger.warning(f"Hall of fame summary failed: {e}")
        
        # LLM strategy designer summary
        if self.llm_enabled and self.strategy_designer.enabled:
            try:
                llm_stats = self.strategy_designer.get_stats()
                self.logger.info("")
                self.logger.info(f"[LLM DESIGNER] Requests: {llm_stats['total_requests']} | "
                               f"Successful: {llm_stats['successful']} | "
                               f"Failed: {llm_stats['failed']} | "
                               f"Fixed: {llm_stats['validation_fixed']}")
            except Exception as e:
                self.logger.warning(f"LLM stats summary failed: {e}")
        
        # Final summary
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("EVOLUTION COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"  Total generations: {self.current_generation + 1}")
        
        if self.mode == 'nsga2':
            # NSGA-II: Show Pareto front summary
            pareto_front = get_pareto_front(list(population.individuals))
            self.logger.info(f"  Pareto front size: {len(pareto_front)}")
            self.logger.info("  Top Pareto-optimal strategies:")
            sorted_front = sorted(pareto_front, key=lambda x: x.objectives[0] if x.objectives else 0, reverse=True)
            for i, ind in enumerate(sorted_front[:5]):
                if ind.objectives and ind.metrics:
                    m = ind.metrics
                    obj_str = ", ".join([f"{self.objectives_config[j]['name']}={ind.objectives[j]:.3f}" 
                                        for j in range(min(len(ind.objectives), len(self.objectives_config)))])
                    self.logger.info(f"    {i+1}. {ind.id}: profit={m.get('profit', 0):.2f}%, drawdown={m.get('max_drawdown', 0):.1%}, sharpe={m.get('sharpe_ratio', 0):.2f}")
        else:
            # Single-objective: Show best individual
            self.logger.info(f"  Best individual: {self.best_individual.id}")
            self.logger.info(f"  Best fitness: {self.best_individual.fitness:.4f}")
            if self.best_individual.metrics:
                m = self.best_individual.metrics
                self.logger.info(f"  Best profit: {m.get('profit', 0):.2f}% | Win rate: {m.get('win_rate', 0):.1%}")
        
        self.logger.info("=" * 70)
        
        # Finalize run diagnostics (close CSV, save timing summary)
        timing_summary = self.diagnostics.end_run(top_strategies=population.get_best(10) if population else None)
        if timing_summary:
            self.logger.info(f"[TIMING] {timing_summary}")
        
        # Close visualization if enabled
        if self.visualizer:
            self.visualizer.close()
        
        # Shutdown parallel evaluator pool (free worker processes)
        if self.parallel_evaluator:
            self.parallel_evaluator.shutdown()
        
        # Stop terminal monitor (prints final summary)
        self.monitor.on_evolution_complete({
            'generations': self.current_generation + 1,
            'best_fitness': self.best_individual.fitness if self.best_individual else None,
        })
        
        # Generate final trade visualizations for top strategies
        if self.trade_visualizer and self.trade_vis_mode == 'final':
            self.logger.info("[TRADE VIS] Generating trade charts for top strategies...")
            top_individuals = population.get_best(self.trade_vis_top_n)
            for idx, ind in enumerate(top_individuals):
                self._visualize_strategy_trades(ind, self.current_generation, idx)
            self.logger.info(f"[TRADE VIS] Generated charts for {len(top_individuals)} strategies")
        
        # Run holdout test on best strategies (if configured)
        holdout_config = self.config.get('holdout_test', {})
        if holdout_config.get('enabled', False):
            self._run_holdout_test(population, holdout_config)
        
        # Return top strategies based on mode
        if self.mode == 'nsga2':
            # Prefer external archive if available
            if pareto_archive is not None and pareto_archive.size > 0:
                self.logger.info(f"[ARCHIVE] Returning {pareto_archive.size} archive members as final solution set")
                return nsga2_crowded_comparison_sort(pareto_archive.get_archive())[:self.pareto_front_size]
            # Fall back to last generation's Pareto front
            pareto_front = get_pareto_front(list(population.individuals))
            return nsga2_crowded_comparison_sort(pareto_front)[:self.pareto_front_size]
        else:
            # Return top by fitness
            population.sort_by_fitness(reverse=True)
            return population.get_best(10)
