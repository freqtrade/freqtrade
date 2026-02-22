"""
Main Evolution Engine

Coordinates the genetic algorithm evolution process.
Supports both single-objective and multi-objective (NSGA-II) optimization.
"""

import random
import yaml
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
from genetic_algorithm.core.crossover import crossover
from genetic_algorithm.core.mutation import mutate
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.core.nsga2 import (
    fast_non_dominated_sort,
    crowding_distance_assignment,
    extract_objectives_from_metrics,
    get_pareto_front,
    nsga2_crowded_comparison_sort,
    DEFAULT_OBJECTIVES
)
from genetic_algorithm.evaluation.parallel import ParallelEvaluator, is_parallel_available


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
        
        # Adaptive parameters
        self.base_mutation_rate = self.mutation_rate
        self.adaptive_mutation = ga_config.get('adaptive_mutation', True)
        self.max_adaptation_factor = ga_config.get('max_adaptation_factor', 2.0)
        self.adaptation_step = ga_config.get('adaptation_step', 0.1)
        
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
        
        # Only add handlers if not already added (avoid duplicate handlers on re-init)
        if not logger.handlers:
            # Create formatter
            log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            formatter = logging.Formatter(log_format)
            
            # Add console handler if enabled
            if log_config.get('console', True):
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
        Create initial population with random strategies.
        
        Returns:
            Initial population
        """
        self.logger.info(f"Initializing population with {self.population_size} individuals")
        
        population = Population(size=self.population_size, generation=0)
        
        for i in range(self.population_size):
            # Generate random strategy
            strategy_gene = self.strategy_generator.generate_random_strategy(
                generation=0,
                individual_id=i
            )
            individual = Individual(strategy_gene=strategy_gene)
            population.add_individual(individual)
        
        self.logger.info("Population initialized successfully")
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
    
    def _evaluate_population_parallel(self, unevaluated: list):
        """
        Evaluate population using parallel workers.
        
        Args:
            unevaluated: List of unevaluated individuals
        """
        self.logger.info(f"[EVAL] Parallel evaluation of {len(unevaluated)} individuals...")
        
        # Create progress callback for tqdm if enabled
        pbar = None
        if self.progress_enabled:
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
        if self.progress_enabled:
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
        self.logger.debug(f"[ELITISM] Preserving top {self.elite_size} individuals")
        for individual in population.get_best(self.elite_size):
            gene_copy = individual.strategy_gene.copy()
            gene_copy.generation = self.current_generation + 1
            elite_copy = Individual(strategy_gene=gene_copy)
            # Carry over fitness and metrics to avoid re-evaluation
            elite_copy.raw_fitness = individual.raw_fitness
            elite_copy.fitness = individual.fitness
            elite_copy.metrics = individual.metrics.copy() if individual.metrics else {}
            elite_copy.evaluated = True
            next_gen.add_individual(elite_copy)
        self.logger.info(f"[ELITISM] Preserved {self.elite_size} elite individuals")
        
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
        
        # Inject random immigrants
        immigrants_before = len(next_gen)
        for _ in range(immigrant_count):
            if len(next_gen) >= self.population_size:
                break
            immigrant_gene = self.strategy_generator.generate_random_strategy(
                generation=self.current_generation + 1,
                individual_id=calculate_next_id()
            )
            next_gen.add_individual(Individual(strategy_gene=immigrant_gene))
        
        actual_immigrants_added = len(next_gen) - immigrants_before
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
                        config=self.config
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
        
        Args:
            stats: Current generation statistics
            
        Returns:
            True if converged, False otherwise
        """
        if self.best_individual is None:
            return False
        
        # Use raw fitness (not shared fitness) for convergence detection
        current_best = stats.best_raw_fitness if stats.best_raw_fitness is not None else stats.best_fitness
        
        # Check for improvement
        if current_best <= self.best_fitness_ever:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
            self.best_fitness_ever = current_best
        
        # Adaptive mutation: increase mutation rate if stuck
        if self.adaptive_mutation and self.no_improvement_count > 0:
            # Gradually increase mutation rate when stuck
            # adaptation_factor = 1.0 + (generations_stuck * adaptation_step)
            # Capped at max_adaptation_factor (default 2.0 = double the rate)
            adaptation_factor = min(
                self.max_adaptation_factor, 
                1.0 + (self.no_improvement_count * self.adaptation_step)
            )
            self.mutation_rate = min(0.5, self.base_mutation_rate * adaptation_factor)
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
        
        return False
    
    def evolve(self) -> List[Individual]:
        """
        Run the complete evolution process.
        
        Returns:
            List of best individuals
        """
        self.logger.info("=" * 70)
        self.logger.info("GENETIC ALGORITHM STARTING")
        self.logger.info("=" * 70)
        self.logger.info(f"  Population: {self.population_size} | Generations: {self.generations}")
        self.logger.info(f"  Mutation: {self.mutation_rate:.2%} | Crossover: {self.crossover_rate:.2%}")
        self.logger.info(f"  Selection: {self.selection_method} | Elite size: {self.elite_size}")
        self.logger.info("=" * 70)
        
        # Initialize population
        population = self.initialize_population()
        
        # Evolution loop
        for gen in range(self.generations):
            self.current_generation = gen
            self.logger.info("")
            self.logger.info(f"{'─'*70}")
            self.logger.info(f"GENERATION {gen + 1}/{self.generations}")
            self.logger.info(f"{'─'*70}")
            
            # Step 1: Evaluate fitness
            self.evaluate_population(population)
            
            # Step 2: Apply ranking based on mode
            distance_matrix = None
            
            if self.mode == 'nsga2':
                # NSGA-II: Non-dominated sorting + crowding distance
                fronts = fast_non_dominated_sort(list(population.individuals))
                for front in fronts:
                    crowding_distance_assignment(front)
                pareto_front = fronts[0] if fronts else []
                self.logger.info(f"[NSGA-II] {len(fronts)} Pareto fronts, front 1 has {len(pareto_front)} individuals")
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
                self.logger.info(f"[NEW BEST] {best.id} with fitness {best.fitness:.4f}")
                
                # Generate trade visualization on improvement
                if self.trade_visualizer and self.trade_vis_mode == 'improvement':
                    self._visualize_strategy_trades(best, gen, 0)
            
            # Generate trade visualization each generation (for top N)
            if self.trade_visualizer and self.trade_vis_mode == 'each_generation':
                top_individuals = population.get_best(self.trade_vis_top_n)
                for idx, ind in enumerate(top_individuals):
                    self._visualize_strategy_trades(ind, gen, idx)
            
            # Check convergence
            if self.check_convergence(stats):
                self.logger.info("[CONVERGENCE] Evolution converged early")
                break
            
            # Create next generation
            if gen < self.generations - 1:  # Don't create next gen on last iteration
                population = self.create_next_generation(population)
        
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
        
        # Close visualization if enabled
        if self.visualizer:
            self.visualizer.close()
        
        # Generate final trade visualizations for top strategies
        if self.trade_visualizer and self.trade_vis_mode == 'final':
            self.logger.info("[TRADE VIS] Generating trade charts for top strategies...")
            top_individuals = population.get_best(self.trade_vis_top_n)
            for idx, ind in enumerate(top_individuals):
                self._visualize_strategy_trades(ind, self.current_generation, idx)
            self.logger.info(f"[TRADE VIS] Generated charts for {len(top_individuals)} strategies")
        
        # Return top strategies based on mode
        if self.mode == 'nsga2':
            # Return entire Pareto front sorted by crowded comparison
            pareto_front = get_pareto_front(list(population.individuals))
            return nsga2_crowded_comparison_sort(pareto_front)[:self.pareto_front_size]
        else:
            # Return top by fitness
            population.sort_by_fitness(reverse=True)
            return population.get_best(10)
