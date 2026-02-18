"""
Main Evolution Engine

Coordinates the genetic algorithm evolution process.
"""

import random
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from genetic_algorithm.core.population import Population, PopulationStats, apply_fitness_sharing
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.selection import select_parents
from genetic_algorithm.core.crossover import crossover
from genetic_algorithm.core.mutation import mutate
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.fitness import FitnessEvaluator


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
        
        # Diversity preservation settings
        self.fitness_sharing = ga_config.get('fitness_sharing', True)
        self.sharing_radius = ga_config.get('sharing_radius', 0.3)
        self.diversity_threshold = ga_config.get('diversity_threshold', 0.15)
        self.allow_self_crossover = ga_config.get('allow_self_crossover', True)
        self.random_immigrants = ga_config.get('random_immigrants', 3)
        
        # Initialize components
        self.strategy_generator = StrategyGenerator(self.config)
        self.fitness_evaluator = FitnessEvaluator(self.config)
        
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
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        log_config = self.config.get('logging', {})
        logger = logging.getLogger('GeneticAlgorithm')
        logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        
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
        
        Args:
            population: Population to evaluate
        """
        unevaluated = [ind for ind in population if not ind.evaluated]
        
        if not unevaluated:
            return
        
        self.logger.info(f"Evaluating {len(unevaluated)} individuals...")
        
        for i, individual in enumerate(unevaluated):
            strategy_name = individual.id
            self.logger.debug(f"Evaluating individual {i+1}/{len(unevaluated)}: {strategy_name}")
            
            try:
                # Evaluate fitness
                fitness, metrics = self.fitness_evaluator.evaluate(
                    individual.strategy_gene,
                    strategy_name=strategy_name
                )
                individual.set_fitness(fitness, metrics)
                
                self.logger.debug(f"  Fitness: {fitness:.4f}, Profit: {metrics.get('profit', 0):.2f}%")
                
            except Exception as e:
                # Handle evaluation errors gracefully
                self.logger.error(f"Failed to evaluate {strategy_name}: {e}")
                # Set zero fitness for failed evaluation
                individual.set_fitness(0.0, {
                    'profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'error': str(e)
                })
    
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
    
    def create_next_generation(self, population: Population) -> Population:
        """
        Create next generation through selection, crossover, and mutation.
        
        Args:
            population: Current population
            
        Returns:
            Next generation population
        """
        self.logger.info(f"Creating generation {self.current_generation + 1}")
        
        # Sort by fitness
        population.sort_by_fitness(reverse=True)
        
        # Create next generation
        next_gen = Population(size=self.population_size, generation=self.current_generation + 1)
        
        # Elitism: keep top performers
        for individual in population.get_best(self.elite_size):
            gene_copy = individual.strategy_gene.copy()
            gene_copy.generation = self.current_generation + 1
            next_gen.add_individual(Individual(strategy_gene=gene_copy))
        
        # Inject random immigrants to maintain diversity
        # Get current generation stats to check diversity
        stats = population.get_stats()
        immigrant_count = self.random_immigrants
        
        # Double immigrant count if diversity is low
        if stats.genetic_diversity is not None and stats.genetic_diversity < self.diversity_threshold:
            immigrant_count = self.random_immigrants * 2
            self.logger.info(f"Low diversity ({stats.genetic_diversity:.4f} < {self.diversity_threshold:.4f}), doubling immigrant count")
        
        # Inject random immigrants
        actual_immigrants_added = 0
        for i in range(immigrant_count):
            if len(next_gen) >= self.population_size:
                break
            immigrant_gene = self.strategy_generator.generate_random_strategy(
                generation=self.current_generation + 1,
                individual_id=len(next_gen)  # Use current length for unique ID
            )
            next_gen.add_individual(Individual(strategy_gene=immigrant_gene))
            actual_immigrants_added += 1
        
        self.logger.info(f"Injected {actual_immigrants_added} random immigrants")
        
        # Helper to create child from parent gene
        def create_child(parent_gene, ind_id):
            gene = parent_gene.copy()
            gene.generation = self.current_generation + 1
            gene.individual_id = ind_id
            return Individual(strategy_gene=gene)
        
        # Fill rest with offspring
        offspring_count = 0
        while len(next_gen) < self.population_size:
            # Select parents
            parent1, parent2 = select_parents(
                population, num_parents=2,
                method=self.selection_method,
                tournament_size=self.tournament_size,
                allow_duplicates=self.allow_self_crossover
            )
            
            # Crossover or copy
            try:
                if random.random() < self.crossover_rate:
                    child1, child2 = crossover(
                        parent1, parent2,
                        generation=self.current_generation + 1,
                        ind_id=len(next_gen),  # Use current length for unique ID
                        config=self.config
                    )
                else:
                    child1 = create_child(parent1.strategy_gene, len(next_gen))
                    child2 = create_child(parent2.strategy_gene, len(next_gen) + 1)
            except (ValueError, KeyError, AttributeError, TypeError) as e:
                # If crossover fails, use clones of parents instead
                self.logger.warning(f"Crossover failed: {e}. Using parent clones instead.")
                child1 = create_child(parent1.strategy_gene, len(next_gen))
                child2 = create_child(parent2.strategy_gene, len(next_gen) + 1)
            
            # Mutation - call unconditionally, mutate() handles internal probability checks
            # Previously had double-gating: outer random.random() + internal mutation sampling
            for child in [child1, child2]:
                try:
                    child = mutate(child, self.mutation_rate, self.config)
                    next_gen.add_individual(child)
                except (ValueError, KeyError, AttributeError, TypeError) as e:
                    # If mutation or adding fails, log and skip this child
                    self.logger.warning(f"Failed to mutate/add child: {e}. Skipping this individual.")
                    # Continue with the next child
                    continue
                
                if len(next_gen) >= self.population_size:
                    break
            
            offspring_count += 2
        
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
        self.logger.info("Starting evolution...")
        self.logger.info(f"Population size: {self.population_size}")
        self.logger.info(f"Generations: {self.generations}")
        self.logger.info(f"Mutation rate: {self.mutation_rate}")
        self.logger.info(f"Crossover rate: {self.crossover_rate}")
        
        # Initialize population
        population = self.initialize_population()
        
        # Evolution loop
        for gen in range(self.generations):
            self.current_generation = gen
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Generation {gen + 1}/{self.generations}")
            self.logger.info(f"{'='*60}")
            
            # Evaluate fitness
            self.evaluate_population(population)
            
            # Apply fitness sharing to preserve diversity
            if self.fitness_sharing:
                apply_fitness_sharing(population, sigma_share=self.sharing_radius)
                self.logger.info("Applied fitness sharing for diversity preservation")
            
            # Get statistics
            stats = population.get_stats()
            self.generation_stats.append(stats)
            
            # Log statistics
            self.logger.info(f"Best fitness: {stats.best_fitness:.4f}")
            self.logger.info(f"Avg fitness: {stats.avg_fitness:.4f}")
            self.logger.info(f"Fitness diversity: {stats.diversity_score:.4f}")
            if stats.genetic_diversity is not None:
                self.logger.info(f"Genetic diversity: {stats.genetic_diversity:.4f}")
            
            # Update visualization if enabled
            if self.visualizer:
                self.visualizer.update(gen, stats, population)
            
            # Update best individual
            best = population.get_best(1)[0]
            if self._should_update_best_individual(best):
                self.best_individual = best
                self.logger.info(f"New best individual: {best.id} with fitness {best.fitness:.4f}")
            
            # Check convergence
            if self.check_convergence(stats):
                break
            
            # Create next generation
            if gen < self.generations - 1:  # Don't create next gen on last iteration
                population = self.create_next_generation(population)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Evolution complete!")
        self.logger.info(f"Best individual: {self.best_individual.id}")
        self.logger.info(f"Best fitness: {self.best_individual.fitness:.4f}")
        self.logger.info("="*60)
        
        # Close visualization if enabled
        if self.visualizer:
            self.visualizer.close()
        
        # Return top strategies
        population.sort_by_fitness(reverse=True)
        return population.get_best(10)
