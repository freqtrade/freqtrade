#!/usr/bin/env python3
"""
Parallel Evaluation Benchmark Script

This script benchmarks the parallel evaluation feature by comparing
sequential vs parallel strategy evaluation times.

Usage:
    python genetic_algorithm/benchmark_parallel.py [--workers N] [--strategies N]

Example:
    python genetic_algorithm/benchmark_parallel.py --workers 4 --strategies 20
"""

import argparse
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.evaluation.parallel import (
    ParallelEvaluator,
    is_parallel_available,
    get_recommended_workers
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "genetic_algorithm/config/ga_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_test_individuals(config: dict, count: int) -> List[Individual]:
    """Create test individuals for benchmarking."""
    generator = StrategyGenerator(config)
    individuals = []
    
    logger.info(f"Generating {count} random strategies...")
    for i in range(count):
        strategy_gene = generator.generate_random_strategy(
            generation=0,
            individual_id=i
        )
        individual = Individual(strategy_gene=strategy_gene)
        individuals.append(individual)
    
    return individuals


def benchmark_sequential(
    config: dict,
    individuals: List[Individual]
) -> Tuple[float, int, int]:
    """
    Benchmark sequential evaluation.
    
    Returns:
        Tuple of (time_seconds, successful_count, failed_count)
    """
    logger.info("=" * 60)
    logger.info("SEQUENTIAL EVALUATION")
    logger.info("=" * 60)
    
    evaluator = FitnessEvaluator(config)
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, ind in enumerate(individuals):
        try:
            fitness, metrics = evaluator.evaluate(ind.strategy_gene)
            ind.set_fitness(fitness, metrics)
            successful += 1
            
            if (i + 1) % 5 == 0 or i == len(individuals) - 1:
                logger.info(f"Progress: {i+1}/{len(individuals)} - fitness: {fitness:.4f}")
                
        except Exception as e:
            logger.warning(f"Failed to evaluate strategy {i}: {e}")
            failed += 1
    
    elapsed = time.time() - start_time
    
    logger.info(f"Sequential: {successful} succeeded, {failed} failed in {elapsed:.2f}s")
    return elapsed, successful, failed


def benchmark_parallel(
    config: dict,
    individuals: List[Individual],
    num_workers: int
) -> Tuple[float, int, int]:
    """
    Benchmark parallel evaluation.
    
    Returns:
        Tuple of (time_seconds, successful_count, failed_count)
    """
    logger.info("=" * 60)
    logger.info(f"PARALLEL EVALUATION ({num_workers} workers)")
    logger.info("=" * 60)
    
    evaluator = ParallelEvaluator(config, num_workers=num_workers)
    
    start_time = time.time()
    result = evaluator.evaluate_batch(individuals)
    elapsed = time.time() - start_time
    
    logger.info(f"Parallel: {result.successful} succeeded, {result.failed} failed in {elapsed:.2f}s")
    return elapsed, result.successful, result.failed


def run_benchmark(
    config_path: str,
    num_strategies: int,
    num_workers: int,
    skip_sequential: bool = False
):
    """Run the complete benchmark."""
    logger.info("=" * 70)
    logger.info("PARALLEL EVALUATION BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Strategies to evaluate: {num_strategies}")
    logger.info(f"Workers: {num_workers}")
    logger.info(f"CPU cores available: {os.cpu_count()}")
    logger.info(f"Parallel available: {is_parallel_available()}")
    logger.info("=" * 70)
    
    # Load config
    config = load_config(config_path)
    
    # Check if parallel is available
    if not is_parallel_available():
        logger.error("Parallel evaluation is not available on this system!")
        return
    
    # Create test individuals
    individuals_seq = create_test_individuals(config, num_strategies)
    
    # Clone individuals for parallel test (to ensure fair comparison)
    individuals_par = [
        Individual(strategy_gene=ind.strategy_gene.copy())
        for ind in individuals_seq
    ]
    
    results = {}
    
    # Run sequential benchmark
    if not skip_sequential:
        seq_time, seq_success, seq_failed = benchmark_sequential(config, individuals_seq)
        results['sequential'] = {
            'time': seq_time,
            'successful': seq_success,
            'failed': seq_failed
        }
    
    # Run parallel benchmark
    par_time, par_success, par_failed = benchmark_parallel(config, individuals_par, num_workers)
    results['parallel'] = {
        'time': par_time,
        'successful': par_success,
        'failed': par_failed,
        'workers': num_workers
    }
    
    # Calculate speedup
    logger.info("")
    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 70)
    
    if not skip_sequential:
        speedup = seq_time / par_time if par_time > 0 else 0
        efficiency = (speedup / num_workers) * 100
        
        logger.info(f"Sequential time:    {seq_time:.2f}s")
        logger.info(f"Parallel time:      {par_time:.2f}s")
        logger.info(f"Speedup:            {speedup:.2f}x")
        logger.info(f"Efficiency:         {efficiency:.1f}%")
        logger.info(f"Time saved:         {seq_time - par_time:.2f}s")
        
        results['speedup'] = speedup
        results['efficiency'] = efficiency
    else:
        logger.info(f"Parallel time:      {par_time:.2f}s ({num_workers} workers)")
        avg_per_strategy = par_time / num_strategies if num_strategies > 0 else 0
        logger.info(f"Avg per strategy:   {avg_per_strategy:.2f}s")
    
    logger.info("")
    logger.info("RECOMMENDATION:")
    if not skip_sequential and speedup > 1.5:
        logger.info(f"✅ Parallel evaluation provides {speedup:.1f}x speedup - RECOMMENDED")
        logger.info(f"   Enable in ga_config.yaml:")
        logger.info(f"   parallel_evaluation:")
        logger.info(f"     enabled: true")
        logger.info(f"     num_workers: {num_workers}")
    elif not skip_sequential:
        logger.info(f"⚠️  Speedup is only {speedup:.1f}x - may not be worth the overhead")
        logger.info(f"   Consider using sequential evaluation for small populations")
    
    logger.info("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark parallel evaluation performance'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='genetic_algorithm/config/ga_config.yaml',
        help='Path to GA config file'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of worker processes (default: auto, recommended: {get_recommended_workers()})'
    )
    parser.add_argument(
        '--strategies',
        type=int,
        default=20,
        help='Number of strategies to evaluate (default: 20)'
    )
    parser.add_argument(
        '--skip-sequential',
        action='store_true',
        help='Skip sequential benchmark (faster, but no speedup calculation)'
    )
    
    args = parser.parse_args()
    
    num_workers = args.workers if args.workers else get_recommended_workers()
    
    try:
        results = run_benchmark(
            config_path=args.config,
            num_strategies=args.strategies,
            num_workers=num_workers,
            skip_sequential=args.skip_sequential
        )
        
        if results:
            print("\n✅ Benchmark completed successfully!")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
