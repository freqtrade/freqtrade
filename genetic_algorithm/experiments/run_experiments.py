#!/usr/bin/env python3
"""
Experiment Runner for GA Parameter Testing

Runs multiple GA evolution experiments with different configurations
and logs results for comparison analysis.

Usage:
    # Run single experiment
    python run_experiments.py exp01_baseline
    
    # Run all experiments sequentially
    python run_experiments.py --all
    
    # Run specific experiments
    python run_experiments.py exp01_baseline exp02_high_mutation
    
    # List available experiments
    python run_experiments.py --list
"""

import os
import sys
import json
import time
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genetic_algorithm.core.evolution import GeneticAlgorithm


EXPERIMENTS_DIR = Path(__file__).parent
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"


def get_available_experiments():
    """Get list of available experiment configs."""
    configs = list(CONFIGS_DIR.glob("exp*.yaml"))
    return sorted([c.stem for c in configs])


def run_experiment(exp_name: str, verbose: bool = True):
    """
    Run a single experiment and save results.
    
    Args:
        exp_name: Name of experiment (without .yaml extension)
        verbose: Print progress updates
        
    Returns:
        dict with experiment results
    """
    config_path = CONFIGS_DIR / f"{exp_name}.yaml"
    
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return None
    
    # Load config for summary
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create result dict
    result = {
        "experiment": exp_name,
        "config_path": str(config_path),
        "start_time": datetime.now().isoformat(),
        "config_summary": {
            "population_size": config.get("genetic_algorithm", {}).get("population_size"),
            "generations": config.get("genetic_algorithm", {}).get("generations"),
            "mutation_rate": config.get("genetic_algorithm", {}).get("mutation_rate"),
            "crossover_rate": config.get("genetic_algorithm", {}).get("crossover_rate"),
            "selection_method": config.get("genetic_algorithm", {}).get("selection_method"),
            "tournament_size": config.get("genetic_algorithm", {}).get("tournament_size"),
            "elite_size": config.get("genetic_algorithm", {}).get("elite_size"),
            "adaptive_mutation": config.get("genetic_algorithm", {}).get("adaptive_mutation"),
            "fitness_sharing": config.get("genetic_algorithm", {}).get("fitness_sharing"),
            "regime_aware": config.get("regime_aware", {}).get("enabled"),
            "parallel": config.get("parallel_evaluation", {}).get("enabled"),
        }
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"{'='*70}")
        print(f"Population: {result['config_summary']['population_size']}")
        print(f"Generations: {result['config_summary']['generations']}")
        print(f"Mutation: {result['config_summary']['mutation_rate']}")
        print(f"Selection: {result['config_summary']['selection_method']}")
        print(f"{'='*70}\n")
    
    # Run evolution
    start_time = time.time()
    
    try:
        ga = GeneticAlgorithm(str(config_path))
        best_individuals = ga.evolve()
        best_individual = best_individuals[0] if best_individuals else None
        
        elapsed = time.time() - start_time
        
        # Extract generation stats
        gen_stats = []
        for i, stats in enumerate(ga.generation_stats):
            gen_stats.append({
                "generation": i + 1,
                "best_fitness": stats.best_fitness,
                "avg_fitness": stats.avg_fitness,
                "worst_fitness": stats.worst_fitness,
                "genetic_diversity": stats.genetic_diversity,
            })
        
        # Calculate improvement metrics
        if len(gen_stats) >= 2:
            first_gen = gen_stats[0]
            last_gen = gen_stats[-1]
            
            improvement = {
                "best_fitness_change": last_gen["best_fitness"] - first_gen["best_fitness"],
                "best_fitness_percent": ((last_gen["best_fitness"] / first_gen["best_fitness"]) - 1) * 100 if first_gen["best_fitness"] > 0 else 0,
                "avg_fitness_change": last_gen["avg_fitness"] - first_gen["avg_fitness"],
                "generations_with_improvement": sum(1 for i in range(1, len(gen_stats)) 
                                                     if gen_stats[i]["best_fitness"] > gen_stats[i-1]["best_fitness"]),
            }
        else:
            improvement = {}
        
        result.update({
            "success": True,
            "elapsed_seconds": elapsed,
            "end_time": datetime.now().isoformat(),
            "generation_stats": gen_stats,
            "improvement": improvement,
            "best_individual": {
                "id": best_individual.id if best_individual else None,
                "fitness": best_individual.fitness if best_individual else None,
                "metrics": best_individual.metrics if best_individual else None,
            }
        })
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"RESULTS: {exp_name}")
            print(f"{'='*70}")
            print(f"Elapsed: {elapsed:.1f}s")
            print(f"Best fitness: {result['best_individual']['fitness']:.4f}")
            if improvement:
                print(f"Fitness improvement: {improvement['best_fitness_change']:+.4f} ({improvement['best_fitness_percent']:+.1f}%)")
                print(f"Generations with improvement: {improvement['generations_with_improvement']}/{len(gen_stats)-1}")
            print(f"{'='*70}\n")
            
    except Exception as e:
        elapsed = time.time() - start_time
        result.update({
            "success": False,
            "elapsed_seconds": elapsed,
            "end_time": datetime.now().isoformat(),
            "error": str(e),
        })
        print(f"ERROR in {exp_name}: {e}")
        import traceback
        traceback.print_exc()
    
    # Save result
    result_path = RESULTS_DIR / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"Results saved to: {result_path}")
    
    return result


def compare_results(results: list):
    """Compare multiple experiment results."""
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Experiment':<30} {'Best Fit':>10} {'Improve':>10} {'Time':>8} {'Gen/Imp':>8}")
    print("-" * 80)
    
    for r in results:
        if not r or not r.get("success"):
            continue
            
        name = r["experiment"][:30]
        best_fit = r.get("best_individual", {}).get("fitness", 0)
        improve = r.get("improvement", {}).get("best_fitness_change", 0)
        elapsed = r.get("elapsed_seconds", 0)
        gens_imp = r.get("improvement", {}).get("generations_with_improvement", 0)
        
        print(f"{name:<30} {best_fit:>10.4f} {improve:>+10.4f} {elapsed:>7.1f}s {gens_imp:>8}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Run GA parameter experiments")
    parser.add_argument("experiments", nargs="*", help="Experiment names to run")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--compare", action="store_true", help="Compare results after running")
    
    args = parser.parse_args()
    
    available = get_available_experiments()
    
    if args.list:
        print("Available experiments:")
        for exp in available:
            print(f"  - {exp}")
        return
    
    if args.all:
        experiments = available
    elif args.experiments:
        experiments = args.experiments
    else:
        print("Usage: python run_experiments.py [experiment_names...] [--all] [--list]")
        print("\nAvailable experiments:")
        for exp in available:
            print(f"  - {exp}")
        return
    
    # Run experiments
    results = []
    for exp in experiments:
        if exp not in available:
            print(f"WARNING: Unknown experiment '{exp}', skipping")
            continue
        result = run_experiment(exp)
        results.append(result)
    
    # Compare if multiple results
    if len(results) > 1 or args.compare:
        compare_results(results)


if __name__ == "__main__":
    main()
