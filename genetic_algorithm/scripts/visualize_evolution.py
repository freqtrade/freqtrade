#!/usr/bin/env python3
"""
Evolution Visualization Script

Generates comprehensive visualization plots for GA evolution runs.
Use after completing a GA run to analyze:
- Evolution progress (fitness over generations)
- Regime-aware performance
- Strategy diversity
- Top performer analysis

Usage:
    python genetic_algorithm/scripts/visualize_evolution.py --config fast
    python genetic_algorithm/scripts/visualize_evolution.py --config medium
    python genetic_algorithm/scripts/visualize_evolution.py --config deep
    python genetic_algorithm/scripts/visualize_evolution.py --output-dir genetic_algorithm/output/custom_run

Author: GA System
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_evolution_stats(output_dir: Path) -> dict:
    """Load evolution statistics from JSON file."""
    stats_file = output_dir / 'evolution_stats.json'
    if not stats_file.exists():
        print(f"Warning: Stats file not found: {stats_file}")
        return None
    
    with open(stats_file, 'r') as f:
        return json.load(f)


def plot_fitness_evolution(stats: dict, output_dir: Path):
    """
    Plot fitness evolution over generations.
    Shows best, average, and worst fitness with diversity overlay.
    """
    if not stats or 'generations' not in stats:
        print("No generation data found in stats")
        return None
    
    generations = stats.get('generations', [])
    if not generations:
        print("Empty generations data")
        return None
    
    # Extract data
    gen_nums = [g.get('generation', i) for i, g in enumerate(generations)]
    best_fitness = [g.get('best_fitness', 0) for g in generations]
    avg_fitness = [g.get('avg_fitness', 0) for g in generations]
    worst_fitness = [g.get('worst_fitness', 0) for g in generations]
    diversity = [g.get('diversity', 0) for g in generations]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    fig.suptitle('Evolution Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Fitness curves
    ax1.plot(gen_nums, best_fitness, 'g-', linewidth=2, label='Best Fitness', marker='o', markersize=4)
    ax1.plot(gen_nums, avg_fitness, 'b-', linewidth=1.5, label='Average Fitness', alpha=0.8)
    ax1.fill_between(gen_nums, worst_fitness, best_fitness, alpha=0.2, color='blue', label='Fitness Range')
    
    # Mark best generation
    best_gen_idx = np.argmax(best_fitness)
    ax1.axvline(x=gen_nums[best_gen_idx], color='green', linestyle='--', alpha=0.5)
    ax1.annotate(f'Best: {best_fitness[best_gen_idx]:.4f}', 
                xy=(gen_nums[best_gen_idx], best_fitness[best_gen_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='green')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Score')
    ax1.set_title('Fitness Evolution')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Diversity
    ax2.fill_between(gen_nums, 0, diversity, alpha=0.4, color='purple')
    ax2.plot(gen_nums, diversity, 'purple', linewidth=1.5, label='Genetic Diversity')
    ax2.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Diversity Threshold')
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Diversity')
    ax2.set_title('Population Diversity')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(diversity) * 1.2 if diversity else 1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'plots' / 'fitness_evolution.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    
    plt.close()
    return plot_path


def plot_regime_performance(stats: dict, output_dir: Path):
    """
    Plot strategy performance across different market regimes.
    Shows how fitness varies in bullish, bearish, and sideways markets.
    """
    if not stats or 'regime_performance' not in stats:
        print("No regime performance data found")
        return None
    
    regime_data = stats['regime_performance']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Strategy Performance by Market Regime', fontsize=14, fontweight='bold')
    
    regimes = ['bullish', 'bearish', 'sideways']
    colors = {'bullish': 'green', 'bearish': 'red', 'sideways': 'gray'}
    
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        if regime not in regime_data:
            ax.set_title(f'{regime.title()}: No Data')
            continue
        
        data = regime_data[regime]
        
        # Box plot or histogram of fitness scores
        if isinstance(data, list):
            ax.hist(data, bins=20, color=colors[regime], alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(data), color='black', linestyle='--', label=f'Mean: {np.mean(data):.3f}')
        else:
            # Single value display
            ax.bar([regime], [data], color=colors[regime])
            ax.text(0, data, f'{data:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{regime.title()} Regime')
        ax.set_xlabel('Fitness Score')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / 'plots' / 'regime_performance.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    
    plt.close()
    return plot_path


def plot_top_strategies(stats: dict, output_dir: Path):
    """
    Plot performance metrics of top strategies.
    Shows radar chart of metrics for top performers.
    """
    if not stats or 'top_strategies' not in stats:
        print("No top strategies data found")
        return None
    
    top_strategies = stats['top_strategies']
    if not top_strategies:
        return None
    
    # Metrics to display
    metrics = ['profit', 'sharpe_ratio', 'win_rate', 'profit_factor', 'max_drawdown']
    
    # Create figure
    n_strategies = min(len(top_strategies), 5)
    fig, axes = plt.subplots(1, n_strategies, figsize=(4 * n_strategies, 5))
    fig.suptitle('Top Strategy Performance Metrics', fontsize=14, fontweight='bold')
    
    if n_strategies == 1:
        axes = [axes]
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_strategies))
    
    for idx, (strategy, ax) in enumerate(zip(top_strategies[:n_strategies], axes)):
        # Extract metrics
        values = []
        labels = []
        for metric in metrics:
            if metric in strategy:
                value = strategy[metric]
                # Normalize drawdown (negative -> positive for display)
                if metric == 'max_drawdown':
                    value = abs(value)
                values.append(value)
                labels.append(metric.replace('_', ' ').title())
        
        if not values:
            continue
        
        # Bar chart for each strategy
        bars = ax.bar(range(len(values)), values, color=colors[idx])
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'Strategy #{idx + 1}\nFitness: {strategy.get("fitness", 0):.4f}')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / 'plots' / 'top_strategies.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    
    plt.close()
    return plot_path


def plot_metrics_evolution(stats: dict, output_dir: Path):
    """
    Plot how individual metrics evolve over generations.
    Shows profit, sharpe ratio, win rate, drawdown trends.
    """
    if not stats or 'generations' not in stats:
        return None
    
    generations = stats.get('generations', [])
    if not generations:
        return None
    
    # Check if metrics data exists
    first_gen = generations[0]
    if 'best_metrics' not in first_gen:
        print("No metrics data in generations")
        return None
    
    # Extract metrics per generation
    gen_nums = list(range(len(generations)))
    
    metrics_to_plot = {
        'profit': {'label': 'Best Profit (%)', 'color': 'green'},
        'sharpe_ratio': {'label': 'Best Sharpe Ratio', 'color': 'blue'},
        'win_rate': {'label': 'Best Win Rate (%)', 'color': 'orange'},
        'max_drawdown': {'label': 'Best Max Drawdown (%)', 'color': 'red'},
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metrics Evolution Over Generations', fontsize=14, fontweight='bold')
    
    for ax, (metric, config) in zip(axes.flat, metrics_to_plot.items()):
        values = []
        for gen in generations:
            if 'best_metrics' in gen and metric in gen['best_metrics']:
                val = gen['best_metrics'][metric]
                # Convert to percentage for display
                if metric in ['profit', 'win_rate', 'max_drawdown']:
                    val *= 100
                values.append(val)
            else:
                values.append(0)
        
        ax.plot(gen_nums[:len(values)], values, color=config['color'], linewidth=2, marker='o', markersize=3)
        ax.fill_between(gen_nums[:len(values)], 0, values, alpha=0.2, color=config['color'])
        ax.set_xlabel('Generation')
        ax.set_ylabel(config['label'])
        ax.set_title(config['label'])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / 'plots' / 'metrics_evolution.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    
    plt.close()
    return plot_path


def generate_summary_report(stats: dict, output_dir: Path):
    """Generate a text summary report of the evolution run."""
    if not stats:
        return None
    
    report_lines = [
        "=" * 60,
        "EVOLUTION RUN SUMMARY REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    # Config summary
    if 'config' in stats:
        config = stats['config']
        report_lines.extend([
            "CONFIGURATION:",
            f"  Population Size: {config.get('population_size', 'N/A')}",
            f"  Generations: {config.get('generations', 'N/A')}",
            f"  Mutation Rate: {config.get('mutation_rate', 'N/A')}",
            f"  Regime Detection: {config.get('regime_method', 'N/A')}",
            "",
        ])
    
    # Evolution summary
    if 'generations' in stats:
        generations = stats['generations']
        if generations:
            best_gen = max(generations, key=lambda g: g.get('best_fitness', 0))
            
            report_lines.extend([
                "EVOLUTION RESULTS:",
                f"  Total Generations: {len(generations)}",
                f"  Best Fitness: {best_gen.get('best_fitness', 0):.4f} (Gen {best_gen.get('generation', 0)})",
                f"  Final Fitness: {generations[-1].get('best_fitness', 0):.4f}",
                f"  Improvement: {((generations[-1].get('best_fitness', 0) - generations[0].get('best_fitness', 0)) / max(generations[0].get('best_fitness', 0.001), 0.001)) * 100:.1f}%",
                "",
            ])
    
    # Regime performance
    if 'regime_performance' in stats:
        report_lines.append("REGIME PERFORMANCE:")
        for regime, value in stats['regime_performance'].items():
            if isinstance(value, list):
                report_lines.append(f"  {regime.title()}: Mean={np.mean(value):.4f}, Std={np.std(value):.4f}")
            else:
                report_lines.append(f"  {regime.title()}: {value:.4f}")
        report_lines.append("")
    
    # Top strategies
    if 'top_strategies' in stats:
        report_lines.append("TOP STRATEGIES:")
        for idx, strategy in enumerate(stats['top_strategies'][:5], 1):
            report_lines.append(f"  #{idx}: Fitness={strategy.get('fitness', 0):.4f}")
            if 'profit' in strategy:
                report_lines.append(f"       Profit={strategy['profit']*100:.2f}%, Sharpe={strategy.get('sharpe_ratio', 0):.2f}")
        report_lines.append("")
    
    report_lines.extend([
        "=" * 60,
        "VISUALIZATION FILES:",
        f"  {output_dir / 'plots'}",
        "=" * 60,
    ])
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_path = output_dir / 'evolution_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"Saved: {report_path}")
    
    # Also print to console
    print("\n" + report_text)
    
    return report_path


def create_demo_stats(output_dir: Path):
    """Create demo stats for testing visualization when no real data exists."""
    np.random.seed(42)
    
    generations = []
    for i in range(20):
        # Simulate improving fitness
        base_fitness = 0.2 + (i * 0.03) + np.random.normal(0, 0.01)
        
        generations.append({
            'generation': i,
            'best_fitness': min(base_fitness + 0.05, 0.95),
            'avg_fitness': base_fitness,
            'worst_fitness': max(base_fitness - 0.15, 0.05),
            'diversity': max(0.3 - (i * 0.01) + np.random.normal(0, 0.02), 0.1),
            'best_metrics': {
                'profit': (i * 0.8 + np.random.normal(0, 2)) / 100,
                'sharpe_ratio': 0.5 + i * 0.1 + np.random.normal(0, 0.2),
                'win_rate': 0.35 + i * 0.015 + np.random.normal(0, 0.02),
                'max_drawdown': -(0.25 - i * 0.005 + np.random.normal(0, 0.02)),
            }
        })
    
    stats = {
        'config': {
            'population_size': 50,
            'generations': 20,
            'mutation_rate': 0.15,
            'regime_method': 'adx_di_hysteresis',
        },
        'generations': generations,
        'regime_performance': {
            'bullish': [0.6 + np.random.normal(0, 0.1) for _ in range(10)],
            'bearish': [0.4 + np.random.normal(0, 0.1) for _ in range(10)],
            'sideways': [0.5 + np.random.normal(0, 0.1) for _ in range(10)],
        },
        'top_strategies': [
            {'fitness': 0.85, 'profit': 0.15, 'sharpe_ratio': 2.1, 'win_rate': 0.58, 'profit_factor': 1.8, 'max_drawdown': -0.12},
            {'fitness': 0.82, 'profit': 0.12, 'sharpe_ratio': 1.9, 'win_rate': 0.55, 'profit_factor': 1.6, 'max_drawdown': -0.15},
            {'fitness': 0.80, 'profit': 0.10, 'sharpe_ratio': 1.7, 'win_rate': 0.52, 'profit_factor': 1.5, 'max_drawdown': -0.18},
        ]
    }
    
    # Save demo stats
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'evolution_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Visualize GA evolution progress and results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize fast run results
    python visualize_evolution.py --config fast
    
    # Visualize medium run results  
    python visualize_evolution.py --config medium
    
    # Visualize deep search results
    python visualize_evolution.py --config deep
    
    # Custom output directory
    python visualize_evolution.py --output-dir genetic_algorithm/output/my_run
    
    # Create demo visualization (for testing)
    python visualize_evolution.py --config fast --demo
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        choices=['fast', 'medium', 'deep'],
        help='Configuration name to visualize'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Custom output directory path'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Create demo visualization with synthetic data (for testing)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.config:
        config_dirs = {
            'fast': Path('genetic_algorithm/output/fast_run'),
            'medium': Path('genetic_algorithm/output/medium_run'),
            'deep': Path('genetic_algorithm/output/deep_run'),
        }
        output_dir = config_dirs[args.config]
    else:
        print("Error: Please specify --config or --output-dir")
        parser.print_help()
        return 1
    
    print(f"Visualizing evolution from: {output_dir}")
    
    # Load or create stats
    if args.demo:
        print("Creating demo visualization with synthetic data...")
        stats = create_demo_stats(output_dir)
    else:
        stats = load_evolution_stats(output_dir)
        if stats is None:
            print(f"No evolution stats found in {output_dir}")
            print("Run a GA evolution first or use --demo for synthetic data")
            return 1
    
    # Generate plots
    print("\nGenerating visualization plots...")
    
    plots_created = []
    
    plot = plot_fitness_evolution(stats, output_dir)
    if plot:
        plots_created.append(plot)
    
    plot = plot_metrics_evolution(stats, output_dir)
    if plot:
        plots_created.append(plot)
    
    plot = plot_regime_performance(stats, output_dir)
    if plot:
        plots_created.append(plot)
    
    plot = plot_top_strategies(stats, output_dir)
    if plot:
        plots_created.append(plot)
    
    # Generate summary report
    generate_summary_report(stats, output_dir)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*50}")
    print(f"Plots created: {len(plots_created)}")
    print(f"Output directory: {output_dir / 'plots'}")
    print(f"\nOpen the plots directory to view results:")
    print(f"  xdg-open {output_dir / 'plots'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
