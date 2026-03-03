"""
Live Visualization for Genetic Algorithm Evolution

Provides real-time plotting of fitness values, metrics, and strategy performance
during the evolution process.
"""

import logging
import matplotlib
# Set a safe default backend before any pyplot import.
# __init__() will override this if needed (e.g. TkAgg for interactive mode).
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from genetic_algorithm.core.population import PopulationStats, Population
from genetic_algorithm.core.individual import Individual

logger = logging.getLogger(__name__)


class GAVisualizer:
    """
    Real-time visualization of genetic algorithm evolution.
    
    Displays live updates of:
    - Best, average, and worst fitness per generation
    - Fitness diversity
    - Top strategies performance metrics
    - Distribution of fitness values
    """
    
    def __init__(self, enabled: bool = False, interactive: bool = True, 
                 save_plots: bool = True, output_dir: Optional[Path] = None):
        """
        Initialize the visualizer.
        
        Args:
            enabled: Whether visualization is enabled
            interactive: Whether to show live interactive plots
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots (defaults to genetic_algorithm/output/plots)
        """
        self.enabled = enabled
        self.interactive = interactive
        self.save_plots = save_plots
        self.output_dir = output_dir or Path("genetic_algorithm/output/plots")
        
        if not self.enabled:
            return
        
        # Set up matplotlib for interactive or non-interactive mode.
        # Backend was already set to 'Agg' at module import time; switch
        # only if interactive mode is requested and available.
        if self.interactive:
            try:
                plt.switch_backend('TkAgg')
            except (ImportError, ModuleNotFoundError):
                logger.warning("Interactive backend (TkAgg) not available. Falling back to non-interactive mode.")
                self.interactive = False
                plt.switch_backend('Agg')

        if self.interactive:
            plt.ion()  # Enable interactive mode
        
        # Create output directory if saving plots
        if self.save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage for plotting
        self.generations: List[int] = []
        self.best_fitness: List[float] = []
        self.avg_fitness: List[float] = []
        self.worst_fitness: List[float] = []
        self.diversity: List[float] = []
        
        # Metrics tracking
        self.best_profit: List[float] = []
        self.best_sharpe: List[float] = []
        self.best_win_rate: List[float] = []
        self.best_drawdown: List[float] = []
        
        # Initialize figure and subplots
        self.fig = None
        self.axes = None
        
        logger.info("GAVisualizer initialized (enabled=%s, interactive=%s)", 
                   self.enabled, self.interactive)
    
    def setup_plots(self):
        """Set up the figure and subplots for visualization."""
        if not self.enabled:
            return
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Genetic Algorithm Evolution Progress', fontsize=16, fontweight='bold')

        # Pre-create the twin axis for the performance metrics subplot
        # to avoid leaking a new Axes on every update() call.
        self._perf_twin_ax = self.axes[1, 0].twinx()
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        if self.interactive:
            plt.show(block=False)
        
        logger.info("Plots initialized")
    
    def update(self, generation: int, stats: PopulationStats, population: Population):
        """
        Update visualization with new generation data.
        
        Args:
            generation: Current generation number
            stats: Population statistics for this generation
            population: Current population (for extracting best individual metrics)
        """
        if not self.enabled:
            return
        
        # Initialize plots on first update
        if self.fig is None:
            self.setup_plots()
        
        # Store generation data
        self.generations.append(generation)
        self.best_fitness.append(stats.best_fitness or 0)
        self.avg_fitness.append(stats.avg_fitness or 0)
        self.worst_fitness.append(stats.worst_fitness or 0)
        self.diversity.append(stats.diversity_score or 0)
        
        # Get best individual metrics
        best_individual = population.get_best(1)[0] if len(population) > 0 else None
        if best_individual and best_individual.metrics:
            metrics = best_individual.metrics
            self.best_profit.append(metrics.get('profit', 0))
            self.best_sharpe.append(metrics.get('sharpe_ratio', 0))
            self.best_win_rate.append(metrics.get('win_rate', 0) * 100)  # Convert to percentage
            self.best_drawdown.append(metrics.get('max_drawdown', 0) * 100)  # Convert to percentage
        else:
            # Append zeros if no metrics available
            self.best_profit.append(0)
            self.best_sharpe.append(0)
            self.best_win_rate.append(0)
            self.best_drawdown.append(0)
        
        # Update all plots
        self._plot_fitness_evolution()
        self._plot_diversity()
        self._plot_performance_metrics()
        self._plot_fitness_distribution(population)
        
        # Refresh the display
        if self.interactive:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.1)  # Longer pause to ensure update is visible
        
        # Always save intermediate plot for non-interactive mode
        if not self.interactive and self.save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = self.output_dir / f"ga_evolution_gen{generation}_{timestamp}.png"
            self.fig.savefig(intermediate_file, dpi=100, bbox_inches='tight')
            logger.debug(f"Saved intermediate plot for generation {generation}")
        
        logger.debug("Visualization updated for generation %d", generation)
    
    def _plot_fitness_evolution(self):
        """Plot fitness evolution over generations."""
        ax = self.axes[0, 0]
        ax.clear()
        
        x = self.generations
        ax.plot(x, self.best_fitness, 'g-', linewidth=2, label='Best Fitness', marker='o')
        ax.plot(x, self.avg_fitness, 'b--', linewidth=1.5, label='Average Fitness', marker='s')
        ax.plot(x, self.worst_fitness, 'r:', linewidth=1, label='Worst Fitness', marker='x')
        
        # Fill area between best and worst
        ax.fill_between(x, self.worst_fitness, self.best_fitness, alpha=0.2, color='gray')
        
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Fitness Score', fontweight='bold')
        ax.set_title('Fitness Evolution', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add current values as text
        if self.best_fitness:
            current_gen = self.generations[-1]
            current_best = self.best_fitness[-1]
            current_avg = self.avg_fitness[-1]
            ax.text(0.02, 0.98, f'Gen {current_gen}\nBest: {current_best:.4f}\nAvg: {current_avg:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_diversity(self):
        """Plot fitness diversity over generations."""
        ax = self.axes[0, 1]
        ax.clear()
        
        x = self.generations
        ax.plot(x, self.diversity, 'purple', linewidth=2, marker='d')
        ax.fill_between(x, 0, self.diversity, alpha=0.3, color='purple')
        
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Diversity Score', fontweight='bold')
        ax.set_title('Population Diversity', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add current value
        if self.diversity:
            current_diversity = self.diversity[-1]
            ax.text(0.98, 0.98, f'Current: {current_diversity:.4f}',
                   transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    
    def _plot_performance_metrics(self):
        """Plot key performance metrics of best strategy over generations."""
        ax = self.axes[1, 0]
        ax.clear()
        
        x = self.generations
        
        # Reuse the pre-created twin axis (created in initialize_plots)
        ax2 = self._perf_twin_ax
        ax2.clear()
        
        # Plot profit and sharpe on left axis
        line1 = ax.plot(x, self.best_profit, 'g-', linewidth=2, label='Profit %', marker='o')
        line2 = ax.plot(x, self.best_sharpe, 'b-', linewidth=2, label='Sharpe Ratio', marker='s')
        
        # Plot win rate and drawdown on right axis
        line3 = ax2.plot(x, self.best_win_rate, 'orange', linewidth=2, label='Win Rate %', marker='^', linestyle='--')
        line4 = ax2.plot(x, self.best_drawdown, 'r', linewidth=2, label='Max Drawdown %', marker='v', linestyle=':')
        
        ax.set_xlabel('Generation', fontweight='bold')
        ax.set_ylabel('Profit % / Sharpe Ratio', fontweight='bold', color='k')
        ax2.set_ylabel('Win Rate % / Drawdown %', fontweight='bold', color='k')
        ax.set_title('Best Strategy Performance Metrics', fontweight='bold')
        
        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        ax.grid(True, alpha=0.3)
        
        # Add current values
        if self.best_profit:
            current_profit = self.best_profit[-1]
            current_sharpe = self.best_sharpe[-1]
            current_winrate = self.best_win_rate[-1]
            current_dd = self.best_drawdown[-1]
            
            info_text = (f'Profit: {current_profit:.2f}%\n'
                        f'Sharpe: {current_sharpe:.2f}\n'
                        f'Win Rate: {current_winrate:.1f}%\n'
                        f'Drawdown: {current_dd:.1f}%')
            
            ax.text(0.98, 0.02, info_text,
                   transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    def _plot_fitness_distribution(self, population: Population):
        """Plot histogram of current population fitness distribution."""
        ax = self.axes[1, 1]
        ax.clear()
        
        # Get all fitness values
        fitness_values = [ind.fitness for ind in population if ind.fitness is not None]
        
        if not fitness_values:
            ax.text(0.5, 0.5, 'No fitness data available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create histogram
        n, bins, patches = ax.hist(fitness_values, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        
        # Color the bars based on value (gradient from red to green)
        cm = plt.cm.RdYlGn
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        ax.set_xlabel('Fitness Score', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Current Population Fitness Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_fitness = np.mean(fitness_values)
        median_fitness = np.median(fitness_values)
        std_fitness = np.std(fitness_values)
        
        stats_text = (f'Mean: {mean_fitness:.4f}\n'
                     f'Median: {median_fitness:.4f}\n'
                     f'Std Dev: {std_fitness:.4f}\n'
                     f'N: {len(fitness_values)}')
        
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def save_final_plot(self, filename: Optional[str] = None):
        """
        Save the final plot to a file.
        
        Args:
            filename: Optional filename for the plot (auto-generated if not provided)
        """
        if not self.enabled or not self.save_plots:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ga_evolution_{timestamp}.png"
        
        filepath = self.output_dir / filename
        
        if self.fig:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info("Final plot saved to: %s", filepath)
            print(f"\n✓ Visualization saved to: {filepath}")
    
    def close(self):
        """Close the visualization and clean up."""
        if not self.enabled:
            return
        
        if self.fig:
            # Save final plot before closing
            self.save_final_plot()
            
            if self.interactive:
                plt.ioff()  # Turn off interactive mode
            
            # Keep the plot open for a moment before closing in interactive mode
            if self.interactive:
                print("\nVisualization complete. Close the plot window to continue...")
                plt.show()  # Block until user closes
            
            plt.close(self.fig)
        
        logger.info("Visualizer closed")
