"""
Feature Importance Tracker

Tracks which indicators and conditions appear in high-performing vs
low-performing strategies across generations. This identifies which
building blocks the GA finds most useful and enables future
adaptive indicator probability weighting.

Usage:
    tracker = FeatureImportanceTracker()
    # After each generation:
    tracker.update(population)
    # At end of evolution:
    report = tracker.get_report()
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Accumulated statistics for a single indicator/feature."""
    appearances_top: int = 0      # Times this feature appeared in top 20%
    appearances_bottom: int = 0   # Times this feature appeared in bottom 20%
    appearances_total: int = 0    # Total appearances
    fitness_sum_when_present: float = 0.0  # Sum of fitness when present
    fitness_count_when_present: int = 0    # Count for averaging
    generations_seen: int = 0     # Number of generations it appeared in
    
    @property
    def avg_fitness_when_present(self) -> float:
        if self.fitness_count_when_present == 0:
            return 0.0
        return self.fitness_sum_when_present / self.fitness_count_when_present
    
    @property
    def importance_score(self) -> float:
        """
        Feature importance score.
        
        Higher = more associated with top-performing strategies.
        Score considers:
        - Frequency in top 20% vs bottom 20% (selection differential)
        - Average fitness when present
        """
        if self.appearances_total == 0:
            return 0.0
        
        top_ratio = self.appearances_top / max(1, self.appearances_total)
        bottom_ratio = self.appearances_bottom / max(1, self.appearances_total)
        
        # Selection differential: how much more likely to appear in top vs bottom
        differential = top_ratio - bottom_ratio
        
        # Combine with average fitness contribution
        return differential * 0.6 + min(1.0, self.avg_fitness_when_present) * 0.4


class FeatureImportanceTracker:
    """
    Tracks indicator/condition importance across GA generations.
    
    After each generation, call update() with the evaluated population.
    The tracker identifies which indicators consistently appear in
    top-performing strategies.
    """
    
    def __init__(self):
        self.indicator_stats: Dict[str, FeatureStats] = defaultdict(FeatureStats)
        self.operator_stats: Dict[str, FeatureStats] = defaultdict(FeatureStats)
        self.condition_pattern_stats: Dict[str, FeatureStats] = defaultdict(FeatureStats)
        self.generation_history: List[Dict[str, Any]] = []
        self.total_generations = 0
    
    def update(self, population) -> None:
        """
        Update feature importance data from a completed generation.
        
        Args:
            population: Evaluated Population object with fitness scores assigned.
        """
        individuals = [ind for ind in population if ind.evaluated and ind.fitness is not None]
        if len(individuals) < 5:
            return  # Not enough data for meaningful analysis
        
        self.total_generations += 1
        
        # Sort by fitness
        sorted_inds = sorted(individuals, key=lambda x: x.fitness or 0, reverse=True)
        n = len(sorted_inds)
        top_cutoff = max(1, int(n * 0.2))  # Top 20%
        bottom_cutoff = max(1, int(n * 0.2))  # Bottom 20%
        
        top_set = set(id(ind) for ind in sorted_inds[:top_cutoff])
        bottom_set = set(id(ind) for ind in sorted_inds[-bottom_cutoff:])
        
        # Track which features appear in each tier
        gen_indicator_counts = defaultdict(lambda: {'top': 0, 'bottom': 0, 'total': 0})
        
        for ind in sorted_inds:
            gene = ind.strategy_gene
            fitness = ind.fitness or 0
            is_top = id(ind) in top_set
            is_bottom = id(ind) in bottom_set
            
            # Track indicator types
            indicator_types_seen = set()
            for indicator in gene.indicators:
                ind_type = indicator.type
                indicator_types_seen.add(ind_type)
                
                stats = self.indicator_stats[ind_type]
                stats.appearances_total += 1
                stats.fitness_sum_when_present += fitness
                stats.fitness_count_when_present += 1
                if is_top:
                    stats.appearances_top += 1
                if is_bottom:
                    stats.appearances_bottom += 1
                
                gen_indicator_counts[ind_type]['total'] += 1
                if is_top:
                    gen_indicator_counts[ind_type]['top'] += 1
                if is_bottom:
                    gen_indicator_counts[ind_type]['bottom'] += 1
            
            # Track operator types
            for cond in gene.entry_conditions + gene.exit_conditions:
                op = cond.operator
                stats = self.operator_stats[op]
                stats.appearances_total += 1
                stats.fitness_sum_when_present += fitness
                stats.fitness_count_when_present += 1
                if is_top:
                    stats.appearances_top += 1
                if is_bottom:
                    stats.appearances_bottom += 1
            
            # Track condition patterns (indicator + operator combos)
            for cond in gene.entry_conditions:
                # Extract base type
                base_type = cond.indicator.split('_')[0] if '_' in cond.indicator else cond.indicator
                if base_type.startswith('CDL'):
                    base_type = cond.indicator  # Keep full CDL name
                pattern = f"ENTRY:{base_type}:{cond.operator}"
                stats = self.condition_pattern_stats[pattern]
                stats.appearances_total += 1
                stats.fitness_sum_when_present += fitness
                stats.fitness_count_when_present += 1
                if is_top:
                    stats.appearances_top += 1
                if is_bottom:
                    stats.appearances_bottom += 1
            
            for cond in gene.exit_conditions:
                base_type = cond.indicator.split('_')[0] if '_' in cond.indicator else cond.indicator
                if base_type.startswith('CDL'):
                    base_type = cond.indicator
                pattern = f"EXIT:{base_type}:{cond.operator}"
                stats = self.condition_pattern_stats[pattern]
                stats.appearances_total += 1
                stats.fitness_sum_when_present += fitness
                stats.fitness_count_when_present += 1
                if is_top:
                    stats.appearances_top += 1
                if is_bottom:
                    stats.appearances_bottom += 1
            
            # Update generations_seen
            for ind_type in indicator_types_seen:
                self.indicator_stats[ind_type].generations_seen = self.total_generations
        
        # Store generation snapshot
        top_indicators = sorted(
            gen_indicator_counts.items(),
            key=lambda x: x[1]['top'] - x[1]['bottom'],
            reverse=True
        )[:5]
        
        self.generation_history.append({
            'generation': self.total_generations,
            'population_size': n,
            'top_indicators': [(name, counts) for name, counts in top_indicators],
            'avg_fitness': sum(ind.fitness or 0 for ind in sorted_inds) / n,
            'best_fitness': sorted_inds[0].fitness if sorted_inds else 0,
        })
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive feature importance report.
        
        Returns:
            Dictionary with ranked features and statistics.
        """
        # Rank indicators by importance score
        ranked_indicators = sorted(
            self.indicator_stats.items(),
            key=lambda x: x[1].importance_score,
            reverse=True
        )
        
        # Rank operators
        ranked_operators = sorted(
            self.operator_stats.items(),
            key=lambda x: x[1].importance_score,
            reverse=True
        )
        
        # Rank condition patterns
        ranked_patterns = sorted(
            self.condition_pattern_stats.items(),
            key=lambda x: x[1].importance_score,
            reverse=True
        )
        
        report = {
            'total_generations_analyzed': self.total_generations,
            'indicators': [
                {
                    'name': name,
                    'importance_score': round(stats.importance_score, 4),
                    'avg_fitness': round(stats.avg_fitness_when_present, 4),
                    'top_20pct_appearances': stats.appearances_top,
                    'bottom_20pct_appearances': stats.appearances_bottom,
                    'total_appearances': stats.appearances_total,
                }
                for name, stats in ranked_indicators
            ],
            'operators': [
                {
                    'name': name,
                    'importance_score': round(stats.importance_score, 4),
                    'total_appearances': stats.appearances_total,
                }
                for name, stats in ranked_operators
            ],
            'top_condition_patterns': [
                {
                    'pattern': name,
                    'importance_score': round(stats.importance_score, 4),
                    'avg_fitness': round(stats.avg_fitness_when_present, 4),
                    'total_appearances': stats.appearances_total,
                }
                for name, stats in ranked_patterns[:20]  # Top 20 patterns
            ],
        }
        
        return report
    
    def log_summary(self, top_n: int = 10) -> None:
        """Log a summary of feature importance to the logger."""
        report = self.get_report()
        
        logger.info("=" * 60)
        logger.info("FEATURE IMPORTANCE REPORT")
        logger.info(f"Analyzed {report['total_generations_analyzed']} generations")
        logger.info("-" * 60)
        
        logger.info("\nTop Indicators (by importance score):")
        for ind in report['indicators'][:top_n]:
            logger.info(f"  {ind['name']:15s}  score={ind['importance_score']:+.4f}  "
                        f"avg_fit={ind['avg_fitness']:.4f}  "
                        f"top={ind['top_20pct_appearances']}  "
                        f"bot={ind['bottom_20pct_appearances']}  "
                        f"total={ind['total_appearances']}")
        
        logger.info("\nTop Condition Patterns:")
        for pat in report['top_condition_patterns'][:top_n]:
            logger.info(f"  {pat['pattern']:30s}  score={pat['importance_score']:+.4f}  "
                        f"avg_fit={pat['avg_fitness']:.4f}")
        
        logger.info("=" * 60)
    
    def get_indicator_weights(self) -> Dict[str, float]:
        """
        Get adaptive indicator sampling weights based on importance.
        
        Indicators with higher importance scores get sampled more frequently
        in random strategy generation and mutation.
        
        Returns:
            Dict mapping indicator type to weight (higher = sample more often).
        """
        if not self.indicator_stats:
            return {}
        
        weights = {}
        for name, stats in self.indicator_stats.items():
            # Base weight of 1.0, scaled by importance
            # Ensure no indicator gets weight below 0.3 (always some chance)
            score = stats.importance_score
            weights[name] = max(0.3, 1.0 + score * 2.0)
        
        return weights
