"""
Overfitting Analysis Utilities

Provides structured overfitting detection, classification, and reporting.
Combines walk-forward train-val gap, holdout degradation, and Monte Carlo
robustness into a unified overfitting assessment per individual.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Overfitting Classification
# ============================================================================

# Classification labels
LABEL_SAFE = "SAFE"
LABEL_WARNING = "WARNING"
LABEL_OVERFIT = "OVERFIT"
LABEL_UNKNOWN = "UNKNOWN"


@dataclass
class OverfitThresholds:
    """Configurable thresholds for overfitting classification."""
    
    # Holdout degradation thresholds (fraction, e.g. 0.3 = 30%)
    holdout_degradation_warning: float = 0.25
    holdout_degradation_overfit: float = 0.50
    
    # Walk-forward train-val gap thresholds (fraction)
    wf_gap_warning: float = 0.15
    wf_gap_overfit: float = 0.30
    
    # Monte Carlo robustness thresholds (fraction of profitable permutations)
    mc_robustness_warning: float = 0.70
    mc_robustness_overfit: float = 0.50
    
    # Composite score thresholds (0-1 scale, higher = more overfit)
    composite_warning: float = 0.25
    composite_overfit: float = 0.50
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OverfitThresholds':
        """Load thresholds from config dict (overfit_analysis section)."""
        oa_config = config.get('overfit_analysis', {})
        thresholds = oa_config.get('thresholds', {})
        return cls(
            holdout_degradation_warning=thresholds.get('holdout_degradation_warning', 0.25),
            holdout_degradation_overfit=thresholds.get('holdout_degradation_overfit', 0.50),
            wf_gap_warning=thresholds.get('wf_gap_warning', 0.15),
            wf_gap_overfit=thresholds.get('wf_gap_overfit', 0.30),
            mc_robustness_warning=thresholds.get('mc_robustness_warning', 0.70),
            mc_robustness_overfit=thresholds.get('mc_robustness_overfit', 0.50),
            composite_warning=thresholds.get('composite_warning', 0.25),
            composite_overfit=thresholds.get('composite_overfit', 0.50),
        )


@dataclass
class OverfitAssessment:
    """Complete overfitting assessment for a single individual."""
    
    individual_id: str
    fitness: float
    
    # Holdout metrics
    holdout_fitness: Optional[float] = None
    holdout_degradation: Optional[float] = None
    holdout_profit: Optional[float] = None
    holdout_trades: Optional[int] = None
    holdout_drawdown: Optional[float] = None
    holdout_label: str = LABEL_UNKNOWN
    
    # Walk-forward metrics
    train_val_gap: Optional[float] = None
    wf_label: str = LABEL_UNKNOWN
    
    # Monte Carlo metrics
    mc_robustness: Optional[float] = None
    mc_mean_profit: Optional[float] = None
    mc_profit_p5: Optional[float] = None
    mc_profit_p95: Optional[float] = None
    mc_profit_std: Optional[float] = None
    mc_label: str = LABEL_UNKNOWN
    
    # Composite
    composite_score: Optional[float] = None  # 0 = no overfitting, 1 = severe
    overall_label: str = LABEL_UNKNOWN
    
    # Strategy details
    indicator_count: Optional[int] = None
    condition_count: Optional[int] = None
    profit: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    num_trades: Optional[int] = None
    win_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def classify_overfitting(
    metrics: Dict[str, Any],
    fitness: float,
    thresholds: Optional[OverfitThresholds] = None,
    strategy_gene=None,
) -> OverfitAssessment:
    """
    Classify a single individual's overfitting risk.
    
    Combines holdout degradation, walk-forward train-val gap, and Monte Carlo
    robustness into a unified assessment with a composite overfit score (0-1).
    
    Args:
        metrics: Individual's metrics dict (from individual.metrics)
        fitness: Individual's fitness score
        thresholds: Classification thresholds (uses defaults if None)
        strategy_gene: Optional StrategyGene for structural info
    
    Returns:
        OverfitAssessment with per-signal labels and composite score
    """
    if thresholds is None:
        thresholds = OverfitThresholds()
    
    assessment = OverfitAssessment(
        individual_id=metrics.get('id', 'unknown'),
        fitness=fitness,
        profit=metrics.get('profit', None),
        sharpe_ratio=metrics.get('sharpe_ratio', None),
        max_drawdown=metrics.get('max_drawdown', None),
        num_trades=metrics.get('num_trades', None),
        win_rate=metrics.get('win_rate', None),
    )
    
    # Strategy complexity from gene
    if strategy_gene is not None:
        assessment.indicator_count = len(getattr(strategy_gene, 'indicators', []))
        assessment.condition_count = len(getattr(strategy_gene, 'entry_conditions', []))
    
    signal_scores = []  # Each signal contributes a 0-1 score
    
    # --- Holdout Degradation ---
    holdout_fitness = metrics.get('holdout_fitness')
    holdout_degradation = metrics.get('holdout_degradation')
    
    if holdout_degradation is not None:
        assessment.holdout_fitness = holdout_fitness
        assessment.holdout_degradation = holdout_degradation
        assessment.holdout_profit = metrics.get('holdout_profit')
        assessment.holdout_trades = metrics.get('holdout_trades')
        assessment.holdout_drawdown = metrics.get('holdout_drawdown')
        
        if holdout_degradation >= thresholds.holdout_degradation_overfit:
            assessment.holdout_label = LABEL_OVERFIT
            signal_scores.append(min(holdout_degradation / thresholds.holdout_degradation_overfit, 1.0))
        elif holdout_degradation >= thresholds.holdout_degradation_warning:
            assessment.holdout_label = LABEL_WARNING
            signal_scores.append(holdout_degradation)
        else:
            assessment.holdout_label = LABEL_SAFE
            signal_scores.append(max(holdout_degradation, 0.0))
    
    # --- Walk-Forward Train-Val Gap ---
    train_val_gap = metrics.get('train_val_gap')
    
    if train_val_gap is not None:
        assessment.train_val_gap = train_val_gap
        
        if train_val_gap >= thresholds.wf_gap_overfit:
            assessment.wf_label = LABEL_OVERFIT
            signal_scores.append(min(train_val_gap / thresholds.wf_gap_overfit, 1.0))
        elif train_val_gap >= thresholds.wf_gap_warning:
            assessment.wf_label = LABEL_WARNING
            signal_scores.append(train_val_gap)
        else:
            assessment.wf_label = LABEL_SAFE
            signal_scores.append(max(train_val_gap, 0.0))
    
    # --- Monte Carlo Robustness ---
    mc_robustness = metrics.get('mc_robustness')
    
    if mc_robustness is not None:
        assessment.mc_robustness = mc_robustness
        assessment.mc_mean_profit = metrics.get('mc_mean_profit')
        assessment.mc_profit_p5 = metrics.get('mc_profit_p5')
        assessment.mc_profit_p95 = metrics.get('mc_profit_p95', metrics.get('mc_profit_std'))
        assessment.mc_profit_std = metrics.get('mc_profit_std')
        
        # Invert: low robustness = high overfitting risk
        mc_overfit_signal = 1.0 - mc_robustness
        
        if mc_robustness < thresholds.mc_robustness_overfit:
            assessment.mc_label = LABEL_OVERFIT
        elif mc_robustness < thresholds.mc_robustness_warning:
            assessment.mc_label = LABEL_WARNING
        else:
            assessment.mc_label = LABEL_SAFE
        
        signal_scores.append(mc_overfit_signal)
    
    # --- Composite Score (weighted average) ---
    # Holdout is the strongest overfitting signal; MC can mask problems when used
    # as equal-weight mean. Use explicit weights per signal type.
    if signal_scores:
        has_holdout = holdout_degradation is not None
        has_wf = train_val_gap is not None
        has_mc = mc_robustness is not None
        
        # Assign weights based on which signals are available
        weights = []
        if has_holdout and has_wf and has_mc:
            weights = [0.50, 0.30, 0.20]  # holdout, wf, mc
        elif has_holdout and has_mc:
            weights = [0.70, 0.30]  # holdout, mc
        elif has_holdout and has_wf:
            weights = [0.65, 0.35]  # holdout, wf
        elif has_wf and has_mc:
            weights = [0.55, 0.45]  # wf, mc
        else:
            weights = [1.0]  # single signal
        
        # Weights must match len(signal_scores)
        if len(weights) != len(signal_scores):
            # Fallback to equal weights if mismatch
            weights = [1.0 / len(signal_scores)] * len(signal_scores)
        
        assessment.composite_score = sum(w * s for w, s in zip(weights, signal_scores))
        
        # Hard override: severe holdout degradation with negative holdout profit
        # is definitive overfitting regardless of other signals
        if (has_holdout 
            and holdout_degradation >= thresholds.holdout_degradation_overfit
            and assessment.holdout_profit is not None 
            and assessment.holdout_profit < 0):
            assessment.overall_label = LABEL_OVERFIT
            assessment.composite_score = max(assessment.composite_score, 
                                              thresholds.composite_overfit)
        elif assessment.composite_score >= thresholds.composite_overfit:
            assessment.overall_label = LABEL_OVERFIT
        elif assessment.composite_score >= thresholds.composite_warning:
            assessment.overall_label = LABEL_WARNING
        else:
            assessment.overall_label = LABEL_SAFE
    else:
        assessment.composite_score = None
        assessment.overall_label = LABEL_UNKNOWN
    
    return assessment


# ============================================================================
# Generation-Level Holdout Tracking
# ============================================================================

@dataclass
class GenerationHoldoutStats:
    """Holdout metrics for a single generation's monitoring check."""
    generation: int
    avg_degradation: float
    best_degradation: float
    worst_degradation: float
    num_evaluated: int
    num_profitable: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Structured Report Export
# ============================================================================

def generate_detailed_results(
    top_strategies: list,
    config: Dict[str, Any],
    generation_holdout_history: Optional[List[GenerationHoldoutStats]] = None,
    generation_stats: Optional[list] = None,
    thresholds: Optional[OverfitThresholds] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive results dictionary for JSON export.
    
    Combines in-sample fitness, holdout validation, Monte Carlo robustness,
    walk-forward metrics, and overfitting classification into one report.
    
    Args:
        top_strategies: List of top Individual objects
        config: Full GA config dict
        generation_holdout_history: List of per-generation holdout stats
        generation_stats: List of PopulationStats (or dicts)
        thresholds: Overfitting classification thresholds
    
    Returns:
        Dict ready for JSON serialization
    """
    if thresholds is None:
        thresholds = OverfitThresholds.from_config(config)
    
    # Assess each individual
    assessments = []
    for rank, individual in enumerate(top_strategies, 1):
        metrics = getattr(individual, 'metrics', {})
        fitness = getattr(individual, 'fitness', 0.0)
        gene = getattr(individual, 'strategy_gene', None)
        
        assessment = classify_overfitting(
            metrics=metrics,
            fitness=fitness,
            thresholds=thresholds,
            strategy_gene=gene,
        )
        assessment.individual_id = getattr(individual, 'id', f'rank_{rank}')
        
        assessments.append({
            'rank': rank,
            'assessment': assessment.to_dict(),
        })
    
    # Build report
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'ga_version': '1.0',
            'config_summary': {
                'population_size': config.get('genetic_algorithm', {}).get('population_size'),
                'generations': config.get('genetic_algorithm', {}).get('generations'),
                'mutation_rate': config.get('genetic_algorithm', {}).get('mutation_rate'),
                'crossover_rate': config.get('genetic_algorithm', {}).get('crossover_rate'),
                'pairs': config.get('backtesting', {}).get('pairs', []),
                'timerange': config.get('backtesting', {}).get('timerange', ''),
                'walk_forward_enabled': config.get('walk_forward', {}).get('enabled', False),
                'holdout_enabled': config.get('holdout_validation', {}).get('enabled', False),
                'holdout_pct': config.get('holdout_validation', {}).get('holdout_pct'),
                'monte_carlo_enabled': config.get('monte_carlo', {}).get('enabled', False),
                'regime_aware': config.get('regime_aware', {}).get('enabled', False),
                'parallel_enabled': config.get('parallel_evaluation', {}).get('enabled', False),
            },
            'thresholds': asdict(thresholds),
        },
        'strategies': assessments,
        'summary': _compute_summary(assessments),
    }
    
    # Add generation holdout history if available
    if generation_holdout_history:
        report['holdout_history'] = [h.to_dict() for h in generation_holdout_history]
    
    # Add generation stats summary if available
    if generation_stats:
        report['generation_fitness_history'] = _extract_fitness_history(generation_stats)
    
    return report


def _compute_summary(assessments: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate summary across all assessed strategies."""
    if not assessments:
        return {'total': 0, 'safe': 0, 'warning': 0, 'overfit': 0, 'unknown': 0}
    
    labels = [a['assessment']['overall_label'] for a in assessments]
    composites = [a['assessment']['composite_score'] for a in assessments 
                  if a['assessment']['composite_score'] is not None]
    
    return {
        'total': len(assessments),
        'safe': labels.count(LABEL_SAFE),
        'warning': labels.count(LABEL_WARNING),
        'overfit': labels.count(LABEL_OVERFIT),
        'unknown': labels.count(LABEL_UNKNOWN),
        'avg_composite_score': sum(composites) / len(composites) if composites else None,
        'max_composite_score': max(composites) if composites else None,
    }


def _extract_fitness_history(generation_stats: list) -> List[Dict[str, Any]]:
    """Extract a compact fitness history from generation stats."""
    history = []
    for s in generation_stats:
        if isinstance(s, dict):
            entry = {
                'generation': s.get('generation'),
                'best_fitness': s.get('best_fitness'),
                'avg_fitness': s.get('avg_fitness'),
                'genetic_diversity': s.get('genetic_diversity'),
                'holdout_avg_degradation': s.get('holdout_avg_degradation'),
                'holdout_best_degradation': s.get('holdout_best_degradation'),
            }
        else:
            # PopulationStats dataclass
            entry = {
                'generation': getattr(s, 'generation', None),
                'best_fitness': getattr(s, 'best_fitness', None),
                'avg_fitness': getattr(s, 'avg_fitness', None),
                'genetic_diversity': getattr(s, 'genetic_diversity', None),
                'holdout_avg_degradation': getattr(s, 'holdout_avg_degradation', None),
                'holdout_best_degradation': getattr(s, 'holdout_best_degradation', None),
            }
        history.append(entry)
    return history


def save_detailed_results(
    report: Dict[str, Any],
    output_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """
    Save detailed results to JSON file.
    
    Args:
        report: Report dict from generate_detailed_results()
        output_dir: Directory to save to
        filename: Optional filename (auto-generated if None)
    
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_detailed_{timestamp}.json"
    
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Detailed results saved to: {filepath}")
    return filepath


def print_overfit_summary(assessments: List[OverfitAssessment], logger_instance=None):
    """
    Print a formatted overfitting summary to console and optionally to logger.
    
    Args:
        assessments: List of OverfitAssessment objects
        logger_instance: Optional logger to also log the output
    """
    log = logger_instance or logger
    
    header = f"\n{'='*80}\nOVERFITTING ANALYSIS SUMMARY\n{'='*80}"
    print(header)
    log.info(header.strip())
    
    print(f"{'Rank':<6} {'ID':<25} {'Fitness':>8} {'Holdout':>8} {'Degrad':>7} "
          f"{'WF-Gap':>7} {'MC-Rob':>7} {'Score':>6} {'Label':<8}")
    print("-" * 95)
    
    for i, a in enumerate(assessments, 1):
        holdout_str = f"{a.holdout_fitness:.4f}" if a.holdout_fitness is not None else "  N/A "
        degrad_str = f"{a.holdout_degradation:.1%}" if a.holdout_degradation is not None else "  N/A "
        wf_str = f"{a.train_val_gap:.1%}" if a.train_val_gap is not None else "  N/A "
        mc_str = f"{a.mc_robustness:.1%}" if a.mc_robustness is not None else "  N/A "
        score_str = f"{a.composite_score:.3f}" if a.composite_score is not None else " N/A "
        
        label_icon = {"SAFE": "✓", "WARNING": "⚠", "OVERFIT": "✗", "UNKNOWN": "?"}.get(a.overall_label, "?")
        
        line = (f"{i:<6} {a.individual_id:<25} {a.fitness:>8.4f} {holdout_str:>8} {degrad_str:>7} "
                f"{wf_str:>7} {mc_str:>7} {score_str:>6} {label_icon} {a.overall_label:<8}")
        print(line)
        log.info(line)
    
    # Summary counts
    labels = [a.overall_label for a in assessments]
    composites = [a.composite_score for a in assessments if a.composite_score is not None]
    
    summary = (f"\n  Total: {len(assessments)} | "
               f"SAFE: {labels.count(LABEL_SAFE)} | "
               f"WARNING: {labels.count(LABEL_WARNING)} | "
               f"OVERFIT: {labels.count(LABEL_OVERFIT)} | "
               f"UNKNOWN: {labels.count(LABEL_UNKNOWN)}")
    
    if composites:
        summary += f"\n  Avg composite score: {sum(composites)/len(composites):.3f} (0=no overfit, 1=severe)"
    
    print(summary)
    log.info(summary.strip())
    print()
