"""
Fitness evaluation and metrics.

Includes:
- FitnessEvaluator: Single-threaded strategy evaluation
- ParallelEvaluator: Multi-process parallel evaluation
- DirectBacktester: Direct FreqTrade API integration
- RegimeAwareEvaluator: Regime-balanced strategy evaluation
"""

from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.evaluation.parallel import (
    ParallelEvaluator,
    ParallelEvaluationResult,
    is_parallel_available,
    get_recommended_workers
)
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester, BacktestResult
from genetic_algorithm.evaluation.regime_aware import (
    RegimeAwareEvaluator,
    RegimeEvaluationResult,
    create_regime_aware_evaluator,
)

__all__ = [
    'FitnessEvaluator',
    'ParallelEvaluator',
    'ParallelEvaluationResult',
    'DirectBacktester',
    'BacktestResult',
    'is_parallel_available',
    'get_recommended_workers',
    'RegimeAwareEvaluator',
    'RegimeEvaluationResult',
    'create_regime_aware_evaluator',
]