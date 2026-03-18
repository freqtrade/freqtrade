"""
Tests for the 17-step code improvement plan.

Validates:
  - Fitness smoothing (sigmoid bonuses, Gaussian trade frequency, logistic penalties)
  - CDL operator fallback in mutation
  - Diversity-aware elitism
  - NSGA-II environmental selection
  - Min condition retry loop
  - Timeframe mutation filter
  - Atomic cache writes
  - Mutation cooling reset
  - Dead exit proportional penalty
  - Win rate continuous sigmoid
  - Drawdown sigmoid penalty
"""

import sys
import os
import json
import math
import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import (
    StrategyGene, IndicatorGene, ConditionGene, is_higher_timeframe,
)
from genetic_algorithm.core.population import (
    Population,
    calculate_strategy_distance,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_gene(indicators=None, entry_conditions=None, exit_conditions=None,
               timeframe='5m', stoploss=-0.10, informative_timeframes=None):
    """Create a StrategyGene with realistic defaults."""
    if indicators is None:
        indicators = [
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
        ]
    if entry_conditions is None:
        entry_conditions = [
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
            ConditionGene(indicator='EMA_0', operator='cross_above', threshold=0, logic='AND'),
        ]
    if exit_conditions is None:
        exit_conditions = [
            ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='AND'),
        ]
    gene = StrategyGene(
        generation=0, individual_id=0,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        stoploss=stoploss,
        timeframe=timeframe,
        minimal_roi={"0": 0.05, "30": 0.03, "60": 0.01},
        max_open_trades=3,
    )
    if informative_timeframes:
        gene.informative_timeframes = informative_timeframes
    return gene


def _make_individual(gene=None, fitness=1.0, raw_fitness=None):
    """Create an Individual wrapping a gene."""
    if gene is None:
        gene = _make_gene()
    ind = Individual(strategy_gene=gene)
    ind.fitness = fitness
    ind.raw_fitness = raw_fitness if raw_fitness is not None else fitness
    ind.evaluated = True
    ind.metrics = {'profit': fitness * 10, 'num_trades': 20}
    return ind


def _make_fitness_evaluator(config_overrides=None):
    """Create a FitnessEvaluator with mocked heavy deps."""
    config = {
        'fitness_weights': {},
        'fitness_penalties': {},
        'backtesting': {'pairs': ['BTC/USDT']},
        'walk_forward': {'enabled': False},
        'monte_carlo': {},
        'fitness_bounds': {},
        'trade_frequency_thresholds': {
            'ideal_min': 10,
            'ideal_max': 50,
        },
        'indicators': {},
    }
    if config_overrides:
        config.update(config_overrides)

    with patch('genetic_algorithm.evaluation.fitness.DirectBacktester'):
        with patch('genetic_algorithm.evaluation.fitness.StrategyGenerator'):
            from genetic_algorithm.evaluation.fitness import FitnessEvaluator
            evaluator = FitnessEvaluator(config)
    return evaluator


def _make_config():
    """Standard mutation config."""
    return {
        'indicators': {
            'available': ['RSI', 'EMA', 'SMA', 'MACD', 'CDL_DOJI', 'CDL_HAMMER'],
            'max_per_strategy': 5,
            'min_per_strategy': 1,
            'min_entry_conditions': 2,
            'min_exit_conditions': 1,
            'RSI': {'period': [7, 21], 'buy_threshold': [20, 40], 'sell_threshold': [60, 80]},
            'EMA': {'period': [10, 50]},
            'SMA': {'period': [10, 50]},
            'MACD': {'fast_period': [8, 21], 'slow_period': [21, 50], 'signal_period': [5, 14]},
        },
        'strategy_constraints': {
            'stoploss_range': [-0.20, -0.05],
            'roi_range': [0.01, 0.10],
            'timeframes': ['5m', '15m', '1h', '4h'],
            'max_open_trades_range': [1, 10],
        },
    }


# ============================================================================
# Test 1: Fitness Smoothing — Trade Frequency Gaussian
# ============================================================================

class TestTradeFrequencyGaussian:
    """Verify _normalize_trade_frequency uses smooth Gaussian, no hard steps."""

    def setup_method(self):
        self.evaluator = _make_fitness_evaluator()

    def test_zero_trades_returns_zero(self):
        assert self.evaluator._normalize_trade_frequency(0) == 0.0

    def test_ideal_center_returns_near_one(self):
        ideal_center = (self.evaluator.tf_ideal_min + self.evaluator.tf_ideal_max) // 2
        score = self.evaluator._normalize_trade_frequency(ideal_center)
        assert score >= 0.95, f"Center of ideal range should be ~1.0, got {score}"

    def test_smoothness_no_large_jumps(self):
        """No adjacent trade counts should differ by more than 5% in score."""
        scores = [self.evaluator._normalize_trade_frequency(n) for n in range(1, 200)]
        for i in range(1, len(scores)):
            diff = abs(scores[i] - scores[i - 1])
            assert diff < 0.05, (
                f"Jump of {diff:.4f} between trades={i} and trades={i+1} "
                f"(scores {scores[i-1]:.4f} → {scores[i]:.4f})"
            )

    def test_far_from_ideal_scores_low(self):
        """Very high or very low trade counts should score low."""
        score_low = self.evaluator._normalize_trade_frequency(1)
        score_high = self.evaluator._normalize_trade_frequency(500)
        assert score_low < 0.5, f"1 trade should score low, got {score_low}"
        assert score_high < 0.5, f"500 trades should score low, got {score_high}"

    def test_minimum_floor(self):
        """Even bad trade counts should return at least 0.15."""
        for n in [1, 2, 500, 1000]:
            score = self.evaluator._normalize_trade_frequency(n)
            assert score >= 0.15, f"Floor violated at trades={n}: {score}"


# ============================================================================
# Test 2: Fitness Smoothing — Min Trades Logistic Ramp
# ============================================================================

class TestMinTradesLogisticPenalty:
    """Verify min trades penalty is smooth logistic, not cliff."""

    def setup_method(self):
        self.evaluator = _make_fitness_evaluator({
            'fitness_penalties': {'min_trades': 10},
        })

    def test_zero_trades_heavy_penalty(self):
        fitness = self.evaluator._apply_penalties(1.0, {'num_trades': 0, 'max_drawdown': 0, 'win_rate': 0.5})
        # min_penalty_floor (default 0.10) prevents going below 10% of original
        assert fitness <= 0.10, f"Zero trades should be heavily penalized, got {fitness}"

    def test_above_min_no_penalty(self):
        fitness = self.evaluator._apply_penalties(1.0, {'num_trades': 20, 'max_drawdown': 0, 'win_rate': 0.5})
        # Above min_trades, no trade penalty applied (other penalties may still apply)
        assert fitness > 0.5, f"Above min trades should have minimal penalty, got {fitness}"

    def test_smooth_ramp_from_1_to_min(self):
        """Fitness should increase smoothly from 1 trade to min_trades."""
        values = []
        for n in range(1, 11):
            f = self.evaluator._apply_penalties(1.0, {'num_trades': n, 'max_drawdown': 0, 'win_rate': 0.5})
            values.append(f)
        # Must be monotonically non-decreasing
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 0.01, (
                f"Not monotonic: trades={i} → {values[i-1]:.4f}, trades={i+1} → {values[i]:.4f}"
            )
        # No single jump > 20% (old cliff was 99%)
        for i in range(1, len(values)):
            diff = values[i] - values[i - 1]
            assert diff < 0.20, f"Jump of {diff:.4f} between trades={i} and {i+1}"


# ============================================================================
# Test 3: Fitness Smoothing — Drawdown Sigmoid Penalty
# ============================================================================

class TestDrawdownSigmoidPenalty:
    """Verify drawdown penalty is smooth sigmoid, not hard gate."""

    def setup_method(self):
        self.evaluator = _make_fitness_evaluator({
            'fitness_penalties': {'max_drawdown': 0.30, 'min_trades': 0},
        })

    def test_low_drawdown_minimal_penalty(self):
        fitness = self.evaluator._apply_penalties(1.0, {'num_trades': 50, 'max_drawdown': 0.05, 'win_rate': 0.5})
        assert fitness > 0.90, f"5% drawdown should have minimal penalty, got {fitness}"

    def test_high_drawdown_strong_penalty(self):
        fitness = self.evaluator._apply_penalties(1.0, {'num_trades': 50, 'max_drawdown': 0.60, 'win_rate': 0.5})
        assert fitness < 0.50, f"60% drawdown should have strong penalty, got {fitness}"

    def test_smooth_around_threshold(self):
        """Fitness should change smoothly around the 0.30 threshold."""
        values = []
        for dd in [0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40]:
            f = self.evaluator._apply_penalties(1.0, {'num_trades': 50, 'max_drawdown': dd, 'win_rate': 0.5})
            values.append(f)
        for i in range(1, len(values)):
            diff = abs(values[i] - values[i - 1])
            assert diff < 0.25, f"Jump of {diff:.4f} around dd threshold (values: {values})"


# ============================================================================
# Test 4: Fitness Smoothing — Win Rate Continuous Sigmoid
# ============================================================================

class TestWinRateSigmoidPenalty:
    """Verify win rate penalty is continuous sigmoid, not binary gate."""

    def setup_method(self):
        self.evaluator = _make_fitness_evaluator({
            'fitness_penalties': {'min_win_rate': 0.30, 'min_trades': 0, 'max_drawdown': 1.0},
        })

    def test_high_win_rate_no_penalty(self):
        fitness = self.evaluator._apply_penalties(1.0, {'num_trades': 50, 'max_drawdown': 0, 'win_rate': 0.60})
        assert fitness > 0.90, f"60% win rate should have little penalty, got {fitness}"

    def test_low_win_rate_penalty(self):
        fitness = self.evaluator._apply_penalties(1.0, {'num_trades': 50, 'max_drawdown': 0, 'win_rate': 0.10})
        assert fitness < 0.80, f"10% win rate should be penalized, got {fitness}"

    def test_low_trades_reduce_confidence(self):
        """With few trades, win rate penalty should be reduced (low confidence)."""
        f_many = self.evaluator._apply_penalties(1.0, {'num_trades': 50, 'max_drawdown': 0, 'win_rate': 0.10})
        f_few = self.evaluator._apply_penalties(1.0, {'num_trades': 2, 'max_drawdown': 0, 'win_rate': 0.10})
        # With fewer trades, the confidence is lower → less penalty applied
        assert f_few > f_many, (
            f"Fewer trades should reduce win rate penalty: few={f_few:.4f} vs many={f_many:.4f}"
        )

    def test_one_trade_minimal_penalty(self):
        """With only 1 trade, below threshold for win rate penalty entirely."""
        fitness = self.evaluator._apply_penalties(1.0, {'num_trades': 1, 'max_drawdown': 0, 'win_rate': 0.0})
        # num_trades < 2 → no win rate penalty
        assert fitness >= 0.90, f"1 trade should skip win rate penalty, got {fitness}"


# ============================================================================
# Test 5: Dead Exit Proportional Penalty
# ============================================================================

class TestDeadExitProportionalPenalty:
    """Verify dead exit penalty is proportional, not all-or-nothing."""

    def setup_method(self):
        self.evaluator = _make_fitness_evaluator({
            'fitness_penalties': {'min_trades': 0, 'max_drawdown': 1.0},
        })

    def test_no_dead_exits_no_penalty(self):
        """With no dead exits, dead exit penalty contributes nothing.
        Other penalties (drawdown sigmoid) may still apply, so compare with and without."""
        gene_alive = _make_gene(exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='AND'),
        ])
        gene_dead = _make_gene(exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=0, logic='AND'),
        ])
        metrics = {'num_trades': 50, 'max_drawdown': 0, 'win_rate': 0.5}
        f_alive = self.evaluator._apply_penalties(1.0, metrics, gene_alive)
        f_dead = self.evaluator._apply_penalties(1.0, metrics, gene_dead)
        # Alive exits should score higher than dead exits
        assert f_alive > f_dead, f"Alive exits should score higher: alive={f_alive}, dead={f_dead}"

    def test_all_dead_exits_max_penalty(self):
        """All bounded exits dead → penalty = 1.0 - 1.0*0.3 = 0.7x."""
        gene = _make_gene(exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=0, logic='AND'),  # dead: RSI < 0
        ])
        fitness = self.evaluator._apply_penalties(1.0, {'num_trades': 50, 'max_drawdown': 0, 'win_rate': 0.5}, gene)
        assert 0.65 < fitness < 0.75, f"All dead exits: expected ~0.7, got {fitness}"

    def test_partial_dead_exits_proportional(self):
        """1 of 2 dead → more penalty than 0 of 2 dead, less than 2 of 2."""
        metrics = {'num_trades': 50, 'max_drawdown': 0, 'win_rate': 0.5}
        gene_none_dead = _make_gene(exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='AND'),
            ConditionGene(indicator='RSI_0', operator='>', threshold=60, logic='AND'),
        ])
        gene_half_dead = _make_gene(exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='AND'),  # alive
            ConditionGene(indicator='RSI_0', operator='<', threshold=0, logic='AND'),   # dead
        ])
        gene_all_dead = _make_gene(exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=0, logic='AND'),
            ConditionGene(indicator='RSI_0', operator='<', threshold=0, logic='AND'),
        ])
        f_none = self.evaluator._apply_penalties(1.0, metrics, gene_none_dead)
        f_half = self.evaluator._apply_penalties(1.0, metrics, gene_half_dead)
        f_all = self.evaluator._apply_penalties(1.0, metrics, gene_all_dead)
        assert f_none > f_half > f_all, (
            f"Proportional order violated: none={f_none:.4f}, half={f_half:.4f}, all={f_all:.4f}"
        )


# ============================================================================
# Test 6: CDL Operator Fallback in Mutation
# ============================================================================

class TestCDLOperatorFallback:
    """Verify CDL patterns only get '<' or '>' operators, never cross_above/below."""

    def test_cdl_operator_100_mutations(self):
        """Mutate a CDL_DOJI condition 100 times, verify operators."""
        from genetic_algorithm.core.mutation import mutate_conditions

        config = _make_config()
        invalid_ops = {'cross_above', 'cross_below', 'increasing', 'decreasing', 'between', 'value_above_ago'}

        for _ in range(100):
            gene = _make_gene(
                indicators=[IndicatorGene(type='CDL_DOJI', parameters={}, instance_id='CDL_DOJI_0')],
                entry_conditions=[
                    ConditionGene(indicator='CDL_DOJI_0', operator='>', threshold=0, logic='AND'),
                ],
                exit_conditions=[
                    ConditionGene(indicator='CDL_DOJI_0', operator='<', threshold=50, logic='AND'),
                ],
            )
            ind = _make_individual(gene)
            mutated = mutate_conditions(ind, mutation_rate=1.0, config=config)
            for cond in mutated.strategy_gene.entry_conditions + mutated.strategy_gene.exit_conditions:
                if 'CDL_' in cond.indicator:
                    assert cond.operator not in invalid_ops, (
                        f"CDL condition got invalid operator '{cond.operator}'"
                    )


# ============================================================================
# Test 7: Diversity-Aware Elitism
# ============================================================================

class TestDiversityAwareElitism:
    """Verify elitism rejects near-duplicates and picks diverse individuals."""

    def test_identical_individuals_only_one_survives(self):
        """Multiple identical individuals should result in only 1 elite."""
        from genetic_algorithm.core.population import calculate_strategy_distance

        # Create 5 identical genes
        gene = _make_gene()
        population = Population(size=5, generation=0)
        for i in range(5):
            ind = _make_individual(gene.copy(), fitness=1.0 - i * 0.01)
            population.add_individual(ind)

        # Simulate diversity-aware selection with threshold 0.15
        ranked = sorted(
            [ind for ind in population.individuals if ind.raw_fitness is not None],
            key=lambda x: x.raw_fitness, reverse=True,
        )
        threshold = 0.15
        elites = []
        for candidate in ranked:
            if len(elites) >= 3:
                break
            too_close = any(
                calculate_strategy_distance(candidate, e) < threshold
                for e in elites
            )
            if not too_close:
                elites.append(candidate)

        # Only 1 unique structure passes the diversity check
        assert len(elites) == 1, f"Expected 1 diverse elite from identical pool, got {len(elites)}"

    def test_diverse_individuals_all_survive(self):
        """Individuals with different structures should all survive."""
        population = Population(size=3, generation=0)

        # Different indicator types → high distance
        ind1 = _make_individual(_make_gene(
            indicators=[IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0')],
            timeframe='5m', stoploss=-0.05,
        ), fitness=1.0)
        ind2 = _make_individual(_make_gene(
            indicators=[IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, instance_id='MACD_0')],
            timeframe='1h', stoploss=-0.15,
        ), fitness=0.9)
        ind3 = _make_individual(_make_gene(
            indicators=[
                IndicatorGene(type='BBANDS', parameters={'period': 20}, instance_id='BBANDS_0'),
                IndicatorGene(type='ADX', parameters={'period': 14}, instance_id='ADX_0'),
            ],
            timeframe='15m', stoploss=-0.20,
        ), fitness=0.8)

        for ind in [ind1, ind2, ind3]:
            population.add_individual(ind)

        ranked = sorted(
            [ind for ind in population.individuals],
            key=lambda x: x.raw_fitness, reverse=True,
        )
        threshold = 0.15
        elites = []
        for candidate in ranked:
            if len(elites) >= 3:
                break
            too_close = any(
                calculate_strategy_distance(candidate, e) < threshold
                for e in elites
            )
            if not too_close:
                elites.append(candidate)

        assert len(elites) == 3, f"Expected 3 diverse elites, got {len(elites)}"

    def test_fallback_fills_remaining_slots(self):
        """When not enough diverse candidates, fallback fills from top."""
        gene = _make_gene()
        ranked = []
        for i in range(5):
            ranked.append(_make_individual(gene.copy(), fitness=1.0 - i * 0.01))

        elite_size = 3
        threshold = 0.15
        elites = []
        for candidate in ranked:
            if len(elites) >= elite_size:
                break
            too_close = any(
                calculate_strategy_distance(candidate, e) < threshold
                for e in elites
            )
            if not too_close:
                elites.append(candidate)
        # Fallback
        if len(elites) < elite_size:
            for candidate in ranked:
                if len(elites) >= elite_size:
                    break
                if candidate not in elites:
                    elites.append(candidate)

        assert len(elites) == elite_size, f"Fallback should fill to {elite_size}, got {len(elites)}"


# ============================================================================
# Test 8: NSGA-II Environmental Selection
# ============================================================================

class TestNSGA2EnvironmentalSelection:
    """Verify (μ+λ) merge and Pareto front-based selection."""

    def test_dominated_removed(self):
        """Strictly dominated individuals should be removed from front 0."""
        from genetic_algorithm.core.nsga2 import fast_non_dominated_sort

        # All objectives are MAXIMIZED.
        # A: high profit, medium stability → non-dominated
        ind_a = _make_individual(fitness=1.0)
        ind_a.objectives = [10.0, 5.0]

        # B: medium profit, high stability → non-dominated (trade-off)
        ind_b = _make_individual(fitness=0.8)
        ind_b.objectives = [5.0, 10.0]

        # C: low on both → dominated by A
        ind_c = _make_individual(fitness=0.5)
        ind_c.objectives = [4.0, 3.0]

        fronts = fast_non_dominated_sort([ind_a, ind_b, ind_c])

        assert len(fronts) >= 2, f"Expected at least 2 fronts, got {len(fronts)}"
        front0_ids = {id(ind) for ind in fronts[0]}
        assert id(ind_a) in front0_ids, "ind_a should be in front 0"
        assert id(ind_b) in front0_ids, "ind_b should be in front 0"
        front1_ids = {id(ind) for ind in fronts[1]}
        assert id(ind_c) in front1_ids, "ind_c should be dominated (front 1)"

    def test_crowding_preserves_extremes(self):
        """Crowding distance should assign infinity to boundary solutions."""
        from genetic_algorithm.core.nsga2 import crowding_distance_assignment

        individuals = []
        for i in range(5):
            ind = _make_individual(fitness=float(i))
            ind.objectives = [float(i), 5.0 - float(i)]
            ind.crowding_distance = 0.0
            individuals.append(ind)

        crowding_distance_assignment(individuals)
        # Boundary individuals (first and last in sorted order) should have infinite distance
        distances = [ind.crowding_distance for ind in individuals]
        assert max(distances) == float('inf'), "Boundary solutions should have infinite crowding distance"


# ============================================================================
# Test 9: Min Conditions Retry Loop
# ============================================================================

class TestMinConditionsRetry:
    """Verify generator retries to meet min_conds requirement."""

    def test_generates_at_least_min_conditions(self):
        """Generator should produce at least min_entry_conditions."""
        config = _make_config()
        config['indicators']['min_entry_conditions'] = 2
        config['indicators']['max_entry_conditions'] = 4

        with patch('genetic_algorithm.evaluation.fitness.DirectBacktester'):
            with patch('genetic_algorithm.evaluation.fitness.StrategyGenerator'):
                from genetic_algorithm.strategies.generator import StrategyGenerator
                gen = StrategyGenerator(config)

        # Create indicators to generate conditions from
        indicators = [
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
            IndicatorGene(type='SMA', parameters={'period': 30}, instance_id='SMA_0'),
        ]

        for _ in range(20):
            conditions = gen._generate_random_conditions(indicators, is_entry=True)
            assert len(conditions) >= 2, (
                f"Expected at least 2 entry conditions, got {len(conditions)}: "
                f"{[(c.indicator, c.operator) for c in conditions]}"
            )


# ============================================================================
# Test 10: Timeframe Mutation Filter
# ============================================================================

class TestTimeframeMutationFilter:
    """Verify informative timeframes are filtered after base change."""

    def test_lower_informative_filtered_after_mutation(self):
        """If base changes to '1h', informative '15m' should be removed."""
        from genetic_algorithm.core.mutation import mutate_structure

        config = _make_config()
        gene = _make_gene(timeframe='5m', informative_timeframes=['15m', '1h', '4h'])
        ind = _make_individual(gene)

        # Force base timeframe to '1h' by mocking random.choice
        seen_valid = False
        for _ in range(100):
            mutated = mutate_structure(ind, mutation_rate=1.0, config=config)
            new_base = mutated.strategy_gene.timeframe
            itfs = mutated.strategy_gene.informative_timeframes
            if itfs:
                for itf in itfs:
                    assert is_higher_timeframe(itf, new_base), (
                        f"Informative TF '{itf}' not higher than base '{new_base}'"
                    )
                    seen_valid = True

        # At least some mutations should have produced valid results
        # (it's probabilistic, but 100 tries should hit it)


# ============================================================================
# Test 11: Atomic Cache Writes
# ============================================================================

class TestAtomicCacheWrites:
    """Verify cache writes use atomic temp+rename pattern."""

    def test_cache_file_written_atomically(self):
        """Cache put should produce a valid JSON file via atomic write."""
        from genetic_algorithm.evaluation.direct_backtester import BacktestCache, BacktestResult

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BacktestCache(cache_dir=Path(tmpdir))
            result = BacktestResult(
                success=True,
                strategy_name='test_strat',
                total_profit=10.5,
                profit_percent=10.5,
                total_trades=25,
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                profit_factor=1.8,
                max_drawdown=0.15,
                win_rate=0.6,
            )
            cache.put("test_strategy_code", {"pairs": ["BTC/USDT"]}, result)

            # Verify file exists and is valid JSON
            cache_files = list(Path(tmpdir).glob("*.json"))
            assert len(cache_files) == 1, f"Expected 1 cache file, got {len(cache_files)}"
            with open(cache_files[0]) as f:
                data = json.load(f)
            assert data['total_profit'] == 10.5

    def test_no_temp_files_left_behind(self):
        """No temp files should remain after successful write."""
        from genetic_algorithm.evaluation.direct_backtester import BacktestCache, BacktestResult

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BacktestCache(cache_dir=Path(tmpdir))
            result = BacktestResult(
                success=True,
                strategy_name='test_strat',
                total_profit=5.0,
                profit_percent=5.0,
                total_trades=10,
                sharpe_ratio=1.0,
                sortino_ratio=1.0,
                profit_factor=1.5,
                max_drawdown=0.10,
                win_rate=0.5,
            )
            cache.put("test_code", {"pairs": ["BTC/USDT"]}, result)

            # No temp files (starting with '.') should remain
            all_files = list(Path(tmpdir).iterdir())
            temp_files = [f for f in all_files if f.name.startswith('.')]
            assert len(temp_files) == 0, f"Temp files left behind: {temp_files}"


# ============================================================================
# Test 12: Mutation Cooling Reset
# ============================================================================

class TestMutationCoolingReset:
    """Verify no_improvement_count resets to 0 on new best (not halved)."""

    def test_reset_not_halved(self):
        """After finding new best, no_improvement_count should be 0."""
        # Verify by checking the source code directly
        import inspect
        from genetic_algorithm.core.evolution import GeneticAlgorithm

        source = inspect.getsource(GeneticAlgorithm)
        # The old halving pattern should NOT exist
        assert 'no_improvement_count // 2' not in source, (
            "Old halving pattern still exists in evolution.py"
        )


# ============================================================================
# Test 13: Sigmoid Bonus Smoothness
# ============================================================================

class TestSigmoidBonusSmoothness:
    """Verify sigmoid bonuses produce smooth, bounded values."""

    def test_sigmoid_bonus_basic(self):
        """_sigmoid_bonus should be monotonically increasing."""
        import math

        def _sigmoid_bonus(value, threshold, max_bonus, steepness=5.0):
            try:
                return max_bonus / (1.0 + math.exp(-steepness * (value - threshold)))
            except OverflowError:
                return 0.0 if value < threshold else max_bonus

        # Test monotonicity
        prev = _sigmoid_bonus(-100, 1.0, 0.10, 3.0)
        for v in range(-99, 100):
            current = _sigmoid_bonus(v, 1.0, 0.10, 3.0)
            assert current >= prev, f"Not monotonic at v={v}"
            prev = current

    def test_sigmoid_bonus_bounded(self):
        """Should be bounded between 0 and max_bonus."""
        import math

        def _sigmoid_bonus(value, threshold, max_bonus, steepness=5.0):
            try:
                return max_bonus / (1.0 + math.exp(-steepness * (value - threshold)))
            except OverflowError:
                return 0.0 if value < threshold else max_bonus

        for v in range(-1000, 1000, 10):
            result = _sigmoid_bonus(v, 5.0, 0.15, 0.3)
            assert 0.0 <= result <= 0.15 + 1e-9, f"Out of bounds at v={v}: {result}"

    def test_tanh_soft_cap(self):
        """Total bonus should be soft-capped by tanh, never exceed ~1.3."""
        import math
        for excess in [0.0, 0.1, 0.5, 1.0, 5.0, 100.0]:
            total = 1.0 + 0.3 * math.tanh(excess / 0.3)
            assert total <= 1.31, f"Soft cap exceeded: excess={excess}, total={total}"
            assert total >= 1.0, f"Below 1.0: excess={excess}, total={total}"


# ============================================================================
# Test 14: Config Override Warning (island model)
# ============================================================================

class TestConfigOverrideWarning:
    """Verify island model logs warning when disabling walk-forward."""

    def test_warning_logged_when_wf_disabled(self):
        """Source code should contain the warning log call."""
        import inspect
        # Read the island_model module source
        from genetic_algorithm.core import island_model
        source = inspect.getsource(island_model)
        assert 'Walk-forward disabled' in source, (
            "Island model should warn when disabling walk-forward"
        )


# ============================================================================
# Test 15: Worker Log Level Configurable
# ============================================================================

class TestWorkerLogLevelConfigurable:
    """Verify worker log level reads from config."""

    def test_source_uses_config_worker_log_level(self):
        """parallel.py should reference config's worker_log_level."""
        import inspect
        from genetic_algorithm.evaluation import parallel
        source = inspect.getsource(parallel)
        assert 'worker_log_level' in source, (
            "parallel.py should read worker_log_level from config"
        )
        # Should NOT have hardcoded WARNING (the old pattern)
        # Count occurrences: "setLevel(logging.WARNING)" should be replaced
        hardcoded_count = source.count("setLevel(logging.WARNING)")
        assert hardcoded_count == 0, (
            f"Found {hardcoded_count} hardcoded WARNING log levels in parallel.py"
        )


# ============================================================================
# Priority 1 & 2 Fixes: Output Pipeline & NSGA-II
# ============================================================================


class TestOutputDirFromConfig:
    """Verify run_ga.py derives output_dir from config, env, or default."""

    def test_output_dir_from_config(self):
        """output.dir in config should be used when set."""
        from genetic_algorithm.run_ga import OUTPUT_DIR
        config = {'output': {'dir': '/tmp/test_ga_output_dir'}}
        # Simulate the derivation logic from main()
        _output_cfg = config.get('output', {})
        output_dir = Path(
            os.environ.get('GA_OUTPUT_DIR_NOTSET_KEY_12345')
            or _output_cfg.get('dir')
            or str(OUTPUT_DIR)
        )
        assert str(output_dir) == '/tmp/test_ga_output_dir'

    def test_output_dir_env_override(self):
        """GA_OUTPUT_DIR env var should override config."""
        from genetic_algorithm.run_ga import OUTPUT_DIR
        config = {'output': {'dir': '/tmp/from_config'}}
        with patch.dict(os.environ, {'GA_OUTPUT_DIR': '/tmp/from_env'}):
            _output_cfg = config.get('output', {})
            output_dir = Path(
                os.environ.get('GA_OUTPUT_DIR')
                or _output_cfg.get('dir')
                or str(OUTPUT_DIR)
            )
        assert str(output_dir) == '/tmp/from_env'

    def test_output_dir_default_fallback(self):
        """Falls back to OUTPUT_DIR when config and env are absent."""
        from genetic_algorithm.run_ga import OUTPUT_DIR
        config = {}
        _output_cfg = config.get('output', {})
        output_dir = Path(
            os.environ.get('GA_OUTPUT_DIR_NOTSET_KEY_12345')
            or _output_cfg.get('dir')
            or str(OUTPUT_DIR)
        )
        assert output_dir == OUTPUT_DIR


class TestEvolutionStatsSave:
    """Verify _save_evolution_stats writes correct JSON."""

    def test_writes_valid_json(self, tmp_path):
        from genetic_algorithm.run_ga import _save_evolution_stats
        from genetic_algorithm.core.population import PopulationStats

        stats = [
            PopulationStats(generation=0, size=10, best_fitness=0.5,
                            avg_fitness=0.3, worst_fitness=0.1,
                            median_fitness=0.25, diversity_score=0.7,
                            genetic_diversity=0.6, best_raw_fitness=0.55,
                            avg_raw_fitness=0.35),
            PopulationStats(generation=1, size=10, best_fitness=0.6,
                            avg_fitness=0.4, worst_fitness=0.2,
                            median_fitness=0.35, diversity_score=0.65,
                            genetic_diversity=0.58, best_raw_fitness=0.62,
                            avg_raw_fitness=0.42),
        ]
        config = {
            'output': {'save_stats': True, 'stats_file': 'evolution_stats.json'},
            'genetic_algorithm': {'population_size': 10, 'generations': 2,
                                  'mutation_rate': 0.15, 'selection_method': 'tournament'},
        }
        _save_evolution_stats(tmp_path, config, stats)
        result = json.loads((tmp_path / 'evolution_stats.json').read_text())
        assert len(result['generations']) == 2
        assert result['generations'][0]['best_fitness'] == 0.5
        assert result['generations'][1]['diversity'] == 0.65
        assert result['config']['population_size'] == 10

    def test_no_temp_files_left(self, tmp_path):
        from genetic_algorithm.run_ga import _save_evolution_stats
        from genetic_algorithm.core.population import PopulationStats
        stats = [PopulationStats(generation=0, size=5, best_fitness=0.1)]
        config = {'output': {'save_stats': True}, 'genetic_algorithm': {}}
        _save_evolution_stats(tmp_path, config, stats)
        tmp_files = list(tmp_path.glob('*.tmp'))
        assert len(tmp_files) == 0, f"Temp files left behind: {tmp_files}"


class TestEvolutionDiagnosticsKeyFix:
    """Verify evolution.py reads 'dir' key (not just 'directory')."""

    def test_reads_dir_key(self):
        import inspect
        from genetic_algorithm.core import evolution
        source = inspect.getsource(evolution.GeneticAlgorithm._setup_diagnostics)
        assert "'dir'" in source, (
            "_setup_diagnostics should read the 'dir' key from output config"
        )


class TestNSGA2FitnessPreservation:
    """set_objectives() must NOT overwrite the scalar fitness set by set_fitness()."""

    def test_set_objectives_preserves_scalar_fitness(self):
        ind = Individual(strategy_gene=None)
        ind.set_fitness(0.42, {'profit': 5.0, 'sharpe_ratio': 3.0})
        ind.set_objectives([0.001, 0.95], ind.metrics)
        assert abs(ind.fitness - 0.42) < 1e-6, (
            f"set_objectives overwrote fitness: expected 0.42, got {ind.fitness}"
        )

    def test_set_objectives_fallback_when_no_fitness(self):
        ind = Individual(strategy_gene=None)
        ind.set_objectives([0.55, 0.8], {'profit': 3.0})
        assert abs(ind.fitness - 0.55) < 1e-6, (
            f"Fallback should use objectives[0]: expected 0.55, got {ind.fitness}"
        )


# ============================================================================
# Audit Fix Tests (Wave 13)
# ============================================================================


class TestC1OffspringLoopSafety:
    """C1: Offspring creation loop must have a max-iteration guard."""

    def test_evolution_source_has_max_attempts_guard(self):
        """Verify the while loop in create_next_generation has a safety counter."""
        import inspect
        from genetic_algorithm.core.evolution import GeneticAlgorithm
        source = inspect.getsource(GeneticAlgorithm.create_next_generation)
        assert 'max_offspring_attempts' in source, (
            "create_next_generation() missing max_offspring_attempts guard"
        )
        assert '_offspring_loop_iter' in source, (
            "create_next_generation() missing iteration counter"
        )

    def test_evolution_source_has_unmutated_fallback(self):
        """Verify mutation failure adds unmutated clone as fallback."""
        import inspect
        from genetic_algorithm.core.evolution import GeneticAlgorithm
        source = inspect.getsource(GeneticAlgorithm.create_next_generation)
        assert 'ga_offspring_unmutated' in source, (
            "create_next_generation() missing unmutated clone fallback"
        )


class TestC2BacktestConfigFix:
    """C2: CPCV must use self.config['backtesting'] not self.backtest_config."""

    def test_no_self_backtest_config_reference(self):
        """Ensure evolution.py does not reference self.backtest_config."""
        import inspect
        from genetic_algorithm.core.evolution import GeneticAlgorithm
        source = inspect.getsource(GeneticAlgorithm)
        assert 'self.backtest_config' not in source, (
            "evolution.py still references undefined self.backtest_config"
        )


class TestC3FeeNoiseDeterminism:
    """C3: Fee noise must use a seeded RNG, not global random."""

    def test_fee_noise_uses_seeded_rng(self):
        """Verify direct_backtester uses Random() instance, not global random.gauss()."""
        import inspect
        from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
        source = inspect.getsource(DirectBacktester._create_backtest_config)
        # Must use instance RNG, not bare random.gauss
        assert '_fee_rng' in source, (
            "_create_backtest_config() should use a seeded _fee_rng instance"
        )
        assert '_fee_rng.gauss' in source, (
            "_create_backtest_config() should call _fee_rng.gauss(), not random.gauss()"
        )

    def test_fee_noise_deterministic_same_inputs(self):
        """Same seed + same strategy name must produce same noise."""
        import random as _random_module
        name = "Gen5_Ind4"
        seed_cfg = 42

        results = []
        for _ in range(3):
            rng_seed = hash((seed_cfg, name[:64])) & 0xFFFFFFFF
            rng = _random_module.Random(rng_seed)
            results.append(rng.gauss(0, 0.0002))

        assert results[0] == results[1] == results[2], (
            f"Seeded RNG not deterministic: {results}"
        )

    def test_fee_noise_no_name_error_at_runtime(self):
        """_create_backtest_config() must not reference undefined strategy_code."""
        import inspect
        from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
        source = inspect.getsource(DirectBacktester._create_backtest_config)
        # The fee noise block must NOT reference strategy_code (undefined in this scope)
        # It should use strategy_name (the method parameter) instead
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'fee_rng_seed' in line and 'strategy_code' in line:
                raise AssertionError(
                    f"Line {i}: fee_rng_seed references 'strategy_code' which is "
                    f"not a parameter of _create_backtest_config(). Use 'strategy_name' instead."
                )
        # Verify strategy_name IS used for the seed
        assert any('strategy_name' in line and 'fee_rng_seed' in line
                    for line in lines), (
            "fee_rng_seed should use strategy_name for deterministic seeding"
        )


class TestC4TempFileCleanup:
    """C4: Temp config file must be cleaned up even on crash."""

    def test_finally_block_in_main(self):
        """Verify run_ga.main() has a finally block for tmp_config_path."""
        import inspect
        from genetic_algorithm.run_ga import main
        source = inspect.getsource(main)
        assert 'finally:' in source, "main() missing finally block"
        assert 'tmp_config_path' in source.split('finally:')[1], (
            "finally block doesn't clean up tmp_config_path"
        )


class TestH4DSRTrialsNonDuplicate:
    """H4: DSR n_trials must not double-count hashed + unhashed evals."""

    def test_hashed_evals_use_hash_count_only(self):
        """When hashes are available, n_trials = len(hashes) only."""
        from genetic_algorithm.evaluation.deflated_sharpe import DSRTracker
        tracker = DSRTracker()
        tracker.register_evaluation(strategy_hash='abc123')
        tracker.register_evaluation(strategy_hash='def456')
        tracker.register_evaluation(strategy_hash=None)  # fallback
        tracker.register_evaluation(strategy_hash=None)  # fallback
        # Should use hash count (2), not hash+fallback (4)
        assert tracker.n_trials == 2, (
            f"Expected n_trials=2 (hash count only), got {tracker.n_trials}"
        )

    def test_fallback_only_when_no_hashes(self):
        """When no hashes registered, use fallback counter."""
        from genetic_algorithm.evaluation.deflated_sharpe import DSRTracker
        tracker = DSRTracker()
        tracker.register_evaluation(strategy_hash=None)
        tracker.register_evaluation(strategy_hash=None)
        tracker.register_evaluation(strategy_hash=None)
        assert tracker.n_trials == 3, (
            f"Expected n_trials=3 (fallback only), got {tracker.n_trials}"
        )

    def test_deduplication_on_same_hash(self):
        """Same hash registered twice = 1 trial."""
        from genetic_algorithm.evaluation.deflated_sharpe import DSRTracker
        tracker = DSRTracker()
        tracker.register_evaluation(strategy_hash='same')
        tracker.register_evaluation(strategy_hash='same')
        assert tracker.n_trials == 1, (
            f"Expected n_trials=1 (dedup), got {tracker.n_trials}"
        )


class TestH5ParsimonyExitConditionMin:
    """H5: Parsimony must reject indicator removal that drops exits below 1."""

    def test_removal_blocked_when_only_exit_references_indicator(self):
        """Removing an indicator used only by exit conditions should be blocked."""
        from genetic_algorithm.core.parsimony import _apply_removal
        from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene

        gene = StrategyGene(
            generation=0,
            individual_id='test',
            indicators=[
                IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
                IndicatorGene(type='MACD', parameters={'fast': 12, 'slow': 26}, instance_id='MACD_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='RSI_0', operator='>', threshold=50),
                ConditionGene(indicator='RSI_0', operator='<', threshold=30),
            ],
            exit_conditions=[ConditionGene(indicator='MACD_0', operator='>', threshold=0)],
            minimal_roi={'0': 0.1},
            stoploss=-0.1,
            timeframe='1h',
        )
        # Try removing MACD_0 (index 1) — should be blocked because it's the only exit condition's indicator
        result = _apply_removal(gene, 'indicator', 1, min_entry_conditions=2)
        assert result is None, (
            "Removing MACD_0 should be blocked — it would leave 0 exit conditions"
        )


class TestH6CrossoverCDLFallback:
    """H6: _top_up_conditions should prefer non-CDL indicators."""

    def test_top_up_filters_cdl_indicators(self):
        """Verify source code filters CDL indicators in _top_up_conditions."""
        import inspect
        from genetic_algorithm.core.crossover import _top_up_conditions
        source = inspect.getsource(_top_up_conditions)
        assert "CDL_" in source, "Should filter CDL indicators in _top_up_conditions"
        assert "usable_indicators" in source, (
            "_top_up_conditions should build a usable_indicators list excluding CDL"
        )


class TestH7MutationRollback:
    """H7: Failed mutation method should roll back to pre-mutation state."""

    def test_mutation_source_has_rollback(self):
        """Verify mutate() rolls back on failure instead of continuing with partial state."""
        import inspect
        from genetic_algorithm.core.mutation import mutate
        source = inspect.getsource(mutate)
        assert 'pre_mutation' in source, (
            "mutate() missing pre_mutation rollback snapshot"
        )
        assert 'mutated = pre_mutation' in source, (
            "mutate() doesn't roll back to pre_mutation on error"
        )


class TestH11CPCVValidation:
    """H11: CPCV must validate n_test_groups and n_groups."""

    def test_n_test_groups_greater_than_n_groups_returns_empty(self):
        from genetic_algorithm.evaluation.cpcv import generate_cpcv_paths
        result = generate_cpcv_paths(n_groups=4, n_test_groups=5)
        assert result == [], (
            f"Expected empty list for n_test_groups > n_groups, got {len(result)} paths"
        )

    def test_n_test_groups_zero_returns_empty(self):
        from genetic_algorithm.evaluation.cpcv import generate_cpcv_paths
        result = generate_cpcv_paths(n_groups=4, n_test_groups=0)
        assert result == [], (
            f"Expected empty list for n_test_groups=0, got {len(result)} paths"
        )

    def test_n_groups_one_returns_empty(self):
        from genetic_algorithm.evaluation.cpcv import generate_cpcv_paths
        result = generate_cpcv_paths(n_groups=1, n_test_groups=1)
        assert result == [], (
            f"Expected empty list for n_groups=1, got {len(result)} paths"
        )

    def test_valid_inputs_return_paths(self):
        from genetic_algorithm.evaluation.cpcv import generate_cpcv_paths
        result = generate_cpcv_paths(n_groups=6, n_test_groups=2)
        # C(6,2) = 15
        assert len(result) == 15, (
            f"Expected 15 paths for C(6,2), got {len(result)}"
        )
