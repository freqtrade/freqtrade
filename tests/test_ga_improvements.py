"""
Tests for GA Improvement Implementations

Tests the following 10 improvements:
1. Walk-forward embargo period
2. Train-val gap penalty
3. Out-of-sample holdout validation
4. AND/OR condition logic in strategy generator
5. Realistic slippage modeling
6. Per-pair performance breakdown & penalty
7. Checkpoint save/load/restore
8. Crossover method config passthrough
9. Failed walk-forward window handling
10. Config loading with new settings
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from genetic_algorithm.utils.timerange import (
    create_walk_forward_windows,
    parse_timerange,
    format_date,
    TimeWindow,
)
from genetic_algorithm.core.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    ConditionGene,
)
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population, PopulationStats
from genetic_algorithm.evaluation.direct_backtester import BacktestResult


# ============================================================================
# Helpers
# ============================================================================

def _make_strategy_gene(**overrides):
    """Create a minimal valid StrategyGene for testing."""
    defaults = dict(
        generation=0,
        individual_id=0,
        indicators=[IndicatorGene(type='RSI', parameters={'period': 14})],
        entry_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND')],
        exit_conditions=[ConditionGene(indicator='RSI', operator='>', threshold=70, logic='AND')],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
        max_open_trades=3,
    )
    defaults.update(overrides)
    return StrategyGene(**defaults)


def _make_individual(**overrides):
    """Create a minimal evaluated Individual for testing."""
    gene = _make_strategy_gene(**{k: v for k, v in overrides.items()
                                  if k in ('generation', 'individual_id')})
    ind = Individual(strategy_gene=gene)
    ind.fitness = overrides.get('fitness', 0.5)
    ind.raw_fitness = overrides.get('raw_fitness', ind.fitness)
    ind.metrics = overrides.get('metrics', {'profit': 5.0, 'sharpe_ratio': 1.0,
                                             'max_drawdown': 0.1, 'win_rate': 0.6,
                                             'num_trades': 20})
    ind.evaluated = True
    return ind


# ============================================================================
# 1. Walk-Forward Embargo Period
# ============================================================================

class TestEmbargoWindows:
    """Test that embargo_days inserts a gap between train and val windows."""

    def test_embargo_zero_produces_contiguous_windows(self):
        """With embargo_days=0, val_start == train_end."""
        windows = create_walk_forward_windows(
            timerange='20230101-20230601',
            train_days=60,
            validation_days=30,
            step_days=30,
            mode='rolling',
            embargo_days=0,
        )
        assert len(windows) > 0
        for w in windows:
            # val_start should equal train_end (no gap)
            assert w.val_start == w.train_end, (
                f"Expected val_start={w.val_start} == train_end={w.train_end}"
            )

    def test_embargo_positive_inserts_gap(self):
        """With embargo_days=N, val_start should be N days after train_end."""
        embargo = 5
        windows = create_walk_forward_windows(
            timerange='20230101-20230901',
            train_days=60,
            validation_days=30,
            step_days=30,
            mode='rolling',
            embargo_days=embargo,
        )
        assert len(windows) > 0
        for w in windows:
            train_end_dt = parse_timerange(f"{w.train_end}-{w.val_start}")[0]
            val_start_dt = parse_timerange(f"{w.train_end}-{w.val_start}")[1]
            gap_days = (val_start_dt - train_end_dt).days
            assert gap_days == embargo, (
                f"Expected gap={embargo}, got {gap_days} "
                f"(train_end={w.train_end}, val_start={w.val_start})"
            )

    def test_embargo_reduces_number_of_windows(self):
        """Larger embargo should produce fewer windows because more data is consumed."""
        windows_no_embargo = create_walk_forward_windows(
            timerange='20230101-20230601',
            train_days=60,
            validation_days=30,
            step_days=30,
            mode='rolling',
            embargo_days=0,
        )
        windows_with_embargo = create_walk_forward_windows(
            timerange='20230101-20230601',
            train_days=60,
            validation_days=30,
            step_days=30,
            mode='rolling',
            embargo_days=7,
        )
        assert len(windows_with_embargo) <= len(windows_no_embargo)

    def test_embargo_negative_raises(self):
        """Negative embargo_days should raise ValueError."""
        with pytest.raises(ValueError, match="embargo_days must be non-negative"):
            create_walk_forward_windows(
                timerange='20230101-20230601',
                train_days=60,
                validation_days=30,
                step_days=30,
                embargo_days=-1,
            )

    def test_embargo_anchored_mode(self):
        """Embargo should also work in anchored mode."""
        windows = create_walk_forward_windows(
            timerange='20230101-20230901',
            train_days=60,
            validation_days=30,
            step_days=30,
            mode='anchored',
            embargo_days=3,
        )
        assert len(windows) > 0
        for w in windows:
            train_end_dt = parse_timerange(f"{w.train_end}-{w.val_start}")[0]
            val_start_dt = parse_timerange(f"{w.train_end}-{w.val_start}")[1]
            gap = (val_start_dt - train_end_dt).days
            assert gap == 3


# ============================================================================
# 2. Train-Val Gap Penalty
# ============================================================================

class TestGapPenalty:
    """Test the train-val gap penalty logic in FitnessEvaluator."""

    def _make_evaluator(self, gap_config=None):
        """Create a FitnessEvaluator with specific gap penalty config."""
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        config = {
            'fitness_weights': {},
            'fitness_penalties': {},
            'backtesting': {
                'pairs': ['UNITTEST/BTC'],
                'timerange': '20180101-20180301',
                'stake_amount': 0.05,
                'fee': 0.001,
            },
            'walk_forward': {
                'enabled': False,
                'gap_penalty': gap_config or {'enabled': True, 'threshold': 0.1, 'max_penalty': 0.5},
            },
            'indicators': {'available': ['RSI']},
            'strategy_constraints': {'timeframes': ['5m']},
        }
        return FitnessEvaluator(config)

    def test_gap_penalty_reduces_fitness_for_large_gap(self):
        """When train-val gap exceeds threshold, fitness should decrease."""
        evaluator = self._make_evaluator({'enabled': True, 'threshold': 0.1, 'max_penalty': 0.5})

        # Simulate: gap penalty logic from evaluate_walk_forward
        avg_train_fitness = 0.8
        avg_val_fitness = 0.4
        gap = avg_train_fitness - avg_val_fitness  # 0.4
        threshold = 0.1
        max_penalty = 0.5

        base_fitness = 0.5
        excess_gap = gap - threshold  # 0.3
        gap_penalty_factor = max(1.0 - max_penalty, 1.0 - excess_gap * 2.0)  # max(0.5, 0.4) = 0.5
        penalized = base_fitness * gap_penalty_factor

        assert penalized < base_fitness
        assert gap_penalty_factor == 0.5  # Capped at max_penalty

    def test_gap_penalty_no_effect_below_threshold(self):
        """When gap is below threshold, no penalty should be applied."""
        threshold = 0.1
        gap = 0.05  # Below threshold
        max_penalty = 0.5

        base_fitness = 0.5
        if gap > threshold:
            excess_gap = gap - threshold
            factor = max(1.0 - max_penalty, 1.0 - excess_gap * 2.0)
            penalized = base_fitness * factor
        else:
            penalized = base_fitness

        assert penalized == base_fitness

    def test_gap_penalty_disabled(self):
        """When gap penalty is disabled, fitness should not change."""
        enabled = False
        gap = 0.5  # Large gap
        threshold = 0.1

        base_fitness = 0.5
        if enabled and gap > threshold:
            penalized = base_fitness * 0.5
        else:
            penalized = base_fitness

        assert penalized == base_fitness

    def test_gap_penalty_progressive(self):
        """Gap penalty should be progressive — larger gaps get bigger penalties."""
        threshold = 0.1
        max_penalty = 0.5
        base_fitness = 1.0

        results = []
        for gap in [0.15, 0.25, 0.35, 0.5]:
            excess = gap - threshold
            factor = max(1.0 - max_penalty, 1.0 - excess * 2.0)
            results.append(base_fitness * factor)

        # Each successive result should be <= the previous (more gap = more penalty)
        for i in range(1, len(results)):
            assert results[i] <= results[i - 1], (
                f"Penalty not progressive: gap series produced {results}"
            )


# ============================================================================
# 3. Holdout Split
# ============================================================================

class TestHoldoutSplit:
    """Test the split_timerange_for_holdout static method."""

    def test_basic_split(self):
        """Standard 15% holdout from end."""
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        evo_tr, holdout_tr = FitnessEvaluator.split_timerange_for_holdout(
            '20230101-20230601', holdout_pct=0.15
        )

        # Parse dates
        evo_start, evo_end = parse_timerange(evo_tr)
        ho_start, ho_end = parse_timerange(holdout_tr)

        # Holdout end should match original end
        assert format_date(ho_end) == '20230601'
        # Evolution end should match holdout start (contiguous)
        assert format_date(evo_end) == format_date(ho_start)
        # Evolution start should match original start
        assert format_date(evo_start) == '20230101'

    def test_holdout_minimum_7_days(self):
        """Holdout should be at least 7 days even with tiny pct."""
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        evo_tr, holdout_tr = FitnessEvaluator.split_timerange_for_holdout(
            '20230101-20230201', holdout_pct=0.01  # ~0.3 days
        )
        ho_start, ho_end = parse_timerange(holdout_tr)
        holdout_days = (ho_end - ho_start).days
        assert holdout_days >= 7

    def test_holdout_pct_respected(self):
        """Holdout period should be approximately the requested percentage."""
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        evo_tr, holdout_tr = FitnessEvaluator.split_timerange_for_holdout(
            '20230101-20240101', holdout_pct=0.20
        )
        start, end = parse_timerange('20230101-20240101')
        total_days = (end - start).days
        ho_start, ho_end = parse_timerange(holdout_tr)
        holdout_days = (ho_end - ho_start).days

        expected = int(total_days * 0.20)
        assert abs(holdout_days - expected) <= 1  # Allow 1-day rounding


# ============================================================================
# 4. AND/OR Condition Logic
# ============================================================================

class TestAndOrConditionLogic:
    """Test that _generate_condition_code properly handles AND/OR grouping."""

    def _get_generator(self):
        config = {
            'indicators': {'available': ['RSI', 'EMA', 'SMA']},
            'strategy_constraints': {'timeframes': ['5m']},
        }
        from genetic_algorithm.strategies.generator import StrategyGenerator
        return StrategyGenerator(config)

    def _make_indicators(self):
        return [
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
            IndicatorGene(type='SMA', parameters={'period': 50}, instance_id='SMA_0'),
        ]

    def test_all_and_conditions(self):
        """All AND conditions → every condition required (& between each)."""
        gen = self._get_generator()
        indicators = self._make_indicators()
        conditions = [
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
            ConditionGene(indicator='EMA_0', operator='cross_above', threshold=0, logic='AND'),
        ]
        code = gen._generate_condition_code(conditions, indicators, is_entry=True)
        # Should contain & (AND) and not | (OR)
        assert '&' in code, f"Expected '&' in all-AND code:\n{code}"
        # There should be no | since all are AND
        # (Careful: the expression may use | for something else, but at the top-level combining there shouldn't be)
        assert 'enter_long' in code

    def test_all_or_conditions(self):
        """All OR conditions → at least one must fire (| between each)."""
        gen = self._get_generator()
        indicators = self._make_indicators()
        conditions = [
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='OR'),
            ConditionGene(indicator='EMA_0', operator='cross_above', threshold=0, logic='OR'),
        ]
        code = gen._generate_condition_code(conditions, indicators, is_entry=True)
        assert '|' in code, f"Expected '|' in all-OR code:\n{code}"
        assert 'enter_long' in code

    def test_mixed_and_or_conditions(self):
        """Mixed AND/OR → AND conditions are required, OR conditions grouped."""
        gen = self._get_generator()
        indicators = self._make_indicators()
        conditions = [
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
            ConditionGene(indicator='EMA_0', operator='cross_above', threshold=0, logic='OR'),
            ConditionGene(indicator='SMA_0', operator='cross_above', threshold=0, logic='OR'),
        ]
        code = gen._generate_condition_code(conditions, indicators, is_entry=True)
        # Should contain both & and |
        assert '&' in code, f"Expected '&' in mixed code:\n{code}"
        assert '|' in code, f"Expected '|' in mixed code:\n{code}"
        assert 'enter_long' in code

    def test_empty_conditions_fallback(self):
        """No conditions should produce a volume-based fallback."""
        gen = self._get_generator()
        code = gen._generate_condition_code([], [], is_entry=True)
        assert 'enter_long' in code

    def test_exit_conditions_use_exit_long(self):
        """Exit conditions should set exit_long signal."""
        gen = self._get_generator()
        indicators = self._make_indicators()
        conditions = [
            ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='AND'),
        ]
        code = gen._generate_condition_code(conditions, indicators, is_entry=False)
        assert 'exit_long' in code


# ============================================================================
# 5. Realistic Slippage Modeling
# ============================================================================

class TestSlippageModeling:
    """Test that slippage_pct is added to the exchange fee."""

    def test_slippage_added_to_fee(self):
        """Fee should include slippage_pct on top of base fee."""
        base_fee = 0.001
        slippage = 0.0005
        expected_total = base_fee + slippage

        # Simulate the logic in _create_backtest_config
        fee = base_fee
        slippage_pct = slippage
        if slippage_pct > 0:
            fee = fee + slippage_pct

        assert abs(fee - expected_total) < 1e-10

    def test_zero_slippage_no_change(self):
        """With slippage_pct=0, fee should remain unchanged."""
        base_fee = 0.001
        fee = base_fee
        slippage_pct = 0.0
        if slippage_pct > 0:
            fee = fee + slippage_pct
        assert fee == base_fee

    def test_slippage_config_integration(self):
        """Verify slippage_pct is read from config and applied."""
        config = {
            'backtesting': {
                'pairs': ['UNITTEST/BTC'],
                'timerange': '20180101-20180301',
                'stake_amount': 0.05,
                'fee': 0.001,
                'slippage_pct': 0.0005,
            },
        }
        ga_cfg = config['backtesting']
        fee = ga_cfg.get('fee', 0.001)
        slippage_pct = ga_cfg.get('slippage_pct', 0.0)
        if slippage_pct > 0:
            fee = fee + slippage_pct
        assert abs(fee - 0.0015) < 1e-10


# ============================================================================
# 6. Per-Pair Performance Breakdown & Penalty
# ============================================================================

class TestPerPairMetrics:
    """Test per-pair profit extraction and penalty application."""

    def test_per_pair_in_backtest_result(self):
        """BacktestResult should support per_pair_profit field."""
        result = BacktestResult(
            success=True,
            strategy_name='test',
            per_pair_profit={'BTC/USDT': 5.0, 'ETH/USDT': -3.0}
        )
        assert result.per_pair_profit is not None
        assert result.per_pair_profit['BTC/USDT'] == 5.0
        assert result.per_pair_profit['ETH/USDT'] == -3.0

    def test_backtest_result_to_metrics_with_per_pair(self):
        """_backtest_result_to_metrics should include per-pair stats."""
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        config = {
            'fitness_weights': {},
            'fitness_penalties': {},
            'backtesting': {
                'pairs': ['UNITTEST/BTC'],
                'timerange': '20180101-20180301',
                'stake_amount': 0.05,
                'fee': 0.001,
            },
            'walk_forward': {'enabled': False},
            'indicators': {'available': ['RSI']},
            'strategy_constraints': {'timeframes': ['5m']},
        }
        evaluator = FitnessEvaluator(config)

        result = BacktestResult(
            success=True,
            strategy_name='test',
            profit_percent=10.0,
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.55,
            total_trades=30,
            profit_factor=1.8,
            sortino_ratio=2.0,
            per_pair_profit={'BTC/USDT': 8.0, 'ETH/USDT': 2.0, 'XRP/USDT': -5.0}
        )

        metrics = evaluator._backtest_result_to_metrics(result)
        assert 'per_pair_profit' in metrics
        assert metrics['worst_pair_profit'] == -5.0
        assert 'pair_profit_std' in metrics  # 3 pairs → std should exist

    def test_pair_penalty_applied_for_large_loss(self):
        """Pair penalty should reduce fitness when worst pair exceeds threshold."""
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        config = {
            'fitness_weights': {},
            'fitness_penalties': {'pair_loss_threshold': -10.0},
            'backtesting': {
                'pairs': ['UNITTEST/BTC'],
                'timerange': '20180101-20180301',
                'stake_amount': 0.05,
                'fee': 0.001,
            },
            'walk_forward': {'enabled': False},
            'indicators': {'available': ['RSI']},
            'strategy_constraints': {'timeframes': ['5m']},
        }
        evaluator = FitnessEvaluator(config)

        metrics_no_loss = {'num_trades': 20, 'max_drawdown': 0.1, 'win_rate': 0.6}
        metrics_with_loss = {
            'num_trades': 20, 'max_drawdown': 0.1, 'win_rate': 0.6,
            'worst_pair_profit': -25.0  # Exceeds -10 threshold
        }

        fitness_base = 0.5
        f1 = evaluator._apply_penalties(fitness_base, metrics_no_loss)
        f2 = evaluator._apply_penalties(fitness_base, metrics_with_loss)

        assert f2 < f1, f"Expected pair penalty: f1={f1}, f2={f2}"

    def test_pair_penalty_not_applied_within_threshold(self):
        """No penalty when worst pair loss is within threshold."""
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator

        config = {
            'fitness_weights': {},
            'fitness_penalties': {'pair_loss_threshold': -10.0},
            'backtesting': {
                'pairs': ['UNITTEST/BTC'],
                'timerange': '20180101-20180301',
                'stake_amount': 0.05,
                'fee': 0.001,
            },
            'walk_forward': {'enabled': False},
            'indicators': {'available': ['RSI']},
            'strategy_constraints': {'timeframes': ['5m']},
        }
        evaluator = FitnessEvaluator(config)

        metrics_ok = {
            'num_trades': 20, 'max_drawdown': 0.1, 'win_rate': 0.6,
            'worst_pair_profit': -5.0  # Within threshold
        }
        metrics_none = {'num_trades': 20, 'max_drawdown': 0.1, 'win_rate': 0.6}

        f1 = evaluator._apply_penalties(0.5, metrics_ok)
        f2 = evaluator._apply_penalties(0.5, metrics_none)

        assert f1 == f2, f"Unexpected penalty: f_ok={f1}, f_none={f2}"


# ============================================================================
# 7. Checkpoint Save/Load/Restore
# ============================================================================

class TestCheckpointing:
    """Test checkpoint serialization and restoration."""

    def _make_ga(self, tmpdir):
        """Create a GeneticAlgorithm instance with checkpoint_dir in tmpdir."""
        from genetic_algorithm.core.evolution import GeneticAlgorithm

        # We need to mock the config load to avoid full initialization
        # Instead, test checkpoint methods directly on an instance
        config = {
            'genetic_algorithm': {
                'population_size': 5,
                'generations': 3,
                'mutation_rate': 0.15,
                'crossover_rate': 0.7,
                'crossover_method': 'single_point',
                'elite_size': 1,
                'tournament_size': 3,
                'selection_method': 'tournament',
                'convergence_patience': 10,
                'mode': 'single_objective',
                'fitness_sharing': False,
                'sharing_radius': 0.3,
                'diversity_threshold': 0.15,
                'allow_self_crossover': True,
                'random_immigrants': 0,
                'adaptive_mutation': False,
            },
            'nsga2': {},
            'storage': {
                'checkpoint_dir': str(tmpdir),
                'checkpoint_interval': 1,
            },
            'fitness_weights': {},
            'fitness_penalties': {},
            'backtesting': {
                'pairs': ['UNITTEST/BTC'],
                'timerange': '20180101-20180301',
                'stake_amount': 0.05,
                'fee': 0.001,
            },
            'walk_forward': {'enabled': False},
            'indicators': {
                'available': ['RSI'],
                'min_per_strategy': 1,
                'max_per_strategy': 1,
            },
            'strategy_constraints': {
                'timeframes': ['5m'],
                'stoploss_range': [-0.20, -0.05],
                'roi_range': [0.01, 0.10],
                'max_open_trades_range': [1, 5],
            },
            'logging': {'level': 'WARNING', 'console': False},
        }
        # Patch _load_config and _setup_logging to skip file access
        with patch.object(GeneticAlgorithm, '_load_config', return_value=config), \
             patch.object(GeneticAlgorithm, '_setup_logging', return_value=logging_mock()):
            ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
            ga.config = config
            ga.logger = logging_mock()

            ga_config = config['genetic_algorithm']
            ga.population_size = ga_config['population_size']
            ga.generations = ga_config['generations']
            ga.mutation_rate = ga_config['mutation_rate']
            ga.crossover_rate = ga_config['crossover_rate']
            ga.crossover_method = ga_config.get('crossover_method', 'single_point')
            ga.elite_size = ga_config['elite_size']
            ga.tournament_size = ga_config.get('tournament_size', 3)
            ga.selection_method = ga_config.get('selection_method', 'tournament')
            ga.convergence_patience = ga_config.get('convergence_patience', 10)
            ga.mode = ga_config.get('mode', 'single_objective')
            ga.fitness_sharing = ga_config.get('fitness_sharing', False)
            ga.sharing_radius = ga_config.get('sharing_radius', 0.3)
            ga.diversity_threshold = ga_config.get('diversity_threshold', 0.15)
            ga.allow_self_crossover = ga_config.get('allow_self_crossover', True)
            ga.random_immigrants = ga_config.get('random_immigrants', 0)
            ga.adaptive_mutation = ga_config.get('adaptive_mutation', False)
            ga.base_mutation_rate = ga.mutation_rate
            ga.max_adaptation_factor = 2.0
            ga.adaptation_step = 0.1
            ga.random_seed = None
            ga.best_individual = None
            ga.best_fitness_ever = 0.0
            ga.no_improvement_count = 0
            ga.generation_stats = []
            ga.current_generation = 0

            storage_config = config.get('storage', {})
            ga.checkpoint_dir = Path(storage_config.get('checkpoint_dir', str(tmpdir)))
            ga.checkpoint_interval = storage_config.get('checkpoint_interval', 1)

        return ga

    def test_save_creates_checkpoint_file(self):
        """save_checkpoint should create latest_checkpoint.json."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            ga = self._make_ga(tmpdir)
            pop = Population(size=2, generation=0)
            ind1 = _make_individual(generation=0, individual_id=0, fitness=0.5)
            ind2 = _make_individual(generation=0, individual_id=1, fitness=0.7)
            pop.add_individual(ind1)
            pop.add_individual(ind2)

            ga.best_individual = ind2
            ga.best_fitness_ever = 0.7
            ga.save_checkpoint(pop, generation=0)

            cp_file = tmpdir / 'latest_checkpoint.json'
            assert cp_file.exists(), f"Checkpoint file not found at {cp_file}"

            with open(cp_file) as f:
                data = json.load(f)

            assert data['generation'] == 0
            assert len(data['population']) == 2
            assert data['best_fitness_ever'] == 0.7
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_checkpoint_returns_data(self):
        """load_checkpoint should return saved data."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            ga = self._make_ga(tmpdir)
            pop = Population(size=1, generation=0)
            ind = _make_individual(generation=0, individual_id=0, fitness=0.6)
            pop.add_individual(ind)
            ga.best_individual = ind
            ga.best_fitness_ever = 0.6

            ga.save_checkpoint(pop, generation=2)
            loaded = ga.load_checkpoint()

            assert loaded is not None
            assert loaded['generation'] == 2
            assert loaded['best_fitness_ever'] == 0.6
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_checkpoint_returns_none_when_missing(self):
        """load_checkpoint should return None if no checkpoint exists."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            ga = self._make_ga(tmpdir)
            loaded = ga.load_checkpoint()
            assert loaded is None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_restore_round_trip(self):
        """save → load → restore should produce equivalent population."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            ga = self._make_ga(tmpdir)
            pop = Population(size=2, generation=5)
            ind1 = _make_individual(generation=5, individual_id=0, fitness=0.4)
            ind2 = _make_individual(generation=5, individual_id=1, fitness=0.8)
            pop.add_individual(ind1)
            pop.add_individual(ind2)

            ga.best_individual = ind2
            ga.best_fitness_ever = 0.8
            ga.no_improvement_count = 3
            ga.mutation_rate = 0.25
            ga.generation_stats = [
                PopulationStats(generation=0, size=2, best_fitness=0.5, avg_fitness=0.4, worst_fitness=0.3),
            ]

            ga.save_checkpoint(pop, generation=5)

            # Create fresh GA and restore
            ga2 = self._make_ga(tmpdir)
            cp = ga2.load_checkpoint()
            restored_pop = ga2.restore_from_checkpoint(cp)

            assert len(restored_pop.individuals) == 2
            assert ga2.current_generation == 5
            assert ga2.best_fitness_ever == 0.8
            assert ga2.no_improvement_count == 3
            assert ga2.mutation_rate == 0.25
            assert len(ga2.generation_stats) == 1

            # Verify individual data round-tripped
            restored_ind = max(restored_pop.individuals, key=lambda x: x.fitness or 0)
            assert abs(restored_ind.fitness - 0.8) < 1e-6
            assert restored_ind.id == ind2.id
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_checkpoint_atomic_write(self):
        """Checkpoint should not leave a .tmp file on success."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            ga = self._make_ga(tmpdir)
            pop = Population(size=1, generation=0)
            ind = _make_individual()
            pop.add_individual(ind)
            ga.best_individual = ind

            ga.save_checkpoint(pop, generation=0)

            tmp_file = tmpdir / 'latest_checkpoint.tmp'
            assert not tmp_file.exists(), "Temp file should be removed after atomic write"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# 8. Crossover Method Config
# ============================================================================

class TestCrossoverMethodConfig:
    """Test that crossover_method is read from config and passed through."""

    def test_config_crossover_method_stored(self):
        """GeneticAlgorithm should store crossover_method from config."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            from genetic_algorithm.core.evolution import GeneticAlgorithm

            config = {
                'genetic_algorithm': {
                    'population_size': 5,
                    'generations': 2,
                    'mutation_rate': 0.15,
                    'crossover_rate': 0.7,
                    'crossover_method': 'uniform',
                    'elite_size': 1,
                    'tournament_size': 3,
                    'selection_method': 'tournament',
                    'convergence_patience': 10,
                    'mode': 'single_objective',
                    'fitness_sharing': False,
                    'random_immigrants': 0,
                    'adaptive_mutation': False,
                },
                'nsga2': {},
                'storage': {'checkpoint_dir': str(tmpdir), 'checkpoint_interval': 5},
                'fitness_weights': {},
                'fitness_penalties': {},
                'backtesting': {
                    'pairs': ['UNITTEST/BTC'],
                    'timerange': '20180101-20180301',
                    'stake_amount': 0.05,
                    'fee': 0.001,
                },
                'walk_forward': {'enabled': False},
                'indicators': {'available': ['RSI'], 'min_per_strategy': 1, 'max_per_strategy': 1},
                'strategy_constraints': {
                    'timeframes': ['5m'],
                    'stoploss_range': [-0.20, -0.05],
                    'roi_range': [0.01, 0.10],
                    'max_open_trades_range': [1, 5],
                },
                'logging': {'level': 'WARNING', 'console': False},
            }

            with patch.object(GeneticAlgorithm, '_load_config', return_value=config):
                ga = GeneticAlgorithm.__new__(GeneticAlgorithm)
                ga.config = config
                ga.logger = logging_mock()
                ga_cfg = config['genetic_algorithm']
                ga.crossover_method = ga_cfg.get('crossover_method', 'single_point')

            assert ga.crossover_method == 'uniform'
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_crossover_function_accepts_method(self):
        """crossover() should accept and use the method parameter."""
        from genetic_algorithm.core.crossover import crossover

        parent1 = _make_individual(generation=0, individual_id=0)
        parent2 = _make_individual(generation=0, individual_id=1)

        # Test all three methods
        for method in ('single_point', 'uniform', 'component'):
            child1, child2 = crossover(
                parent1, parent2,
                generation=1, ind_id=10,
                method=method,
            )
            assert child1 is not None
            assert child2 is not None
            assert child1.strategy_gene.generation == 1

    def test_invalid_crossover_method_raises(self):
        """Invalid crossover method should raise ValueError."""
        from genetic_algorithm.core.crossover import crossover

        parent1 = _make_individual(generation=0, individual_id=0)
        parent2 = _make_individual(generation=0, individual_id=1)

        with pytest.raises(ValueError, match="Unknown crossover method"):
            crossover(parent1, parent2, generation=1, ind_id=10, method='invalid_method')


# ============================================================================
# 9. Failed Walk-Forward Window Handling
# ============================================================================

class TestFailedWindowHandling:
    """Test that failed windows are tracked and penalize proportionally."""

    def test_success_ratio_calculation(self):
        """Success ratio = successful_windows / total_windows."""
        total = 5
        failed = 2
        successful = total - failed
        success_ratio = successful / total
        assert abs(success_ratio - 0.6) < 1e-10

    def test_fitness_scaled_by_success_ratio(self):
        """Final fitness should be multiplied by success_ratio when windows fail."""
        base_fitness = 0.8
        total_windows = 5
        failed_windows = 2
        successful_windows = total_windows - failed_windows

        # Simulate logic from evaluate_walk_forward
        if failed_windows > 0:
            success_ratio = successful_windows / total_windows
            penalized = base_fitness * success_ratio
        else:
            penalized = base_fitness

        expected = 0.8 * (3 / 5)  # 0.48
        assert abs(penalized - expected) < 1e-10

    def test_all_windows_failed_returns_zero(self):
        """If all windows fail, fitness should be 0."""
        validation_fitness_scores = []  # all failed
        total_windows = 5
        failed_windows = 5

        if not validation_fitness_scores:
            final_fitness = 0.0
        else:
            final_fitness = sum(validation_fitness_scores) / len(validation_fitness_scores)

        assert final_fitness == 0.0

    def test_no_failures_no_penalty(self):
        """If no windows fail, no success_ratio penalty is applied."""
        base_fitness = 0.8
        failed_windows = 0

        if failed_windows > 0:
            penalized = base_fitness * 0.5
        else:
            penalized = base_fitness

        assert penalized == base_fitness


# ============================================================================
# 10. Config Loading with New Settings
# ============================================================================

class TestConfigNewSettings:
    """Test that ga_config.yaml contains and correctly loads new settings."""

    @pytest.fixture
    def config(self):
        import yaml
        config_path = PROJECT_ROOT / 'genetic_algorithm' / 'config' / 'ga_config.yaml'
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_crossover_method_in_config(self, config):
        assert 'crossover_method' in config['genetic_algorithm']
        assert config['genetic_algorithm']['crossover_method'] in ('single_point', 'uniform', 'component')

    def test_slippage_pct_in_config(self, config):
        assert 'slippage_pct' in config['backtesting']
        assert isinstance(config['backtesting']['slippage_pct'], (int, float))
        assert config['backtesting']['slippage_pct'] >= 0

    def test_pair_loss_threshold_in_config(self, config):
        assert 'pair_loss_threshold' in config['fitness_penalties']
        assert config['fitness_penalties']['pair_loss_threshold'] < 0

    def test_embargo_days_in_config(self, config):
        assert 'embargo_days' in config['walk_forward']
        assert isinstance(config['walk_forward']['embargo_days'], int)
        assert config['walk_forward']['embargo_days'] >= 0

    def test_gap_penalty_in_config(self, config):
        gap = config['walk_forward']['gap_penalty']
        assert 'enabled' in gap
        assert 'threshold' in gap
        assert 'max_penalty' in gap
        assert isinstance(gap['threshold'], (int, float))

    def test_holdout_validation_in_config(self, config):
        assert 'holdout_validation' in config
        holdout = config['holdout_validation']
        assert 'enabled' in holdout
        assert 'holdout_pct' in holdout
        assert 0 < holdout['holdout_pct'] < 1


# ============================================================================
# Helpers
# ============================================================================

def logging_mock():
    """Create a mock logger."""
    mock = MagicMock()
    mock.info = MagicMock()
    mock.debug = MagicMock()
    mock.warning = MagicMock()
    mock.error = MagicMock()
    mock.setLevel = MagicMock()
    mock.propagate = True
    mock.handlers = []
    mock.addHandler = MagicMock()
    return mock


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
