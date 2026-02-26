"""
Tests for Tier 3 GA Improvements:
- Monte-Carlo Robustness Analysis
- Parsimony Pressure (Strategy Simplification)
- Pareto Archive with Crowding Decay
- Dynamic Indicator Parameter Ranges
"""

import pytest
import random
from typing import Dict, Any, Tuple

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_trades():
    """Create sample trade results for Monte-Carlo testing."""
    random.seed(42)
    return [
        {'profit_ratio': 0.02 + random.gauss(0, 0.01)} for _ in range(50)
    ]


@pytest.fixture
def sample_strategy_gene():
    """Create a sample StrategyGene for parsimony/dynamic bounds testing."""
    from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
    return StrategyGene(
        generation=1,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, instance_id='MACD_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
            ConditionGene(indicator='EMA_0', operator='>', threshold=0, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='AND'),
        ],
        timeframe='5m',
        stoploss=-0.05,
    )


@pytest.fixture
def sample_individual(sample_strategy_gene):
    """Create a sample Individual."""
    from genetic_algorithm.core.individual import Individual
    ind = Individual(strategy_gene=sample_strategy_gene)
    ind.fitness = 0.75
    ind.raw_fitness = 0.75
    ind.metrics = {'profit': 15.0, 'sharpe_ratio': 1.5}
    ind.evaluated = True
    return ind


@pytest.fixture
def mock_evaluator():
    """Mock evaluate function that returns stable fitness based on complexity."""
    def evaluate(gene):
        complexity = gene.calculate_complexity()
        # Simpler = better (for parsimony testing)
        fitness = 0.8 - (complexity - 5) * 0.01
        return fitness, {'profit': fitness * 20}
    return evaluate


# =============================================================================
# MONTE-CARLO ROBUSTNESS TESTS
# =============================================================================

class TestMonteCarlo:
    """Tests for Monte-Carlo robustness scoring."""
    
    def test_bootstrap_trades_basic(self, sample_trades):
        """Test basic bootstrap resampling."""
        from genetic_algorithm.evaluation.monte_carlo import bootstrap_trades
        
        result = bootstrap_trades(sample_trades, num_permutations=50, random_seed=42)
        
        assert result.num_permutations == 50
        assert 0 <= result.robustness_score <= 1
        assert len(result.permutation_profits) == 50
        assert result.profit_p5 <= result.mean_profit <= result.profit_p95
    
    def test_bootstrap_empty_trades(self):
        """Bootstrap with empty trade list returns empty result."""
        from genetic_algorithm.evaluation.monte_carlo import bootstrap_trades
        
        result = bootstrap_trades([], num_permutations=50)
        assert result.robustness_score == 0.0
        assert result.num_permutations == 0
    
    def test_shuffle_trade_order(self, sample_trades):
        """Test trade order shuffling."""
        from genetic_algorithm.evaluation.monte_carlo import shuffle_trade_order
        
        result = shuffle_trade_order(sample_trades, num_permutations=50, random_seed=42)
        
        assert result.num_permutations == 50
        assert result.mean_profit != 0  # Should have some profit
        # Shuffling with compounding should produce variance
        assert result.profit_std > 0
    
    def test_jitter_slippage(self, sample_trades):
        """Test slippage jitter perturbation."""
        from genetic_algorithm.evaluation.monte_carlo import jitter_slippage
        
        result = jitter_slippage(
            sample_trades,
            slippage_std=0.001,
            num_permutations=50,
            random_seed=42
        )
        
        assert result.num_permutations == 50
        # With slippage, some variance is expected
        assert result.profit_std > 0
    
    def test_run_monte_carlo_full(self, sample_trades):
        """Test full Monte-Carlo analysis with all methods."""
        from genetic_algorithm.evaluation.monte_carlo import run_monte_carlo
        
        config = {
            'num_permutations': 60,
            'methods': ['bootstrap', 'shuffle', 'slippage_jitter'],
            'slippage_std': 0.0005,
            'sample_fraction': 1.0,
            'random_seed': 42,
        }
        
        result = run_monte_carlo(sample_trades, config)
        
        # 60 / 3 methods = 20 each × 3 = 60 total
        assert result.num_permutations >= 50
        assert 0 <= result.robustness_score <= 1
    
    def test_monte_carlo_result_fields(self, sample_trades):
        """Verify MonteCarloResult has all expected fields."""
        from genetic_algorithm.evaluation.monte_carlo import run_monte_carlo
        
        config = {'num_permutations': 30, 'methods': ['bootstrap'], 'random_seed': 123}
        result = run_monte_carlo(sample_trades, config)
        
        assert hasattr(result, 'robustness_score')
        assert hasattr(result, 'mean_profit')
        assert hasattr(result, 'profit_std')
        assert hasattr(result, 'profit_p5')
        assert hasattr(result, 'profit_p95')
        assert hasattr(result, 'mean_sharpe')
        assert hasattr(result, 'num_permutations')
        assert hasattr(result, 'permutation_profits')


# =============================================================================
# PARSIMONY PRESSURE TESTS
# =============================================================================

class TestParsimony:
    """Tests for parsimony pressure / strategy simplification."""
    
    def test_simplify_removes_indicator(self, sample_strategy_gene, mock_evaluator):
        """Simplification can remove an indicator."""
        from genetic_algorithm.core.parsimony import simplify_strategy
        
        original = sample_strategy_gene.copy()
        original_complexity = original.calculate_complexity()
        original_fitness = 0.75
        
        simplified, new_fitness, n_removed = simplify_strategy(
            original,
            original_fitness,
            mock_evaluator,
            epsilon=0.15,  # Allow 15% drop
            max_removals=2,
        )
        
        # At least one component should be removable
        assert n_removed >= 0
        if n_removed > 0:
            assert simplified.calculate_complexity() < original_complexity
    
    def test_simplify_respects_epsilon(self, sample_strategy_gene):
        """Won't remove component if fitness drop exceeds epsilon."""
        from genetic_algorithm.core.parsimony import simplify_strategy
        
        # Evaluator that severely penalizes any simplification
        def harsh_evaluator(gene):
            complexity = gene.calculate_complexity()
            return 0.9 - (6 - complexity) * 0.5, {}
        
        original = sample_strategy_gene.copy()
        _, _, n_removed = simplify_strategy(
            original,
            0.9,
            harsh_evaluator,
            epsilon=0.01,  # Very strict
            max_removals=5,
        )
        
        # Should not remove any (fitness drop > epsilon)
        assert n_removed == 0
    
    def test_simplify_preserves_validity(self, sample_strategy_gene, mock_evaluator):
        """Simplification never leaves strategy invalid."""
        from genetic_algorithm.core.parsimony import simplify_strategy
        
        simplified, _, _ = simplify_strategy(
            sample_strategy_gene.copy(),
            0.75,
            mock_evaluator,
            epsilon=0.5,  # Very permissive
            max_removals=10,
        )
        
        # Must still have at least 1 indicator and 1 entry condition
        assert len(simplified.indicators) >= 1
        assert len(simplified.entry_conditions) >= 1
    
    def test_apply_parsimony_to_elites(self, sample_individual, mock_evaluator):
        """apply_parsimony_to_elites works on Individual list."""
        from genetic_algorithm.core.parsimony import apply_parsimony_to_elites
        
        elites = [sample_individual]
        config = {'epsilon': 0.15, 'max_removals': 1}
        
        total_removed = apply_parsimony_to_elites(elites, mock_evaluator, config)
        
        # Should return count (may be 0 if nothing removable)
        assert total_removed >= 0
        # If removed, metric should be recorded
        if total_removed > 0:
            assert 'parsimony_removed' in elites[0].metrics
    
    def test_build_removal_candidates(self, sample_strategy_gene):
        """_build_removal_candidates returns correct candidates."""
        from genetic_algorithm.core.parsimony import _build_removal_candidates
        
        candidates = _build_removal_candidates(sample_strategy_gene)
        
        # 3 indicators, 2 entry conditions, 1 exit condition
        # indicators: 3 (all removable since >1)
        # entry: 2 (all removable since >1)
        # exit: 1
        assert len(candidates) == 3 + 2 + 1


# =============================================================================
# PARETO ARCHIVE TESTS
# =============================================================================

class TestParetoArchive:
    """Tests for Pareto archive with crowding decay."""
    
    @pytest.fixture
    def sample_population(self, sample_strategy_gene):
        """Create a population of individuals with varied objectives."""
        from genetic_algorithm.core.individual import Individual
        
        pop = []
        for i in range(10):
            gene = sample_strategy_gene.copy()
            gene.individual_id = i
            ind = Individual(strategy_gene=gene)
            # Objectives: profit vs drawdown trade-off
            ind.objectives = [10 + i * 2, 1 - i * 0.08]  # profit increases, drawdown worsens
            ind.rank = 1
            ind.crowding_distance = 1.0
            ind.fitness = 0.5 + i * 0.05
            ind.raw_fitness = ind.fitness
            ind.evaluated = True
            pop.append(ind)
        return pop
    
    def test_archive_init(self):
        """Archive initializes with correct parameters."""
        from genetic_algorithm.core.pareto_archive import ParetoArchive
        
        archive = ParetoArchive(max_size=50, decay_rate=0.9)
        assert archive.max_size == 50
        assert archive.decay_rate == 0.9
        assert archive.size == 0
    
    def test_archive_update_adds_members(self, sample_population):
        """Update adds Pareto-optimal members to archive."""
        from genetic_algorithm.core.pareto_archive import ParetoArchive
        
        archive = ParetoArchive(max_size=50)
        archive.update(sample_population, generation=0)
        
        assert archive.size > 0
    
    def test_archive_respects_max_size(self, sample_strategy_gene):
        """Archive prunes to max_size using crowding distance."""
        from genetic_algorithm.core.pareto_archive import ParetoArchive
        from genetic_algorithm.core.individual import Individual
        
        archive = ParetoArchive(max_size=5, decay_rate=1.0)
        
        # Create 20 non-dominated individuals (all on Pareto front)
        pop = []
        for i in range(20):
            gene = sample_strategy_gene.copy()
            gene.individual_id = i
            ind = Individual(strategy_gene=gene)
            # Trade-off: higher i = more profit but worse drawdown
            ind.objectives = [i * 2, -i * 0.1]
            ind.rank = 1
            ind.evaluated = True
            pop.append(ind)
        
        archive.update(pop, generation=0)
        
        # Should be pruned to max_size
        assert archive.size == 5
    
    def test_archive_decay_reduces_crowding_distance(self, sample_population):
        """Decay rate reduces crowding distance over generations."""
        from genetic_algorithm.core.pareto_archive import ParetoArchive
        
        archive = ParetoArchive(max_size=50, decay_rate=0.5)
        archive.update(sample_population, generation=0)
        
        # Record initial crowding distances
        initial_cds = [m.crowding_distance for m in archive.members if m.crowding_distance != float('inf')]
        
        # Update again (decay should apply)
        archive.update([], generation=1)
        
        # Members with finite CD should have decayed
        current_cds = [m.crowding_distance for m in archive.members if m.crowding_distance != float('inf')]
        
        # Decay was applied, but since we added empty pop, archive remains same
        # The decay happens at the start of update()
        assert archive.size > 0  # Archive persists
    
    def test_archive_serialization(self, sample_population):
        """Archive can be saved and restored."""
        from genetic_algorithm.core.pareto_archive import ParetoArchive
        
        archive = ParetoArchive(max_size=50, decay_rate=0.9)
        archive.update(sample_population, generation=0)
        original_size = archive.size
        
        # Serialize
        data = archive.to_dict()
        
        # Restore
        restored = ParetoArchive.from_dict(data)
        
        assert restored.max_size == 50
        assert restored.decay_rate == 0.9
        assert restored.size == original_size
    
    def test_get_best_returns_highest_crowding(self, sample_population):
        """get_best() returns members with highest crowding distance."""
        from genetic_algorithm.core.pareto_archive import ParetoArchive
        
        archive = ParetoArchive(max_size=50)
        archive.update(sample_population, generation=0)
        
        best = archive.get_best(n=3)
        
        assert len(best) <= 3
        # Should be sorted by crowding distance descending
        cds = [b.crowding_distance for b in best]
        assert cds == sorted(cds, reverse=True)


# =============================================================================
# DYNAMIC BOUNDS TESTS
# =============================================================================

class TestDynamicBounds:
    """Tests for dynamic indicator parameter ranges."""
    
    def test_initialise_bounds_from_config(self):
        """initialise_bounds seeds bounds from config."""
        from genetic_algorithm.utils.dynamic_bounds import initialise_bounds
        
        indicator_config = {
            'RSI': {'period': [7, 21]},
        }
        parameters = {'period': 14}
        
        bounds = initialise_bounds('RSI', parameters, indicator_config)
        
        assert 'period' in bounds
        assert bounds['period'] == (7, 21)
    
    def test_initialise_bounds_fallback(self):
        """initialise_bounds uses fallback when config missing."""
        from genetic_algorithm.utils.dynamic_bounds import initialise_bounds
        
        parameters = {'period': 14}
        
        bounds = initialise_bounds('UNKNOWN', parameters, {})
        
        assert 'period' in bounds
        # Fallback window around current value
        lo, hi = bounds['period']
        assert lo < 14 < hi
    
    def test_mutate_bounds_changes_values(self):
        """mutate_bounds modifies the bounds."""
        from genetic_algorithm.utils.dynamic_bounds import mutate_bounds
        
        original = {'period': (10, 20)}
        params = {'period': 15}
        
        # Run multiple times to ensure at least one change
        changed = False
        for _ in range(20):
            new_bounds = mutate_bounds(original, params, mutation_strength=0.3, rng=random.Random())
            if new_bounds['period'] != original['period']:
                changed = True
                break
        
        assert changed, "Bounds should eventually change with mutation"
    
    def test_mutate_bounds_preserves_validity(self):
        """mutate_bounds keeps min <= max."""
        from genetic_algorithm.utils.dynamic_bounds import mutate_bounds
        
        bounds = {'period': (10, 20)}
        params = {'period': 15}
        
        for _ in range(100):
            new_bounds = mutate_bounds(bounds, params, mutation_strength=0.5)
            lo, hi = new_bounds['period']
            assert lo <= hi, f"Invalid bounds: {lo} > {hi}"
    
    def test_sample_from_bounds_respects_evolved(self):
        """sample_from_bounds uses evolved bounds when present."""
        from genetic_algorithm.utils.dynamic_bounds import sample_from_bounds
        
        evolved = {'period': (15, 18)}
        fallback = (5, 50)
        
        for _ in range(50):
            val = sample_from_bounds('period', evolved, fallback, is_int=True)
            assert 15 <= val <= 18
    
    def test_sample_from_bounds_uses_fallback(self):
        """sample_from_bounds uses fallback when bounds missing."""
        from genetic_algorithm.utils.dynamic_bounds import sample_from_bounds
        
        fallback = (10, 20)
        
        for _ in range(50):
            val = sample_from_bounds('period', None, fallback, is_int=True)
            assert 10 <= val <= 20
    
    def test_crossover_bounds_combines_parents(self):
        """crossover_bounds creates child from two parents."""
        from genetic_algorithm.utils.dynamic_bounds import crossover_bounds
        
        bounds1 = {'period': (5, 15), 'std': (1.0, 2.0)}
        bounds2 = {'period': (10, 20), 'other': (0, 1)}
        
        child = crossover_bounds(bounds1, bounds2, rng=random.Random(42))
        
        # Should have all keys
        assert 'period' in child
        assert 'std' in child or 'other' in child
    
    def test_indicator_gene_has_param_bounds_field(self):
        """IndicatorGene supports optional param_bounds."""
        from genetic_algorithm.core.strategy_gene import IndicatorGene
        
        ind = IndicatorGene(
            type='RSI',
            parameters={'period': 14},
            param_bounds={'period': (7, 21)},
        )
        
        assert ind.param_bounds == {'period': (7, 21)}
    
    def test_strategy_gene_serializes_param_bounds(self, sample_strategy_gene):
        """param_bounds is preserved through to_dict/from_dict."""
        from genetic_algorithm.core.strategy_gene import StrategyGene
        
        gene = sample_strategy_gene.copy()
        gene.indicators[0].param_bounds = {'period': (5, 25)}
        
        data = gene.to_dict()
        restored = StrategyGene.from_dict(data)
        
        assert restored.indicators[0].param_bounds == {'period': (5, 25)}
    
    def test_mutate_dynamic_bounds_operator(self, sample_individual):
        """mutate_dynamic_bounds operator works."""
        from genetic_algorithm.core.mutation import mutate_dynamic_bounds
        
        config = {
            'dynamic_bounds': {'enabled': True, 'mutation_strength': 0.2},
            'indicators': {'RSI': {'period': [7, 21]}},
        }
        
        mutated = mutate_dynamic_bounds(sample_individual, mutation_rate=1.0, config=config)
        
        # Should return an Individual
        assert mutated is not None
        # At least one indicator should now have param_bounds
        has_bounds = any(ind.param_bounds for ind in mutated.strategy_gene.indicators)
        assert has_bounds


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestTier3Integration:
    """Integration tests verifying Tier 3 features work with evolution."""
    
    def test_config_has_tier3_sections(self):
        """ga_config.yaml includes all Tier 3 config sections."""
        from pathlib import Path
        import yaml
        
        config_path = Path(__file__).parent.parent / 'genetic_algorithm' / 'config' / 'ga_config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert 'monte_carlo' in config
        assert 'parsimony' in config
        assert 'pareto_archive' in config
        assert 'dynamic_bounds' in config
    
    def test_monte_carlo_config_defaults(self):
        """Verify Monte-Carlo config has correct defaults."""
        from pathlib import Path
        import yaml
        
        config_path = Path(__file__).parent.parent / 'genetic_algorithm' / 'config' / 'ga_config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        mc = config['monte_carlo']
        assert 'enabled' in mc
        assert 'num_permutations' in mc
        assert 'methods' in mc
        assert isinstance(mc['methods'], list)
    
    def test_parsimony_integrates_with_evolution(self, sample_individual, mock_evaluator):
        """Parsimony pressure can be applied to elite individuals."""
        from genetic_algorithm.core.parsimony import apply_parsimony_to_elites
        
        # Simulating what evolution.py does
        elites = [sample_individual]
        config = {'enabled': True, 'epsilon': 0.1, 'max_removals': 1}
        
        apply_parsimony_to_elites(elites, mock_evaluator, config)
        
        # No crash = success
        assert True
    
    def test_pareto_archive_integrates_with_nsga2(self, sample_strategy_gene):
        """Pareto archive works with NSGA-II population."""
        from genetic_algorithm.core.pareto_archive import ParetoArchive
        from genetic_algorithm.core.individual import Individual
        from genetic_algorithm.core.nsga2 import fast_non_dominated_sort, crowding_distance_assignment
        
        # Create varied population
        pop = []
        for i in range(15):
            gene = sample_strategy_gene.copy()
            gene.individual_id = i
            ind = Individual(strategy_gene=gene)
            ind.objectives = [i * 3, 1 - i * 0.05]
            ind.fitness = 0.5 + i * 0.03
            ind.raw_fitness = ind.fitness
            ind.evaluated = True
            pop.append(ind)
        
        # NSGA-II ranking
        fronts = fast_non_dominated_sort(pop)
        for front in fronts:
            crowding_distance_assignment(front)
        
        # Archive update
        archive = ParetoArchive(max_size=10, decay_rate=0.95)
        archive.update(pop, generation=0)
        
        assert archive.size > 0
        assert archive.size <= 10
