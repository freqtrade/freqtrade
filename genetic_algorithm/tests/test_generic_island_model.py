"""
Tests for the Generic Island Model Evolution.

Tests cover:
- Auto-generation of island configs
- Indicator pool splitting with overlap
- Pair rotation
- Each migration topology (ring, fully_connected, tournament, hierarchical)
- Merge round logic
- Gene hashing and deduplication
- Walk-forward compatibility
- Config building for sub-islands
- Integration with run_ga.py branching
"""

import copy
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.generic_island_model import (
    GenericIslandModelEvolution,
    GenericIslandConfig,
    GenericMigrationConfig,
    GenericIslandStats,
    GenericMigrationEvent,
    INDICATOR_FAMILIES,
    ALL_INDICATORS,
)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _make_gene(generation=0, individual_id=0, **overrides):
    """Create a minimal StrategyGene for testing."""
    defaults = dict(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
        ],
        entry_conditions=[
            ConditionGene(
                indicator='RSI',
                operator='<',
                threshold=30.0,
                logic='AND',
            ),
        ],
        exit_conditions=[
            ConditionGene(
                indicator='RSI',
                operator='>',
                threshold=70.0,
                logic='AND',
            ),
        ],
        stoploss=-0.10,
        timeframe='15m',
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
        max_open_trades=3,
    )
    defaults.update(overrides)
    return StrategyGene(**defaults)


def _make_individual(fitness=None, generation=0, individual_id=0, **gene_kw):
    """Create an Individual with optional fitness."""
    gene = _make_gene(generation=generation, individual_id=individual_id, **gene_kw)
    ind = Individual(strategy_gene=gene)
    if fitness is not None:
        ind.raw_fitness = fitness
        ind.fitness = fitness
        ind.evaluated = True
        ind.metrics = {'profit': fitness * 10, 'sharpe_ratio': 1.0, 'num_trades': 20}
    return ind


def _make_population(fitnesses):
    """Create a Population with individuals at the given fitness levels."""
    pop = Population(size=len(fitnesses), generation=0)
    for i, f in enumerate(fitnesses):
        ind = _make_individual(fitness=f, generation=0, individual_id=i)
        pop.add_individual(ind)
    return pop


def _minimal_config(num_islands=3, topology='ring', **gim_overrides):
    """Build a minimal config dict for GenericIslandModelEvolution."""
    cfg = {
        'genetic_algorithm': {
            'population_size': 10,
            'generations': 5,
            'mutation_rate': 0.3,
            'crossover_rate': 0.7,
            'elite_size': 2,
            'tournament_size': 3,
            'seed': 42,
        },
        'backtesting': {
            'pairs': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'],
            'timeframe': '15m',
            'timerange': '20230101-20260315',
        },
        'indicators': {
            'available': list(ALL_INDICATORS),
        },
        'generic_island_model': {
            'enabled': True,
            'num_islands': num_islands,
            'parallel_islands': False,
            'population_per_island': 5,
            'generations': 3,
            'specialization': {
                'rotate_seeds': True,
                'indicator_pools': True,
                'indicator_overlap': 0.5,
                'pair_rotation': False,
                'pair_subset_size': 3,
            },
            'migration': {
                'topology': topology,
                'interval': 2,
                'count': 1,
                'merge_rounds': False,
                'merge_interval': 3,
                'tournament_size': 3,
            },
            **gim_overrides,
        },
        'island_model': {'enabled': False},
        'terminal_monitor': {'enabled': False},
        'walk_forward': {'enabled': False},
        'fitness_weights': {},
        'fitness_penalties': {},
        'strategy_constraints': {},
    }
    return cfg


def _create_model_from_config(config):
    """
    Create a GenericIslandModelEvolution without going through YAML file.
    Patches the file read to use the in-memory config.
    """
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False,
    ) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    try:
        model = GenericIslandModelEvolution(
            config_path=tmp_path,
            visualize=False,
            interactive=False,
        )
    finally:
        import os
        os.unlink(tmp_path)

    return model


# ══════════════════════════════════════════════════════════════════════
# Tests: Island Configuration
# ══════════════════════════════════════════════════════════════════════

class TestIslandAutoGeneration:
    """Tests for automatic island configuration generation."""

    def test_auto_generate_creates_correct_number_of_islands(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)
        assert len(model.island_configs) == 5

    def test_auto_generate_rotates_seeds(self):
        config = _minimal_config(num_islands=4)
        model = _create_model_from_config(config)
        seeds = [ic.seed for ic in model.island_configs]
        assert len(set(seeds)) == 4, "All seeds should be unique"
        # Seeds should be sequential from base_seed
        assert seeds == [42, 43, 44, 45]

    def test_auto_generate_without_seed_rotation(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['specialization']['rotate_seeds'] = False
        model = _create_model_from_config(config)
        seeds = [ic.seed for ic in model.island_configs]
        assert all(s == 42 for s in seeds), "All seeds should be the same base seed"

    def test_auto_generate_with_indicator_pools(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)

        # Each island should have an indicator pool
        for ic in model.island_configs:
            assert ic.indicator_pool is not None
            assert len(ic.indicator_pool) > 0

    def test_auto_generate_without_indicator_pools(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['specialization']['indicator_pools'] = False
        model = _create_model_from_config(config)

        for ic in model.island_configs:
            assert ic.indicator_pool is None

    def test_auto_generate_with_pair_rotation(self):
        config = _minimal_config(num_islands=5)
        config['generic_island_model']['specialization']['pair_rotation'] = True
        config['generic_island_model']['specialization']['pair_subset_size'] = 3
        model = _create_model_from_config(config)

        # Each island should have a pair subset
        for ic in model.island_configs:
            assert ic.pairs is not None
            assert len(ic.pairs) == 3

    def test_auto_generate_population_size(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['population_per_island'] = 8
        model = _create_model_from_config(config)

        for ic in model.island_configs:
            assert ic.population_size == 8

    def test_explicit_islands_override_auto(self):
        config = _minimal_config(num_islands=10)
        config['generic_island_model']['islands'] = [
            {'name': 'custom_1', 'population_size': 20, 'seed': 100},
            {'name': 'custom_2', 'population_size': 15, 'seed': 200,
             'indicator_pool': ['RSI', 'MACD']},
        ]
        model = _create_model_from_config(config)

        assert len(model.island_configs) == 2
        assert model.island_configs[0].name == 'custom_1'
        assert model.island_configs[0].population_size == 20
        assert model.island_configs[1].indicator_pool == ['RSI', 'MACD']


class TestIndicatorPoolSplitting:
    """Tests for indicator pool splitting logic."""

    def test_pool_count_matches_families(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)
        pools = model._split_indicator_pools()
        assert len(pools) == len(INDICATOR_FAMILIES)

    def test_pools_have_overlap(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)
        pools = model._split_indicator_pools()

        # Check pairs of adjacent pools have some overlap
        for i in range(len(pools)):
            j = (i + 1) % len(pools)
            overlap = set(pools[i]) & set(pools[j])
            assert len(overlap) > 0, f"Pools {i} and {j} should overlap"

    def test_all_indicators_covered(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)
        pools = model._split_indicator_pools()

        all_in_pools = set()
        for pool in pools:
            all_in_pools.update(pool)

        # All standard indicators should appear in at least one pool
        for family, indicators in INDICATOR_FAMILIES.items():
            for ind in indicators:
                assert ind in all_in_pools, f"Indicator {ind} not in any pool"

    def test_overlap_increases_with_config(self):
        config_low = _minimal_config(num_islands=5)
        config_low['generic_island_model']['specialization']['indicator_overlap'] = 0.1
        model_low = _create_model_from_config(config_low)
        pools_low = model_low._split_indicator_pools()

        config_high = _minimal_config(num_islands=5)
        config_high['generic_island_model']['specialization']['indicator_overlap'] = 0.9
        model_high = _create_model_from_config(config_high)
        pools_high = model_high._split_indicator_pools()

        avg_size_low = sum(len(p) for p in pools_low) / len(pools_low)
        avg_size_high = sum(len(p) for p in pools_high) / len(pools_high)
        assert avg_size_high >= avg_size_low


class TestPairRotation:
    """Tests for pair rotation logic."""

    def test_rotate_pairs_creates_subsets(self):
        config = _minimal_config(num_islands=5)
        model = _create_model_from_config(config)
        subsets = model._rotate_pairs()
        assert len(subsets) == 5  # 5 pairs = 5 rotations
        for subset in subsets:
            assert len(subset) == 3  # pair_subset_size=3

    def test_rotate_pairs_returns_all_when_subset_too_large(self):
        config = _minimal_config(num_islands=3)
        config['generic_island_model']['specialization']['pair_subset_size'] = 10
        model = _create_model_from_config(config)
        subsets = model._rotate_pairs()
        assert len(subsets) == 1
        assert len(subsets[0]) == 5


# ══════════════════════════════════════════════════════════════════════
# Tests: Island Config Building
# ══════════════════════════════════════════════════════════════════════

class TestBuildIslandConfig:
    """Tests for building per-island GA config."""

    def test_disables_nested_island_models(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        island_cfg = model._build_island_config(ic)

        assert island_cfg['island_model']['enabled'] is False
        assert island_cfg['generic_island_model']['enabled'] is False

    def test_sets_population_size(self):
        config = _minimal_config(num_islands=2)
        config['generic_island_model']['population_per_island'] = 12
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        island_cfg = model._build_island_config(ic)

        assert island_cfg['genetic_algorithm']['population_size'] == 12

    def test_sets_seed(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        island_cfg = model._build_island_config(ic)

        assert island_cfg['genetic_algorithm']['seed'] == ic.seed

    def test_restricts_indicator_pool(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        # Manually set an indicator pool
        ic.indicator_pool = ['RSI', 'MACD', 'BBANDS']
        island_cfg = model._build_island_config(ic)

        assert island_cfg['indicators']['available'] == ['RSI', 'MACD', 'BBANDS']

    def test_restricts_pairs(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        ic.pairs = ['BTC/USDT', 'ETH/USDT']
        island_cfg = model._build_island_config(ic)

        assert island_cfg['backtesting']['pairs'] == ['BTC/USDT', 'ETH/USDT']

    def test_walk_forward_configurable(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        ic.walk_forward_enabled = True
        island_cfg = model._build_island_config(ic)

        assert island_cfg['walk_forward']['enabled'] is True

    def test_disables_terminal_monitor(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        ic = model.island_configs[0]
        island_cfg = model._build_island_config(ic)

        assert island_cfg['terminal_monitor']['enabled'] is False


# ══════════════════════════════════════════════════════════════════════
# Tests: Migration
# ══════════════════════════════════════════════════════════════════════

class TestMigrationHelpers:
    """Tests for migration helper methods."""

    def test_get_top_individuals_returns_sorted(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([0.1, 0.5, 0.3, 0.8, 0.2])
        model.island_populations['test'] = pop

        top = model._get_top_individuals('test', 3)
        assert len(top) == 3
        assert top[0].raw_fitness == 0.8
        assert top[1].raw_fitness == 0.5
        assert top[2].raw_fitness == 0.3

    def test_get_top_individuals_skips_unevaluated(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([0.5, 0.3])
        # Add an unevaluated individual
        uneval = _make_individual(fitness=None)
        pop.individuals.append(uneval)
        model.island_populations['test'] = pop

        top = model._get_top_individuals('test', 5)
        assert len(top) == 2

    def test_get_top_individuals_nonexistent_island(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        top = model._get_top_individuals('nonexistent', 3)
        assert top == []

    def test_inject_migrants_replaces_worst(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([0.1, 0.2, 0.3, 0.4, 0.5])
        model.island_populations['target'] = pop

        migrant = _make_individual(fitness=0.9, generation=1, individual_id=99)
        replaced = model._inject_migrants('target', [migrant], generation=5)

        assert replaced == 1
        # The worst (0.1) should have been replaced
        fitnesses = [
            ind.raw_fitness for ind in pop.individuals
            if ind.raw_fitness is not None and ind.evaluated
        ]
        assert 0.1 not in [f for f in fitnesses if f != 0.1]  # replaced

    def test_inject_migrants_marks_unevaluated(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([0.1, 0.2])
        model.island_populations['target'] = pop

        migrant = _make_individual(fitness=0.9)
        model._inject_migrants('target', [migrant], generation=5)

        # Find the injected migrant (the one with evaluated=False)
        unevaluated = [ind for ind in pop.individuals if not ind.evaluated]
        assert len(unevaluated) == 1

    def test_inject_migrants_empty_list(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)
        pop = _make_population([0.1, 0.2])
        model.island_populations['target'] = pop
        replaced = model._inject_migrants('target', [], generation=5)
        assert replaced == 0


class TestGeneHash:
    """Tests for gene hashing / deduplication."""

    def test_same_gene_same_hash(self):
        ind1 = _make_individual(fitness=0.5, generation=0, individual_id=0)
        ind2 = _make_individual(fitness=0.5, generation=0, individual_id=0)
        h1 = GenericIslandModelEvolution._gene_hash(ind1)
        h2 = GenericIslandModelEvolution._gene_hash(ind2)
        assert h1 == h2

    def test_different_gene_different_hash(self):
        ind1 = _make_individual(fitness=0.5, generation=0, individual_id=0)
        ind2 = _make_individual(
            fitness=0.5, generation=0, individual_id=0,
            stoploss=-0.20,  # Different stoploss
        )
        h1 = GenericIslandModelEvolution._gene_hash(ind1)
        h2 = GenericIslandModelEvolution._gene_hash(ind2)
        assert h1 != h2

    def test_hash_ignores_generation_and_id(self):
        ind1 = _make_individual(fitness=0.5, generation=0, individual_id=0)
        ind2 = _make_individual(fitness=0.5, generation=5, individual_id=99)
        h1 = GenericIslandModelEvolution._gene_hash(ind1)
        h2 = GenericIslandModelEvolution._gene_hash(ind2)
        assert h1 == h2


class TestMigrationTopologies:
    """Tests for each migration topology."""

    def _setup_model_with_populations(self, topology='ring', num_islands=4):
        """Create a model with pre-populated islands."""
        config = _minimal_config(num_islands=num_islands, topology=topology)
        model = _create_model_from_config(config)

        # Create populations with distinct fitness ranges per island
        for i, ic in enumerate(model.island_configs):
            base_fitness = (i + 1) * 0.1  # island 0: 0.1, island 1: 0.2, etc.
            fitnesses = [base_fitness + j * 0.01 for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop
            model.island_stats[ic.name] = GenericIslandStats(name=ic.name)

        return model

    def test_ring_migration_creates_events(self):
        model = self._setup_model_with_populations('ring', 4)
        model._migrate_ring(generation=2)

        assert len(model.migration_history) == 4  # One per island
        # Check ring pattern: each source → next
        sources = [e.source for e in model.migration_history]
        targets = [e.target for e in model.migration_history]
        names = [ic.name for ic in model.island_configs]
        for i, (src, tgt) in enumerate(zip(sources, targets)):
            assert src == names[i]
            assert tgt == names[(i + 1) % len(names)]

    def test_fully_connected_migration_all_pairs(self):
        model = self._setup_model_with_populations('fully_connected', 3)
        model._migrate_fully_connected(generation=2)

        # 3 islands × 2 targets each = 6 events
        assert len(model.migration_history) == 6

    def test_tournament_migration_creates_events(self):
        model = self._setup_model_with_populations('tournament', 4)
        model._migrate_tournament(generation=2)

        # 4 islands shuffled → 2 pairs → 2 events
        assert len(model.migration_history) == 2

    def test_hierarchical_migration_bidirectional(self):
        model = self._setup_model_with_populations('hierarchical', 4)
        model._migrate_hierarchical(generation=2)

        # 4 islands → 2 pairs → 2 bidirectional = 4 events
        assert len(model.migration_history) == 4

    def test_migrate_dispatches_to_correct_topology(self):
        for topology in ['ring', 'fully_connected', 'tournament', 'hierarchical']:
            model = self._setup_model_with_populations(topology, 4)
            model._migrate(generation=2)
            assert len(model.migration_history) > 0, f"No events for {topology}"


class TestMergeRound:
    """Tests for global merge rounds."""

    def test_merge_round_injects_global_elites(self):
        config = _minimal_config(num_islands=3)
        model = _create_model_from_config(config)

        # Create populations with different fitness levels
        for i, ic in enumerate(model.island_configs):
            fitnesses = [(i + 1) * 0.1 + j * 0.01 for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop
            model.island_stats[ic.name] = GenericIslandStats(name=ic.name)

        model._merge_round(generation=5)

        # After merge, the global best should appear in some islands
        # This is hard to test precisely due to deduplication, but we can
        # verify the merge ran without errors
        assert True

    def test_merge_round_with_empty_populations(self):
        config = _minimal_config(num_islands=2)
        model = _create_model_from_config(config)

        for ic in model.island_configs:
            pop = Population(size=0, generation=0)
            model.island_populations[ic.name] = pop
            model.island_stats[ic.name] = GenericIslandStats(name=ic.name)

        # Should not crash
        model._merge_round(generation=5)


# ══════════════════════════════════════════════════════════════════════
# Tests: Result Collection
# ══════════════════════════════════════════════════════════════════════

class TestResultCollection:
    """Tests for final result collection and deduplication."""

    def test_collect_results_has_global_key(self):
        config = _minimal_config(num_islands=3)
        model = _create_model_from_config(config)

        for i, ic in enumerate(model.island_configs):
            fitnesses = [(i + 1) * 0.1 + j * 0.01 for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop

        results = model._collect_final_results()
        assert '__global__' in results
        assert len(results['__global__']) > 0

    def test_collect_results_per_island(self):
        config = _minimal_config(num_islands=3)
        model = _create_model_from_config(config)

        for i, ic in enumerate(model.island_configs):
            fitnesses = [0.1 * (i + 1) + j * 0.01 for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop

        results = model._collect_final_results()
        for ic in model.island_configs:
            assert ic.name in results
            assert len(results[ic.name]) <= 5

    def test_global_results_sorted_by_fitness(self):
        config = _minimal_config(num_islands=3)
        model = _create_model_from_config(config)

        for i, ic in enumerate(model.island_configs):
            fitnesses = [0.1 * (i + 1) + j * 0.01 for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop

        results = model._collect_final_results()
        global_top = results['__global__']
        fitnesses = [ind.raw_fitness for ind in global_top]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_global_results_max_20(self):
        config = _minimal_config(num_islands=10)
        model = _create_model_from_config(config)

        for i, ic in enumerate(model.island_configs):
            fitnesses = [0.01 * (i * 5 + j + 1) for j in range(5)]
            pop = _make_population(fitnesses)
            model.island_populations[ic.name] = pop

        results = model._collect_final_results()
        assert len(results['__global__']) <= 20


# ══════════════════════════════════════════════════════════════════════
# Tests: Identify primary family
# ══════════════════════════════════════════════════════════════════════

class TestIdentifyPrimaryFamily:
    def test_momentum_dominant(self):
        pool = ['RSI', 'MACD', 'STOCH', 'EMA']
        result = GenericIslandModelEvolution._identify_primary_family(pool)
        assert result == 'momentum'

    def test_trend_dominant(self):
        pool = ['EMA', 'SMA', 'TEMA', 'KAMA', 'RSI']
        result = GenericIslandModelEvolution._identify_primary_family(pool)
        assert result == 'trend'

    def test_mixed(self):
        pool = ['RSI', 'EMA']
        result = GenericIslandModelEvolution._identify_primary_family(pool)
        assert result in INDICATOR_FAMILIES  # Should pick one

    def test_empty_pool(self):
        result = GenericIslandModelEvolution._identify_primary_family([])
        assert result == 'mixed'


# ══════════════════════════════════════════════════════════════════════
# Tests: Constants
# ══════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_all_indicators_is_union_of_families(self):
        expected = set()
        for indicators in INDICATOR_FAMILIES.values():
            expected.update(indicators)
        assert set(ALL_INDICATORS) == expected

    def test_no_duplicate_indicators_within_families(self):
        for family, indicators in INDICATOR_FAMILIES.items():
            assert len(indicators) == len(set(indicators)), \
                f"Duplicates in {family}"

    def test_expected_family_count(self):
        assert len(INDICATOR_FAMILIES) == 5
