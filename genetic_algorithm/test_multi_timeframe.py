"""
Tests for Multi-Timeframe Strategy Implementation

Tests cover:
1. StrategyGene multi-TF fields (informative_timeframes, indicator timeframe)
2. Instance ID assignment with multi-TF indicators
3. Strategy generation with multi-TF enabled
4. Code generation with informative pairs and merge_informative_pair
5. Mutation operators for multi-TF (add/remove/change timeframes)
6. Crossover operators preserving multi-TF state
7. Serialization (to_dict / from_dict) roundtrip
8. Timeframe ordering helpers
"""

import random
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import (
    StrategyGene, IndicatorGene, ConditionGene,
    timeframe_to_minutes, is_higher_timeframe, TIMEFRAME_ORDER,
)
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.mutation import mutate, mutate_timeframes
from genetic_algorithm.core.crossover import crossover
from genetic_algorithm.strategies.generator import StrategyGenerator


# ---------------------------------------------------------------------------
# Helper: minimal config
# ---------------------------------------------------------------------------
def _make_config(multi_tf_enabled=False):
    """Return a minimal GA config dict for testing."""
    cfg = {
        'indicators': {
            'available': ['RSI', 'EMA', 'SMA', 'MACD', 'BBANDS', 'ADX', 'ATR', 'CCI', 'STOCH'],
            'max_per_strategy': 5,
            'min_per_strategy': 1,
            'RSI': {'period': [7, 21], 'buy_threshold': [20, 40], 'sell_threshold': [60, 80]},
            'EMA': {'period': [5, 50]},
            'SMA': {'period': [10, 100]},
            'MACD': {'fast_period': [8, 21], 'slow_period': [21, 50], 'signal_period': [5, 14]},
            'BBANDS': {'period': [15, 30], 'std_dev': [1.5, 3.0]},
            'ADX': {'period': [10, 20], 'threshold': [20, 40]},
            'ATR': {'period': [10, 20]},
            'CCI': {'period': [10, 30], 'buy_threshold': [-200, -100], 'sell_threshold': [100, 200]},
            'STOCH': {'k_period': [5, 21], 'd_period': [3, 14], 'k_threshold': [20, 40], 'd_threshold': [60, 80]},
        },
        'strategy_constraints': {
            'timeframes': ['5m', '15m', '1h'],
            'stoploss_range': [-0.20, -0.05],
            'roi_range': [0.01, 0.10],
        },
        'multi_timeframe': {
            'enabled': multi_tf_enabled,
            'available': ['15m', '1h', '4h'],
            'max_timeframes': 2,
            'higher_timeframe_preference': ['EMA', 'SMA', 'ADX', 'RSI'],
        },
    }
    return cfg


def _make_base_strategy():
    """Create a simple single-TF strategy for testing."""
    return StrategyGene(
        generation=0, individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='EMA', parameters={'period': 20}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='cross_below', threshold=30),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='cross_above', threshold=70),
        ],
        timeframe='5m',
    )


def _make_multi_tf_strategy():
    """Create a multi-TF strategy for testing."""
    return StrategyGene(
        generation=0, individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='EMA', parameters={'period': 20}, timeframe='1h'),
            IndicatorGene(type='ADX', parameters={'period': 14}, timeframe='4h'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='cross_below', threshold=30),
            ConditionGene(indicator='EMA', operator='cross_above', threshold=0, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='cross_above', threshold=70),
        ],
        timeframe='5m',
        informative_timeframes=['1h', '4h'],
    )


# ===========================================================================
# 1. Timeframe ordering helpers
# ===========================================================================
def test_timeframe_to_minutes():
    assert timeframe_to_minutes('1m') == 1
    assert timeframe_to_minutes('5m') == 5
    assert timeframe_to_minutes('1h') == 60
    assert timeframe_to_minutes('4h') == 240
    assert timeframe_to_minutes('1d') == 1440
    assert timeframe_to_minutes('unknown') == 0


def test_is_higher_timeframe():
    assert is_higher_timeframe('1h', '5m') is True
    assert is_higher_timeframe('5m', '1h') is False
    assert is_higher_timeframe('5m', '5m') is False
    assert is_higher_timeframe('4h', '1h') is True
    assert is_higher_timeframe('1d', '4h') is True


# ===========================================================================
# 2. IndicatorGene timeframe field
# ===========================================================================
def test_indicator_gene_timeframe_default():
    ind = IndicatorGene(type='RSI', parameters={'period': 14})
    assert ind.timeframe is None


def test_indicator_gene_timeframe_set():
    ind = IndicatorGene(type='EMA', parameters={'period': 20}, timeframe='1h')
    assert ind.timeframe == '1h'


# ===========================================================================
# 3. StrategyGene multi-TF fields
# ===========================================================================
def test_strategy_gene_informative_timeframes_default():
    sg = _make_base_strategy()
    assert sg.informative_timeframes == []


def test_strategy_gene_informative_timeframes_set():
    sg = _make_multi_tf_strategy()
    assert sg.informative_timeframes == ['1h', '4h']


def test_strategy_gene_get_base_indicators():
    sg = _make_multi_tf_strategy()
    base = sg.get_base_indicators()
    assert len(base) == 1
    assert base[0].type == 'RSI'


def test_strategy_gene_get_informative_indicators():
    sg = _make_multi_tf_strategy()
    inf = sg.get_informative_indicators()
    assert len(inf) == 2
    types = {i.type for i in inf}
    assert types == {'EMA', 'ADX'}


def test_strategy_gene_get_indicators_by_timeframe():
    sg = _make_multi_tf_strategy()
    assert len(sg.get_indicators_by_timeframe(None)) == 1
    assert len(sg.get_indicators_by_timeframe('1h')) == 1
    assert len(sg.get_indicators_by_timeframe('4h')) == 1
    assert len(sg.get_indicators_by_timeframe('15m')) == 0


# ===========================================================================
# 4. Assign instance IDs with multi-TF
# ===========================================================================
def test_assign_instance_ids_multi_tf():
    sg = _make_multi_tf_strategy()
    sg.assign_instance_ids()
    
    ids = [ind.instance_id for ind in sg.indicators]
    # Base RSI => 'RSI_0', informative EMA => 'EMA_1h_0', informative ADX => 'ADX_4h_0'
    assert ids[0] == 'RSI_0'
    assert ids[1] == 'EMA_1h_0'
    assert ids[2] == 'ADX_4h_0'


def test_assign_instance_ids_multi_tf_same_type():
    """Two RSI indicators on different TFs get distinct IDs."""
    sg = StrategyGene(
        generation=0, individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='RSI', parameters={'period': 21}, timeframe='1h'),
        ],
        entry_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=30)],
        timeframe='5m',
        informative_timeframes=['1h'],
    )
    sg.assign_instance_ids()
    ids = [ind.instance_id for ind in sg.indicators]
    assert ids[0] == 'RSI_0'
    assert ids[1] == 'RSI_1h_0'


# ===========================================================================
# 5. Serialization roundtrip (to_dict / from_dict)
# ===========================================================================
def test_to_dict_from_dict_roundtrip():
    sg = _make_multi_tf_strategy()
    sg.assign_instance_ids()
    
    d = sg.to_dict()
    assert d['informative_timeframes'] == ['1h', '4h']
    assert d['indicators'][1]['timeframe'] == '1h'
    assert d['indicators'][2]['timeframe'] == '4h'
    
    sg2 = StrategyGene.from_dict(d)
    assert sg2.informative_timeframes == ['1h', '4h']
    assert sg2.indicators[1].timeframe == '1h'
    assert sg2.indicators[2].timeframe == '4h'


def test_copy_preserves_multi_tf():
    sg = _make_multi_tf_strategy()
    sg.assign_instance_ids()
    
    sg2 = sg.copy()
    assert sg2.informative_timeframes == sg.informative_timeframes
    assert sg2.indicators[1].timeframe == '1h'


# ===========================================================================
# 6. Strategy generator with multi-TF
# ===========================================================================
def test_generator_without_multi_tf():
    """Without multi-TF enabled, no informative indicators are created."""
    random.seed(42)
    config = _make_config(multi_tf_enabled=False)
    gen = StrategyGenerator(config)
    strategy = gen.generate_random_strategy(0, 0)
    
    assert strategy.informative_timeframes == []
    assert all(ind.timeframe is None for ind in strategy.indicators)


def test_generator_with_multi_tf():
    """With multi-TF enabled, informative indicators should be created."""
    random.seed(42)
    config = _make_config(multi_tf_enabled=True)
    gen = StrategyGenerator(config)
    
    # Generate several strategies; at least some should have informative TFs
    has_informative = False
    for i in range(20):
        strategy = gen.generate_random_strategy(0, i)
        if strategy.informative_timeframes:
            has_informative = True
            # Verify all informative TFs are higher than the base TF
            for itf in strategy.informative_timeframes:
                assert is_higher_timeframe(itf, strategy.timeframe), \
                    f"{itf} should be higher than {strategy.timeframe}"
            # Verify informative indicators exist
            inf_inds = strategy.get_informative_indicators()
            assert len(inf_inds) >= 1
            break
    
    assert has_informative, "At least one strategy should have informative timeframes"


# ===========================================================================
# 7. Code generation with multi-TF
# ===========================================================================
def test_code_generation_single_tf():
    """Single-TF strategy code should not reference informative pair logic."""
    config = _make_config()
    gen = StrategyGenerator(config)
    sg = _make_base_strategy()
    
    code = gen.generate_strategy_code(sg)
    assert 'merge_informative_pair' in code  # Import always present in generated code
    assert 'def informative_pairs(self):' in code
    assert 'return []' in code  # No informative pairs for single-TF


def test_code_generation_multi_tf():
    """Multi-TF strategy code should include informative_pairs and merge logic."""
    config = _make_config(multi_tf_enabled=True)
    gen = StrategyGenerator(config)
    sg = _make_multi_tf_strategy()
    
    code = gen.generate_strategy_code(sg)
    
    assert 'merge_informative_pair' in code
    assert 'def informative_pairs(self):' in code
    assert 'self.dp.get_pair_dataframe' in code
    assert "inf_tf = '1h'" in code or "inf_tf = '4h'" in code


def test_code_generation_informative_condition_suffix():
    """Conditions on informative indicators should use the TF suffix in column names."""
    config = _make_config(multi_tf_enabled=True)
    gen = StrategyGenerator(config)
    
    # Create strategy with an EMA on 1h and a condition referencing it
    sg = StrategyGene(
        generation=0, individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='EMA', parameters={'period': 20}, timeframe='1h'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='cross_below', threshold=30),
            ConditionGene(indicator='EMA', operator='cross_above', threshold=0, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='cross_above', threshold=70),
        ],
        timeframe='5m',
        informative_timeframes=['1h'],
    )
    
    code = gen.generate_strategy_code(sg)
    
    # The EMA condition should reference the _1h suffix column
    assert 'ema_20_1h' in code or 'close_1h' in code


# ===========================================================================
# 8. Mutation: mutate_timeframes
# ===========================================================================
def test_mutate_timeframes_disabled():
    """When multi-TF is disabled, mutation should return unchanged individual."""
    config = _make_config(multi_tf_enabled=False)
    sg = _make_base_strategy()
    ind = Individual(strategy_gene=sg)
    
    result = mutate_timeframes(ind, 1.0, config)
    assert result is ind  # Same object returned


def test_mutate_timeframes_add():
    """Mutation should be able to add an informative timeframe."""
    random.seed(10)
    config = _make_config(multi_tf_enabled=True)
    sg = _make_base_strategy()
    sg.assign_instance_ids()
    ind = Individual(strategy_gene=sg)
    
    # Run mutation multiple times to ensure at least one adds a TF
    added = False
    for _ in range(50):
        result = mutate_timeframes(ind, 1.0, config)
        if result.strategy_gene.informative_timeframes:
            added = True
            assert len(result.strategy_gene.get_informative_indicators()) >= 1
            break
    
    assert added, "Timeframe mutation should add an informative TF"


def test_mutate_timeframes_remove():
    """Mutation should be able to remove an informative timeframe."""
    random.seed(42)
    config = _make_config(multi_tf_enabled=True)
    sg = _make_multi_tf_strategy()
    sg.assign_instance_ids()
    ind = Individual(strategy_gene=sg)
    
    # Run mutation multiple times
    removed = False
    for _ in range(50):
        result = mutate_timeframes(ind, 1.0, config)
        if len(result.strategy_gene.informative_timeframes) < 2:
            removed = True
            break
    
    assert removed, "Timeframe mutation should be able to remove a TF"


def test_mutate_full_with_timeframes():
    """Full mutate() should include timeframe mutations when multi-TF enabled."""
    random.seed(42)
    config = _make_config(multi_tf_enabled=True)
    sg = _make_base_strategy()
    sg.assign_instance_ids()
    ind = Individual(strategy_gene=sg)
    
    # Run several mutations to test it doesn't crash
    for _ in range(20):
        mutated = mutate(ind, 0.5, config)
        assert mutated.strategy_gene.indicators  # Still has indicators
        assert mutated.strategy_gene.entry_conditions  # Still has conditions


# ===========================================================================
# 9. Crossover with multi-TF
# ===========================================================================
def test_crossover_preserves_informative_timeframes():
    """Crossover should handle informative_timeframes without crashing."""
    config = _make_config(multi_tf_enabled=True)
    
    sg1 = _make_base_strategy()
    sg1.assign_instance_ids()
    
    sg2 = _make_multi_tf_strategy()
    sg2.assign_instance_ids()
    
    ind1 = Individual(strategy_gene=sg1)
    ind2 = Individual(strategy_gene=sg2)
    
    random.seed(42)
    for method in ['single_point', 'uniform', 'component']:
        child1, child2 = crossover(ind1, ind2, generation=1, ind_id=0,
                                    method=method, config=config)
        # Children should be valid
        assert child1.strategy_gene.indicators
        assert child2.strategy_gene.indicators
        # informative_timeframes should be a list
        assert isinstance(child1.strategy_gene.informative_timeframes, list)
        assert isinstance(child2.strategy_gene.informative_timeframes, list)


# ===========================================================================
# 10. Complexity with multi-TF
# ===========================================================================
def test_complexity_includes_informative_indicators():
    """Complexity should count informative indicators too."""
    sg = _make_multi_tf_strategy()
    # 3 indicators + 2 entry conditions + 1 exit condition = 6
    assert sg.calculate_complexity() == 6


# ===========================================================================
# Run all tests
# ===========================================================================
if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-o', 'addopts='])
