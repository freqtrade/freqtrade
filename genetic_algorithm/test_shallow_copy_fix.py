"""
Test shallow copy fixes in crossover and StrategyGene

Tests that deep copies are properly made during crossover and copy operations,
ensuring that mutations don't corrupt parent strategies.
"""

import pytest
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.crossover import single_point_crossover, uniform_crossover, component_crossover


def create_test_strategy(gen_id, ind_id):
    """Create a test strategy for testing."""
    return StrategyGene(
        generation=gen_id,
        individual_id=ind_id,
        indicators=[
            IndicatorGene(type='RSI', parameters={'timeperiod': 14}),
            IndicatorGene(type='MACD', parameters={'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
            ConditionGene(indicator='MACD', operator='cross_above', threshold=0, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='>', threshold=70),
        ],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
    )


class TestStrategyCopyIsolation:
    """Test that StrategyGene.copy() creates isolated copies."""
    
    def test_copy_isolates_indicator_parameters(self):
        """Test that copying a strategy isolates indicator parameters."""
        # Create original strategy
        original = create_test_strategy(0, 0)
        original_rsi_period = original.indicators[0].parameters['timeperiod']
        
        # Create copy
        copy = original.copy()
        
        # Mutate copy's indicator parameter
        copy.indicators[0].parameters['timeperiod'] = 999
        
        # Verify original is unchanged
        assert original.indicators[0].parameters['timeperiod'] == original_rsi_period
        assert original.indicators[0].parameters['timeperiod'] != 999
        
    def test_copy_isolates_condition_threshold(self):
        """Test that copying a strategy isolates condition thresholds."""
        # Create original strategy
        original = create_test_strategy(0, 0)
        original_threshold = original.entry_conditions[0].threshold
        
        # Create copy
        copy = original.copy()
        
        # Mutate copy's condition threshold
        copy.entry_conditions[0].threshold = 999.0
        
        # Verify original is unchanged
        assert original.entry_conditions[0].threshold == original_threshold
        assert original.entry_conditions[0].threshold != 999.0


class TestCrossoverIsolation:
    """Test that crossover operations create isolated children."""
    
    def test_single_point_crossover_isolates_indicators(self):
        """Test that single-point crossover isolates indicator parameters."""
        # Create parent strategies
        parent1 = Individual(strategy_gene=create_test_strategy(0, 0))
        parent2 = Individual(strategy_gene=create_test_strategy(0, 1))
        
        # Store original values
        p1_rsi_period = parent1.strategy_gene.indicators[0].parameters['timeperiod']
        p2_rsi_period = parent2.strategy_gene.indicators[0].parameters['timeperiod']
        
        # Perform crossover
        child1, child2 = single_point_crossover(parent1, parent2, generation=1, ind_id=0)
        
        # Mutate children's indicator parameters
        if child1.strategy_gene.indicators:
            child1.strategy_gene.indicators[0].parameters['timeperiod'] = 888
        if child2.strategy_gene.indicators:
            child2.strategy_gene.indicators[0].parameters['timeperiod'] = 999
        
        # Verify parents are unchanged
        assert parent1.strategy_gene.indicators[0].parameters['timeperiod'] == p1_rsi_period
        assert parent2.strategy_gene.indicators[0].parameters['timeperiod'] == p2_rsi_period
        
    def test_single_point_crossover_isolates_conditions(self):
        """Test that single-point crossover isolates condition thresholds."""
        # Create parent strategies
        parent1 = Individual(strategy_gene=create_test_strategy(0, 0))
        parent2 = Individual(strategy_gene=create_test_strategy(0, 1))
        
        # Store original values
        p1_threshold = parent1.strategy_gene.entry_conditions[0].threshold
        p2_threshold = parent2.strategy_gene.entry_conditions[0].threshold
        
        # Perform crossover
        child1, child2 = single_point_crossover(parent1, parent2, generation=1, ind_id=0)
        
        # Mutate children's condition thresholds
        if child1.strategy_gene.entry_conditions:
            child1.strategy_gene.entry_conditions[0].threshold = 888.0
        if child2.strategy_gene.entry_conditions:
            child2.strategy_gene.entry_conditions[0].threshold = 999.0
        
        # Verify parents are unchanged
        assert parent1.strategy_gene.entry_conditions[0].threshold == p1_threshold
        assert parent2.strategy_gene.entry_conditions[0].threshold == p2_threshold
        
    def test_uniform_crossover_isolates_indicators(self):
        """Test that uniform crossover isolates indicator parameters."""
        # Create parent strategies
        parent1 = Individual(strategy_gene=create_test_strategy(0, 0))
        parent2 = Individual(strategy_gene=create_test_strategy(0, 1))
        
        # Store original values
        p1_rsi_period = parent1.strategy_gene.indicators[0].parameters['timeperiod']
        p2_rsi_period = parent2.strategy_gene.indicators[0].parameters['timeperiod']
        
        # Perform crossover
        child1, child2 = uniform_crossover(parent1, parent2, generation=1, ind_id=0)
        
        # Mutate children's indicator parameters
        if child1.strategy_gene.indicators:
            child1.strategy_gene.indicators[0].parameters['timeperiod'] = 777
        if child2.strategy_gene.indicators:
            child2.strategy_gene.indicators[0].parameters['timeperiod'] = 666
        
        # Verify parents are unchanged
        assert parent1.strategy_gene.indicators[0].parameters['timeperiod'] == p1_rsi_period
        assert parent2.strategy_gene.indicators[0].parameters['timeperiod'] == p2_rsi_period
        
    def test_component_crossover_isolates_indicators(self):
        """Test that component crossover isolates indicator parameters."""
        # Create parent strategies
        parent1 = Individual(strategy_gene=create_test_strategy(0, 0))
        parent2 = Individual(strategy_gene=create_test_strategy(0, 1))
        
        # Store original values
        p1_rsi_period = parent1.strategy_gene.indicators[0].parameters['timeperiod']
        p2_rsi_period = parent2.strategy_gene.indicators[0].parameters['timeperiod']
        
        # Perform crossover
        child1, child2 = component_crossover(parent1, parent2, generation=1, ind_id=0)
        
        # Mutate children's indicator parameters
        if child1.strategy_gene.indicators:
            child1.strategy_gene.indicators[0].parameters['timeperiod'] = 555
        if child2.strategy_gene.indicators:
            child2.strategy_gene.indicators[0].parameters['timeperiod'] = 444
        
        # Verify parents are unchanged
        assert parent1.strategy_gene.indicators[0].parameters['timeperiod'] == p1_rsi_period
        assert parent2.strategy_gene.indicators[0].parameters['timeperiod'] == p2_rsi_period


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
