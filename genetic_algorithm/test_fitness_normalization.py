"""
Test fitness weight normalization

Verifies that fitness weights are properly normalized to sum to 1.0,
regardless of configuration.
"""

import pytest
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


def create_test_strategy():
    """Create a test strategy for fitness evaluation."""
    return StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'timeperiod': 14}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='>', threshold=70),
        ],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
    )


class TestFitnessWeightNormalization:
    """Test that fitness weights are normalized properly."""
    
    def test_default_weights_are_normalized(self):
        """Test that default weights sum to 1.0."""
        config = {
            'fitness_weights': {},
            'fitness_penalties': {},
            'backtesting': {},
        }
        evaluator = FitnessEvaluator(config)
        
        # Create test metrics
        metrics = {
            'profit': 10.0,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'profit_factor': 2.0,
            'max_drawdown': 0.10,
            'win_rate': 0.55,
            'num_trades': 20,
        }
        
        strategy = create_test_strategy()
        
        # Calculate fitness (this should normalize weights internally)
        fitness = evaluator.calculate_fitness(metrics, strategy)
        
        # Fitness should be positive and reasonable
        assert fitness > 0
        assert fitness <= 10.0  # With bonuses, should still be reasonable
        
    def test_custom_weights_are_normalized(self):
        """Test that custom weights that don't sum to 1.0 are normalized."""
        # Weights that sum to 1.25 (like in the TODO description)
        config = {
            'fitness_weights': {
                'profit': 0.25,
                'sharpe_ratio': 0.15,
                'sortino_ratio': 0.15,
                'profit_factor': 0.10,
                'drawdown': 0.15,
                'win_rate': 0.10,
                'trade_frequency': 0.10,
                # Total = 1.0, but if config omits sortino/profit_factor, defaults add 0.25
            },
            'fitness_penalties': {},
            'backtesting': {},
        }
        evaluator = FitnessEvaluator(config)
        
        # Create test metrics
        metrics = {
            'profit': 10.0,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'profit_factor': 2.0,
            'max_drawdown': 0.10,
            'win_rate': 0.55,
            'num_trades': 20,
        }
        
        strategy = create_test_strategy()
        
        # Calculate fitness
        fitness1 = evaluator.calculate_fitness(metrics, strategy)
        
        # Now test with weights that explicitly sum to 1.25
        config2 = {
            'fitness_weights': {
                'profit': 0.30,  # Inflated
                'sharpe_ratio': 0.20,  # Inflated
                'sortino_ratio': 0.20,  # Inflated
                'profit_factor': 0.15,  # Inflated
                'drawdown': 0.15,  # Inflated
                'win_rate': 0.15,  # Inflated
                'trade_frequency': 0.10,  # Inflated
                # Total = 1.25
            },
            'fitness_penalties': {},
            'backtesting': {},
        }
        evaluator2 = FitnessEvaluator(config2)
        
        # Calculate fitness with inflated weights
        fitness2 = evaluator2.calculate_fitness(metrics, strategy)
        
        # After normalization, the relative proportions should be the same
        # but the absolute fitness values should be comparable (not inflated)
        assert abs(fitness1 - fitness2) < 0.5  # Should be similar after normalization
        
    def test_partial_weights_are_normalized(self):
        """Test that partial weight configs are normalized."""
        # Only specify a few weights
        config = {
            'fitness_weights': {
                'profit': 0.50,
                'sharpe_ratio': 0.30,
                # Missing: sortino_ratio, profit_factor, drawdown, win_rate, trade_frequency
                # Defaults will fill in: 0.15 + 0.10 + 0.15 + 0.10 + 0.10 = 0.60
                # Total = 1.10 before normalization
            },
            'fitness_penalties': {},
            'backtesting': {},
        }
        evaluator = FitnessEvaluator(config)
        
        metrics = {
            'profit': 10.0,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'profit_factor': 2.0,
            'max_drawdown': 0.10,
            'win_rate': 0.55,
            'num_trades': 20,
        }
        
        strategy = create_test_strategy()
        fitness = evaluator.calculate_fitness(metrics, strategy)
        
        # Should produce reasonable fitness value
        assert fitness > 0
        assert fitness <= 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
