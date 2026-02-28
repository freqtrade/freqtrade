"""
Tests for Plan 1 (Overfitting Fixes) and Plan 2 (LLM Integration)

Covers:
- Walk-forward partial credit system
- Holdout fitness penalty
- Composite overfit scoring (weighted average + hard rules)
- Min exit condition enforcement
- Unused-indicator penalty
- LLM provider factory
- LLM prompt builder
- LLM strategy designer (JSON → StrategyGene conversion)
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


# ============================================================================
# Plan 1: Overfitting Fixes
# ============================================================================

class TestCompositeOverfitScoring:
    """Test weighted composite scoring and hard override rules."""
    
    def test_weighted_composite_holdout_mc(self):
        """Holdout + MC should use 70/30 weights, not equal."""
        from genetic_algorithm.utils.overfit_analysis import classify_overfitting, OverfitThresholds
        
        thresholds = OverfitThresholds()
        metrics = {
            'holdout_degradation': 0.80,  # Severe overfitting
            'holdout_fitness': 0.1,
            'holdout_profit': -5.0,
            'mc_robustness': 0.95,  # MC says fine (masking holdout)
        }
        
        assessment = classify_overfitting(metrics, fitness=0.5, thresholds=thresholds)
        
        # With weighted scoring (holdout=0.7, MC=0.3), composite should be high
        # Old: (1.0 + 0.05) / 2 = 0.525 → WARNING  
        # New: 0.7*1.0 + 0.3*0.05 = 0.715 → OVERFIT
        assert assessment.overall_label == "OVERFIT", (
            f"Expected OVERFIT but got {assessment.overall_label} "
            f"(composite={assessment.composite_score})"
        )
    
    def test_hard_override_negative_holdout(self):
        """Severe holdout degradation + negative profit → always OVERFIT."""
        from genetic_algorithm.utils.overfit_analysis import classify_overfitting, OverfitThresholds
        
        thresholds = OverfitThresholds()
        metrics = {
            'holdout_degradation': 0.60,
            'holdout_fitness': 0.05,
            'holdout_profit': -10.0,
            'mc_robustness': 0.99,  # Perfect MC (would have masked before)
        }
        
        assessment = classify_overfitting(metrics, fitness=0.5, thresholds=thresholds)
        assert assessment.overall_label == "OVERFIT"
    
    def test_safe_strategy_unchanged(self):
        """Low degradation + good MC → SAFE."""
        from genetic_algorithm.utils.overfit_analysis import classify_overfitting, OverfitThresholds
        
        thresholds = OverfitThresholds()
        metrics = {
            'holdout_degradation': 0.10,
            'holdout_fitness': 0.4,
            'holdout_profit': 5.0,
            'mc_robustness': 0.85,
        }
        
        assessment = classify_overfitting(metrics, fitness=0.5, thresholds=thresholds)
        assert assessment.overall_label == "SAFE"
    
    def test_tightened_thresholds(self):
        """Default thresholds should be 0.25/0.50 (tightened from 0.35/0.60)."""
        from genetic_algorithm.utils.overfit_analysis import OverfitThresholds
        
        t = OverfitThresholds()
        assert t.composite_warning == 0.25
        assert t.composite_overfit == 0.50
    
    def test_from_config_defaults_match_class_defaults(self):
        """from_config({}) fallback defaults must match class-level defaults."""
        from genetic_algorithm.utils.overfit_analysis import OverfitThresholds
        
        t_class = OverfitThresholds()
        t_config = OverfitThresholds.from_config({})  # No overfit_analysis section
        
        assert t_config.composite_warning == t_class.composite_warning == 0.25
        assert t_config.composite_overfit == t_class.composite_overfit == 0.50
        assert t_config.holdout_degradation_warning == t_class.holdout_degradation_warning
        assert t_config.holdout_degradation_overfit == t_class.holdout_degradation_overfit


class TestMinExitConditions:
    """Test enforcement of minimum exit conditions."""
    
    def test_mutation_preserves_exit_conditions(self):
        """Mutation should not remove exit conditions below minimum."""
        from genetic_algorithm.core.mutation import mutate
        from genetic_algorithm.core.individual import Individual
        
        config = {
            'indicators': {'min_exit_conditions': 1, 'min_entry_conditions': 2, 
                          'available': ['RSI', 'MACD']},
            'strategy_constraints': {},
        }
        
        # Run mutation many times — exit conditions should never drop below 1
        for _ in range(50):
            indicators = [
                IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
                IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, instance_id='MACD_0'),
            ]
            entry_conditions = [
                ConditionGene(indicator='RSI_0', operator='<', threshold=30),
                ConditionGene(indicator='MACD_0', operator='cross_above', threshold=0),
            ]
            exit_conditions = [
                ConditionGene(indicator='RSI_0', operator='>', threshold=70),
            ]
            
            gene = StrategyGene(
                generation=0, individual_id=0,
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
            )
            individual = Individual(strategy_gene=gene)
            
            mutated = mutate(individual, mutation_rate=0.9, config=config)
            mutated_gene = mutated.strategy_gene
            assert len(mutated_gene.exit_conditions) >= 1, (
                f"Exit conditions dropped to {len(mutated_gene.exit_conditions)}"
            )
    
    def test_crossover_enforces_exit_minimum(self):
        """_enforce_min_entry_conditions should also enforce exit conditions."""
        from genetic_algorithm.core.crossover import _enforce_min_entry_conditions
        
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='RSI_0', operator='<', threshold=30),
                ConditionGene(indicator='RSI_0', operator='<', threshold=25),
            ],
            exit_conditions=[],  # Empty!
        )
        
        config = {
            'indicators': {'min_exit_conditions': 1, 'min_entry_conditions': 2, 'available': ['RSI']},
        }
        
        _enforce_min_entry_conditions(gene, config)
        assert len(gene.exit_conditions) >= 1


class TestUnusedIndicatorPenalty:
    """Test the unused-indicator fitness penalty."""
    
    def test_penalty_applied_for_unused_indicators(self):
        """Strategies with unused indicators should receive a penalty."""
        from genetic_algorithm.evaluation.fitness import FitnessEvaluator
        
        # Create gene with 3 indicators but only 1 used in conditions
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
                IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, instance_id='MACD_0'),
                IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='RSI_0', operator='<', threshold=30),
            ],
            exit_conditions=[
                ConditionGene(indicator='RSI_0', operator='>', threshold=70),
            ],
        )
        
        metrics = {
            'num_trades': 20,
            'max_drawdown': 0.10,
            'win_rate': 0.55,
            'profit': 10.0,
        }
        
        # Create evaluator with penalties
        config = {
            'backtesting': {'pairs': ['BTC/USDT'], 'timerange': '20230101-20240101', 'exchange': 'binance'},
            'fitness_weights': {'profit': 1.0},
            'fitness_penalties': {
                'min_trades': 5,
                'max_drawdown': 0.30,
                'min_win_rate': 0.30,
                'complexity_weight': 0,
                'pair_loss_threshold': -99,
                'unused_indicator_weight': 0.02,
            },
            'fitness_bounds': {},
            'trade_frequency_thresholds': {},
            'walk_forward': {'enabled': False},
        }
        
        evaluator = FitnessEvaluator(config)
        
        # Apply penalties with and without the gene
        fitness_with_gene = evaluator._apply_penalties(1.0, metrics, strategy_gene=gene)
        fitness_without_gene = evaluator._apply_penalties(1.0, metrics, strategy_gene=None)
        
        # With 2/3 unused, penalty should be applied
        assert fitness_with_gene < fitness_without_gene


# ============================================================================
# Plan 2: LLM Integration
# ============================================================================

class TestLLMProviderFactory:
    """Test the LLM provider factory pattern."""
    
    def test_factory_creates_grok(self):
        from genetic_algorithm.llm.provider import LLMProviderFactory, GrokProvider
        
        config = {'provider': 'grok', 'api_key': 'test-key'}
        provider = LLMProviderFactory.create(config)
        assert isinstance(provider, GrokProvider)
        assert provider.base_url == 'https://api.x.ai/v1'
    
    def test_factory_creates_openai(self):
        from genetic_algorithm.llm.provider import LLMProviderFactory, OpenAIProvider
        
        config = {'provider': 'openai', 'api_key': 'test-key'}
        provider = LLMProviderFactory.create(config)
        assert isinstance(provider, OpenAIProvider)
    
    def test_factory_creates_anthropic(self):
        from genetic_algorithm.llm.provider import LLMProviderFactory, AnthropicProvider
        
        config = {'provider': 'anthropic', 'api_key': 'test-key'}
        provider = LLMProviderFactory.create(config)
        assert isinstance(provider, AnthropicProvider)
    
    def test_factory_creates_local(self):
        from genetic_algorithm.llm.provider import LLMProviderFactory, LocalProvider
        
        config = {'provider': 'local'}
        provider = LLMProviderFactory.create(config)
        assert isinstance(provider, LocalProvider)
        assert 'localhost' in provider.base_url
    
    def test_factory_rejects_unknown(self):
        from genetic_algorithm.llm.provider import LLMProviderFactory
        
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMProviderFactory.create({'provider': 'nonexistent'})
    
    def test_extract_json_from_markdown(self):
        from genetic_algorithm.llm.provider import LLMProvider
        
        text = '```json\n{"indicators": []}\n```'
        result = LLMProvider._extract_json(text)
        assert result == '{"indicators": []}'
    
    def test_extract_json_raw(self):
        from genetic_algorithm.llm.provider import LLMProvider
        
        text = 'Here is the strategy: {"indicators": []} end.'
        result = LLMProvider._extract_json(text)
        assert '{"indicators": []}' in result


class TestStrategyPromptBuilder:
    """Test prompt construction."""
    
    def test_build_system_prompt(self):
        from genetic_algorithm.llm.prompts import StrategyPromptBuilder
        
        config = {
            'indicators': {'available': ['RSI', 'MACD'], 'min_entry_conditions': 2},
            'strategy_constraints': {'timeframes': ['15m', '1h']},
            'advanced': {'llm': {}},
        }
        
        builder = StrategyPromptBuilder(config)
        system = builder.build_system_prompt()
        
        assert 'quantitative' in system.lower()
        assert 'uncorrelated' in system.lower() or 'multiple' in system.lower()
    
    def test_build_seed_prompt_contains_schema(self):
        from genetic_algorithm.llm.prompts import StrategyPromptBuilder
        
        config = {
            'indicators': {'available': ['RSI', 'MACD'], 'min_entry_conditions': 2},
            'strategy_constraints': {'timeframes': ['15m']},
            'advanced': {'llm': {}},
        }
        
        builder = StrategyPromptBuilder(config)
        prompt = builder.build_seed_prompt(strategy_style='trend_following')
        
        assert 'indicators' in prompt
        assert 'entry_conditions' in prompt
        assert 'exit_conditions' in prompt
        assert 'instance_id' in prompt
        assert 'trend_following' in prompt.lower() or 'trend' in prompt.lower()
    
    def test_diverse_styles(self):
        from genetic_algorithm.llm.prompts import get_diverse_styles
        
        styles = get_diverse_styles(7)
        assert len(styles) == 7
        # Should cycle through all 5 styles
        unique = set(styles)
        assert len(unique) == 5


class TestStrategyDesigner:
    """Test the StrategyDesigner JSON → StrategyGene conversion."""
    
    def _make_config(self):
        return {
            'indicators': {
                'available': ['RSI', 'MACD', 'EMA', 'BBANDS', 'ADX'],
                'min_entry_conditions': 2,
                'max_entry_conditions': 4,
                'min_exit_conditions': 1,
                'min_per_strategy': 2,
                'max_per_strategy': 5,
            },
            'strategy_constraints': {
                'timeframes': ['15m', '1h'],
                'stoploss_range': [-0.20, -0.05],
                'roi_range': [0.01, 0.10],
            },
            'advanced': {'llm': {'enabled': False}},
        }
    
    def test_valid_json_to_strategy_gene(self):
        from genetic_algorithm.llm.designer import StrategyDesigner
        
        config = self._make_config()
        designer = StrategyDesigner(config)
        
        llm_output = {
            "indicators": [
                {"type": "RSI", "instance_id": "RSI_0", "parameters": {"period": 14}, "weight": 1.0, "timeframe": None},
                {"type": "MACD", "instance_id": "MACD_0", "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9}, "weight": 1.0, "timeframe": None},
            ],
            "entry_conditions": [
                {"indicator": "RSI_0", "operator": "<", "threshold": 30, "logic": "AND", "threshold_upper": 0, "lookback": 3},
                {"indicator": "MACD_0", "operator": "cross_above", "threshold": 0, "logic": "AND", "threshold_upper": 0, "lookback": 3},
            ],
            "exit_conditions": [
                {"indicator": "RSI_0", "operator": ">", "threshold": 70, "logic": "AND", "threshold_upper": 0, "lookback": 3},
            ],
            "timeframe": "15m",
            "stoploss": -0.10,
            "minimal_roi": {"0": 0.05, "30": 0.03, "60": 0.01},
            "max_open_trades": 3,
            "trailing_stop": False,
        }
        
        gene = designer._json_to_strategy_gene(llm_output, generation=0, individual_id=0)
        
        assert gene is not None
        assert len(gene.indicators) == 2
        assert len(gene.entry_conditions) == 2
        assert len(gene.exit_conditions) == 1
        assert gene.timeframe == '15m'
        assert gene.stoploss == -0.10
    
    def test_unknown_indicator_filtered(self):
        from genetic_algorithm.llm.designer import StrategyDesigner
        
        config = self._make_config()
        designer = StrategyDesigner(config)
        
        llm_output = {
            "indicators": [
                {"type": "RSI", "instance_id": "RSI_0", "parameters": {"period": 14}},
                {"type": "UNKNOWN_IND", "instance_id": "UNKNOWN_0", "parameters": {}},
                {"type": "MACD", "instance_id": "MACD_0", "parameters": {}},
            ],
            "entry_conditions": [
                {"indicator": "RSI_0", "operator": "<", "threshold": 30},
                {"indicator": "UNKNOWN_0", "operator": ">", "threshold": 50},
                {"indicator": "MACD_0", "operator": "cross_above", "threshold": 0},
            ],
            "exit_conditions": [
                {"indicator": "RSI_0", "operator": ">", "threshold": 70},
            ],
            "timeframe": "15m",
            "stoploss": -0.10,
        }
        
        gene = designer._json_to_strategy_gene(llm_output, generation=0, individual_id=0)
        
        assert gene is not None
        # UNKNOWN_IND should be filtered
        assert all(ind.type != 'UNKNOWN_IND' for ind in gene.indicators)
        # UNKNOWN_0 condition should be filtered
        assert all(c.indicator != 'UNKNOWN_0' for c in gene.entry_conditions)

    def test_missing_exit_conditions_auto_added(self):
        from genetic_algorithm.llm.designer import StrategyDesigner
        
        config = self._make_config()
        designer = StrategyDesigner(config)
        
        llm_output = {
            "indicators": [
                {"type": "RSI", "instance_id": "RSI_0", "parameters": {"period": 14}},
                {"type": "EMA", "instance_id": "EMA_0", "parameters": {"period": 20}},
            ],
            "entry_conditions": [
                {"indicator": "RSI_0", "operator": "<", "threshold": 30},
                {"indicator": "EMA_0", "operator": "cross_above", "threshold": 0},
            ],
            "exit_conditions": [],  # Empty!
            "timeframe": "15m",
            "stoploss": -0.10,
        }
        
        gene = designer._json_to_strategy_gene(llm_output, generation=0, individual_id=0)
        
        assert gene is not None
        assert len(gene.exit_conditions) >= 1  # Should auto-add default
    
    def test_stoploss_clamped_to_range(self):
        from genetic_algorithm.llm.designer import StrategyDesigner
        
        config = self._make_config()
        designer = StrategyDesigner(config)
        
        llm_output = {
            "indicators": [
                {"type": "RSI", "instance_id": "RSI_0", "parameters": {"period": 14}},
                {"type": "MACD", "instance_id": "MACD_0", "parameters": {}},
            ],
            "entry_conditions": [
                {"indicator": "RSI_0", "operator": "<", "threshold": 30},
                {"indicator": "MACD_0", "operator": "cross_above", "threshold": 0},
            ],
            "exit_conditions": [
                {"indicator": "RSI_0", "operator": ">", "threshold": 70},
            ],
            "timeframe": "15m",
            "stoploss": -0.50,  # Too loose — should be clamped to -0.20
        }
        
        gene = designer._json_to_strategy_gene(llm_output, generation=0, individual_id=0)
        
        assert gene is not None
        assert gene.stoploss >= -0.20  # Clamped to config range
    
    def test_disabled_designer_returns_empty(self):
        from genetic_algorithm.llm.designer import StrategyDesigner
        
        config = self._make_config()
        designer = StrategyDesigner(config)
        
        # Should return empty list when disabled
        result = designer.generate_seed_strategies(count=5)
        assert result == []
    
    def test_population_weakness_analysis(self):
        from genetic_algorithm.llm.designer import StrategyDesigner
        
        config = self._make_config()
        designer = StrategyDesigner(config)
        
        # Create mock individuals  
        mock_ind = MagicMock()
        mock_ind.strategy_gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[IndicatorGene(type='RSI', parameters={'period': 14})],
            entry_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=30)],
        )
        mock_ind.metrics = {'num_trades': 5, 'max_drawdown': 0.10}
        mock_ind.fitness = 0.5
        
        weaknesses = designer.get_population_weaknesses([mock_ind])
        
        # Should detect that many indicators are unused
        assert len(weaknesses) > 0
        assert any('MACD' in w for w in weaknesses)  # MACD not used


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
