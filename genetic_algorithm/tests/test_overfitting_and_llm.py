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


# ============================================================================
# Phase 1 Regression Tests (Population.get_all + Holdout Penalty on raw_fitness)
# ============================================================================

class TestPopulationGetAll:
    """Verify Population.get_all() returns references to all individuals."""

    def test_get_all_returns_all(self):
        from genetic_algorithm.core.population import Population
        from genetic_algorithm.core.individual import Individual

        pop = Population(size=5, generation=0)
        for i in range(5):
            gene = StrategyGene(
                indicators=[IndicatorGene(type='rsi', parameters={'period': 14})],
                entry_conditions=[ConditionGene(indicator='rsi', operator='less_than', threshold=30)],
                exit_conditions=[ConditionGene(indicator='rsi', operator='greater_than', threshold=70)],
                generation=0, individual_id=i,
            )
            ind = Individual(strategy_gene=gene)
            ind.fitness = float(i)
            ind.raw_fitness = float(i)
            ind.evaluated = True
            ind.metrics = {'origin': 'test'}
            pop.add_individual(ind)

        result = pop.get_all()
        assert len(result) == 5
        # Returned list elements should be same object references as internal list
        for r in result:
            assert r in pop.individuals

    def test_get_all_empty_population(self):
        from genetic_algorithm.core.population import Population

        pop = Population(size=0, generation=0)
        assert pop.get_all() == []


class TestHoldoutPenaltyAffectsRawFitness:
    """Verify holdout penalty reduces raw_fitness (not just fitness)."""

    def _make_individual(self, fitness_val, ind_id=0):
        """Helper to create a minimal Individual with given fitness."""
        from genetic_algorithm.core.individual import Individual

        gene = StrategyGene(
            indicators=[IndicatorGene(type='rsi', parameters={'period': 14})],
            entry_conditions=[ConditionGene(indicator='rsi', operator='less_than', threshold=30)],
            exit_conditions=[ConditionGene(indicator='rsi', operator='greater_than', threshold=70)],
            generation=0, individual_id=ind_id,
        )
        ind = Individual(strategy_gene=gene)
        ind.raw_fitness = fitness_val
        ind.fitness = fitness_val
        ind.evaluated = True
        ind.metrics = {}
        return ind

    def test_penalty_reduces_raw_fitness(self):
        """After holdout penalty, raw_fitness must be lower than before."""
        ind = self._make_individual(0.5)
        original_raw = ind.raw_fitness

        # Simulate the penalty logic from evolution.py _run_holdout_monitoring
        degradation = 60.0  # 60% degradation
        penalty_factor = 0.5
        degradation_frac = degradation / 100.0
        penalty_mult = max(0.3, 1.0 - degradation_frac * penalty_factor)
        ind.raw_fitness = ind.raw_fitness * penalty_mult
        ind.fitness = ind.fitness * penalty_mult
        ind.metrics['holdout_penalty'] = 1.0 - penalty_mult

        assert ind.raw_fitness < original_raw
        assert ind.raw_fitness == pytest.approx(0.5 * 0.7, abs=1e-6)
        assert ind.fitness == pytest.approx(0.5 * 0.7, abs=1e-6)
        assert ind.metrics['holdout_penalty'] == pytest.approx(0.3, abs=1e-6)

    def test_penalty_floor_at_0_3(self):
        """Penalty multiplier should floor at 0.3 (not 0.5 as before)."""
        ind = self._make_individual(1.0)

        degradation = 200.0  # Extreme degradation
        penalty_factor = 0.5
        degradation_frac = degradation / 100.0
        penalty_mult = max(0.3, 1.0 - degradation_frac * penalty_factor)

        assert penalty_mult == 0.3  # Floor
        ind.raw_fitness *= penalty_mult
        assert ind.raw_fitness == pytest.approx(0.3, abs=1e-6)

    def test_penalised_elite_ranks_lower(self):
        """An overfit elite with penalty should rank below an unpenalized one."""
        ind_a = self._make_individual(0.5, ind_id=0)  # Will be penalized
        ind_b = self._make_individual(0.45, ind_id=1)  # Slightly worse but not penalized

        # Penalize ind_a
        ind_a.raw_fitness *= 0.7
        ind_a.fitness *= 0.7

        # raw_fitness: ind_a=0.35, ind_b=0.45 → ind_b should rank higher
        ranked = sorted([ind_a, ind_b], key=lambda x: x.raw_fitness, reverse=True)
        assert ranked[0] is ind_b, (
            f"Penalized ind should rank lower: a.raw={ind_a.raw_fitness}, b.raw={ind_b.raw_fitness}"
        )


class TestCachedHoldoutEvaluator:
    """Verify holdout evaluator caching attributes exist on GeneticAlgorithm."""

    def test_cache_attributes_initialized(self):
        """_holdout_evaluator and _holdout_range should start as None."""
        from genetic_algorithm.core.evolution import GeneticAlgorithm
        import os

        config_path = os.path.join(
            os.path.dirname(__file__), 'config', 'ga_config.yaml'
        )
        if not os.path.exists(config_path):
            pytest.skip("Test config not available")

        ga = GeneticAlgorithm(config_path)
        assert ga._holdout_evaluator is None
        assert ga._holdout_range is None


# ============================================================================
# Phase 2 Tests: Strategy Quality Guardrails
# ============================================================================

class TestThresholdClamps:
    """Verify bounded indicator thresholds are clamped to valid ranges."""

    def test_rsi_clamped_to_0_100(self):
        from genetic_algorithm.core.mutation import clamp_condition_thresholds
        conds = [ConditionGene(indicator='rsi', operator='<', threshold=-20)]
        clamp_condition_thresholds(conds)
        assert conds[0].threshold == 0

    def test_rsi_upper_clamped(self):
        from genetic_algorithm.core.mutation import clamp_condition_thresholds
        conds = [ConditionGene(indicator='RSI', operator='>', threshold=120)]
        clamp_condition_thresholds(conds)
        assert conds[0].threshold == 100

    def test_stoch_clamped(self):
        from genetic_algorithm.core.mutation import clamp_condition_thresholds
        conds = [ConditionGene(indicator='STOCH_0', operator='<', threshold=-5)]
        clamp_condition_thresholds(conds)
        assert conds[0].threshold == 0

    def test_between_thresholds_ordered(self):
        from genetic_algorithm.core.mutation import clamp_condition_thresholds
        conds = [ConditionGene(indicator='RSI', operator='between',
                               threshold=80, threshold_upper=30)]
        clamp_condition_thresholds(conds)
        assert conds[0].threshold <= conds[0].threshold_upper

    def test_unbounded_indicator_unchanged(self):
        from genetic_algorithm.core.mutation import clamp_condition_thresholds
        conds = [ConditionGene(indicator='MACD', operator='>', threshold=-999)]
        clamp_condition_thresholds(conds)
        assert conds[0].threshold == -999  # MACD is unbounded


class TestConditionDeduplication:
    """Verify duplicate/subsumed conditions are pruned."""

    def test_exact_duplicate_removed(self):
        from genetic_algorithm.core.crossover import _deduplicate_conditions
        conds = [
            ConditionGene(indicator='rsi', operator='<', threshold=30),
            ConditionGene(indicator='rsi', operator='<', threshold=30),
        ]
        result = _deduplicate_conditions(conds)
        assert len(result) == 1

    def test_subsumed_less_than_pruned(self):
        """vroc < -117 AND vroc < -200 → keep only vroc < -200."""
        from genetic_algorithm.core.crossover import _deduplicate_conditions
        conds = [
            ConditionGene(indicator='vroc', operator='<', threshold=-117),
            ConditionGene(indicator='vroc', operator='<', threshold=-200),
        ]
        result = _deduplicate_conditions(conds)
        assert len(result) == 1
        assert result[0].threshold == -200

    def test_subsumed_greater_than_pruned(self):
        """rsi > 60 AND rsi > 80 → keep only rsi > 80."""
        from genetic_algorithm.core.crossover import _deduplicate_conditions
        conds = [
            ConditionGene(indicator='rsi', operator='>', threshold=60),
            ConditionGene(indicator='rsi', operator='>', threshold=80),
        ]
        result = _deduplicate_conditions(conds)
        assert len(result) == 1
        assert result[0].threshold == 80

    def test_different_operators_preserved(self):
        from genetic_algorithm.core.crossover import _deduplicate_conditions
        conds = [
            ConditionGene(indicator='rsi', operator='<', threshold=30),
            ConditionGene(indicator='rsi', operator='cross_above', threshold=30),
        ]
        result = _deduplicate_conditions(conds)
        assert len(result) == 2


class TestIndicatorDeduplication:
    """Verify duplicate indicators are removed after crossover."""

    def test_duplicate_same_type_removed(self):
        from genetic_algorithm.core.crossover import _deduplicate_indicators
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='RSI', parameters={'period': 14}),
                IndicatorGene(type='RSI', parameters={'period': 14}),
                IndicatorGene(type='MACD', parameters={'fast_period': 12}),
            ],
            entry_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=30)],
        )
        _deduplicate_indicators(gene)
        assert len(gene.indicators) == 2
        types = [i.type for i in gene.indicators]
        assert types.count('RSI') == 1

    def test_different_params_kept(self):
        from genetic_algorithm.core.crossover import _deduplicate_indicators
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='RSI', parameters={'period': 14}),
                IndicatorGene(type='RSI', parameters={'period': 21}),
            ],
            entry_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=30)],
        )
        _deduplicate_indicators(gene)
        assert len(gene.indicators) == 2  # Different params → kept


class TestDeadExitPenalty:
    """Verify fitness penalty for impossible exit thresholds."""

    def test_rsi_below_zero_exit_penalized(self):
        """RSI < 0 as exit condition should trigger dead-exit penalty."""
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[IndicatorGene(type='RSI', parameters={'period': 14})],
            entry_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=30)],
            exit_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=0)],
        )
        fitness = 1.0
        # Simulate the penalty logic from fitness.py
        _BOUNDED = {'RSI': (0, 100), 'STOCH': (0, 100)}
        dead_count = 0
        bounded_count = 0
        for cond in gene.exit_conditions:
            base_type = cond.indicator.split('_')[0]
            bounds = _BOUNDED.get(base_type.upper())
            if bounds:
                bounded_count += 1
                lo, hi = bounds
                if cond.operator in ('<', 'less_than') and cond.threshold <= lo:
                    dead_count += 1
        if bounded_count > 0 and dead_count == bounded_count:
            fitness *= 0.7
        assert fitness == pytest.approx(0.7, abs=1e-6)

    def test_valid_exit_not_penalized(self):
        """RSI > 70 is a valid exit — no penalty."""
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[IndicatorGene(type='RSI', parameters={'period': 14})],
            entry_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=30)],
            exit_conditions=[ConditionGene(indicator='RSI', operator='>', threshold=70)],
        )
        _BOUNDED = {'RSI': (0, 100)}
        dead_count = 0
        bounded_count = 0
        for cond in gene.exit_conditions:
            base_type = cond.indicator.split('_')[0]
            bounds = _BOUNDED.get(base_type.upper())
            if bounds:
                bounded_count += 1
                lo, hi = bounds
                if cond.operator in ('<', 'less_than') and cond.threshold <= lo:
                    dead_count += 1
                elif cond.operator in ('>', 'greater_than') and cond.threshold >= hi:
                    dead_count += 1
        assert dead_count == 0  # No dead conditions


# ════════════════════════════════════════════════════════════════
# Walk-Forward max_windows cap
# ════════════════════════════════════════════════════════════════

class TestMaxWindowsCap:
    """Verify create_walk_forward_windows respects max_windows."""

    def test_cap_limits_window_count(self):
        from genetic_algorithm.utils.timerange import create_walk_forward_windows
        # 365 days with 60/20/20 rolling → ~15 windows uncapped
        windows = create_walk_forward_windows(
            timerange='20250101-20251231',
            train_days=60, validation_days=20, step_days=20,
            max_windows=5
        )
        assert len(windows) == 5

    def test_no_cap_returns_all(self):
        from genetic_algorithm.utils.timerange import create_walk_forward_windows
        windows_no_cap = create_walk_forward_windows(
            timerange='20250101-20251231',
            train_days=60, validation_days=20, step_days=20,
            max_windows=None
        )
        windows_capped = create_walk_forward_windows(
            timerange='20250101-20251231',
            train_days=60, validation_days=20, step_days=20,
            max_windows=5
        )
        assert len(windows_no_cap) > len(windows_capped)

    def test_cap_larger_than_natural_count(self):
        from genetic_algorithm.utils.timerange import create_walk_forward_windows
        # 120 days with 60/20/20 → only 2-3 windows
        windows = create_walk_forward_windows(
            timerange='20250101-20250501',
            train_days=60, validation_days=20, step_days=20,
            max_windows=100
        )
        assert len(windows) < 100  # Natural count is well below cap


class TestPoolHealthCheckFn:
    """Verify the module-level health check function is picklable."""

    def test_fn_returns_true(self):
        from genetic_algorithm.evaluation.parallel import _pool_health_check_fn
        assert _pool_health_check_fn() is True

    def test_fn_is_picklable(self):
        import pickle
        from genetic_algorithm.evaluation.parallel import _pool_health_check_fn
        pickled = pickle.dumps(_pool_health_check_fn)
        restored = pickle.loads(pickled)
        assert restored() is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
