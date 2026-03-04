"""
Tests for Phase 1A: LLM Strategy Seeding

Covers:
- LLMClient abstract base and concrete classes (client.py)
- StrategyParser JSON → StrategyGene conversion with validation (parser.py)
- LLMInjector seed population and immigrant generation (injector.py)
"""

import pytest
from unittest.mock import MagicMock, patch
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_config():
    return {
        'indicators': {
            'available': ['RSI', 'MACD', 'EMA', 'SMA', 'BBANDS', 'ADX'],
            'min_entry_conditions': 1,
            'min_exit_conditions': 1,
        },
        'strategy_constraints': {
            'timeframes': ['5m', '15m', '1h'],
            'stoploss_range': [-0.20, -0.05],
        },
        'advanced': {
            'llm': {
                'enabled': False,
                'provider': 'grok',
                'seed_ratio': 0.2,
                'immigrants_per_generation': 2,
                'max_retries': 3,
            }
        },
    }


@pytest.fixture()
def minimal_gene_dict():
    """A minimal valid LLM JSON payload."""
    return {
        'indicators': [
            {'type': 'RSI', 'instance_id': 'RSI_0', 'parameters': {'period': 14}},
        ],
        'entry_conditions': [
            {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30, 'logic': 'AND'},
        ],
        'exit_conditions': [
            {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70, 'logic': 'AND'},
        ],
        'timeframe': '15m',
        'stoploss': -0.10,
        'minimal_roi': {'0': 0.05, '30': 0.02},
        'max_open_trades': 3,
    }


# ============================================================================
# Tests for client.py
# ============================================================================

class TestLLMClientAbstractBase:
    """LLMClient is an abstract class with a generate_strategy method."""

    def test_generate_strategy_delegates_to_generate(self, base_config):
        from genetic_algorithm.llm.client import GrokClient

        client = GrokClient({'api_key': 'test', 'provider': 'grok'})
        client.generate = MagicMock(return_value='{"indicators": []}')

        result = client.generate_strategy('test prompt', 'system')
        client.generate.assert_called_once_with('test prompt', 'system')
        assert result == '{"indicators": []}'

    def test_concrete_clients_inherit_generate_strategy(self):
        from genetic_algorithm.llm.client import (
            GrokClient, OpenAIClient, AnthropicClient, LocalClient, LLMClient
        )
        for cls in [GrokClient, OpenAIClient, AnthropicClient, LocalClient]:
            assert issubclass(cls, LLMClient), f"{cls.__name__} should inherit LLMClient"
            assert hasattr(cls, 'generate_strategy')

    def test_create_client_returns_grok_client(self):
        from genetic_algorithm.llm.client import create_client, GrokClient

        client = create_client({'provider': 'grok', 'api_key': 'test'})
        assert isinstance(client, GrokClient)

    def test_create_client_returns_openai_client(self):
        from genetic_algorithm.llm.client import create_client, OpenAIClient

        client = create_client({'provider': 'openai', 'api_key': 'test'})
        assert isinstance(client, OpenAIClient)

    def test_create_client_returns_anthropic_client(self):
        from genetic_algorithm.llm.client import create_client, AnthropicClient

        client = create_client({'provider': 'anthropic', 'api_key': 'test'})
        assert isinstance(client, AnthropicClient)

    def test_create_client_returns_local_client(self):
        from genetic_algorithm.llm.client import create_client, LocalClient

        client = create_client({'provider': 'local'})
        assert isinstance(client, LocalClient)

    def test_create_client_raises_for_unknown_provider(self):
        from genetic_algorithm.llm.client import create_client

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_client({'provider': 'nonexistent_xyz'})

    def test_grok_default_base_url(self):
        from genetic_algorithm.llm.client import GrokClient

        client = GrokClient({'api_key': 'test', 'provider': 'grok'})
        assert 'api.x.ai' in client.base_url

    def test_local_default_api_key(self):
        from genetic_algorithm.llm.client import LocalClient

        client = LocalClient({'provider': 'local'})
        # Local servers don't need a real API key
        assert client.api_key == 'not-needed'


# ============================================================================
# Tests for parser.py
# ============================================================================

class TestStrategyParser:
    """StrategyParser converts LLM JSON dicts into StrategyGene objects."""

    def test_valid_json_returns_gene(self, base_config, minimal_gene_dict):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        gene = parser.parse(minimal_gene_dict, generation=1, individual_id=0)

        assert gene is not None
        assert isinstance(gene, StrategyGene)
        assert len(gene.indicators) == 1
        assert gene.indicators[0].type == 'RSI'
        assert len(gene.entry_conditions) == 1
        assert len(gene.exit_conditions) == 1

    def test_unknown_indicator_is_removed(self, base_config):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        data = {
            'indicators': [
                {'type': 'UNKNOWN_IND', 'instance_id': 'X_0', 'parameters': {}},
                {'type': 'RSI', 'instance_id': 'RSI_0', 'parameters': {'period': 14}},
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
            ],
            'exit_conditions': [
                {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70},
            ],
            'timeframe': '15m',
            'stoploss': -0.10,
        }
        gene = parser.parse(data, generation=0, individual_id=0)
        assert gene is not None
        assert len(gene.indicators) == 1
        assert gene.indicators[0].type == 'RSI'

    def test_no_valid_indicators_returns_none(self, base_config):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        gene = parser.parse({'indicators': [], 'entry_conditions': [], 'exit_conditions': []},
                            generation=0, individual_id=0)
        assert gene is None

    def test_stoploss_clamped_to_range(self, base_config, minimal_gene_dict):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        minimal_gene_dict['stoploss'] = -0.99  # below min
        gene = parser.parse(minimal_gene_dict, generation=0, individual_id=0)
        assert gene is not None
        assert gene.stoploss >= -0.20

    def test_unknown_timeframe_replaced(self, base_config, minimal_gene_dict):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        minimal_gene_dict['timeframe'] = '999d'
        gene = parser.parse(minimal_gene_dict, generation=0, individual_id=0)
        assert gene is not None
        assert gene.timeframe in base_config['strategy_constraints']['timeframes']

    def test_dangling_condition_reference_removed(self, base_config):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        data = {
            'indicators': [
                {'type': 'RSI', 'instance_id': 'RSI_0', 'parameters': {'period': 14}},
            ],
            'entry_conditions': [
                {'indicator': 'NONEXISTENT_0', 'operator': '<', 'threshold': 30},
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
            ],
            'exit_conditions': [
                {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70},
            ],
            'timeframe': '15m',
            'stoploss': -0.10,
        }
        gene = parser.parse(data, generation=0, individual_id=0)
        assert gene is not None
        assert all(c.indicator in {'RSI_0', 'RSI'} for c in gene.entry_conditions)

    def test_default_exit_added_when_missing(self, base_config):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        data = {
            'indicators': [
                {'type': 'RSI', 'instance_id': 'RSI_0', 'parameters': {'period': 14}},
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
            ],
            'exit_conditions': [],  # missing!
            'timeframe': '15m',
            'stoploss': -0.10,
        }
        gene = parser.parse(data, generation=0, individual_id=0)
        assert gene is not None
        assert len(gene.exit_conditions) >= 1

    def test_parse_with_feedback_returns_error_on_failure(self, base_config):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        gene, error = parser.parse_with_feedback(
            {'indicators': [], 'entry_conditions': [], 'exit_conditions': []},
            generation=0, individual_id=0,
        )
        assert gene is None
        assert error is not None
        assert len(error) > 0

    def test_parse_with_feedback_none_on_success(self, base_config, minimal_gene_dict):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        gene, error = parser.parse_with_feedback(minimal_gene_dict, generation=0, individual_id=0)
        assert gene is not None
        assert error is None

    def test_auto_assigned_instance_id(self, base_config):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        data = {
            'indicators': [
                {'type': 'RSI', 'parameters': {'period': 14}},  # no instance_id
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30},
            ],
            'exit_conditions': [
                {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70},
            ],
            'timeframe': '15m',
            'stoploss': -0.10,
        }
        gene = parser.parse(data, generation=0, individual_id=0)
        assert gene is not None
        assert gene.indicators[0].instance_id == 'RSI_0'

    def test_minimal_roi_string_keys(self, base_config, minimal_gene_dict):
        from genetic_algorithm.llm.parser import StrategyParser

        parser = StrategyParser(base_config)
        minimal_gene_dict['minimal_roi'] = {0: 0.05, 30: 0.02}  # int keys
        gene = parser.parse(minimal_gene_dict, generation=0, individual_id=0)
        assert gene is not None
        assert all(isinstance(k, str) for k in gene.minimal_roi.keys())


# ============================================================================
# Tests for injector.py
# ============================================================================

class TestLLMInjector:
    """LLMInjector wraps client + parser to provide GA-facing seed/immigrant API."""

    def test_disabled_returns_empty_seed_population(self, base_config):
        from genetic_algorithm.llm.injector import LLMInjector

        injector = LLMInjector(base_config)
        assert injector.enabled is False
        assert injector.get_seed_population(10) == []

    def test_disabled_returns_empty_immigrants(self, base_config):
        from genetic_algorithm.llm.injector import LLMInjector

        injector = LLMInjector(base_config)
        assert injector.get_immigrants(count=3) == []

    def test_seed_ratio_and_immigrants_per_gen_from_config(self, base_config):
        from genetic_algorithm.llm.injector import LLMInjector

        injector = LLMInjector(base_config)
        assert injector.seed_ratio == pytest.approx(0.2)
        assert injector.immigrants_per_generation == 2

    def test_enabled_with_mock_client(self, base_config):
        """With a mocked client that always succeeds, LLMInjector generates seeds."""
        from genetic_algorithm.llm.injector import LLMInjector
        from genetic_algorithm.llm.client import LLMClient

        mock_client = MagicMock(spec=LLMClient)
        mock_client.generate_json.return_value = {
            'indicators': [
                {'type': 'RSI', 'instance_id': 'RSI_0', 'parameters': {'period': 14}},
            ],
            'entry_conditions': [
                {'indicator': 'RSI_0', 'operator': '<', 'threshold': 30, 'logic': 'AND'},
            ],
            'exit_conditions': [
                {'indicator': 'RSI_0', 'operator': '>', 'threshold': 70, 'logic': 'AND'},
            ],
            'timeframe': '15m',
            'stoploss': -0.10,
            'minimal_roi': {'0': 0.05},
            'max_open_trades': 3,
        }

        cfg = dict(base_config)
        cfg['advanced'] = {'llm': {**base_config['advanced']['llm'], 'enabled': True}}

        injector = LLMInjector(cfg, client=mock_client)
        injector.enabled = True  # Override since no real provider configured

        seeds = injector.get_seed_population(3)
        assert len(seeds) == 3
        for gene in seeds:
            assert isinstance(gene, StrategyGene)

    def test_immigrants_with_mock_client(self, base_config):
        from genetic_algorithm.llm.injector import LLMInjector
        from genetic_algorithm.llm.client import LLMClient

        mock_client = MagicMock(spec=LLMClient)
        mock_client.generate_json.return_value = {
            'indicators': [
                {'type': 'MACD', 'instance_id': 'MACD_0',
                 'parameters': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}},
            ],
            'entry_conditions': [
                {'indicator': 'MACD_0', 'operator': 'cross_above', 'threshold': 0},
            ],
            'exit_conditions': [
                {'indicator': 'MACD_0', 'operator': 'cross_below', 'threshold': 0},
            ],
            'timeframe': '1h',
            'stoploss': -0.08,
            'minimal_roi': {'0': 0.04},
            'max_open_trades': 2,
        }

        cfg = dict(base_config)
        cfg['advanced'] = {'llm': {**base_config['advanced']['llm'], 'enabled': True}}

        injector = LLMInjector(cfg, client=mock_client)
        injector.enabled = True

        immigrants = injector.get_immigrants(
            count=2,
            generation=5,
            top_performers=[{'fitness': 0.8, 'profit': 5.2, 'max_drawdown': 0.12,
                             'num_trades': 30, 'win_rate': 0.6, 'indicators': ['MACD']}],
            weaknesses=['No RSI strategies'],
        )
        assert len(immigrants) == 2
        for gene in immigrants:
            assert isinstance(gene, StrategyGene)

    def test_parse_failure_retries_with_error_feedback(self, base_config):
        """Injector retries when parser fails; error message appended to prompt."""
        from genetic_algorithm.llm.injector import LLMInjector
        from genetic_algorithm.llm.client import LLMClient

        call_count = {'n': 0}
        valid_response = {
            'indicators': [
                {'type': 'EMA', 'instance_id': 'EMA_0', 'parameters': {'period': 20}},
            ],
            'entry_conditions': [
                {'indicator': 'EMA_0', 'operator': 'cross_above', 'threshold': 0},
            ],
            'exit_conditions': [
                {'indicator': 'EMA_0', 'operator': 'cross_below', 'threshold': 0},
            ],
            'timeframe': '5m',
            'stoploss': -0.07,
            'minimal_roi': {'0': 0.03},
            'max_open_trades': 1,
        }

        def side_effect(prompt, system_prompt):
            call_count['n'] += 1
            if call_count['n'] == 1:
                return {'indicators': []}  # will fail parse
            return valid_response

        mock_client = MagicMock(spec=LLMClient)
        mock_client.generate_json.side_effect = side_effect

        cfg = dict(base_config)
        cfg['advanced'] = {'llm': {**base_config['advanced']['llm'], 'enabled': True,
                                   'max_retries': 3}}

        injector = LLMInjector(cfg, client=mock_client)
        injector.enabled = True

        seeds = injector.get_seed_population(1)
        assert len(seeds) == 1
        assert call_count['n'] == 2  # failed once, succeeded on retry

    def test_stats_tracking(self, base_config):
        from genetic_algorithm.llm.injector import LLMInjector
        from genetic_algorithm.llm.client import LLMClient

        mock_client = MagicMock(spec=LLMClient)
        mock_client.generate_json.return_value = {
            'indicators': [
                {'type': 'SMA', 'instance_id': 'SMA_0', 'parameters': {'period': 50}},
            ],
            'entry_conditions': [
                {'indicator': 'SMA_0', 'operator': 'cross_above', 'threshold': 0},
            ],
            'exit_conditions': [
                {'indicator': 'SMA_0', 'operator': 'cross_below', 'threshold': 0},
            ],
            'timeframe': '1h',
            'stoploss': -0.10,
            'minimal_roi': {'0': 0.05},
            'max_open_trades': 3,
        }

        cfg = dict(base_config)
        cfg['advanced'] = {'llm': {**base_config['advanced']['llm'], 'enabled': True}}

        injector = LLMInjector(cfg, client=mock_client)
        injector.enabled = True

        injector.get_seed_population(2)
        injector.get_immigrants(count=1, generation=3)

        stats = injector.get_stats()
        assert stats['seed_requested'] == 2
        assert stats['seed_generated'] == 2
        assert stats['immigrant_requested'] == 1
        assert stats['immigrant_generated'] == 1

    def test_get_immigrants_defaults_to_config_count(self, base_config):
        """get_immigrants() with no count arg uses immigrants_per_generation."""
        from genetic_algorithm.llm.injector import LLMInjector
        from genetic_algorithm.llm.client import LLMClient

        mock_client = MagicMock(spec=LLMClient)
        mock_client.generate_json.return_value = {
            'indicators': [
                {'type': 'ADX', 'instance_id': 'ADX_0', 'parameters': {'period': 14}},
            ],
            'entry_conditions': [
                {'indicator': 'ADX_0', 'operator': '>', 'threshold': 25},
            ],
            'exit_conditions': [
                {'indicator': 'ADX_0', 'operator': '<', 'threshold': 20},
            ],
            'timeframe': '15m',
            'stoploss': -0.08,
            'minimal_roi': {'0': 0.04},
            'max_open_trades': 2,
        }

        cfg = dict(base_config)
        cfg['advanced'] = {'llm': {**base_config['advanced']['llm'], 'enabled': True,
                                   'immigrants_per_generation': 3}}

        injector = LLMInjector(cfg, client=mock_client)
        injector.enabled = True

        immigrants = injector.get_immigrants(generation=2)
        assert len(immigrants) == 3  # from config default
