"""
LLM Integration for GA Strategy Generation

Provides provider-agnostic LLM interfaces for generating trading strategy
seeds and immigrants during genetic algorithm evolution.
"""

from genetic_algorithm.llm.provider import LLMProvider, LLMProviderFactory
from genetic_algorithm.llm.designer import StrategyDesigner
from genetic_algorithm.llm.client import (
    LLMClient,
    GrokClient,
    OpenAIClient,
    AnthropicClient,
    LocalClient,
    create_client,
)
from genetic_algorithm.llm.parser import StrategyParser
from genetic_algorithm.llm.injector import LLMInjector
from genetic_algorithm.llm.router import LLMProviderRouter, create_provider_or_router
from genetic_algorithm.llm.diagnostics import (
    diagnose_failure_mode,
    diagnose_all_failure_modes,
    select_mutation_objective,
)

__all__ = [
    # Provider layer
    'LLMProvider',
    'LLMProviderFactory',
    'LLMProviderRouter',
    'create_provider_or_router',
    # Designer (main interface for evolution engine)
    'StrategyDesigner',
    # Diagnostics
    'diagnose_failure_mode',
    'diagnose_all_failure_modes',
    'select_mutation_objective',
    # Client layer (alternative API)
    'LLMClient',
    'GrokClient',
    'OpenAIClient',
    'AnthropicClient',
    'LocalClient',
    'create_client',
    'StrategyParser',
    'LLMInjector',
]
