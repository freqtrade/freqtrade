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

__all__ = [
    # Legacy / provider layer
    'LLMProvider',
    'LLMProviderFactory',
    'StrategyDesigner',
    # Phase 1A additions
    'LLMClient',
    'GrokClient',
    'OpenAIClient',
    'AnthropicClient',
    'LocalClient',
    'create_client',
    'StrategyParser',
    'LLMInjector',
]
