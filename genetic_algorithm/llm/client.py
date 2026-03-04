"""
LLM Client Interface

Abstract LLMClient base class and concrete provider implementations for
multi-provider LLM strategy generation. Concrete clients: GrokClient,
OpenAIClient, AnthropicClient, LocalClient.

Config-driven via the ``advanced.llm`` section in ga_config.yaml.
"""

from typing import Dict, Any

from genetic_algorithm.llm.provider import (
    LLMProvider,
    LLMProviderFactory,
    GrokProvider,
    GroqProvider,
    OpenAIProvider,
    AnthropicProvider,
    LocalProvider,
)


class LLMClient(LLMProvider):
    """
    Abstract LLM client base class.

    Extends :class:`LLMProvider` with a ``generate_strategy`` convenience
    method that asks the provider to produce a strategy-description string
    (typically JSON).  Concrete sub-classes inherit the full HTTP retry /
    back-off logic from ``OpenAICompatibleProvider`` or ``AnthropicProvider``.
    """

    def generate_strategy(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate a strategy description string from the LLM.

        This is a thin wrapper around :meth:`generate` that signals the
        *intent* of the call — asking the model to design a trading strategy.
        The returned string is expected to be valid JSON compatible with
        ``StrategyGene.to_dict()``, but callers must parse/validate it.

        Args:
            prompt: User prompt describing what strategy to generate.
            system_prompt: Optional system-level instructions.

        Returns:
            Raw LLM response string (typically JSON).
        """
        return self.generate(prompt, system_prompt)


# ---------------------------------------------------------------------------
# Concrete client classes
# Each inherits both LLMClient (for generate_strategy) and the matching
# provider implementation (for the actual HTTP logic).
# ---------------------------------------------------------------------------

class GrokClient(LLMClient, GrokProvider):
    """xAI Grok LLM client."""

    def __init__(self, config: Dict[str, Any]):
        # MRO: GrokClient → LLMClient → GrokProvider → OpenAICompatibleProvider
        # GrokProvider.__init__ sets defaults then calls OpenAICompatibleProvider.__init__
        GrokProvider.__init__(self, config)


class OpenAIClient(LLMClient, OpenAIProvider):
    """OpenAI LLM client."""

    def __init__(self, config: Dict[str, Any]):
        OpenAIProvider.__init__(self, config)


class AnthropicClient(LLMClient, AnthropicProvider):
    """Anthropic Claude LLM client."""

    def __init__(self, config: Dict[str, Any]):
        AnthropicProvider.__init__(self, config)


class GroqClient(LLMClient, GroqProvider):
    """Groq (groq.com) LPU inference client."""

    def __init__(self, config: Dict[str, Any]):
        GroqProvider.__init__(self, config)


class LocalClient(LLMClient, LocalProvider):
    """Local LLM server client (Ollama / llama.cpp / vLLM)."""

    def __init__(self, config: Dict[str, Any]):
        LocalProvider.__init__(self, config)


# ---------------------------------------------------------------------------
# Client factory — wraps LLMProviderFactory and returns LLMClient instances
# ---------------------------------------------------------------------------

_CLIENT_MAP: Dict[str, type] = {
    'grok': GrokClient,
    'xai': GrokClient,
    'groq': GroqClient,
    'openai': OpenAIClient,
    'anthropic': AnthropicClient,
    'claude': AnthropicClient,
    'local': LocalClient,
    'ollama': LocalClient,
}


def create_client(config: Dict[str, Any]) -> 'LLMClient':
    """
    Create an :class:`LLMClient` from the ``advanced.llm`` config block.

    Falls back to :class:`LLMProviderFactory` for any provider not in the
    client map (e.g., third-party providers registered at runtime), casting
    the result to :class:`LLMClient` dynamically.

    Args:
        config: LLM config dict (the ``advanced.llm`` section).

    Returns:
        Configured :class:`LLMClient` instance.

    Raises:
        ValueError: If the provider name is unknown.
    """
    provider_name = config.get('provider', '').lower()
    client_cls = _CLIENT_MAP.get(provider_name)
    if client_cls is not None:
        return client_cls(config)
    # Fall back to factory; result is an LLMProvider but exposes generate()
    return LLMProviderFactory.create(config)  # type: ignore[return-value]
