"""
LLM Provider Router

Wraps multiple :class:`LLMProvider` instances and routes calls through a
prioritised fallback chain.  When a provider fails (rate limit, timeout,
server error), the router marks it as cooling-down and tries the next one.

Configuration (``advanced.llm`` in ga_config.yaml)::

    providers_list:
      - provider: groq
        api_key: ""
        model: ""
      - provider: grok
        api_key: ""
      - provider: local

If ``providers_list`` is absent the router falls back to creating a single
provider from the top-level ``provider`` / ``api_key`` / ``model`` keys
(backward-compatible).
"""

import logging
import time
from typing import Dict, Any, List, Optional

from genetic_algorithm.llm.provider import LLMProvider, LLMProviderFactory

logger = logging.getLogger(__name__)


class LLMProviderRouter(LLMProvider):
    """
    Routes LLM calls across a prioritised list of providers with
    automatic failover and cooldown.

    Implements the same interface as :class:`LLMProvider` so callers
    (``StrategyDesigner``) need no code changes.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: The ``advanced.llm`` config dict.  Expected keys:

                * ``providers_list`` — list of per-provider config dicts
                  (each must have at least ``provider``).
                * ``cooldown_seconds`` — seconds to skip a provider after
                  failure (default 60).
                * Standard keys (``temperature``, ``max_tokens``, etc.) are
                  used as defaults for each sub-provider.
        """
        # Don't call super().__init__() with an api_key check — the
        # router doesn't hold a single key.  Store config manually.
        self.config = config
        self.model = config.get('model', '')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 4096)
        self.timeout = config.get('timeout', 60)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2.0)

        self._cooldown_seconds: float = config.get('cooldown_seconds', 30.0)

        # Build ordered list of providers
        providers_list = config.get('providers_list', [])
        self._providers: List[LLMProvider] = []
        self._provider_names: List[str] = []
        self._cooldown_until: Dict[int, float] = {}  # idx → timestamp

        # Per-provider stats
        self._provider_stats: Dict[str, Dict[str, int]] = {}
        self._last_used_provider: str = ''

        for pconf in providers_list:
            # Merge defaults from parent config
            merged = dict(config)
            merged.update(pconf)
            name = merged.get('provider', 'unknown')
            try:
                provider = LLMProviderFactory.create(merged)
                self._providers.append(provider)
                self._provider_names.append(name)
                self._provider_stats[name] = {
                    'attempts': 0, 'successes': 0, 'failures': 0,
                }
                logger.info(f"[ROUTER] Registered provider: {name} ({provider.model})")
            except Exception as e:
                logger.warning(f"[ROUTER] Skipping provider '{name}': {e}")

        if not self._providers:
            raise ValueError(
                "LLMProviderRouter: no valid providers could be created "
                f"from providers_list ({len(providers_list)} entries)."
            )

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Try each available provider in priority order."""
        last_error: Optional[Exception] = None

        for idx, provider in enumerate(self._providers):
            name = self._provider_names[idx]

            # Skip providers that are still cooling down
            until = self._cooldown_until.get(idx, 0)
            if time.time() < until:
                logger.debug(
                    "[ROUTER] %s is cooling down (%.0fs left)",
                    name, until - time.time(),
                )
                continue

            self._provider_stats[name]['attempts'] += 1
            try:
                result = provider.generate(prompt, system_prompt)
                self._provider_stats[name]['successes'] += 1
                self._last_used_provider = name
                return result
            except Exception as e:
                self._provider_stats[name]['failures'] += 1
                logger.warning(
                    "[ROUTER] Provider %s failed: %s — trying next", name, e,
                )
                last_error = e
                # Cooldown this provider
                self._cooldown_until[idx] = time.time() + self._cooldown_seconds

        raise RuntimeError(
            f"All {len(self._providers)} providers failed. "
            f"Last error: {last_error}"
        )

    @property
    def provider_name(self) -> str:
        return "LLMProviderRouter"

    @property
    def last_used_provider(self) -> str:
        """Return the name of the sub-provider that served the last successful call."""
        return self._last_used_provider or "unknown"

    def get_router_stats(self) -> Dict[str, Any]:
        """Return per-provider success/failure statistics."""
        return {
            'providers': list(self._provider_names),
            'stats': dict(self._provider_stats),
            'cooldowns_active': sum(
                1 for ts in self._cooldown_until.values()
                if time.time() < ts
            ),
        }


def create_provider_or_router(config: Dict[str, Any]) -> LLMProvider:
    """
    Create either a single :class:`LLMProvider` or an
    :class:`LLMProviderRouter` depending on configuration.

    * If ``providers_list`` is present and non-empty → router.
    * Otherwise → single provider via ``LLMProviderFactory.create()``.

    Args:
        config: The ``advanced.llm`` config dict.

    Returns:
        A :class:`LLMProvider` (or subclass) instance.
    """
    providers_list = config.get('providers_list', [])
    if providers_list:
        return LLMProviderRouter(config)
    return LLMProviderFactory.create(config)
