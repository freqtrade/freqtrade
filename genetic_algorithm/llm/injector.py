"""
LLM Injector

Provides :class:`LLMInjector` — the GA-facing interface for LLM-based
strategy generation.  It wraps the prompt builder, LLM client, and parser
into a single cohesive object that the evolution engine can call to:

* Seed the **initial population** with LLM-generated strategies.
* Inject **immigrants** during evolution (context-aware, guided by the
  current best strategies and identified population weaknesses).

Configuration (``advanced.llm`` in ga_config.yaml):

.. code-block:: yaml

   advanced:
     llm:
       enabled: true
       provider: "grok"          # grok | openai | anthropic | local
       api_key: ""
       seed_ratio: 0.2           # fraction of initial population from LLM
       immigrants_per_generation: 2
       temperature: 0.8
       max_retries: 3
"""

import logging
from typing import Dict, Any, List, Optional

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.llm.client import create_client, LLMClient
from genetic_algorithm.llm.parser import StrategyParser
from genetic_algorithm.llm.prompts import StrategyPromptBuilder, get_diverse_styles

logger = logging.getLogger(__name__)


class LLMInjector:
    """
    Injects LLM-generated strategies into the GA population.

    Usage::

        injector = LLMInjector(config)

        # At initialisation — seed part of the population
        seed_genes = injector.get_seed_population(size=10)

        # Each generation — replace some random immigrants
        immigrant_genes = injector.get_immigrants(
            count=2,
            generation=5,
            top_performers=[...],   # optional context
            weaknesses=[...],       # optional context
        )

    Both methods return :class:`StrategyGene` objects (not
    :class:`Individual`); callers wrap them in ``Individual`` as needed.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        client: Optional[LLMClient] = None,
    ):
        """
        Args:
            config: Full GA config dict.
            client: Pre-configured :class:`LLMClient`.  Created from config
                when ``None``.
        """
        self.config = config
        llm_cfg = config.get('advanced', {}).get('llm', {})

        self.enabled: bool = llm_cfg.get('enabled', False)
        self.seed_ratio: float = llm_cfg.get('seed_ratio', 0.2)
        self.immigrants_per_generation: int = llm_cfg.get('immigrants_per_generation', 2)
        self._max_retries: int = llm_cfg.get('max_retries', 3)

        self._stats = {
            'seed_requested': 0,
            'seed_generated': 0,
            'immigrant_requested': 0,
            'immigrant_generated': 0,
            'parse_failures': 0,
        }

        self._prompt_builder = StrategyPromptBuilder(config)
        self._parser = StrategyParser(config)

        if client is not None:
            self._client: Optional[LLMClient] = client
        elif self.enabled and llm_cfg.get('provider'):
            try:
                self._client = create_client(llm_cfg)
                logger.info(
                    "LLMInjector: provider=%s model=%s",
                    llm_cfg.get('provider'), llm_cfg.get('model', 'default'),
                )
            except Exception as exc:
                logger.error("Failed to create LLM client: %s. Injector disabled.", exc)
                self._client = None
                self.enabled = False
        else:
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_seed_population(
        self,
        size: int,
        generation: int = 0,
        start_id: int = 0,
    ) -> List[StrategyGene]:
        """
        Generate seed strategies for the initial population.

        The number of strategies generated is ``int(size * seed_ratio)``
        unless *size* itself is the desired count — callers may choose either
        convention.  The evolution engine uses the ratio; unit tests pass the
        exact count.

        Args:
            size: Target number of strategies (caller applies the ratio).
            generation: Generation number (0 for initial population).
            start_id: Starting individual ID.

        Returns:
            List of :class:`StrategyGene` objects (may be shorter than *size*
            if the LLM fails for some requests).
        """
        if not self.enabled or not self._client:
            return []

        self._stats['seed_requested'] += size
        styles = get_diverse_styles(size)
        strategies: List[StrategyGene] = []

        for i in range(size):
            gene = self._generate_one(
                generation=generation,
                individual_id=start_id + i,
                strategy_style=styles[i],
                is_seed=True,
            )
            if gene is not None:
                strategies.append(gene)

        self._stats['seed_generated'] += len(strategies)
        logger.info(
            "LLMInjector seed: %d/%d strategies generated.",
            len(strategies), size,
        )
        return strategies

    def get_immigrants(
        self,
        count: Optional[int] = None,
        generation: int = 1,
        start_id: int = 0,
        top_performers: Optional[List[Dict[str, Any]]] = None,
        weaknesses: Optional[List[str]] = None,
    ) -> List[StrategyGene]:
        """
        Generate immigrant strategies to inject during evolution.

        Args:
            count: Number of immigrants.  Defaults to
                ``immigrants_per_generation`` from config.
            generation: Current generation number.
            start_id: Starting individual ID.
            top_performers: Compact dicts summarising current top strategies
                (fitness, profit, indicators, etc.) so the LLM can produce
                "similar but different" designs.
            weaknesses: List of weakness strings from population analysis
                (e.g. "No strategies use MACD").

        Returns:
            List of :class:`StrategyGene` objects.
        """
        if not self.enabled or not self._client:
            return []

        if count is None:
            count = self.immigrants_per_generation

        self._stats['immigrant_requested'] += count
        strategies: List[StrategyGene] = []

        for i in range(count):
            gene = self._generate_one(
                generation=generation,
                individual_id=start_id + i,
                top_performers=top_performers,
                weaknesses=weaknesses,
                is_seed=False,
            )
            if gene is not None:
                strategies.append(gene)

        self._stats['immigrant_generated'] += len(strategies)
        logger.info(
            "LLMInjector immigrants (gen %d): %d/%d generated.",
            generation, len(strategies), count,
        )
        return strategies

    def get_stats(self) -> Dict[str, Any]:
        """Return a copy of the usage statistics dict."""
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_one(
        self,
        generation: int,
        individual_id: int,
        strategy_style: Optional[str] = None,
        top_performers: Optional[List[Dict[str, Any]]] = None,
        weaknesses: Optional[List[str]] = None,
        is_seed: bool = True,
    ) -> Optional[StrategyGene]:
        """Ask the LLM for one strategy; retry with feedback on parse failure."""
        assert self._client is not None, "LLM client not initialized"
        system_prompt = self._prompt_builder.build_system_prompt()
        error_feedback: Optional[str] = None

        for attempt in range(max(1, self._max_retries)):
            if is_seed:
                user_prompt = self._prompt_builder.build_seed_prompt(
                    strategy_style=strategy_style,
                )
            else:
                user_prompt = self._prompt_builder.build_immigrant_prompt(
                    top_performers=top_performers,
                    weaknesses=weaknesses,
                )

            if error_feedback:
                user_prompt = (
                    f"{user_prompt}\n\n"
                    f"CORRECTION REQUIRED (attempt {attempt + 1}):\n{error_feedback}\n"
                    "Please fix and return a valid JSON object."
                )

            data = self._client.generate_json(user_prompt, system_prompt)
            if data is None:
                logger.warning(
                    "LLMInjector: no JSON from provider (attempt %d/%d).",
                    attempt + 1, self._max_retries,
                )
                error_feedback = "The response was not valid JSON. Return only a JSON object."
                continue

            if isinstance(data, list):
                data = data[0] if data else None
            if not data:
                error_feedback = "Received an empty JSON array. Return a single JSON object."
                continue

            gene, error_feedback = self._parser.parse_with_feedback(
                data, generation, individual_id
            )
            if gene is not None:
                return gene

            logger.warning(
                "LLMInjector parse failure (attempt %d/%d): %s",
                attempt + 1, self._max_retries, error_feedback,
            )

        self._stats['parse_failures'] += 1
        return None
