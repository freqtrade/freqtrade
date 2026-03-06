"""
LLM Strategy Designer

Converts LLM-generated JSON into valid StrategyGene objects with full
validation, sanitization, and fallback handling.
"""

import logging
import random
import time
from typing import Dict, Any, List, Optional, Tuple

from genetic_algorithm.core.strategy_gene import (
    StrategyGene, IndicatorGene, ConditionGene
)
from genetic_algorithm.llm.provider import LLMProvider, LLMProviderFactory
from genetic_algorithm.llm.router import create_provider_or_router
from genetic_algorithm.llm.prompts import (
    StrategyPromptBuilder, STRATEGY_STYLES, get_diverse_styles, INDICATOR_REFERENCE
)
from genetic_algorithm.llm.diagnostics import diagnose_failure_mode, select_mutation_objective

logger = logging.getLogger(__name__)


class StrategyDesigner:
    """
    High-level interface for LLM-based strategy generation.
    
    Handles:
    - Prompt construction via StrategyPromptBuilder
    - LLM API calls via LLMProvider
    - JSON → StrategyGene conversion with validation
    - Fallback to random generation on failure
    - Rate limiting and caching
    """
    
    def __init__(self, config: Dict[str, Any], provider: Optional[LLMProvider] = None):
        """
        Initialize the strategy designer.
        
        Args:
            config: Full GA config dict
            provider: Optional pre-configured LLM provider (created from config if None)
        """
        self.config = config
        self.prompt_builder = StrategyPromptBuilder(config)
        self.available_indicators = config.get('indicators', {}).get('available', [])
        
        # LLM config
        llm_config = config.get('advanced', {}).get('llm', {})
        self.enabled = llm_config.get('enabled', False)
        self.seed_ratio = llm_config.get('seed_ratio', 0.4)
        self.immigrant_ratio = llm_config.get('immigrant_ratio', 0.5)
        
        # Rate limiting (adaptive: backs off on failures, resets on success)
        self._last_call_time = 0.0
        self._min_call_interval = llm_config.get('min_call_interval', 3.0)
        self._max_call_interval = llm_config.get('max_call_interval', 12.0)
        self._effective_call_interval = self._min_call_interval
        
        # LLM mutation config
        self.mutation_enabled = llm_config.get('mutation_enabled', False)
        self.mutation_probability = llm_config.get('mutation_probability', 0.1)
        self.mutation_top_k = llm_config.get('mutation_top_k', 3)
        self.mutation_stagnation_threshold = llm_config.get('mutation_stagnation_threshold', 3)

        # Batching config
        self.batch_enabled = llm_config.get('batch_enabled', True)
        self.max_batch_size = llm_config.get('max_batch_size', 5)

        # Budget enforcement
        self._calls_this_generation = 0
        self._calls_this_run = 0
        self._max_calls_per_generation = llm_config.get('max_calls_per_generation', 20)
        self._max_calls_per_run = llm_config.get('max_calls_per_run', 500)

        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'validation_fixed': 0,
            'fallback_used': 0,
            'calls_by_type': {
                'seed': 0, 'immigrant': 0, 'mutation': 0, 'batch_seed': 0,
                'batch_immigrant': 0, 'batch_mutation': 0,
            },
            'improvements_by_type': {
                'seed': 0, 'immigrant': 0, 'mutation': 0,
            },
        }
        
        # Generation-over-generation performance tracking for feedback loop
        self.generation_history: List[Dict[str, Any]] = []
        self.llm_performance = {
            'llm_survived': 0,
            'llm_eliminated': 0,
            'avg_llm_fitness': 0.0,
            'avg_random_fitness': 0.0,
            'best_llm_fitness': 0.0,
            'best_llm_generation': -1,
        }
        
        # Per-provider attribution tracking
        self._last_provider_used: str = ''
        self.provider_attribution: Dict[str, Dict[str, int]] = {}  # provider → {generated, survived, ...}
        
        # Provider — use router if providers_list is configured
        if provider:
            self.provider = provider
        elif self.enabled and (llm_config.get('provider') or llm_config.get('providers_list')):
            try:
                self.provider = create_provider_or_router(llm_config)
            except Exception as e:
                logger.error(f"Failed to create LLM provider: {e}. LLM generation disabled.")
                self.provider = None
                self.enabled = False
        else:
            self.provider = None

        # Run connectivity test at startup when using router
        if self.enabled and self.provider is not None:
            self._test_provider_connectivity()

    # ------------------------------------------------------------------
    # Provider health check
    # ------------------------------------------------------------------

    def _test_provider_connectivity(self):
        """Ping each provider with a minimal prompt at startup.

        If the primary provider fails but a fallback succeeds, the router's
        cooldown mechanism will automatically redirect traffic.  This test
        just logs the status so operators can see which providers are live
        before the run starts.
        """
        from genetic_algorithm.llm.router import LLMProviderRouter

        if not isinstance(self.provider, LLMProviderRouter):
            # Single provider — quick test
            try:
                _ = self.provider.generate("Say 'ok'.", "Respond with only the word ok.")
                logger.info("[LLM HEALTH] Provider %s: OK", self.provider.provider_name)
            except Exception as e:
                logger.warning("[LLM HEALTH] Provider %s: FAILED (%s)",
                               self.provider.provider_name, e)
            return

        # Router — test each sub-provider individually
        for idx, sub in enumerate(self.provider._providers):
            name = self.provider._provider_names[idx]
            try:
                _ = sub.generate("Say 'ok'.", "Respond with only the word ok.")
                logger.info("[LLM HEALTH] %s: OK", name)
            except Exception as e:
                logger.warning("[LLM HEALTH] %s: FAILED (%s) — will use fallback", name, e)
                # Pre-cooldown this provider so the router skips it initially
                self.provider._cooldown_until[idx] = time.time() + self.provider._cooldown_seconds
    
    def generate_seed_strategies(
        self,
        count: int,
        generation: int = 0,
        start_id: int = 0,
    ) -> List[StrategyGene]:
        """
        Generate seed strategies for the initial population.
        
        Args:
            count: Number of strategies to generate
            generation: Generation number (0 for seeds)
            start_id: Starting individual ID
            
        Returns:
            List of StrategyGene objects
        """
        if not self.enabled or not self.provider:
            logger.info("LLM disabled, skipping seed generation")
            return []
        
        strategies = []
        styles = get_diverse_styles(count)
        
        for i in range(count):
            strategy = self._generate_single_strategy(
                generation=generation,
                individual_id=start_id + i,
                strategy_style=styles[i],
                is_seed=True,
            )
            if strategy:
                strategies.append(strategy)
        
        logger.info(f"LLM seed generation: {len(strategies)}/{count} successful")
        return strategies
    
    def generate_immigrants(
        self,
        count: int,
        generation: int,
        start_id: int = 0,
        top_performers: Optional[List[Dict]] = None,
        weaknesses: Optional[List[str]] = None,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> List[StrategyGene]:
        """
        Generate immigrant strategies during evolution.
        
        Args:
            count: Number of immigrants to generate
            generation: Current generation number
            start_id: Starting individual ID
            top_performers: Summaries of current top strategies
            weaknesses: Identified population weaknesses
            feedback: Performance feedback from previous LLM strategies,
                      feature importance data, and evolution progress
            
        Returns:
            List of StrategyGene objects
        """
        if not self.enabled or not self.provider:
            return []
        
        strategies = []
        for i in range(count):
            strategy = self._generate_single_strategy(
                generation=generation,
                individual_id=start_id + i,
                top_performers=top_performers,
                weaknesses=weaknesses,
                feedback=feedback,
                is_seed=False,
            )
            if strategy:
                strategies.append(strategy)
        
        logger.info(f"LLM immigrant generation: {len(strategies)}/{count} successful "
                   f"(gen {generation})")
        return strategies
    
    def _generate_single_strategy(
        self,
        generation: int,
        individual_id: int,
        strategy_style: Optional[str] = None,
        top_performers: Optional[List[Dict]] = None,
        weaknesses: Optional[List[str]] = None,
        feedback: Optional[Dict[str, Any]] = None,
        is_seed: bool = True,
    ) -> Optional[StrategyGene]:
        """
        Generate a single strategy from the LLM.
        
        Args:
            generation: Generation number
            individual_id: Individual ID
            strategy_style: Strategy style hint
            top_performers: Top performer summaries (for immigrants)
            weaknesses: Population weaknesses (for immigrants)
            feedback: Performance feedback and feature importance (for immigrants)
            is_seed: Whether this is a seed or immigrant
            
        Returns:
            StrategyGene or None on failure
        """
        if not self._budget_available():
            return None

        self.stats['total_requests'] += 1
        call_type = 'seed' if is_seed else 'immigrant'
        self.stats['calls_by_type'][call_type] += 1
        
        # Rate limiting
        self._rate_limit()
        
        try:
            # Build prompt
            system_prompt = self.prompt_builder.build_system_prompt()
            # Read island regime from config (injected by island model)
            island_regime = self.config.get('advanced', {}).get('llm', {}).get('island_regime')
            if is_seed:
                user_prompt = self.prompt_builder.build_seed_prompt(
                    strategy_style=strategy_style,
                    island_regime=island_regime,
                )
            else:
                user_prompt = self.prompt_builder.build_immigrant_prompt(
                    top_performers=top_performers,
                    weaknesses=weaknesses,
                    feedback=feedback,
                    island_regime=island_regime,
                )
            
            # Call LLM
            response = self.provider.generate_json(user_prompt, system_prompt)
            # Track which provider served this call
            used_provider = getattr(self.provider, 'last_used_provider', 
                                     getattr(self.provider, 'provider_name', 'unknown'))
            self._last_provider_used = used_provider
            self._record_call()
            
            if response is None:
                logger.warning(f"LLM returned no valid JSON for strategy {individual_id}")
                self.stats['failed'] += 1
                self._on_llm_failure()
                return None
            
            # Handle array response (LLM sometimes returns array even when asked for one)
            if isinstance(response, list):
                response = response[0] if response else None
                if response is None:
                    self.stats['failed'] += 1
                    self._on_llm_failure()
                    return None
            
            # Convert to StrategyGene
            gene = self._json_to_strategy_gene(response, generation, individual_id)
            
            if gene:
                self.stats['successful'] += 1
                self._on_llm_success()
                logger.info(f"LLM generated strategy {individual_id} "
                          f"({strategy_style or 'immigrant'}): "
                          f"{len(gene.indicators)} indicators, "
                          f"{len(gene.entry_conditions)} entry, "
                          f"{len(gene.exit_conditions)} exit conditions")
            else:
                self.stats['failed'] += 1
                self._on_llm_failure()
            
            return gene
            
        except Exception as e:
            logger.error(f"LLM strategy generation failed: {e}")
            self.stats['failed'] += 1
            self._on_llm_failure()
            return None
    
    def _json_to_strategy_gene(
        self,
        data: Dict[str, Any],
        generation: int,
        individual_id: int,
    ) -> Optional[StrategyGene]:
        """
        Convert LLM JSON output to a validated StrategyGene.
        
        Performs validation and auto-fixes for common LLM errors:
        - Unknown indicator types → removed
        - Missing instance_ids → auto-assigned
        - Dangling condition references → removed
        - Missing required fields → defaults
        
        Args:
            data: Parsed JSON dict from LLM
            generation: Generation number
            individual_id: Individual ID
            
        Returns:
            Valid StrategyGene or None if unfixable
        """
        fixes_applied = 0
        
        try:
            # --- Parse Indicators ---
            raw_indicators = data.get('indicators', [])
            indicators = []
            used_ids = set()
            
            for ind_data in raw_indicators:
                ind_type = ind_data.get('type', '')
                
                # Validate indicator type
                if ind_type not in self.available_indicators:
                    logger.debug(f"LLM used unknown indicator '{ind_type}', skipping")
                    fixes_applied += 1
                    continue
                
                # Parse instance_id
                instance_id = ind_data.get('instance_id')
                if not instance_id or instance_id in used_ids:
                    # Auto-assign unique instance_id
                    counter = 0
                    while f"{ind_type}_{counter}" in used_ids:
                        counter += 1
                    instance_id = f"{ind_type}_{counter}"
                    fixes_applied += 1
                
                used_ids.add(instance_id)
                
                # Parse parameters
                parameters = ind_data.get('parameters', {})
                if not isinstance(parameters, dict):
                    parameters = {}
                
                indicators.append(IndicatorGene(
                    type=ind_type,
                    parameters=parameters,
                    weight=float(ind_data.get('weight', 1.0)),
                    instance_id=instance_id,
                    timeframe=ind_data.get('timeframe'),
                ))
            
            if not indicators:
                logger.warning("LLM produced no valid indicators")
                return None
            
            # --- Valid indicator references ---
            valid_refs = {ind.instance_id for ind in indicators}
            valid_refs.update(ind.type for ind in indicators)
            
            # --- Parse Entry Conditions ---
            entry_conditions = self._parse_conditions(
                data.get('entry_conditions', []), valid_refs, 'entry'
            )
            
            # --- Parse Exit Conditions ---
            exit_conditions = self._parse_conditions(
                data.get('exit_conditions', []), valid_refs, 'exit'
            )
            
            # --- Validate minimums ---
            min_entry = self.config.get('indicators', {}).get('min_entry_conditions', 2)
            min_exit = self.config.get('indicators', {}).get('min_exit_conditions', 1)
            
            if len(entry_conditions) < min_entry:
                logger.warning(f"LLM produced {len(entry_conditions)} entry conditions "
                             f"(min {min_entry}), adding defaults")
                self._add_default_conditions(indicators, entry_conditions, 
                                            min_entry - len(entry_conditions), is_entry=True)
                fixes_applied += 1
            
            if len(exit_conditions) < min_exit:
                logger.warning(f"LLM produced {len(exit_conditions)} exit conditions "
                             f"(min {min_exit}), adding defaults")
                self._add_default_conditions(indicators, exit_conditions,
                                            min_exit - len(exit_conditions), is_entry=False)
                fixes_applied += 1
            
            if not entry_conditions:
                logger.warning("LLM produced no usable entry conditions")
                return None
            
            # --- Parse Risk Parameters ---
            timeframe = data.get('timeframe', '15m')
            available_tfs = self.config.get('strategy_constraints', {}).get(
                'timeframes', ['5m', '15m', '1h']
            )
            if timeframe not in available_tfs:
                timeframe = random.choice(available_tfs)
                fixes_applied += 1
            
            stoploss = float(data.get('stoploss', -0.10))
            sl_range = self.config.get('strategy_constraints', {}).get(
                'stoploss_range', [-0.20, -0.05]
            )
            stoploss = max(sl_range[0], min(sl_range[1], stoploss))
            
            minimal_roi = data.get('minimal_roi', {"0": 0.05, "30": 0.03, "60": 0.01})
            if not isinstance(minimal_roi, dict):
                minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.01}
            # Ensure string keys
            minimal_roi = {str(k): float(v) for k, v in minimal_roi.items()}
            
            max_open_trades = int(data.get('max_open_trades', 3))
            max_open_trades = max(1, min(10, max_open_trades))
            
            trailing_stop = bool(data.get('trailing_stop', False))
            trailing_stop_positive = data.get('trailing_stop_positive')
            if trailing_stop_positive is not None:
                trailing_stop_positive = float(trailing_stop_positive)
            trailing_stop_positive_offset = data.get('trailing_stop_positive_offset')
            if trailing_stop_positive_offset is not None:
                trailing_stop_positive_offset = float(trailing_stop_positive_offset)
            
            # --- Build StrategyGene ---
            gene = StrategyGene(
                generation=generation,
                individual_id=individual_id,
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                timeframe=timeframe,
                stoploss=stoploss,
                minimal_roi=minimal_roi,
                max_open_trades=max_open_trades,
                trailing_stop=trailing_stop,
                trailing_stop_positive=trailing_stop_positive,
                trailing_stop_positive_offset=trailing_stop_positive_offset,
            )
            
            if fixes_applied > 0:
                self.stats['validation_fixed'] += 1
                logger.info(f"Applied {fixes_applied} validation fixes to LLM strategy")
            
            return gene
            
        except Exception as e:
            logger.error(f"Failed to parse LLM strategy JSON: {e}")
            return None
    
    def _parse_conditions(
        self,
        raw_conditions: List[Dict],
        valid_refs: set,
        condition_type: str,
    ) -> List[ConditionGene]:
        """Parse and validate conditions from LLM output."""
        valid_operators = {'<', '>', 'cross_above', 'cross_below', 
                          'increasing', 'decreasing', 'between', 'value_above_ago'}
        
        conditions = []
        for cond_data in raw_conditions:
            if not isinstance(cond_data, dict):
                continue
            
            indicator = cond_data.get('indicator', '')
            operator = cond_data.get('operator', '')
            
            # Validate indicator reference
            if indicator not in valid_refs:
                logger.debug(f"Skipping {condition_type} condition: "
                           f"unknown indicator ref '{indicator}'")
                continue
            
            # Validate operator
            if operator not in valid_operators:
                logger.debug(f"Skipping {condition_type} condition: "
                           f"unknown operator '{operator}'")
                continue
            
            try:
                threshold = float(cond_data.get('threshold', 0))
            except (TypeError, ValueError):
                threshold = 0.0
            
            try:
                threshold_upper = float(cond_data.get('threshold_upper', 0))
            except (TypeError, ValueError):
                threshold_upper = 0.0
            
            try:
                lookback = int(cond_data.get('lookback', 3))
            except (TypeError, ValueError):
                lookback = 3
            
            conditions.append(ConditionGene(
                indicator=indicator,
                operator=operator,
                threshold=threshold,
                logic=cond_data.get('logic', 'AND'),
                threshold_upper=threshold_upper,
                lookback=max(1, lookback),
            ))
        
        return conditions
    
    def _add_default_conditions(
        self,
        indicators: List[IndicatorGene],
        conditions: List[ConditionGene],
        needed: int,
        is_entry: bool,
    ) -> None:
        """Add default conditions for indicators that don't have conditions yet."""
        # Find indicators not already referenced
        referenced = {c.indicator for c in conditions}
        unreferenced = [ind for ind in indicators 
                       if ind.instance_id not in referenced and ind.type not in referenced]
        
        # If all are referenced, use any indicator
        if not unreferenced:
            unreferenced = list(indicators)
        
        for i in range(min(needed, len(unreferenced))):
            ind = unreferenced[i]
            ref = INDICATOR_REFERENCE.get(ind.type, {})
            
            if is_entry:
                template = ref.get('typical_entry', {'operator': '>', 'threshold': '50'})
            else:
                template = ref.get('typical_exit', {'operator': '<', 'threshold': '50'})
            
            # Parse threshold from template (may be a range string like "20-40")
            threshold_str = str(template.get('threshold', '50'))
            try:
                # Try to extract first number from string like "20-40 (oversold)"
                import re
                numbers = re.findall(r'-?\d+\.?\d*', threshold_str)
                threshold = float(numbers[0]) if numbers else 50.0
            except (IndexError, ValueError):
                threshold = 50.0
            
            operator = template.get('operator', '>' if is_entry else '<')
            
            conditions.append(ConditionGene(
                indicator=ind.instance_id or ind.type,
                operator=operator,
                threshold=threshold,
                logic='AND',
            ))
    
    def record_llm_performance(self, generation: int, population) -> None:
        """
        Record how LLM-generated strategies performed this generation.
        
        Compares LLM-origin individuals against random-origin individuals
        and tracks survival/elimination across generations.
        
        Args:
            generation: Current generation number
            population: Evaluated Population object
        """
        llm_individuals = []
        random_individuals = []
        
        for ind in population.individuals:
            origin = getattr(ind, 'metrics', {}).get('origin', 'random')
            if origin in ('llm_seed', 'llm_immigrant', 'llm_mutation'):
                llm_individuals.append(ind)
            else:
                random_individuals.append(ind)
        
        # Calculate per-generation stats
        llm_fitnesses = [ind.fitness for ind in llm_individuals if ind.fitness is not None]
        random_fitnesses = [ind.fitness for ind in random_individuals if ind.fitness is not None]
        
        avg_llm = sum(llm_fitnesses) / len(llm_fitnesses) if llm_fitnesses else 0.0
        avg_random = sum(random_fitnesses) / len(random_fitnesses) if random_fitnesses else 0.0
        best_llm = max(llm_fitnesses) if llm_fitnesses else 0.0
        
        # Collect detailed LLM strategy results for feedback
        llm_results = []
        for ind in llm_individuals:
            m = getattr(ind, 'metrics', {})
            gene = getattr(ind, 'strategy_gene', None)
            llm_results.append({
                'fitness': ind.fitness or 0.0,
                'profit': m.get('profit', 0),
                'max_drawdown': m.get('max_drawdown', 0),
                'win_rate': m.get('win_rate', 0),
                'num_trades': m.get('num_trades', 0),
                'origin': m.get('origin', 'llm'),
                'indicators': [i.type for i in gene.indicators] if gene else [],
                'survived': True,  # It's in the population, so it survived
            })
        
        gen_record = {
            'generation': generation,
            'llm_count': len(llm_individuals),
            'random_count': len(random_individuals),
            'avg_llm_fitness': round(avg_llm, 4),
            'avg_random_fitness': round(avg_random, 4),
            'best_llm_fitness': round(best_llm, 4),
            'llm_results': llm_results,
        }
        self.generation_history.append(gen_record)
        
        # Keep only last 10 generations of history to limit memory
        if len(self.generation_history) > 10:
            self.generation_history = self.generation_history[-10:]
        
        # Update cumulative performance stats
        self.llm_performance['llm_survived'] += len(llm_individuals)
        if best_llm > self.llm_performance['best_llm_fitness']:
            self.llm_performance['best_llm_fitness'] = best_llm
            self.llm_performance['best_llm_generation'] = generation
        
        # Running averages (weighted by count)
        total_llm = sum(r['llm_count'] for r in self.generation_history)
        total_random = sum(r['random_count'] for r in self.generation_history)
        if total_llm > 0:
            self.llm_performance['avg_llm_fitness'] = round(
                sum(r['avg_llm_fitness'] * r['llm_count'] for r in self.generation_history) / total_llm, 4
            )
        if total_random > 0:
            self.llm_performance['avg_random_fitness'] = round(
                sum(r['avg_random_fitness'] * r['random_count'] for r in self.generation_history) / total_random, 4
            )
        
        # Log comparison
        advantage = avg_llm - avg_random
        emoji = "+" if advantage >= 0 else ""
        logger.info(f"[LLM FEEDBACK] Gen {generation}: "
                   f"LLM avg={avg_llm:.4f} ({len(llm_individuals)} strategies) | "
                   f"Random avg={avg_random:.4f} ({len(random_individuals)}) | "
                   f"Advantage: {emoji}{advantage:.4f}")
    
    def build_feedback_context(self, feature_report: Optional[Dict] = None,
                                evolution_progress: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Build comprehensive feedback context for LLM prompt enrichment.
        
        Combines:
        1. LLM strategy performance history (what worked/failed)
        2. Feature importance data (which indicators the GA favors)
        3. Evolution progress (generation, fitness trend, diversity)
        
        Args:
            feature_report: Output from FeatureImportanceTracker.get_report()
            evolution_progress: Dict with generation, best_fitness, diversity, etc.
            
        Returns:
            Feedback dict to pass to generate_immigrants()
        """
        feedback: Dict[str, Any] = {}
        
        # 1. Performance history from last 2 generations
        if self.generation_history:
            recent = self.generation_history[-2:]
            performance_entries = []
            for gen_record in recent:
                for result in gen_record.get('llm_results', []):
                    performance_entries.append({
                        'generation': gen_record['generation'],
                        'fitness': result['fitness'],
                        'profit': result['profit'],
                        'max_drawdown': result['max_drawdown'],
                        'win_rate': result['win_rate'],
                        'num_trades': result['num_trades'],
                        'indicators': result['indicators'],
                    })
            feedback['llm_strategy_results'] = performance_entries
            feedback['llm_vs_random'] = {
                'avg_llm_fitness': self.llm_performance['avg_llm_fitness'],
                'avg_random_fitness': self.llm_performance['avg_random_fitness'],
                'best_llm_fitness': self.llm_performance['best_llm_fitness'],
            }
        
        # 2. Feature importance — which indicators the GA selects for
        if feature_report:
            top_indicators = feature_report.get('indicators', [])[:8]
            feedback['feature_importance'] = [
                {
                    'indicator': ind['name'],
                    'importance_score': ind['importance_score'],
                    'avg_fitness': ind['avg_fitness'],
                }
                for ind in top_indicators
            ]
            top_patterns = feature_report.get('top_condition_patterns', [])[:5]
            feedback['top_condition_patterns'] = [
                {
                    'pattern': p['pattern'],
                    'score': p['importance_score'],
                }
                for p in top_patterns
            ]
        
        # 3. Evolution progress context
        if evolution_progress:
            feedback['evolution_progress'] = evolution_progress
        
        return feedback
    
    def _rate_limit(self):
        """Enforce minimum interval between API calls with adaptive backoff.

        On consecutive failures the effective interval increases by 1.5x
        (up to ``_max_call_interval``).  On success, it resets to the
        configured base interval.  A small random jitter (0-25%) is added
        to avoid thundering-herd effects when multiple calls are queued.
        """
        import random as _rand
        interval = self._effective_call_interval
        jitter = interval * _rand.uniform(0, 0.25)
        elapsed = time.time() - self._last_call_time
        wait = (interval + jitter) - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.time()

    def _on_llm_success(self):
        """Reset adaptive rate-limit interval after a successful call."""
        self._effective_call_interval = self._min_call_interval

    def _on_llm_failure(self):
        """Increase adaptive rate-limit interval after a failed call."""
        self._effective_call_interval = min(
            self._effective_call_interval * 1.5,
            self._max_call_interval,
        )

    def _budget_available(self) -> bool:
        """Check if LLM call budget allows another call."""
        if self._calls_this_generation >= self._max_calls_per_generation:
            logger.debug("LLM generation budget exhausted (%d/%d)",
                        self._calls_this_generation, self._max_calls_per_generation)
            return False
        if self._calls_this_run >= self._max_calls_per_run:
            logger.debug("LLM run budget exhausted (%d/%d)",
                        self._calls_this_run, self._max_calls_per_run)
            return False
        return True

    def _record_call(self):
        """Increment call counters."""
        self._calls_this_generation += 1
        self._calls_this_run += 1

    def reset_generation_budget(self):
        """Reset per-generation call counter (call at start of each generation)."""
        self._calls_this_generation = 0

    # ------------------------------------------------------------------
    # LLM-Guided Mutation
    # ------------------------------------------------------------------

    def mutate_strategy(
        self,
        parent_gene: 'StrategyGene',
        metrics: Dict[str, Any],
        generation: int,
        individual_id: int,
        failure_mode: Optional[str] = None,
        objective: Optional[str] = None,
    ) -> Optional['StrategyGene']:
        """
        Use the LLM to propose a targeted mutation of an existing strategy.

        Instead of generating a new strategy from scratch, asks the LLM to
        make 1–3 minimal edits to address a specific diagnosed problem.

        Args:
            parent_gene: The parent StrategyGene to mutate.
            metrics: Backtest metrics of the parent.
            generation: Generation number for the child.
            individual_id: Individual ID for the child.
            failure_mode: Diagnosed failure string. Auto-diagnosed if None.
            objective: Mutation objective string. Auto-selected if None.

        Returns:
            Mutated StrategyGene or None on failure.
        """
        if not self.enabled or not self.provider:
            return None
        if not self._budget_available():
            return None

        self.stats['total_requests'] += 1
        self.stats['calls_by_type']['mutation'] += 1
        self._rate_limit()

        try:
            parent_dict = parent_gene.to_dict()

            # Auto-diagnose if not provided
            if failure_mode is None:
                failure_mode = diagnose_failure_mode(metrics, self.config)
            if failure_mode is None:
                failure_mode = (
                    "No specific failure detected. Make a small improvement: "
                    "try a different indicator combination, adjust thresholds, "
                    "or add a complementary filter."
                )
            if objective is None:
                objective = select_mutation_objective(metrics, self.config)

            system_prompt = self.prompt_builder.build_system_prompt()
            user_prompt = self.prompt_builder.build_mutation_prompt(
                parent_strategy=parent_dict,
                metrics=metrics,
                failure_mode=failure_mode,
                objective=objective,
            )

            response = self.provider.generate_json(user_prompt, system_prompt)
            self._record_call()

            if response is None:
                logger.warning("LLM mutation returned no valid JSON")
                self.stats['failed'] += 1
                return None

            # Handle array response
            if isinstance(response, list):
                response = response[0] if response else None
                if response is None:
                    self.stats['failed'] += 1
                    return None

            gene = self._json_to_strategy_gene(response, generation, individual_id)
            if gene:
                self.stats['successful'] += 1
                logger.info(
                    f"LLM mutation of strategy {parent_gene.individual_id} → {individual_id}: "
                    f"objective={objective}, {len(gene.indicators)} indicators, "
                    f"{len(gene.entry_conditions)} entry, {len(gene.exit_conditions)} exit"
                )
            else:
                self.stats['failed'] += 1

            return gene

        except Exception as e:
            logger.error(f"LLM mutation failed: {e}")
            self.stats['failed'] += 1
            return None

    def mutate_strategies_batch(
        self,
        parents: List[Dict[str, Any]],
        generation: int,
        start_id: int = 0,
    ) -> List[Optional['StrategyGene']]:
        """
        Batch-mutate multiple strategies in a single LLM call.

        Each item in *parents* should have keys:
        ``gene`` (StrategyGene), ``metrics`` (dict), and optionally
        ``failure_mode`` (str) and ``objective`` (str).

        Args:
            parents: List of dicts with parent info.
            generation: Generation number for children.
            start_id: Starting individual ID.

        Returns:
            List of mutated StrategyGene (None for failures).
        """
        if not self.enabled or not self.provider or not parents:
            return []
        if not self._budget_available():
            return []

        self.stats['total_requests'] += 1
        self.stats['calls_by_type']['batch_mutation'] += 1
        self._rate_limit()

        try:
            batch_items = []
            for item in parents:
                gene = item['gene']
                metrics = item['metrics']
                fm = item.get('failure_mode') or diagnose_failure_mode(metrics, self.config)
                if fm is None:
                    fm = "No specific failure. Make a small targeted improvement."
                obj = item.get('objective') or select_mutation_objective(metrics, self.config)
                batch_items.append({
                    'parent': gene.to_dict(),
                    'metrics': metrics,
                    'failure_mode': fm,
                    'objective': obj,
                })

            system_prompt = self.prompt_builder.build_system_prompt()
            user_prompt = self.prompt_builder.build_batch_mutation_prompt(batch_items)

            response = self.provider.generate_json(user_prompt, system_prompt)
            self._record_call()

            if response is None:
                self.stats['failed'] += 1
                return []

            # Expect an array of strategies
            if isinstance(response, dict):
                response = [response]
            if not isinstance(response, list):
                self.stats['failed'] += 1
                return []

            results = []
            for i, resp_item in enumerate(response):
                if i >= len(parents):
                    break
                gene = self._json_to_strategy_gene(resp_item, generation, start_id + i)
                if gene:
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
                results.append(gene)

            logger.info(f"LLM batch mutation: {len([r for r in results if r])}/{len(parents)} successful")
            return results

        except Exception as e:
            logger.error(f"LLM batch mutation failed: {e}")
            self.stats['failed'] += 1
            return []

    # ------------------------------------------------------------------
    # Batched seed / immigrant generation
    # ------------------------------------------------------------------

    def generate_seed_strategies_batch(
        self,
        count: int,
        generation: int = 0,
        start_id: int = 0,
    ) -> List['StrategyGene']:
        """
        Generate multiple seed strategies in a single LLM call.

        Falls back to sequential generation if batch parsing fails.

        Args:
            count: Number of seeds to generate.
            generation: Generation number.
            start_id: Starting individual ID.

        Returns:
            List of StrategyGene objects.
        """
        if not self.enabled or not self.provider:
            return []
        if not self._budget_available():
            return []

        self.stats['total_requests'] += 1
        self.stats['calls_by_type']['batch_seed'] += 1
        self._rate_limit()

        try:
            styles = get_diverse_styles(count)
            system_prompt = self.prompt_builder.build_system_prompt()
            user_prompt = self.prompt_builder.build_batch_seed_prompt(
                count=count, styles=styles
            )

            response = self.provider.generate_json(user_prompt, system_prompt)
            self._record_call()

            if response is None:
                self.stats['failed'] += 1
                logger.warning("Batch seed generation failed, falling back to sequential")
                return self.generate_seed_strategies(count, generation, start_id)

            # Expect array
            if isinstance(response, dict):
                response = [response]
            if not isinstance(response, list):
                self.stats['failed'] += 1
                return self.generate_seed_strategies(count, generation, start_id)

            strategies = []
            for i, item in enumerate(response):
                gene = self._json_to_strategy_gene(item, generation, start_id + i)
                if gene:
                    strategies.append(gene)
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1

            logger.info(f"LLM batch seed generation: {len(strategies)}/{count} successful")
            return strategies

        except Exception as e:
            logger.error(f"Batch seed generation failed: {e}")
            self.stats['failed'] += 1
            return self.generate_seed_strategies(count, generation, start_id)

    def generate_immigrants_batch(
        self,
        count: int,
        generation: int,
        start_id: int = 0,
        top_performers: Optional[List[Dict]] = None,
        weaknesses: Optional[List[str]] = None,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> List['StrategyGene']:
        """
        Generate multiple immigrants in a single LLM call.

        Falls back to sequential generation if batch parsing fails.

        Args:
            count: Number of immigrants to generate.
            generation: Current generation number.
            start_id: Starting individual ID.
            top_performers: Top strategy summaries.
            weaknesses: Population weakness gaps.
            feedback: Performance feedback and feature importance.

        Returns:
            List of StrategyGene objects.
        """
        if not self.enabled or not self.provider:
            return []
        if not self._budget_available():
            return []

        self.stats['total_requests'] += 1
        self.stats['calls_by_type']['batch_immigrant'] += 1
        self._rate_limit()

        try:
            system_prompt = self.prompt_builder.build_system_prompt()
            user_prompt = self.prompt_builder.build_batch_immigrant_prompt(
                count=count,
                top_performers=top_performers,
                weaknesses=weaknesses,
                feedback=feedback,
            )

            response = self.provider.generate_json(user_prompt, system_prompt)
            self._record_call()

            if response is None:
                self.stats['failed'] += 1
                logger.warning("Batch immigrant generation failed, falling back to sequential")
                return self.generate_immigrants(
                    count, generation, start_id,
                    top_performers, weaknesses, feedback
                )

            if isinstance(response, dict):
                response = [response]
            if not isinstance(response, list):
                self.stats['failed'] += 1
                return self.generate_immigrants(
                    count, generation, start_id,
                    top_performers, weaknesses, feedback
                )

            strategies = []
            for i, item in enumerate(response):
                gene = self._json_to_strategy_gene(item, generation, start_id + i)
                if gene:
                    strategies.append(gene)
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1

            logger.info(f"LLM batch immigrant generation: {len(strategies)}/{count} successful "
                       f"(gen {generation})")
            return strategies

        except Exception as e:
            logger.error(f"Batch immigrant generation failed: {e}")
            self.stats['failed'] += 1
            return self.generate_immigrants(
                count, generation, start_id,
                top_performers, weaknesses, feedback
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Return generation statistics including LLM-vs-random performance."""
        stats = dict(self.stats)
        stats['llm_performance'] = dict(self.llm_performance)
        stats['generations_tracked'] = len(self.generation_history)
        stats['calls_this_run'] = self._calls_this_run
        stats['budget_remaining'] = max(0, self._max_calls_per_run - self._calls_this_run)
        return stats
    
    def get_population_weaknesses(self, top_individuals: list) -> List[str]:
        """
        Analyze current top individuals to identify population weaknesses.
        
        This helps the LLM generate strategies that fill gaps.
        
        Args:
            top_individuals: List of top Individual objects
            
        Returns:
            List of weakness descriptions
        """
        weaknesses = []
        
        if not top_individuals:
            return ["No existing strategies to analyze"]
        
        # Collect indicator usage across top strategies
        indicator_usage = {}
        for ind in top_individuals:
            gene = getattr(ind, 'strategy_gene', None)
            if gene:
                for indicator in gene.indicators:
                    indicator_usage[indicator.type] = indicator_usage.get(indicator.type, 0) + 1
        
        # Find underused indicators
        for ind_type in self.available_indicators:
            if indicator_usage.get(ind_type, 0) == 0:
                weaknesses.append(f"No strategies use {ind_type} indicator")
        
        # Check for missing strategy types
        metrics_list = [getattr(ind, 'metrics', {}) for ind in top_individuals]
        
        avg_trades = sum(m.get('num_trades', 0) for m in metrics_list) / max(len(metrics_list), 1)
        if avg_trades > 50:
            weaknesses.append("Population skews toward high-frequency; need strategies with fewer, higher-quality trades")
        elif avg_trades < 10:
            weaknesses.append("Population skews toward low-frequency; need strategies with more trade opportunities")
        
        avg_dd = sum(m.get('max_drawdown', 0) for m in metrics_list) / max(len(metrics_list), 1)
        if avg_dd > 0.20:
            weaknesses.append("High average drawdown; need conservative strategies with tighter risk management")
        
        return weaknesses
    
    def get_top_performer_summaries(self, top_individuals: list) -> List[Dict]:
        """
        Create compact summaries of top individuals for LLM context.
        
        Args:
            top_individuals: List of top Individual objects
            
        Returns:
            List of summary dicts
        """
        summaries = []
        for ind in top_individuals[:5]:
            gene = getattr(ind, 'strategy_gene', None)
            metrics = getattr(ind, 'metrics', {})
            
            summary = {
                'fitness': getattr(ind, 'fitness', 0.0),
                'profit': metrics.get('profit', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'num_trades': metrics.get('num_trades', 0),
                'win_rate': metrics.get('win_rate', 0),
                'indicators': [i.type for i in gene.indicators] if gene else [],
            }
            summaries.append(summary)
        
        return summaries
