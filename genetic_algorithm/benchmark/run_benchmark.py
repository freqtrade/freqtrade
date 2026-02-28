#!/usr/bin/env python3
"""
GA Benchmark & Measurement Harness

Runs an end-to-end evolution and collects comprehensive metrics:
  - Correctness: fitness values, code validity, NaN/crash detection
  - Performance: timing per phase, memory usage, throughput
  - Quality: profit distribution, win rates, trade counts
  - Evolutionary dynamics: selection pressure, diversity, elite stability

Produces a structured JSON report + human-readable summary.
"""

import sys
import os
import gc
import json
import time
import math
import random
import logging
import traceback
import tempfile
import textwrap
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Data classes for structured metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TimingMetrics:
    """Wall-clock timing for one generation."""
    generation: int = 0
    total_sec: float = 0.0
    evaluate_sec: float = 0.0
    wf_posthoc_sec: float = 0.0
    selection_crossover_mutation_sec: float = 0.0
    stats_sec: float = 0.0
    overhead_sec: float = 0.0  # total - sum of above


@dataclass
class CorrectnessMetrics:
    """Correctness checks for one generation."""
    generation: int = 0
    nan_fitness_count: int = 0
    inf_fitness_count: int = 0
    negative_fitness_count: int = 0
    zero_fitness_count: int = 0
    valid_fitness_count: int = 0
    failed_evaluations: int = 0
    code_syntax_errors: int = 0
    code_syntax_ok: int = 0
    fitness_values: List[float] = field(default_factory=list)


@dataclass
class QualityMetrics:
    """Quality of strategies in one generation."""
    generation: int = 0
    best_fitness: float = 0.0          # Best shared fitness (after fitness sharing)
    best_raw_fitness: float = 0.0      # Best raw fitness (true strategy quality)
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    median_fitness: float = 0.0
    fitness_std: float = 0.0
    genetic_diversity: Optional[float] = None
    best_profit: float = 0.0
    avg_profit: float = 0.0
    best_sharpe: float = 0.0
    best_win_rate: float = 0.0
    best_drawdown: float = 0.0
    avg_trades: float = 0.0
    zero_trade_count: int = 0
    # Selection pressure = best / avg (higher = more pressure)
    selection_pressure: float = 0.0
    # Exploitation ratio = elite fitness / population avg
    exploitation_ratio: float = 0.0
    # Number of unique indicator types across population
    unique_indicator_types: int = 0
    # Number of unique strategy "fingerprints" (approximate)
    unique_strategies: int = 0


@dataclass
class EvolutionaryDynamics:
    """Tracks evolutionary dynamics across generations."""
    elite_turnover: List[float] = field(default_factory=list)  # Identity-based turnover (by individual_id)
    elite_fitness_turnover: List[float] = field(default_factory=list)  # Fitness-based turnover (by raw_fitness value)
    best_fitness_trajectory: List[float] = field(default_factory=list)  # Shared fitness
    best_raw_fitness_trajectory: List[float] = field(default_factory=list)  # Raw fitness (true quality)
    avg_fitness_trajectory: List[float] = field(default_factory=list)
    diversity_trajectory: List[Optional[float]] = field(default_factory=list)
    mutation_rate_trajectory: List[float] = field(default_factory=list)
    selection_pressure_trajectory: List[float] = field(default_factory=list)
    best_changed_at: List[int] = field(default_factory=list)  # Gens where best improved


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_per_generation: List[float] = field(default_factory=list)
    cpu_count: int = 0
    parallel_workers: int = 0
    python_version: str = ""
    platform: str = ""


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str = ""
    config_file: str = ""
    config_summary: Dict[str, Any] = field(default_factory=dict)
    duration_sec: float = 0.0
    completed: bool = False
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Per-generation metrics
    timing: List[Dict] = field(default_factory=list)
    correctness: List[Dict] = field(default_factory=list)
    quality: List[Dict] = field(default_factory=list)
    
    # Aggregate metrics
    dynamics: Dict = field(default_factory=dict)
    system: Dict = field(default_factory=dict)
    
    # Final results
    top_strategies: List[Dict] = field(default_factory=list)
    
    # Correctness summary
    correctness_summary: Dict = field(default_factory=dict)
    
    # Performance summary
    performance_summary: Dict = field(default_factory=dict)
    
    # Diagnostic warnings
    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patching the evolution loop to capture per-phase timing
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentedGA:
    """
    Wraps the GeneticAlgorithm class with instrumentation for detailed metrics.
    
    Instead of modifying evolution.py, we monkey-patch the evolve() loop to
    capture per-phase timing and per-generation metrics. This avoids touching
    production code while giving us deep visibility.
    """
    
    def __init__(self, config_path: str):
        from genetic_algorithm.core.evolution import GeneticAlgorithm
        self.ga = GeneticAlgorithm(config_path, visualize=False, interactive=False)
        
        # Metrics storage
        self.timing_data: List[TimingMetrics] = []
        self.correctness_data: List[CorrectnessMetrics] = []
        self.quality_data: List[QualityMetrics] = []
        self.dynamics = EvolutionaryDynamics()
        self.system_metrics = SystemMetrics()
        
        # Track previous elites for turnover computation
        self._prev_elite_ids: set = set()
        self._prev_elite_fitnesses: set = set()
        
        # Collect system info
        import platform
        self.system_metrics.python_version = platform.python_version()
        self.system_metrics.platform = platform.platform()
        self.system_metrics.cpu_count = os.cpu_count() or 1
        if self.ga.parallel_evaluator:
            self.system_metrics.parallel_workers = self.ga.parallel_evaluator.num_workers
    
    def _get_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # ru_maxrss is in KB on Linux
        except Exception:
            return 0.0
    
    def _check_code_syntax(self, gene) -> bool:
        """Check if generated strategy code is valid Python."""
        try:
            code = self.ga.strategy_generator.generate_strategy_code(gene)
            compile(code, '<strategy>', 'exec')
            return True
        except Exception:
            return False
    
    def _strategy_fingerprint(self, gene) -> str:
        """Create a rough fingerprint for a strategy for uniqueness counting."""
        indicators = sorted([f"{i.type}_{i.timeframe}" for i in gene.indicators])
        conditions = sorted([f"{c.indicator}_{c.operator}" for c in gene.entry_conditions])
        return "|".join(indicators + conditions)
    
    def _collect_correctness(self, generation: int, population) -> CorrectnessMetrics:
        """Collect correctness metrics for a generation."""
        cm = CorrectnessMetrics(generation=generation)
        
        for ind in population.individuals:
            f = ind.fitness
            if f is None or (isinstance(f, float) and math.isnan(f)):
                cm.nan_fitness_count += 1
            elif isinstance(f, float) and math.isinf(f):
                cm.inf_fitness_count += 1
            elif f < 0:
                cm.negative_fitness_count += 1
            elif f == 0:
                cm.zero_fitness_count += 1
            else:
                cm.valid_fitness_count += 1
            
            if isinstance(f, (int, float)) and not math.isnan(f) and not math.isinf(f):
                cm.fitness_values.append(float(f))
            
            if not ind.evaluated:
                cm.failed_evaluations += 1
        
        # Check code syntax for a sample (not all — too slow)
        sample_size = min(5, len(population.individuals))
        sample = random.sample(list(population.individuals), sample_size)
        for ind in sample:
            if self._check_code_syntax(ind.strategy_gene):
                cm.code_syntax_ok += 1
            else:
                cm.code_syntax_errors += 1
        
        return cm
    
    def _collect_quality(self, generation: int, population, stats) -> QualityMetrics:
        """Collect quality metrics for a generation."""
        qm = QualityMetrics(generation=generation)
        
        fitnesses = [ind.fitness for ind in population.individuals 
                     if ind.fitness is not None and not math.isnan(ind.fitness)]
        
        raw_fitnesses = [ind.raw_fitness for ind in population.individuals 
                         if ind.raw_fitness is not None and not math.isnan(ind.raw_fitness)]
        
        if fitnesses:
            qm.best_fitness = max(fitnesses)
            qm.worst_fitness = min(fitnesses)
            qm.avg_fitness = sum(fitnesses) / len(fitnesses)
            sorted_f = sorted(fitnesses)
            mid = len(sorted_f) // 2
            qm.median_fitness = sorted_f[mid] if len(sorted_f) % 2 else (sorted_f[mid - 1] + sorted_f[mid]) / 2
            qm.fitness_std = (sum((f - qm.avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
            qm.selection_pressure = qm.best_fitness / qm.avg_fitness if qm.avg_fitness > 0 else 0
        
        if raw_fitnesses:
            qm.best_raw_fitness = max(raw_fitnesses)
        
        qm.genetic_diversity = stats.genetic_diversity
        
        # Collect strategy-level metrics
        profits = []
        sharpes = []
        win_rates = []
        drawdowns = []
        trades = []
        indicator_types = set()
        fingerprints = set()
        
        for ind in population.individuals:
            m = ind.metrics or {}
            profits.append(m.get('profit', 0))
            sharpes.append(m.get('sharpe_ratio', 0))
            win_rates.append(m.get('win_rate', 0))
            drawdowns.append(m.get('max_drawdown', 0))
            num_trades = m.get('num_trades', 0)
            trades.append(num_trades)
            if num_trades == 0:
                qm.zero_trade_count += 1
            
            for indicator in ind.strategy_gene.indicators:
                indicator_types.add(indicator.type)
            
            fingerprints.add(self._strategy_fingerprint(ind.strategy_gene))
        
        if profits:
            qm.best_profit = max(profits)
            qm.avg_profit = sum(profits) / len(profits)
        if sharpes:
            qm.best_sharpe = max(sharpes)
        if win_rates:
            qm.best_win_rate = max(win_rates)
        if drawdowns:
            qm.best_drawdown = min(drawdowns)  # Best drawdown = lowest
        if trades:
            qm.avg_trades = sum(trades) / len(trades)
        
        qm.unique_indicator_types = len(indicator_types)
        qm.unique_strategies = len(fingerprints)
        
        # Exploitation ratio
        elite_size = self.ga.elite_size
        top_fitnesses = sorted(fitnesses, reverse=True)[:elite_size]
        if top_fitnesses and qm.avg_fitness > 0:
            qm.exploitation_ratio = (sum(top_fitnesses) / len(top_fitnesses)) / qm.avg_fitness
        
        return qm
    
    def _compute_elite_turnover(self, population) -> tuple:
        """Compute elite turnover by both identity and fitness value.
        
        Returns:
            (id_turnover, fitness_turnover) — each in [0, 1].
            id_turnover: fraction of individual_ids that changed.
            fitness_turnover: fraction of raw_fitness values that changed.
        """
        # Select elites by raw_fitness (same logic as evolution.py elitism)
        ranked = sorted(
            [ind for ind in population.individuals if ind.raw_fitness is not None],
            key=lambda x: x.raw_fitness,
            reverse=True,
        )
        top = ranked[:self.ga.elite_size]
        
        current_ids = set()
        current_fitnesses = set()
        for ind in top:
            # Use individual_id (strips generation prefix) for identity tracking
            current_ids.add(ind.strategy_gene.individual_id)
            current_fitnesses.add(round(ind.raw_fitness, 6))
        
        if not self._prev_elite_ids:
            self._prev_elite_ids = current_ids
            self._prev_elite_fitnesses = current_fitnesses
            return 1.0, 1.0  # First generation = 100% turnover
        
        id_overlap = len(current_ids & self._prev_elite_ids)
        id_turnover = 1.0 - (id_overlap / max(len(current_ids), 1))
        
        fit_overlap = len(current_fitnesses & self._prev_elite_fitnesses)
        fit_turnover = 1.0 - (fit_overlap / max(len(current_fitnesses), 1))
        
        self._prev_elite_ids = current_ids
        self._prev_elite_fitnesses = current_fitnesses
        return id_turnover, fit_turnover
    
    def run_instrumented_evolution(self) -> BenchmarkReport:
        """
        Run the full evolution with detailed instrumentation.
        
        Returns a BenchmarkReport with all metrics.
        """
        report = BenchmarkReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config_summary={
                'population_size': self.ga.population_size,
                'generations': self.ga.generations,
                'mutation_rate': self.ga.mutation_rate,
                'crossover_rate': self.ga.crossover_rate,
                'elite_size': self.ga.elite_size,
                'selection_method': self.ga.selection_method,
                'crossover_method': self.ga.crossover_method,
                'parallel': self.ga.parallel_enabled,
                'walk_forward': self.ga.config.get('walk_forward', {}).get('enabled', False),
                'regime_aware': self.ga.regime_aware_enabled,
                'fitness_sharing': self.ga.fitness_sharing,
                'parsimony': self.ga.config.get('parsimony', {}).get('enabled', False),
            }
        )
        
        overall_start = time.time()
        logger = logging.getLogger('benchmark')
        
        try:
            # ── Initialize population ──
            logger.info("=" * 70)
            logger.info("BENCHMARK: Initializing population")
            t0 = time.time()
            population = self.ga.initialize_population()
            init_time = time.time() - t0
            logger.info(f"BENCHMARK: Population initialized in {init_time:.2f}s")
            
            # Validate initial population
            for ind in population.individuals:
                if ind.strategy_gene is None:
                    report.warnings.append("Individual with None strategy_gene found")
            
            # Initialize Pareto archive (mirroring evolve())
            archive_config = self.ga.config.get('pareto_archive', {})
            pareto_archive = None
            if self.ga.mode == 'nsga2' and archive_config.get('enabled', False):
                from genetic_algorithm.core.pareto_archive import ParetoArchive
                pareto_archive = ParetoArchive(
                    max_size=archive_config.get('max_size', 100),
                    decay_rate=archive_config.get('decay_rate', 0.95),
                )
            
            prev_best_fitness = -float('inf')
            
            # ── Evolution loop ──
            for gen in range(self.ga.generations):
                self.ga.current_generation = gen
                gen_start = time.time()
                tm = TimingMetrics(generation=gen)
                
                logger.info(f"\n{'─'*70}")
                logger.info(f"BENCHMARK: Generation {gen + 1}/{self.ga.generations}")
                
                # Phase 1: Evaluate fitness
                t_eval_start = time.time()
                self.ga.evaluate_population(population)
                tm.evaluate_sec = time.time() - t_eval_start
                
                # Phase 1b: Post-hoc walk-forward
                t_wf_start = time.time()
                self.ga._post_hoc_walk_forward_validation(population)
                tm.wf_posthoc_sec = time.time() - t_wf_start
                
                # Phase 2: Stats & ranking (mirrors evolve())
                t_stats_start = time.time()
                from genetic_algorithm.core.population import calculate_pairwise_distances
                from genetic_algorithm.core.population import apply_fitness_sharing
                
                distance_matrix = None
                if self.ga.mode == 'nsga2':
                    from genetic_algorithm.core.nsga2 import fast_non_dominated_sort, crowding_distance_assignment
                    fronts = fast_non_dominated_sort(list(population.individuals))
                    for front in fronts:
                        crowding_distance_assignment(front)
                else:
                    if self.ga.fitness_sharing or len(population.individuals) >= 2:
                        distance_matrix = calculate_pairwise_distances(list(population.individuals))
                    if self.ga.fitness_sharing:
                        apply_fitness_sharing(population, sigma_share=self.ga.sharing_radius,
                                            distance_matrix=distance_matrix)
                
                stats = population.get_stats(distance_matrix=distance_matrix)
                self.ga.generation_stats.append(stats)
                tm.stats_sec = time.time() - t_stats_start
                
                # ── Collect metrics ──
                cm = self._collect_correctness(gen, population)
                qm = self._collect_quality(gen, population, stats)
                id_turnover, fit_turnover = self._compute_elite_turnover(population)
                mem_mb = self._get_memory_mb()
                self.system_metrics.memory_per_generation.append(mem_mb)
                
                # ── Update dynamics ──
                self.dynamics.best_fitness_trajectory.append(qm.best_fitness)
                self.dynamics.best_raw_fitness_trajectory.append(qm.best_raw_fitness)
                self.dynamics.avg_fitness_trajectory.append(qm.avg_fitness)
                self.dynamics.diversity_trajectory.append(qm.genetic_diversity)
                self.dynamics.elite_turnover.append(id_turnover)
                self.dynamics.elite_fitness_turnover.append(fit_turnover)
                self.dynamics.mutation_rate_trajectory.append(self.ga.mutation_rate)
                self.dynamics.selection_pressure_trajectory.append(qm.selection_pressure)
                
                if qm.best_fitness > prev_best_fitness:
                    self.dynamics.best_changed_at.append(gen)
                    prev_best_fitness = qm.best_fitness
                
                # ── Update best individual (mirrors evolve()) ──
                best = population.get_best(1)[0]
                if self.ga._should_update_best_individual(best):
                    self.ga.best_individual = best
                
                # ── Feature tracking ──
                try:
                    self.ga.feature_tracker.update(population)
                    indicator_weights = self.ga.feature_tracker.get_indicator_weights()
                    if indicator_weights:
                        self.ga.config['_indicator_weights'] = indicator_weights
                except Exception:
                    pass
                
                # ── Hall of Fame ──
                try:
                    self.ga.hall_of_fame.update(population, gen)
                except Exception:
                    pass
                
                # ── Convergence check ──
                converged = self.ga.check_convergence(stats)
                
                # ── Create next generation ──
                t_next_start = time.time()
                if gen < self.ga.generations - 1 and not converged:
                    population = self.ga.create_next_generation(population)
                tm.selection_crossover_mutation_sec = time.time() - t_next_start
                
                # ── Finalize timing ──
                tm.total_sec = time.time() - gen_start
                tm.overhead_sec = tm.total_sec - (tm.evaluate_sec + tm.wf_posthoc_sec + 
                                                   tm.stats_sec + tm.selection_crossover_mutation_sec)
                
                # ── Store ──
                self.timing_data.append(tm)
                self.correctness_data.append(cm)
                self.quality_data.append(qm)
                
                # ── Log summary ──
                logger.info(
                    f"BENCHMARK: Gen {gen+1} | "
                    f"Best={qm.best_fitness:.4f} Avg={qm.avg_fitness:.4f} | "
                    f"Div={qm.genetic_diversity or 0:.4f} | "
                    f"Eval={tm.evaluate_sec:.1f}s Total={tm.total_sec:.1f}s | "
                    f"Mem={mem_mb:.0f}MB | "
                    f"NaN={cm.nan_fitness_count} Zero={cm.zero_fitness_count} "
                    f"ZeroTrades={qm.zero_trade_count}"
                )
                
                if converged:
                    logger.info("BENCHMARK: Early convergence detected")
                    break
            
            # ── Gather final results ──
            top = population.get_best(min(5, len(population.individuals)))
            for ind in top:
                report.top_strategies.append({
                    'id': ind.id,
                    'fitness': round(ind.fitness, 6) if ind.fitness else 0,
                    'profit': round(ind.metrics.get('profit', 0), 4) if ind.metrics else 0,
                    'sharpe_ratio': round(ind.metrics.get('sharpe_ratio', 0), 4) if ind.metrics else 0,
                    'max_drawdown': round(ind.metrics.get('max_drawdown', 0), 4) if ind.metrics else 0,
                    'win_rate': round(ind.metrics.get('win_rate', 0), 4) if ind.metrics else 0,
                    'num_trades': ind.metrics.get('num_trades', 0) if ind.metrics else 0,
                    'profit_factor': round(ind.metrics.get('profit_factor', 0), 4) if ind.metrics else 0,
                    'indicators': [i.type for i in ind.strategy_gene.indicators],
                    'entry_conditions': len(ind.strategy_gene.entry_conditions),
                    'exit_conditions': len(ind.strategy_gene.exit_conditions),
                    'timeframe': ind.strategy_gene.timeframe,
                    'stoploss': round(ind.strategy_gene.stoploss, 4),
                })
            
            report.completed = True
            
        except Exception as e:
            report.error = str(e)
            report.error_traceback = traceback.format_exc()
            logger.error(f"BENCHMARK FAILED: {e}")
            logger.error(traceback.format_exc())
        
        report.duration_sec = time.time() - overall_start
        
        # ── Compile report ──
        report.timing = [self._timing_to_dict(t) for t in self.timing_data]
        report.correctness = [self._correctness_to_dict(c) for c in self.correctness_data]
        report.quality = [self._quality_to_dict(q) for q in self.quality_data]
        report.dynamics = self._dynamics_to_dict()
        report.system = self._system_to_dict()
        
        # ── Generate summaries ──
        report.correctness_summary = self._correctness_summary()
        report.performance_summary = self._performance_summary()
        report.warnings.extend(self._generate_warnings())
        
        # Cleanup parallel evaluator
        if self.ga.parallel_evaluator:
            try:
                self.ga.parallel_evaluator.shutdown()
            except Exception:
                pass
        
        return report
    
    # ── Serialization helpers ──
    
    def _timing_to_dict(self, tm: TimingMetrics) -> dict:
        return {
            'generation': tm.generation,
            'total_sec': round(tm.total_sec, 3),
            'evaluate_sec': round(tm.evaluate_sec, 3),
            'wf_posthoc_sec': round(tm.wf_posthoc_sec, 3),
            'selection_crossover_mutation_sec': round(tm.selection_crossover_mutation_sec, 3),
            'stats_sec': round(tm.stats_sec, 3),
            'overhead_sec': round(tm.overhead_sec, 3),
        }
    
    def _correctness_to_dict(self, cm: CorrectnessMetrics) -> dict:
        return {
            'generation': cm.generation,
            'nan_fitness': cm.nan_fitness_count,
            'inf_fitness': cm.inf_fitness_count,
            'negative_fitness': cm.negative_fitness_count,
            'zero_fitness': cm.zero_fitness_count,
            'valid_fitness': cm.valid_fitness_count,
            'failed_evaluations': cm.failed_evaluations,
            'code_syntax_ok': cm.code_syntax_ok,
            'code_syntax_errors': cm.code_syntax_errors,
            'fitness_min': round(min(cm.fitness_values), 6) if cm.fitness_values else None,
            'fitness_max': round(max(cm.fitness_values), 6) if cm.fitness_values else None,
            'fitness_mean': round(sum(cm.fitness_values) / len(cm.fitness_values), 6) if cm.fitness_values else None,
        }
    
    def _quality_to_dict(self, qm: QualityMetrics) -> dict:
        return {
            'generation': qm.generation,
            'best_fitness': round(qm.best_fitness, 6),
            'best_raw_fitness': round(qm.best_raw_fitness, 6),
            'avg_fitness': round(qm.avg_fitness, 6),
            'worst_fitness': round(qm.worst_fitness, 6),
            'median_fitness': round(qm.median_fitness, 6),
            'fitness_std': round(qm.fitness_std, 6),
            'genetic_diversity': round(qm.genetic_diversity, 6) if qm.genetic_diversity else None,
            'best_profit': round(qm.best_profit, 4),
            'avg_profit': round(qm.avg_profit, 4),
            'best_sharpe': round(qm.best_sharpe, 4),
            'best_win_rate': round(qm.best_win_rate, 4),
            'best_drawdown': round(qm.best_drawdown, 4),
            'avg_trades': round(qm.avg_trades, 2),
            'zero_trade_count': qm.zero_trade_count,
            'selection_pressure': round(qm.selection_pressure, 4),
            'exploitation_ratio': round(qm.exploitation_ratio, 4),
            'unique_indicator_types': qm.unique_indicator_types,
            'unique_strategies': qm.unique_strategies,
        }
    
    def _dynamics_to_dict(self) -> dict:
        d = self.dynamics
        return {
            'elite_turnover': [round(x, 4) for x in d.elite_turnover],
            'elite_fitness_turnover': [round(x, 4) for x in d.elite_fitness_turnover],
            'best_fitness_trajectory': [round(x, 6) for x in d.best_fitness_trajectory],
            'best_raw_fitness_trajectory': [round(x, 6) for x in d.best_raw_fitness_trajectory],
            'avg_fitness_trajectory': [round(x, 6) for x in d.avg_fitness_trajectory],
            'diversity_trajectory': [round(x, 6) if x else None for x in d.diversity_trajectory],
            'mutation_rate_trajectory': [round(x, 4) for x in d.mutation_rate_trajectory],
            'selection_pressure_trajectory': [round(x, 4) for x in d.selection_pressure_trajectory],
            'best_changed_at_generations': d.best_changed_at,
            'total_improvements': len(d.best_changed_at),
            'avg_elite_turnover_id': round(sum(d.elite_turnover) / len(d.elite_turnover), 4) if d.elite_turnover else 0,
            'avg_elite_turnover_fitness': round(sum(d.elite_fitness_turnover) / len(d.elite_fitness_turnover), 4) if d.elite_fitness_turnover else 0,
        }
    
    def _system_to_dict(self) -> dict:
        s = self.system_metrics
        mems = s.memory_per_generation
        return {
            'python_version': s.python_version,
            'platform': s.platform,
            'cpu_count': s.cpu_count,
            'parallel_workers': s.parallel_workers,
            'peak_memory_mb': round(max(mems), 1) if mems else 0,
            'avg_memory_mb': round(sum(mems) / len(mems), 1) if mems else 0,
            'memory_per_generation': [round(m, 1) for m in mems],
        }
    
    def _correctness_summary(self) -> dict:
        """Generate overall correctness summary."""
        total_nan = sum(c.nan_fitness_count for c in self.correctness_data)
        total_inf = sum(c.inf_fitness_count for c in self.correctness_data)
        total_neg = sum(c.negative_fitness_count for c in self.correctness_data)
        total_zero = sum(c.zero_fitness_count for c in self.correctness_data)
        total_valid = sum(c.valid_fitness_count for c in self.correctness_data)
        total_failed = sum(c.failed_evaluations for c in self.correctness_data)
        total_syntax_ok = sum(c.code_syntax_ok for c in self.correctness_data)
        total_syntax_err = sum(c.code_syntax_errors for c in self.correctness_data)
        total = total_nan + total_inf + total_neg + total_zero + total_valid
        
        return {
            'total_evaluations': total,
            'valid_fitness_pct': round(100 * total_valid / total, 1) if total else 0,
            'nan_fitness_total': total_nan,
            'inf_fitness_total': total_inf,
            'negative_fitness_total': total_neg,
            'zero_fitness_total': total_zero,
            'failed_evaluations_total': total_failed,
            'code_syntax_ok_total': total_syntax_ok,
            'code_syntax_error_total': total_syntax_err,
            'syntax_check_pass_rate': round(100 * total_syntax_ok / (total_syntax_ok + total_syntax_err), 1) if (total_syntax_ok + total_syntax_err) else 0,
        }
    
    def _performance_summary(self) -> dict:
        """Generate overall performance summary."""
        if not self.timing_data:
            return {}
        
        total_eval = sum(t.evaluate_sec for t in self.timing_data)
        total_wf = sum(t.wf_posthoc_sec for t in self.timing_data)
        total_scm = sum(t.selection_crossover_mutation_sec for t in self.timing_data)
        total_stats = sum(t.stats_sec for t in self.timing_data)
        total_overhead = sum(t.overhead_sec for t in self.timing_data)
        total_all = sum(t.total_sec for t in self.timing_data)
        
        gen_times = [t.total_sec for t in self.timing_data]
        eval_times = [t.evaluate_sec for t in self.timing_data]
        
        return {
            'total_evolution_sec': round(total_all, 2),
            'avg_generation_sec': round(total_all / len(self.timing_data), 2),
            'min_generation_sec': round(min(gen_times), 2),
            'max_generation_sec': round(max(gen_times), 2),
            'time_breakdown': {
                'evaluation_sec': round(total_eval, 2),
                'evaluation_pct': round(100 * total_eval / total_all, 1) if total_all else 0,
                'walk_forward_posthoc_sec': round(total_wf, 2),
                'walk_forward_posthoc_pct': round(100 * total_wf / total_all, 1) if total_all else 0,
                'selection_crossover_mutation_sec': round(total_scm, 2),
                'selection_crossover_mutation_pct': round(100 * total_scm / total_all, 1) if total_all else 0,
                'stats_sec': round(total_stats, 2),
                'stats_pct': round(100 * total_stats / total_all, 1) if total_all else 0,
                'overhead_sec': round(total_overhead, 2),
                'overhead_pct': round(100 * total_overhead / total_all, 1) if total_all else 0,
            },
            'avg_eval_sec': round(total_eval / len(self.timing_data), 2),
            'min_eval_sec': round(min(eval_times), 2),
            'max_eval_sec': round(max(eval_times), 2),
            'throughput_strategies_per_sec': round(
                sum(len(self.quality_data) and self.ga.population_size or 0 for _ in self.timing_data) / total_all, 2
            ) if total_all else 0,
        }
    
    def _generate_warnings(self) -> List[str]:
        """Generate diagnostic warnings based on collected metrics."""
        warnings = []
        
        # Correctness warnings
        cs = self._correctness_summary()
        if cs.get('nan_fitness_total', 0) > 0:
            warnings.append(f"WARNING: {cs['nan_fitness_total']} NaN fitness values detected across all generations")
        if cs.get('inf_fitness_total', 0) > 0:
            warnings.append(f"WARNING: {cs['inf_fitness_total']} Inf fitness values detected")
        if cs.get('code_syntax_error_total', 0) > 0:
            warnings.append(f"WARNING: {cs['code_syntax_error_total']} code syntax errors found in sampled strategies")
        if cs.get('zero_fitness_total', 0) > cs.get('total_evaluations', 1) * 0.5:
            warnings.append("WARNING: >50% of strategies have zero fitness — evaluation may be broken")
        
        # Quality warnings
        if self.quality_data:
            last = self.quality_data[-1]
            if last.zero_trade_count > self.ga.population_size * 0.5:
                warnings.append(f"WARNING: {last.zero_trade_count}/{self.ga.population_size} strategies produced 0 trades in final gen")
            if last.unique_strategies < 3:
                warnings.append("WARNING: Very low strategy diversity — population may have converged prematurely")
            if last.genetic_diversity is not None and last.genetic_diversity < 0.05:
                warnings.append(f"WARNING: Genetic diversity extremely low ({last.genetic_diversity:.4f})")
        
        # Dynamics warnings — use raw fitness (not shared) to avoid false
        # positives when fitness sharing pushes shared values down.
        if self.dynamics.best_raw_fitness_trajectory:
            first_best = self.dynamics.best_raw_fitness_trajectory[0]
            last_best = self.dynamics.best_raw_fitness_trajectory[-1]
            if last_best <= first_best and len(self.dynamics.best_raw_fitness_trajectory) > 2:
                warnings.append("WARNING: Best raw fitness did not improve from gen 0 — evolution may be ineffective")
        
        # Performance warnings
        if self.timing_data:
            ps = self._performance_summary()
            eval_pct = ps.get('time_breakdown', {}).get('evaluation_pct', 0)
            if eval_pct < 50:
                warnings.append(f"WARNING: Evaluation is only {eval_pct:.0f}% of total time — high overhead")
        
        return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Report printing
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: BenchmarkReport):
    """Print a human-readable benchmark report."""
    print("\n" + "=" * 80)
    print("  GA BENCHMARK REPORT")
    print("=" * 80)
    print(f"  Timestamp:  {report.timestamp}")
    print(f"  Duration:   {report.duration_sec:.1f}s")
    print(f"  Status:     {'COMPLETED' if report.completed else 'FAILED'}")
    if report.error:
        print(f"  Error:      {report.error}")
    
    # Config summary
    cs = report.config_summary
    print(f"\n{'─'*80}")
    print("  CONFIGURATION")
    print(f"{'─'*80}")
    print(f"  Population: {cs.get('population_size')}  |  Generations: {cs.get('generations')}")
    print(f"  Mutation: {cs.get('mutation_rate'):.0%}  |  Crossover: {cs.get('crossover_rate'):.0%}")
    print(f"  Parallel: {cs.get('parallel')}  |  Walk-Forward: {cs.get('walk_forward')}")
    print(f"  Fitness Sharing: {cs.get('fitness_sharing')}  |  Parsimony: {cs.get('parsimony')}")
    
    # Correctness
    cc = report.correctness_summary
    print(f"\n{'─'*80}")
    print("  CORRECTNESS")
    print(f"{'─'*80}")
    print(f"  Total evaluations:     {cc.get('total_evaluations', 0)}")
    print(f"  Valid fitness:         {cc.get('valid_fitness_pct', 0):.1f}%")
    print(f"  NaN fitness:           {cc.get('nan_fitness_total', 0)}")
    print(f"  Inf fitness:           {cc.get('inf_fitness_total', 0)}")
    print(f"  Negative fitness:      {cc.get('negative_fitness_total', 0)}")
    print(f"  Zero fitness:          {cc.get('zero_fitness_total', 0)}")
    print(f"  Failed evaluations:    {cc.get('failed_evaluations_total', 0)}")
    print(f"  Code syntax pass rate: {cc.get('syntax_check_pass_rate', 0):.1f}%")
    
    # Performance
    ps = report.performance_summary
    print(f"\n{'─'*80}")
    print("  PERFORMANCE")
    print(f"{'─'*80}")
    if ps:
        print(f"  Total evolution time:  {ps.get('total_evolution_sec', 0):.1f}s")
        print(f"  Avg generation time:   {ps.get('avg_generation_sec', 0):.1f}s")
        print(f"  Min/Max gen time:      {ps.get('min_generation_sec', 0):.1f}s / {ps.get('max_generation_sec', 0):.1f}s")
        tb = ps.get('time_breakdown', {})
        print(f"\n  Time breakdown:")
        print(f"    Evaluation:          {tb.get('evaluation_sec', 0):>7.1f}s  ({tb.get('evaluation_pct', 0):>5.1f}%)")
        print(f"    WF Post-hoc:         {tb.get('walk_forward_posthoc_sec', 0):>7.1f}s  ({tb.get('walk_forward_posthoc_pct', 0):>5.1f}%)")
        print(f"    Selection/XO/Mut:    {tb.get('selection_crossover_mutation_sec', 0):>7.1f}s  ({tb.get('selection_crossover_mutation_pct', 0):>5.1f}%)")
        print(f"    Stats:               {tb.get('stats_sec', 0):>7.1f}s  ({tb.get('stats_pct', 0):>5.1f}%)")
        print(f"    Overhead:            {tb.get('overhead_sec', 0):>7.1f}s  ({tb.get('overhead_pct', 0):>5.1f}%)")
    
    # System
    sy = report.system
    print(f"\n  System:")
    print(f"    Python:    {sy.get('python_version')}")
    print(f"    CPUs:      {sy.get('cpu_count')}")
    print(f"    Workers:   {sy.get('parallel_workers')}")
    print(f"    Peak mem:  {sy.get('peak_memory_mb', 0):.0f} MB")
    print(f"    Avg mem:   {sy.get('avg_memory_mb', 0):.0f} MB")
    
    # Quality / Evolutionary Dynamics
    print(f"\n{'─'*80}")
    print("  EVOLUTIONARY DYNAMICS")
    print(f"{'─'*80}")
    if report.quality:
        first_q = report.quality[0]
        last_q = report.quality[-1]
        print(f"  Best fitness:    {first_q['best_fitness']:.4f} -> {last_q['best_fitness']:.4f}  (delta: {last_q['best_fitness'] - first_q['best_fitness']:+.4f})")
        print(f"  Avg fitness:     {first_q['avg_fitness']:.4f} -> {last_q['avg_fitness']:.4f}  (delta: {last_q['avg_fitness'] - first_q['avg_fitness']:+.4f})")
        div_first = first_q.get('genetic_diversity')
        div_last = last_q.get('genetic_diversity')
        if div_first is not None and div_last is not None:
            print(f"  Diversity:       {div_first:.4f} -> {div_last:.4f}  (delta: {div_last - div_first:+.4f})")
        print(f"  Strategy uniqueness: {first_q.get('unique_strategies', 0)} -> {last_q.get('unique_strategies', 0)}")
        print(f"  Zero-trade strats:   {first_q.get('zero_trade_count', 0)} -> {last_q.get('zero_trade_count', 0)}")
    
    dyn = report.dynamics
    if dyn:
        print(f"\n  Improvements at gens: {dyn.get('best_changed_at_generations', [])}")
        print(f"  Total improvements:   {dyn.get('total_improvements', 0)}")
        avg_id = dyn.get('avg_elite_turnover_id', dyn.get('avg_elite_turnover', 0))
        avg_fit = dyn.get('avg_elite_turnover_fitness', 0)
        print(f"  Avg elite turnover (by ID):      {avg_id:.1%}")
        print(f"  Avg elite turnover (by fitness):  {avg_fit:.1%}")
        
        # Fitness trajectory (compact)
        bf = dyn.get('best_fitness_trajectory', [])
        brf = dyn.get('best_raw_fitness_trajectory', [])
        if brf:
            traj = " -> ".join(f"{f:.4f}" for f in brf)
            print(f"\n  Best RAW fitness trajectory (true strategy quality):")
            print(f"    {traj}")
        if bf:
            traj = " -> ".join(f"{f:.4f}" for f in bf)
            print(f"  Best SHARED fitness trajectory (after fitness sharing):")
            print(f"    {traj}")
        
        mr = dyn.get('mutation_rate_trajectory', [])
        if mr:
            traj = " -> ".join(f"{r:.3f}" for r in mr)
            print(f"  Mutation rate trajectory:")
            print(f"    {traj}")
    
    # Quality per generation table
    if report.quality:
        print(f"\n{'─'*80}")
        print("  PER-GENERATION QUALITY TABLE")
        print(f"{'─'*80}")
        header = f"  {'Gen':>3} | {'Best':>8} | {'Avg':>8} | {'Std':>6} | {'Div':>6} | {'Profit%':>8} | {'Sharpe':>6} | {'WR%':>5} | {'Trades':>6} | {'0Trade':>6} | {'Unique':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for q in report.quality:
            div = f"{q['genetic_diversity']:.4f}" if q.get('genetic_diversity') else "  N/A "
            print(f"  {q['generation']+1:>3} | {q['best_fitness']:>8.4f} | {q['avg_fitness']:>8.4f} | "
                  f"{q['fitness_std']:>6.4f} | {div} | {q['best_profit']:>8.2f} | "
                  f"{q['best_sharpe']:>6.2f} | {q['best_win_rate']*100:>5.1f} | "
                  f"{q['avg_trades']:>6.1f} | {q['zero_trade_count']:>6} | {q['unique_strategies']:>6}")
    
    # Performance per generation table
    if report.timing:
        print(f"\n{'─'*80}")
        print("  PER-GENERATION TIMING TABLE")
        print(f"{'─'*80}")
        header = f"  {'Gen':>3} | {'Total':>7} | {'Eval':>7} | {'WF':>7} | {'S/X/M':>7} | {'Stats':>7} | {'Over':>7} | {'Mem MB':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i, t in enumerate(report.timing):
            mem = report.system.get('memory_per_generation', [0] * len(report.timing))
            m = mem[i] if i < len(mem) else 0
            print(f"  {t['generation']+1:>3} | {t['total_sec']:>7.1f} | {t['evaluate_sec']:>7.1f} | "
                  f"{t['wf_posthoc_sec']:>7.1f} | {t['selection_crossover_mutation_sec']:>7.1f} | "
                  f"{t['stats_sec']:>7.1f} | {t['overhead_sec']:>7.1f} | {m:>7.0f}")
    
    # Top strategies
    if report.top_strategies:
        print(f"\n{'─'*80}")
        print("  TOP STRATEGIES (Final Generation)")
        print(f"{'─'*80}")
        for i, s in enumerate(report.top_strategies, 1):
            print(f"\n  #{i}: {s['id']}")
            print(f"    Fitness: {s['fitness']:.4f}  |  Profit: {s['profit']:.2f}%  |  Sharpe: {s['sharpe_ratio']:.2f}")
            print(f"    Win Rate: {s['win_rate']:.1%}  |  Max DD: {s['max_drawdown']:.1%}  |  Trades: {s['num_trades']}")
            print(f"    PF: {s['profit_factor']:.2f}  |  TF: {s['timeframe']}  |  SL: {s['stoploss']:.2%}")
            print(f"    Indicators: {', '.join(s['indicators'])}")
            print(f"    Conditions: {s['entry_conditions']} entry, {s['exit_conditions']} exit")
    
    # Warnings
    if report.warnings:
        print(f"\n{'─'*80}")
        print("  WARNINGS & DIAGNOSTICS")
        print(f"{'─'*80}")
        for w in report.warnings:
            print(f"  {w}")
    
    print(f"\n{'='*80}")
    
    # ── VERDICT ──
    issues = 0
    cc = report.correctness_summary
    if cc.get('nan_fitness_total', 0) > 0: issues += 1
    if cc.get('inf_fitness_total', 0) > 0: issues += 1
    if cc.get('code_syntax_error_total', 0) > 0: issues += 1
    if not report.completed: issues += 1
    
    if issues == 0:
        print("  VERDICT: ALL CHECKS PASSED")
    else:
        print(f"  VERDICT: {issues} ISSUE(S) DETECTED — see warnings above")
    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run the benchmark."""
    config_path = str(PROJECT_ROOT / "genetic_algorithm" / "config" / "ga_config_benchmark.yaml")
    output_dir = PROJECT_ROOT / "genetic_algorithm" / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "benchmark.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w'),
        ]
    )
    logger = logging.getLogger('benchmark')
    
    logger.info("=" * 70)
    logger.info("GA BENCHMARK STARTING")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)
    
    # Run instrumented evolution
    harness = InstrumentedGA(config_path)
    report = harness.run_instrumented_evolution()
    report.config_file = config_path
    
    # Save JSON report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': report.timestamp,
            'config_file': report.config_file,
            'config_summary': report.config_summary,
            'duration_sec': round(report.duration_sec, 2),
            'completed': report.completed,
            'error': report.error,
            'error_traceback': report.error_traceback,
            'timing': report.timing,
            'correctness': report.correctness,
            'quality': report.quality,
            'dynamics': report.dynamics,
            'system': report.system,
            'top_strategies': report.top_strategies,
            'correctness_summary': report.correctness_summary,
            'performance_summary': report.performance_summary,
            'warnings': report.warnings,
        }, f, indent=2)
    
    logger.info(f"JSON report saved to: {json_path}")
    
    # Print human-readable report
    print_report(report)
    
    # Also save the text report
    import io
    text_path = output_dir / f"benchmark_{timestamp}.txt"
    # Re-run print_report to capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    print_report(report)
    sys.stdout = old_stdout
    with open(text_path, 'w') as f:
        f.write(buffer.getvalue())
    
    logger.info(f"Text report saved to: {text_path}")
    
    return 0 if report.completed else 1


if __name__ == '__main__':
    sys.exit(main())
