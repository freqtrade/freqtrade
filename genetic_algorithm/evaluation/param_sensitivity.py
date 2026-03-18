"""
Parameter Sensitivity Analysis

Post-hoc analysis that tests how robust a strategy's fitness is to small
changes in its numeric parameters.  A strategy is "robust" if small
perturbations (±5%, ±10%, ±20%) do not cause large fitness drops.

Usage:
    from genetic_algorithm.evaluation.param_sensitivity import (
        ParameterSensitivityAnalyzer,
    )
    analyzer = ParameterSensitivityAnalyzer(config)
    report = analyzer.analyze(strategy_gene, base_fitness)
    print(report.overall_robustness)
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.fitness import FitnessEvaluator

logger = logging.getLogger(__name__)

# Default perturbation levels (fraction of original value)
DEFAULT_PERTURBATION_PCTS = [0.05, 0.10, 0.20]

# Thresholds for classifying parameters
FRAGILE_DROP_THRESHOLD = 0.20  # >20% fitness drop at ±10% = fragile
STABLE_DROP_THRESHOLD = 0.05   # <5% fitness drop at ±20% = stable


@dataclass
class ParameterResult:
    """Sensitivity result for a single parameter."""
    name: str  # e.g. "indicator[0].parameters.period" or "stoploss"
    original_value: float
    perturbations: List[Dict[str, Any]]  # [{pct, value, fitness, delta_pct}]
    sensitivity: float  # avg |delta_pct|
    max_drop: float  # worst-case fitness drop (fraction)
    is_fragile: bool
    is_stable: bool


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis report for a strategy."""
    strategy_name: str
    base_fitness: float
    parameters_tested: int
    perturbations_per_param: int
    total_backtests: int
    param_results: List[ParameterResult]
    fragile_params: List[str]
    stable_params: List[str]
    overall_sensitivity: float  # avg sensitivity across all params
    overall_robustness: float   # 1.0 - normalized_sensitivity [0=fragile, 1=solid]
    robustness_band: float      # fitness range across ALL perturbations


class ParameterSensitivityAnalyzer:
    """Analyzes parameter sensitivity of evolved trading strategies."""

    def __init__(
        self,
        config: Dict[str, Any],
        perturbation_pcts: Optional[List[float]] = None,
        evaluator: Optional[FitnessEvaluator] = None,
    ):
        self.config = config
        self.perturbation_pcts = perturbation_pcts or DEFAULT_PERTURBATION_PCTS
        self.evaluator = evaluator or FitnessEvaluator(config)
        self.logger = logging.getLogger(f"{__name__}.Analyzer")

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        gene: StrategyGene,
        base_fitness: float,
        strategy_name: Optional[str] = None,
    ) -> SensitivityReport:
        """
        Run sensitivity analysis on a strategy gene.

        For each extractable numeric parameter, creates perturbed variants
        (original ± pct for each pct in perturbation_pcts), backtests each,
        and computes stability metrics.

        Args:
            gene: The strategy gene to analyze.
            base_fitness: The known fitness of the unperturbed strategy.
            strategy_name: Optional display name.

        Returns:
            SensitivityReport with per-parameter and overall metrics.
        """
        if strategy_name is None:
            strategy_name = f"Gen{gene.generation}_Ind{gene.individual_id}"

        self.logger.info(
            "Starting sensitivity analysis for %s (base_fitness=%.4f)",
            strategy_name, base_fitness,
        )

        # Extract testable parameters
        params = self.extract_parameters(gene)
        self.logger.info("  Extracted %d numeric parameters", len(params))

        if not params:
            return SensitivityReport(
                strategy_name=strategy_name,
                base_fitness=base_fitness,
                parameters_tested=0,
                perturbations_per_param=0,
                total_backtests=0,
                param_results=[],
                fragile_params=[],
                stable_params=[],
                overall_sensitivity=0.0,
                overall_robustness=1.0,
                robustness_band=0.0,
            )

        # Generate and evaluate perturbations
        param_results: List[ParameterResult] = []
        all_fitnesses = [base_fitness]
        total_backtests = 0

        for param_name, original_value in params:
            if original_value == 0:
                self.logger.debug("  Skipping %s (value=0)", param_name)
                continue

            perturbations = []
            for pct in self.perturbation_pcts:
                for direction in [-1, +1]:
                    delta = original_value * pct * direction
                    new_value = original_value + delta

                    # Apply perturbation to a copy of the gene
                    perturbed_gene = gene.copy()
                    self._set_parameter(perturbed_gene, param_name, new_value)

                    # Evaluate
                    try:
                        fitness, _ = self.evaluator.evaluate(
                            perturbed_gene,
                            strategy_name=f"{strategy_name}_sens_{param_name}_{pct}_{direction}",
                        )
                    except Exception as e:
                        self.logger.warning(
                            "  Backtest failed for %s %.0f%%: %s",
                            param_name, pct * 100 * direction, e,
                        )
                        fitness = 0.0

                    total_backtests += 1
                    all_fitnesses.append(fitness)

                    delta_pct = (
                        (fitness - base_fitness) / base_fitness
                        if base_fitness > 0 else 0.0
                    )
                    perturbations.append({
                        'pct': pct * direction,
                        'value': new_value,
                        'fitness': fitness,
                        'delta_pct': delta_pct,
                    })

            if not perturbations:
                continue

            # Compute per-parameter metrics
            abs_deltas = [abs(p['delta_pct']) for p in perturbations]
            sensitivity = sum(abs_deltas) / len(abs_deltas)
            max_drop = max(
                (-p['delta_pct'] for p in perturbations),
                default=0.0,
            )

            # Fragile: >20% drop at ±10%
            is_fragile = any(
                -p['delta_pct'] > FRAGILE_DROP_THRESHOLD
                for p in perturbations
                if abs(p['pct']) <= 0.10
            )

            # Stable: <5% drop even at ±20%
            is_stable = all(
                abs(p['delta_pct']) < STABLE_DROP_THRESHOLD
                for p in perturbations
            )

            result = ParameterResult(
                name=param_name,
                original_value=original_value,
                perturbations=perturbations,
                sensitivity=sensitivity,
                max_drop=max_drop,
                is_fragile=is_fragile,
                is_stable=is_stable,
            )
            param_results.append(result)

            self.logger.info(
                "  %s: sensitivity=%.3f max_drop=%.1f%% %s",
                param_name, sensitivity, max_drop * 100,
                "FRAGILE" if is_fragile else ("STABLE" if is_stable else ""),
            )

        # Overall metrics
        fragile_params = [r.name for r in param_results if r.is_fragile]
        stable_params = [r.name for r in param_results if r.is_stable]

        overall_sensitivity = (
            sum(r.sensitivity for r in param_results) / len(param_results)
            if param_results else 0.0
        )
        overall_robustness = max(0.0, min(1.0, 1.0 - overall_sensitivity))
        robustness_band = max(all_fitnesses) - min(all_fitnesses) if all_fitnesses else 0.0

        report = SensitivityReport(
            strategy_name=strategy_name,
            base_fitness=base_fitness,
            parameters_tested=len(param_results),
            perturbations_per_param=len(self.perturbation_pcts) * 2,
            total_backtests=total_backtests,
            param_results=param_results,
            fragile_params=fragile_params,
            stable_params=stable_params,
            overall_sensitivity=overall_sensitivity,
            overall_robustness=overall_robustness,
            robustness_band=robustness_band,
        )

        self.logger.info(
            "Sensitivity analysis complete: %d params, %d backtests, "
            "robustness=%.3f, fragile=%d, stable=%d",
            report.parameters_tested, report.total_backtests,
            report.overall_robustness, len(fragile_params), len(stable_params),
        )

        return report

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_parameters(gene: StrategyGene) -> List[Tuple[str, float]]:
        """
        Extract all testable numeric parameters from a strategy gene.

        Returns a list of (path_name, current_value) tuples.
        Path names use dotted notation: e.g. "indicators[0].parameters.period".
        """
        params: List[Tuple[str, float]] = []

        # 1. Indicator parameters
        for i, ind in enumerate(gene.indicators):
            for pname, pval in ind.parameters.items():
                if isinstance(pval, (int, float)):
                    params.append((f"indicators[{i}].parameters.{pname}", float(pval)))

        # 2. Entry condition thresholds
        for i, cond in enumerate(gene.entry_conditions):
            params.append((f"entry_conditions[{i}].threshold", float(cond.threshold)))
            if cond.threshold_upper != 0:
                params.append((
                    f"entry_conditions[{i}].threshold_upper",
                    float(cond.threshold_upper),
                ))

        # 3. Exit condition thresholds
        for i, cond in enumerate(gene.exit_conditions):
            params.append((f"exit_conditions[{i}].threshold", float(cond.threshold)))
            if cond.threshold_upper != 0:
                params.append((
                    f"exit_conditions[{i}].threshold_upper",
                    float(cond.threshold_upper),
                ))

        # 4. Short entry/exit conditions
        for i, cond in enumerate(gene.short_entry_conditions):
            params.append((f"short_entry_conditions[{i}].threshold", float(cond.threshold)))
        for i, cond in enumerate(gene.short_exit_conditions):
            params.append((f"short_exit_conditions[{i}].threshold", float(cond.threshold)))

        # 5. Risk parameters
        params.append(("stoploss", float(gene.stoploss)))

        # 6. ROI values
        for key, val in gene.minimal_roi.items():
            params.append((f"minimal_roi.{key}", float(val)))

        # 7. Trailing stop parameters
        if gene.trailing_stop_positive is not None:
            params.append((
                "trailing_stop_positive",
                float(gene.trailing_stop_positive),
            ))
        if gene.trailing_stop_positive_offset is not None:
            params.append((
                "trailing_stop_positive_offset",
                float(gene.trailing_stop_positive_offset),
            ))

        return params

    # ------------------------------------------------------------------
    # Parameter setting
    # ------------------------------------------------------------------

    @staticmethod
    def _set_parameter(gene: StrategyGene, path: str, value: float):
        """
        Set a parameter value in a strategy gene by dotted path name.
        """
        # Parse path: "indicators[0].parameters.period"
        if path.startswith("indicators["):
            idx = int(path.split("[")[1].split("]")[0])
            rest = path.split(".", 1)[1]  # "parameters.period"
            parts = rest.split(".")
            if parts[0] == "parameters" and len(parts) == 2:
                pname = parts[1]
                # Preserve int type if original was int
                original = gene.indicators[idx].parameters.get(pname)
                if isinstance(original, int):
                    gene.indicators[idx].parameters[pname] = round(value)
                else:
                    gene.indicators[idx].parameters[pname] = value
            return

        if path.startswith("entry_conditions["):
            idx = int(path.split("[")[1].split("]")[0])
            field_name = path.split(".")[-1]
            setattr(gene.entry_conditions[idx], field_name, value)
            return

        if path.startswith("exit_conditions["):
            idx = int(path.split("[")[1].split("]")[0])
            field_name = path.split(".")[-1]
            setattr(gene.exit_conditions[idx], field_name, value)
            return

        if path.startswith("short_entry_conditions["):
            idx = int(path.split("[")[1].split("]")[0])
            field_name = path.split(".")[-1]
            setattr(gene.short_entry_conditions[idx], field_name, value)
            return

        if path.startswith("short_exit_conditions["):
            idx = int(path.split("[")[1].split("]")[0])
            field_name = path.split(".")[-1]
            setattr(gene.short_exit_conditions[idx], field_name, value)
            return

        if path.startswith("minimal_roi."):
            roi_key = path.split(".", 1)[1]
            gene.minimal_roi[roi_key] = value
            return

        # Direct attributes: stoploss, trailing_stop_positive, etc.
        if hasattr(gene, path):
            setattr(gene, path, value)
