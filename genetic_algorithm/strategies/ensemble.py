"""
Ensemble Strategy Generator

Creates an ensemble strategy that combines multiple top-performing strategies
from the Hall of Fame using weighted voting for entry/exit signals.

The ensemble strategy:
1. Runs all component strategies' indicators
2. Evaluates each strategy's entry/exit conditions
3. Uses weighted majority voting to decide final signals
4. Weights are based on each strategy's fitness score
"""

import logging
from typing import Dict, Any, List, Optional

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.strategies.generator import StrategyGenerator

logger = logging.getLogger(__name__)


class EnsembleGenerator:
    """
    Generate an ensemble strategy from multiple individual strategies.
    
    The ensemble strategy embeds all component strategies' logic and uses
    weighted voting for signal generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generator = StrategyGenerator(config)
    
    def generate_ensemble_code(
        self,
        strategies: List[Dict[str, Any]],
        vote_threshold: float = 0.5,
        strategy_name: str = "GAEnsembleStrategy"
    ) -> str:
        """
        Generate an ensemble strategy from multiple strategy entries.
        
        Args:
            strategies: List of dicts with 'strategy_gene' (dict) and 'fitness' (float)
            vote_threshold: Fraction of weighted votes needed to trigger signal (0.0-1.0)
            strategy_name: Name for the ensemble strategy class
            
        Returns:
            Python code string for the ensemble strategy
        """
        if not strategies:
            raise ValueError("Need at least one strategy for ensemble")
        
        # Parse strategy genes
        genes_with_weights = []
        for entry in strategies:
            gene_dict = entry.get('strategy_gene', entry.get('strategy_gene_dict', {}))
            fitness = entry.get('fitness', 0.0)
            gene = StrategyGene.from_dict(gene_dict)
            genes_with_weights.append((gene, max(fitness, 0.01)))  # Ensure positive weight
        
        # Use the best strategy's parameters as the base
        best_gene = genes_with_weights[0][0]
        
        # Collect all unique indicators across strategies
        all_indicators = {}
        for gene, _ in genes_with_weights:
            for ind in gene.get_base_indicators():
                key = f"{ind.type}_{ind.parameters.get('period', '')}"
                if key not in all_indicators:
                    all_indicators[key] = ind
        
        # Generate indicator code for all indicators
        all_indicator_list = list(all_indicators.values())
        indicator_code = self.generator._generate_indicator_code(all_indicator_list)
        
        # Generate individual strategy condition blocks
        strategy_blocks = []
        total_weight = sum(w for _, w in genes_with_weights)
        
        for i, (gene, weight) in enumerate(genes_with_weights):
            norm_weight = weight / total_weight
            
            # Generate entry conditions for this strategy 
            entry_code = self.generator._generate_condition_code(
                gene.entry_conditions, gene.indicators, is_entry=True,
                signal_col_override=f'_vote_entry_{i}'
            )
            exit_code = self.generator._generate_condition_code(
                gene.exit_conditions, gene.indicators, is_entry=False,
                signal_col_override=f'_vote_exit_{i}'
            )
            
            strategy_blocks.append({
                'index': i,
                'weight': norm_weight,
                'entry_code': entry_code,
                'exit_code': exit_code,
                'fitness': weight,
            })
        
        # Build voting logic
        n = len(strategy_blocks)
        
        # Entry voting code
        entry_vote_lines = []
        for sb in strategy_blocks:
            entry_vote_lines.append(sb['entry_code'])
        entry_vote_lines.append(f"        # Weighted majority vote for entry")
        entry_vote_lines.append(f"        _entry_votes = pd.Series(0.0, index=dataframe.index)")
        for sb in strategy_blocks:
            i = sb['index']
            w = sb['weight']
            entry_vote_lines.append(
                f"        _entry_votes += dataframe['_vote_entry_{i}'].fillna(0) * {w:.4f}"
            )
        entry_vote_lines.append(
            f"        dataframe.loc[_entry_votes >= {vote_threshold:.4f}, 'enter_long'] = 1"
        )
        
        # Exit voting code
        exit_vote_lines = []
        for sb in strategy_blocks:
            exit_vote_lines.append(sb['exit_code'])
        exit_vote_lines.append(f"        # Weighted majority vote for exit")
        exit_vote_lines.append(f"        _exit_votes = pd.Series(0.0, index=dataframe.index)")
        for sb in strategy_blocks:
            i = sb['index']
            w = sb['weight']
            exit_vote_lines.append(
                f"        _exit_votes += dataframe['_vote_exit_{i}'].fillna(0) * {w:.4f}"
            )
        exit_vote_lines.append(
            f"        dataframe.loc[_exit_votes >= {vote_threshold:.4f}, 'exit_long'] = 1"
        )
        
        entry_code_combined = '\n'.join(entry_vote_lines)
        exit_code_combined = '\n'.join(exit_vote_lines)
        
        # Generate trailing stop params
        trailing_stop_params = ""
        if best_gene.trailing_stop and best_gene.trailing_stop_positive is not None:
            trailing_stop_params = f"""
    trailing_stop_positive = {best_gene.trailing_stop_positive}
    trailing_stop_positive_offset = {best_gene.trailing_stop_positive_offset}"""
        
        # Strategy weights comment
        weights_comment = "\n".join(
            f"#   Strategy {i}: fitness={sb['fitness']:.4f}, weight={sb['weight']:.4f}"
            for i, sb in enumerate(strategy_blocks)
        )
        
        code = f'''"""
Ensemble Strategy - Generated by Genetic Algorithm
Combines {n} strategies using weighted majority voting.

{weights_comment}
Vote threshold: {vote_threshold}
"""

from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import pandas as pd
import talib.abstract as ta
import numpy as np

class {strategy_name}(IStrategy):
    """GA Ensemble Strategy - {n} component strategies with weighted voting"""
    
    INTERFACE_VERSION = 3
    
    timeframe = '{best_gene.timeframe}'
    stoploss = {best_gene.stoploss}
    minimal_roi = {best_gene.minimal_roi}
    trailing_stop = {best_gene.trailing_stop}{trailing_stop_params}
    max_open_trades = {best_gene.max_open_trades}
    
    def informative_pairs(self):
        return []
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add all indicators needed by component strategies"""
{indicator_code}
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signals via weighted majority voting"""
        dataframe['enter_long'] = 0
{entry_code_combined}
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit signals via weighted majority voting"""
        dataframe['exit_long'] = 0
{exit_code_combined}
        return dataframe
'''
        
        logger.info(f"Generated ensemble strategy '{strategy_name}' with {n} components, "
                    f"vote_threshold={vote_threshold}")
        
        return code
    
    def generate_from_hall_of_fame(
        self,
        hof_path: str,
        top_n: int = 5,
        vote_threshold: float = 0.5,
        strategy_name: str = "GAEnsembleStrategy"
    ) -> str:
        """
        Generate ensemble from a Hall of Fame JSON file.
        
        Args:
            hof_path: Path to hall_of_fame.json
            top_n: Number of top strategies to include
            vote_threshold: Voting threshold
            strategy_name: Output strategy name
            
        Returns:
            Python code string
        """
        import json
        from pathlib import Path
        
        hof_file = Path(hof_path)
        if not hof_file.exists():
            raise FileNotFoundError(f"Hall of Fame file not found: {hof_path}")
        
        with open(hof_file) as f:
            hof_data = json.load(f)
        
        entries = hof_data.get('entries', [])
        if not entries:
            raise ValueError("Hall of Fame is empty")
        
        # Sort by fitness descending and take top_n
        entries.sort(key=lambda e: e.get('fitness', 0), reverse=True)
        top_entries = entries[:top_n]
        
        logger.info(f"Creating ensemble from top {len(top_entries)} strategies "
                    f"(fitness range: {top_entries[-1].get('fitness', 0):.4f} - "
                    f"{top_entries[0].get('fitness', 0):.4f})")
        
        return self.generate_ensemble_code(
            strategies=top_entries,
            vote_threshold=vote_threshold,
            strategy_name=strategy_name
        )
