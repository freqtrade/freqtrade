"""
Fitness Evaluator

Evaluates the fitness of trading strategies through backtesting
and calculating performance metrics.
"""

import logging
from typing import Tuple, Dict, Any

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
from genetic_algorithm.evaluation.backtester import BacktestResult
from genetic_algorithm.strategies.generator import StrategyGenerator

logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """
    Evaluates strategy fitness through backtesting.
    
    Responsible for:
    - Running FreqTrade backtests
    - Parsing backtest results
    - Calculating fitness score
    - Computing performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fitness evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fitness_weights = config.get('fitness_weights', {})
        self.fitness_penalties = config.get('fitness_penalties', {})
        self.backtest_config = config.get('backtesting', {})
        
        # Initialize direct backtester and strategy generator
        self.backtester = DirectBacktester(config)
        self.strategy_generator = StrategyGenerator(config)
    
    def evaluate(self, strategy_gene: StrategyGene, strategy_name: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a strategy's fitness through backtesting.
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name for the strategy (auto-generated if not provided)
            
        Returns:
            Tuple of (fitness_score, metrics_dict)
        """
        # Generate strategy name if not provided
        if strategy_name is None:
            import uuid
            strategy_name = f"GA_Strategy_{uuid.uuid4().hex[:8]}"
        
        try:
            # Generate strategy code (strategy name is auto-generated from gene info)
            strategy_code = self.strategy_generator.generate_strategy_code(strategy_gene)
            
            # Extract the generated strategy name from the gene
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            if strategy_name is None:
                strategy_name = generated_name
            
            # Run backtest
            backtest_result = self.backtester.backtest_strategy(strategy_code, generated_name)
            
            # Check if backtest was successful
            if not backtest_result.success:
                logger.warning(f"Backtest failed for {strategy_name}: {backtest_result.error_message}")
                # Return very low fitness for failed strategies
                return 0.0, {
                    'profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'error': backtest_result.error_message
                }
            
            # Convert backtest result to metrics dictionary
            metrics = self._backtest_result_to_metrics(backtest_result)
            
            # Calculate fitness
            fitness = self.calculate_fitness(metrics)
            
            logger.info(f"Strategy {strategy_name}: fitness={fitness:.4f}, "
                       f"profit={metrics['profit']:.2f}%, trades={metrics['num_trades']}")
            
            return fitness, metrics
            
        except Exception as e:
            logger.error(f"Error evaluating strategy {strategy_name}: {e}", exc_info=True)
            # Return zero fitness on error
            return 0.0, {
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'error': str(e)
            }
    
    def _backtest_result_to_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """
        Convert BacktestResult to metrics dictionary for fitness calculation.
        
        Args:
            result: BacktestResult object
            
        Returns:
            Dictionary of metrics
        """
        return {
            'profit': result.profit_percent,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'num_trades': result.total_trades,
            'profit_factor': result.profit_factor,
            'sortino_ratio': result.sortino_ratio,
        }
    
    def calculate_fitness(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall fitness score from metrics.
        
        Uses weighted combination of metrics with penalties.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Fitness score (higher is better)
        """
        # Extract metrics
        profit = metrics.get('profit', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        num_trades = metrics.get('num_trades', 0)
        
        # Normalize metrics to 0-1 range
        norm_profit = self._normalize_profit(profit)
        norm_sharpe = self._normalize_sharpe(sharpe_ratio)
        norm_drawdown = 1 - min(max_drawdown, 1.0)  # Lower drawdown is better
        norm_win_rate = win_rate
        norm_trade_freq = self._normalize_trade_frequency(num_trades)
        
        # Get weights
        w = self.fitness_weights
        w_profit = w.get('profit', 0.3)
        w_sharpe = w.get('sharpe_ratio', 0.25)
        w_drawdown = w.get('drawdown', 0.2)
        w_win_rate = w.get('win_rate', 0.15)
        w_trade_freq = w.get('trade_frequency', 0.1)
        
        # Calculate weighted fitness
        fitness = (
            w_profit * norm_profit +
            w_sharpe * norm_sharpe +
            w_drawdown * norm_drawdown +
            w_win_rate * norm_win_rate +
            w_trade_freq * norm_trade_freq
        )
        
        # Apply penalties
        fitness = self._apply_penalties(fitness, metrics)
        
        return max(0, fitness)  # Ensure non-negative
    
    def _normalize_profit(self, profit: float) -> float:
        """Normalize profit to 0-1 range."""
        # Assume profit range of -50% to +100%
        return (profit + 50) / 150
    
    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe ratio to 0-1 range."""
        # Sharpe ratio typically ranges from -3 to 3
        return (sharpe + 3) / 6
    
    def _normalize_trade_frequency(self, num_trades: int) -> float:
        """Normalize trade frequency to 0-1 range."""
        # Prefer 20-50 trades, penalize too few or too many
        optimal_min = 20
        optimal_max = 50
        
        if num_trades < optimal_min:
            return num_trades / optimal_min
        elif num_trades > optimal_max:
            return max(0, 1 - (num_trades - optimal_max) / 100)
        else:
            return 1.0
    
    def _apply_penalties(self, fitness: float, metrics: Dict[str, float]) -> float:
        """Apply penalties for constraint violations."""
        penalties = self.fitness_penalties
        
        # Penalty for too few trades
        min_trades = penalties.get('min_trades', 10)
        if metrics.get('num_trades', 0) < min_trades:
            fitness *= 0.5
        
        # Penalty for excessive drawdown
        max_dd = penalties.get('max_drawdown', 0.25)
        if metrics.get('max_drawdown', 0) > max_dd:
            fitness *= 0.7
        
        # Penalty for low win rate
        min_wr = penalties.get('min_win_rate', 0.35)
        if metrics.get('win_rate', 0) < min_wr:
            fitness *= 0.8
        
        return fitness
    

