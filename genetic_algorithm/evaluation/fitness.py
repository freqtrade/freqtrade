"""
Fitness Evaluator

Evaluates the fitness of trading strategies through backtesting
and calculating performance metrics.
"""

import logging
from typing import Tuple, Dict, Any

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester, BacktestResult
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
        
        Uses weighted combination of metrics with penalties and robustness scoring.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Fitness score (higher is better)
        """
        # Extract and normalize metrics
        profit = metrics.get('profit', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)  # New: downside risk focus
        profit_factor = metrics.get('profit_factor', 0)  # New: win/loss ratio
        drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        trades = metrics.get('num_trades', 0)
        
        # Clamp values to reasonable ranges to avoid extreme outliers
        profit = max(-50, min(profit, 200))  # -50% to +200%
        sharpe = max(-5, min(sharpe, 10))  # -5 to 10
        sortino = max(-5, min(sortino, 12))  # Sortino often higher than Sharpe
        profit_factor = max(0, min(profit_factor, 10))  # 0 to 10
        drawdown = min(drawdown, 1.0)  # 0 to 100%
        win_rate = max(0, min(win_rate, 1.0))  # 0 to 100%
        
        # Normalize to 0-1 range with better scaling
        norm_profit = (profit + 50) / 250  # -50% to +200%
        norm_sharpe = (sharpe + 5) / 15  # -5 to 10
        norm_sortino = (sortino + 5) / 17  # -5 to 12
        norm_profit_factor = min(1.0, profit_factor / 3.0)  # >3.0 is excellent
        norm_drawdown = 1 - drawdown  # Lower drawdown is better
        norm_win_rate = win_rate  # Already 0-1
        norm_trades = self._normalize_trade_frequency(trades)
        
        # Clamp normalized values
        norm_profit = max(0, min(norm_profit, 1))
        norm_sharpe = max(0, min(norm_sharpe, 1))
        norm_sortino = max(0, min(norm_sortino, 1))
        norm_profit_factor = max(0, min(norm_profit_factor, 1))
        norm_drawdown = max(0, min(norm_drawdown, 1))
        
        # Get weights with defaults (adjusted to include new metrics)
        w = self.fitness_weights
        w_profit = w.get('profit', 0.25)
        w_sharpe = w.get('sharpe_ratio', 0.15)
        w_sortino = w.get('sortino_ratio', 0.15)  # New weight
        w_profit_factor = w.get('profit_factor', 0.10)  # New weight
        w_drawdown = w.get('drawdown', 0.15)
        w_win_rate = w.get('win_rate', 0.10)
        w_trades = w.get('trade_frequency', 0.10)
        
        # Calculate weighted fitness
        fitness = (
            w_profit * norm_profit + 
            w_sharpe * norm_sharpe + 
            w_sortino * norm_sortino +
            w_profit_factor * norm_profit_factor +
            w_drawdown * norm_drawdown + 
            w_win_rate * norm_win_rate + 
            w_trades * norm_trades
        )
        
        # Robustness bonus: reward consistency (good Sortino and profit factor together)
        if sortino > 1.0 and profit_factor > 1.5:
            robustness_bonus = 1.0 + (0.05 * min(sortino, 3.0))  # Up to 15% bonus
            fitness *= robustness_bonus
        
        # Bonus for positive profit (encourage profitable strategies)
        if profit > 0:
            fitness *= 1.1  # 10% bonus for any positive profit
        
        # Extra bonus for significantly profitable strategies
        # Note: This is cumulative with above, so total bonus is 32% (1.1 * 1.2) for >10% profit
        if profit > 10:
            fitness *= 1.2  # Additional 20% bonus (32% total with previous bonus)
        
        # Risk-adjusted excellence bonus: reward exceptional risk-adjusted returns
        if sharpe > 2.0 and drawdown < 0.15:
            fitness *= 1.15  # 15% bonus for excellent risk management
        
        # Apply penalties and return
        penalized_fitness = self._apply_penalties(fitness, metrics)
        
        # Ensure non-negative
        return max(0, penalized_fitness)
    
    def _normalize_trade_frequency(self, num_trades: int) -> float:
        """
        Normalize trade frequency to 0-1 range.
        
        Prefers 10-50 trades for most strategies. Too few trades = unreliable,
        too many trades = overtrading and high fees.
        """
        if num_trades == 0:
            return 0.0
        elif num_trades < 5:
            # Very few trades - heavily penalized
            return num_trades / 10
        elif 5 <= num_trades < 10:
            # Few trades - some penalty
            return 0.5 + (num_trades - 5) / 10
        elif 10 <= num_trades <= 50:
            # Ideal range - full score
            return 1.0
        elif 50 < num_trades <= 100:
            # Moderate overtrading - slight penalty
            return 1.0 - (num_trades - 50) / 100
        else:
            # Excessive trading - significant penalty
            return max(0.3, 1.0 - (num_trades - 50) / 200)
    
    def _apply_penalties(self, fitness: float, metrics: Dict[str, float]) -> float:
        """
        Apply penalties for constraint violations.
        
        Penalties are applied multiplicatively to reduce fitness for strategies
        that violate important constraints.
        """
        penalties = self.fitness_penalties
        
        num_trades = metrics.get('num_trades', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # Soft penalty for low trade count (gradual instead of harsh)
        min_trades = penalties.get('min_trades', 5)
        if num_trades < min_trades:
            if num_trades == 0:
                fitness *= 0.1  # Very low fitness for no trades
            else:
                # Gradual penalty: 50% at 1 trade, increasing to full at min_trades
                # Formula: 0.5 + (num_trades / min_trades) * 0.5
                # E.g., with min_trades=5: 1 trade=60%, 2=70%, 3=80%, 4=90%, 5+=100%
                trade_penalty = 0.5 + (num_trades / min_trades) * 0.5
                fitness *= trade_penalty
        
        # Penalty for excessive drawdown
        max_dd_threshold = penalties.get('max_drawdown', 0.30)
        if max_drawdown > max_dd_threshold:
            # Progressive penalty: worse drawdown = worse penalty
            dd_excess = max_drawdown - max_dd_threshold
            dd_penalty = max(0.3, 1.0 - dd_excess * 2)
            fitness *= dd_penalty
        
        # Penalty for low win rate (but not too harsh)
        min_win_rate = penalties.get('min_win_rate', 0.30)
        if win_rate < min_win_rate and num_trades >= 5:  # Only penalize if enough trades
            # Gradual penalty for low win rate
            wr_penalty = max(0.6, win_rate / min_win_rate)
            fitness *= wr_penalty
        
        return fitness
    

