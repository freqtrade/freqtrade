"""
Trade Visualization Module

Displays strategy trades on candlestick charts with entry/exit markers.
Supports multiple pairs and timeframes with proper labeling.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Must be set before importing pyplot
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import pandas as pd
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib or pandas not available - trade visualization disabled")


class TradeVisualizer:
    """
    Visualizes strategy trades on candlestick charts.
    
    Features:
    - Candlestick OHLCV charts per pair/timeframe
    - Entry markers (buy arrows)
    - Exit markers (sell arrows)
    - Trade profit/loss coloring
    - Support for multiple pairs and timeframes
    """
    
    def __init__(self, config_or_output_dir: Optional[Any] = None, enabled: bool = True):
        """
        Initialize trade visualizer.
        
        Args:
            config_or_output_dir: Either a config dict with 'trade_visualization' section,
                                  a Path to output directory, or None for default
            enabled: Whether visualization is enabled
        """
        # Handle both config dict and direct Path/str arguments
        if isinstance(config_or_output_dir, dict):
            # It's a config dict - extract trade_visualization settings
            trade_vis_config = config_or_output_dir.get('trade_visualization', {})
            output_dir_str = trade_vis_config.get('output_dir', 'genetic_algorithm/output/trade_plots')
            self.output_dir = Path(output_dir_str)
            enabled = trade_vis_config.get('enabled', True) and enabled
        elif isinstance(config_or_output_dir, (str, Path)):
            self.output_dir = Path(config_or_output_dir)
        else:
            self.output_dir = Path("genetic_algorithm/output/trade_plots")
        
        self.enabled = enabled and MATPLOTLIB_AVAILABLE
        
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"TradeVisualizer initialized, output: {self.output_dir}")
        else:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("TradeVisualizer disabled - matplotlib not available")
    
    def visualize_trades(
        self,
        strategy_name: str,
        trades: List[Dict[str, Any]],
        ohlcv_data: Dict[str, pd.DataFrame],
        generation: int = 0,
        individual_idx: int = 0,
        show_plot: bool = False
    ) -> List[Path]:
        """
        Create trade visualization charts for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            trades: List of trade dictionaries with keys:
                    - pair: Trading pair (e.g., "BTC/USDT")
                    - open_date: Entry datetime
                    - close_date: Exit datetime
                    - open_rate: Entry price
                    - close_rate: Exit price
                    - profit_ratio: Profit as ratio (0.05 = 5%)
                    - is_short: Whether short trade
            ohlcv_data: Dict mapping "pair_timeframe" to OHLCV DataFrame
            generation: Generation number (for filename)
            individual_idx: Individual index (for filename)
            show_plot: Whether to display plot interactively
            
        Returns:
            List of paths to saved chart files
        """
        if not self.enabled:
            return []
        
        if not trades:
            logger.debug(f"No trades to visualize for {strategy_name}")
            return []
        
        saved_files = []
        
        # Group trades by pair
        trades_by_pair = {}
        for trade in trades:
            pair = trade.get('pair', 'UNKNOWN')
            if pair not in trades_by_pair:
                trades_by_pair[pair] = []
            trades_by_pair[pair].append(trade)
        
        # Create chart for each pair
        for pair, pair_trades in trades_by_pair.items():
            # Find matching OHLCV data
            ohlcv_df = None
            timeframe = None
            
            for key, df in ohlcv_data.items():
                if pair.replace('/', '_') in key or pair in key:
                    ohlcv_df = df
                    # Extract timeframe from key (e.g., "BTC_USDT_5m" -> "5m")
                    parts = key.split('_')
                    if len(parts) >= 3:
                        timeframe = parts[-1]
                    break
            
            if ohlcv_df is None or len(ohlcv_df) == 0:
                logger.warning(f"No OHLCV data found for {pair}")
                continue
            
            # Create the chart
            try:
                filepath = self._create_trade_chart(
                    strategy_name=strategy_name,
                    pair=pair,
                    timeframe=timeframe or "unknown",
                    trades=pair_trades,
                    ohlcv_df=ohlcv_df,
                    generation=generation,
                    individual_idx=individual_idx,
                    show_plot=show_plot
                )
                if filepath:
                    saved_files.append(filepath)
            except Exception as e:
                logger.error(f"Error creating chart for {pair}: {e}")
                import traceback
                traceback.print_exc()
        
        return saved_files
    
    def _create_trade_chart(
        self,
        strategy_name: str,
        pair: str,
        timeframe: str,
        trades: List[Dict[str, Any]],
        ohlcv_df: pd.DataFrame,
        generation: int,
        individual_idx: int,
        show_plot: bool,
        max_candles: int = 2000
    ) -> Optional[Path]:
        """
        Create a single trade chart for one pair.
        
        Args:
            strategy_name: Strategy name
            pair: Trading pair
            timeframe: Timeframe string
            trades: List of trades for this pair
            ohlcv_df: OHLCV DataFrame with columns: date, open, high, low, close, volume
            generation: Generation number
            individual_idx: Individual index
            show_plot: Whether to show interactive plot
            max_candles: Maximum number of candles to display (for performance)
            
        Returns:
            Path to saved chart file
        """
        # Ensure we have required columns
        required_cols = ['date', 'open', 'high', 'low', 'close']
        ohlcv_df = ohlcv_df.copy()
        
        # Handle different column naming conventions
        column_mapping = {
            'Date': 'date', 'Open': 'open', 'High': 'high', 
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }
        ohlcv_df.rename(columns=column_mapping, inplace=True)
        
        for col in required_cols:
            if col not in ohlcv_df.columns:
                logger.error(f"Missing required column '{col}' in OHLCV data for {pair}")
                return None
        
        # Convert date column if needed
        if not pd.api.types.is_datetime64_any_dtype(ohlcv_df['date']):
            ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date'])
        
        # Limit candles for performance - focus on period with trades
        if len(ohlcv_df) > max_candles and trades:
            # Find trade time range
            trade_dates = []
            for trade in trades:
                open_date = trade.get('open_date')
                close_date = trade.get('close_date')
                if open_date:
                    if isinstance(open_date, str):
                        trade_dates.append(pd.to_datetime(open_date))
                    else:
                        trade_dates.append(pd.to_datetime(open_date))
                if close_date:
                    if isinstance(close_date, str):
                        trade_dates.append(pd.to_datetime(close_date))
                    else:
                        trade_dates.append(pd.to_datetime(close_date))
            
            if trade_dates:
                min_trade_date = min(trade_dates)
                max_trade_date = max(trade_dates)
                
                # Add some padding around trade range
                padding_candles = max_candles // 10
                
                # Filter to trade range with padding
                ohlcv_df = ohlcv_df[
                    (ohlcv_df['date'] >= min_trade_date - pd.Timedelta(hours=padding_candles)) &
                    (ohlcv_df['date'] <= max_trade_date + pd.Timedelta(hours=padding_candles))
                ]
                
                # If still too many, sample to max_candles
                if len(ohlcv_df) > max_candles:
                    step = len(ohlcv_df) // max_candles
                    ohlcv_df = ohlcv_df.iloc[::step].head(max_candles)
                    
                logger.info(f"Limited chart to {len(ohlcv_df)} candles around trade period")
            else:
                # No trade dates, just take last max_candles
                ohlcv_df = ohlcv_df.tail(max_candles)
                logger.info(f"Limited chart to last {len(ohlcv_df)} candles")
        elif len(ohlcv_df) > max_candles:
            ohlcv_df = ohlcv_df.tail(max_candles)
            logger.info(f"Limited chart to last {len(ohlcv_df)} candles")
        
        # Create figure with two subplots (price + volume)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 10), 
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )
        
        # Title with strategy info
        title = f"{strategy_name}\n{pair} | {timeframe} | Gen {generation}"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Plot candlesticks
        self._plot_candlesticks(ax1, ohlcv_df)
        
        # Plot volume
        self._plot_volume(ax2, ohlcv_df)
        
        # Plot trades
        self._plot_trades(ax1, trades, ohlcv_df)
        
        # Add trade statistics
        self._add_trade_stats(fig, trades)
        
        # Format axes
        ax1.set_ylabel('Price', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        ax1.legend(loc='upper left')
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pair_safe = pair.replace('/', '_')
        filename = f"trades_gen{generation}_ind{individual_idx}_{pair_safe}_{timeframe}_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved trade chart: {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return filepath
    
    def _plot_candlesticks(self, ax, df: pd.DataFrame):
        """Plot OHLCV candlesticks."""
        # Calculate candle width based on average time delta
        if len(df) > 1:
            time_delta = (df['date'].iloc[1] - df['date'].iloc[0]).total_seconds()
            width = time_delta / (24 * 3600) * 0.8  # 80% of candle spacing
        else:
            width = 0.6
        
        # Separate up and down candles
        up = df[df['close'] >= df['open']]
        down = df[df['close'] < df['open']]
        
        # Plot up candles (green)
        ax.bar(up['date'], up['close'] - up['open'], width, bottom=up['open'], 
               color='#26a69a', edgecolor='#26a69a', alpha=0.9)
        ax.bar(up['date'], up['high'] - up['close'], width * 0.1, bottom=up['close'],
               color='#26a69a', alpha=0.9)
        ax.bar(up['date'], up['open'] - up['low'], width * 0.1, bottom=up['low'],
               color='#26a69a', alpha=0.9)
        
        # Plot down candles (red)
        ax.bar(down['date'], down['close'] - down['open'], width, bottom=down['open'],
               color='#ef5350', edgecolor='#ef5350', alpha=0.9)
        ax.bar(down['date'], down['high'] - down['open'], width * 0.1, bottom=down['open'],
               color='#ef5350', alpha=0.9)
        ax.bar(down['date'], down['close'] - down['low'], width * 0.1, bottom=down['low'],
               color='#ef5350', alpha=0.9)
    
    def _plot_volume(self, ax, df: pd.DataFrame):
        """Plot volume bars."""
        if 'volume' not in df.columns:
            return
        
        # Calculate width
        if len(df) > 1:
            time_delta = (df['date'].iloc[1] - df['date'].iloc[0]).total_seconds()
            width = time_delta / (24 * 3600) * 0.8
        else:
            width = 0.6
        
        # Color by price direction (vectorized - much faster than iterrows)
        colors = np.where(df['close'] >= df['open'], '#26a69a', '#ef5350')
        
        ax.bar(df['date'], df['volume'], width, color=colors, alpha=0.7)
    
    def _plot_trades(self, ax, trades: List[Dict[str, Any]], ohlcv_df: pd.DataFrame):
        """Plot trade entry and exit markers."""
        if not trades:
            return
        
        # Get price range for marker sizing
        price_range = ohlcv_df['high'].max() - ohlcv_df['low'].min()
        marker_offset = price_range * 0.03
        
        winning_entries = []
        winning_exits = []
        losing_entries = []
        losing_exits = []
        
        for trade in trades:
            try:
                # Parse trade data
                open_date = pd.to_datetime(trade.get('open_date'))
                close_date = pd.to_datetime(trade.get('close_date'))
                open_rate = float(trade.get('open_rate', 0))
                close_rate = float(trade.get('close_rate', 0))
                profit = float(trade.get('profit_ratio', 0))
                is_short = trade.get('is_short', False)
                
                # Categorize by profit
                if profit >= 0:
                    winning_entries.append((open_date, open_rate))
                    winning_exits.append((close_date, close_rate))
                else:
                    losing_entries.append((open_date, open_rate))
                    losing_exits.append((close_date, close_rate))
                
                # Draw connection line between entry and exit
                line_color = '#26a69a' if profit >= 0 else '#ef5350'
                ax.plot([open_date, close_date], [open_rate, close_rate], 
                       color=line_color, linestyle='--', alpha=0.5, linewidth=1)
                
            except Exception as e:
                logger.debug(f"Could not plot trade: {e}")
                continue
        
        # Plot markers
        marker_size = 100
        
        # Winning trades
        if winning_entries:
            dates, prices = zip(*winning_entries)
            ax.scatter(dates, prices, marker='^', s=marker_size, c='#00ff00', 
                      edgecolors='black', linewidths=1, label='Entry (Win)', zorder=5)
        if winning_exits:
            dates, prices = zip(*winning_exits)
            ax.scatter(dates, prices, marker='v', s=marker_size, c='#00ff00',
                      edgecolors='black', linewidths=1, label='Exit (Win)', zorder=5)
        
        # Losing trades
        if losing_entries:
            dates, prices = zip(*losing_entries)
            ax.scatter(dates, prices, marker='^', s=marker_size, c='#ff0000',
                      edgecolors='black', linewidths=1, label='Entry (Loss)', zorder=5)
        if losing_exits:
            dates, prices = zip(*losing_exits)
            ax.scatter(dates, prices, marker='v', s=marker_size, c='#ff0000',
                      edgecolors='black', linewidths=1, label='Exit (Loss)', zorder=5)
    
    def _add_trade_stats(self, fig, trades: List[Dict[str, Any]]):
        """Add trade statistics text box to figure."""
        if not trades:
            return
        
        # Calculate statistics
        total_trades = len(trades)
        wins = sum(1 for t in trades if t.get('profit_ratio', 0) >= 0)
        losses = total_trades - wins
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.get('profit_ratio', 0) for t in trades)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        # Create stats text
        stats_text = (
            f"Trades: {total_trades} | "
            f"Wins: {wins} | Losses: {losses} | "
            f"Win Rate: {win_rate:.1%} | "
            f"Total P/L: {total_profit:.2%} | "
            f"Avg P/L: {avg_profit:.2%}"
        )
        
        # Add text at bottom of figure
        fig.text(0.5, 0.02, stats_text, ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def visualize_strategy_from_backtest(
        self,
        strategy_name: str,
        backtest_result: 'BacktestResult',
        generation: int = 0,
        individual_idx: int = 0
    ) -> List[Path]:
        """
        Convenience method to visualize trades from a BacktestResult.
        
        Args:
            strategy_name: Name of strategy
            backtest_result: BacktestResult object with trades and ohlcv_data
            generation: Generation number
            individual_idx: Individual index
            
        Returns:
            List of saved chart paths
        """
        if not hasattr(backtest_result, 'trades') or not backtest_result.trades:
            logger.debug(f"No trades in backtest result for {strategy_name}")
            return []
        
        if not hasattr(backtest_result, 'ohlcv_data') or not backtest_result.ohlcv_data:
            logger.warning(f"No OHLCV data in backtest result for {strategy_name}")
            return []
        
        return self.visualize_trades(
            strategy_name=strategy_name,
            trades=backtest_result.trades,
            ohlcv_data=backtest_result.ohlcv_data,
            generation=generation,
            individual_idx=individual_idx
        )


def create_trade_plots_for_strategy(
    strategy_file: Path,
    config_path: str = "genetic_algorithm/config/ga_config.yaml",
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    Create trade plots for a specific strategy file.
    
    This is a standalone function that can be called from command line
    or scripts to visualize trades for any strategy.
    
    Args:
        strategy_file: Path to the strategy Python file
        config_path: Path to GA config file
        output_dir: Output directory for plots
        
    Returns:
        List of saved chart paths
    """
    import yaml
    from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Read strategy code
    with open(strategy_file, 'r') as f:
        strategy_code = f.read()
    
    strategy_name = strategy_file.stem
    
    # Create backtester and run backtest with trade data
    backtester = DirectBacktester(config)
    result = backtester.backtest_strategy_with_trades(strategy_code, strategy_name)
    
    if not result.success:
        logger.error(f"Backtest failed for {strategy_name}: {result.error_message}")
        return []
    
    # Create visualizer
    visualizer = TradeVisualizer(output_dir=output_dir)
    
    return visualizer.visualize_strategy_from_backtest(
        strategy_name=strategy_name,
        backtest_result=result,
        generation=0,
        individual_idx=0
    )
