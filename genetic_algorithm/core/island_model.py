"""
Island Model Evolution

Orchestrates multiple independent GA populations (islands) that evolve
on regime-specific data segments and periodically exchange individuals
through migration.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    Master Island                     │
    │          (balanced data, finds generalists)          │
    │    ┌──────────┐   ┌──────────┐   ┌──────────┐      │
    │    │  migrate  │   │  migrate  │   │  migrate  │      │
    │    └────▲─────┘   └────▲─────┘   └────▲─────┘      │
    ├─────────┼──────────────┼──────────────┼──────────────┤
    │  ┌──────┴──────┐ ┌─────┴──────┐ ┌────┴───────┐     │
    │  │  Bullish    │ │  Bearish    │ │  Sideways   │     │
    │  │  Island     │ │  Island     │ │  Island     │     │
    │  └─────────────┘ └─────────────┘ └─────────────┘     │
    └─────────────────────────────────────────────────────┘

Usage:
    from genetic_algorithm.core.island_model import IslandModelEvolution
    evo = IslandModelEvolution("config/ga_config_island.yaml")
    results = evo.evolve()
"""

import copy
import json
import logging
import os
import signal
import time
import threading
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population
from genetic_algorithm.core.hall_of_fame import HallOfFame
from genetic_algorithm.utils.regime_detector import (
    RegimeDetector,
    RegimeSegment,
    RegimeType,
    load_ohlcv_data,
)
from genetic_algorithm.utils.mtf_regime_detector import MTFRegimeDetector

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class IslandConfig:
    """Configuration for a single island."""
    name: str
    data_regime: str  # 'bullish', 'bearish', 'sideways', 'balanced'
    population_size: int = 25
    segments: List[RegimeSegment] = field(default_factory=list)
    holdout_segments: List[RegimeSegment] = field(default_factory=list)


@dataclass
class MigrationConfig:
    """Configuration for migration between islands."""
    # Specialist-to-specialist migration
    specialist_interval: int = 3
    specialist_count: int = 2
    specialist_topology: str = 'fully_connected'  # 'ring', 'fully_connected'

    # Master exchange
    master_interval: int = 2
    master_receive_count: int = 3   # specialists → master
    master_send_count: int = 3      # master → specialists


@dataclass
class IslandStats:
    """Statistics for one island across the evolution."""
    name: str
    regime: str
    best_fitness: float = 0.0
    best_profit: float = 0.0
    avg_fitness: float = 0.0
    generations_completed: int = 0
    migrants_sent: int = 0
    migrants_received: int = 0


@dataclass
class MigrationEvent:
    """Record of a migration event."""
    generation: int
    source: str
    target: str
    count: int
    fitnesses: List[float]


@dataclass
class _AggregateStats:
    """Lightweight aggregate stats for the terminal monitor."""
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    genetic_diversity: Optional[float] = None
    generation: int = 0
    # Fields the monitor may access
    best_raw_fitness: Optional[float] = None
    median_fitness: Optional[float] = None
    diversity_score: Optional[float] = None
    holdout_avg_degradation: Optional[float] = None
    holdout_best_degradation: Optional[float] = None
    holdout_num_evaluated: Optional[int] = None
    holdout_num_profitable: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════
# Island Model Evolution
# ══════════════════════════════════════════════════════════════════════

class IslandModelEvolution:
    """
    Orchestrates multiple GeneticAlgorithm instances as independent
    islands with periodic migration.

    Each island evolves on regime-specific data segments, producing
    specialist strategies.  A 'master' island receives migrants from
    all specialists and evaluates on balanced (all-regime) data.
    """

    def __init__(
        self,
        config_path: str,
        visualize: bool = False,
        interactive: bool = True,
    ):
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.visualize = visualize
        self.interactive = interactive
        self.logger = logging.getLogger("GeneticAlgorithm.IslandModel")

        # ── Parse island model config ──
        island_cfg = self.config.get('island_model', {})
        if not island_cfg.get('enabled', False):
            raise ValueError(
                "island_model.enabled must be true in the config to use "
                "IslandModelEvolution."
            )

        # Build island configs
        self.island_configs: List[IslandConfig] = []
        for isl in island_cfg.get('islands', []):
            self.island_configs.append(IslandConfig(
                name=isl['name'],
                data_regime=isl.get('data_regime', 'balanced'),
                population_size=isl.get('population_size', 25),
            ))

        if not self.island_configs:
            raise ValueError("island_model.islands list is empty or missing.")

        # Migration config
        mig_cfg = island_cfg.get('migration', {})
        spec_cfg = mig_cfg.get('specialist', {})
        master_cfg = mig_cfg.get('master', {})
        self.migration = MigrationConfig(
            specialist_interval=spec_cfg.get('interval', 3),
            specialist_count=spec_cfg.get('count', 2),
            specialist_topology=spec_cfg.get('topology', 'fully_connected'),
            master_interval=master_cfg.get('interval', 2),
            master_receive_count=master_cfg.get('count', 3),
            master_send_count=master_cfg.get('send_count',
                                              master_cfg.get('count', 3)),
        )

        # GA settings (shared)
        ga_cfg = self.config.get('genetic_algorithm', {})
        self.generations = ga_cfg.get('generations', 25)

        # Regime detection config
        regime_det_cfg = island_cfg.get('regime_detection', {})
        self.regime_pair = regime_det_cfg.get('pair', 'BTC/USDT')
        self.regime_timeframe = regime_det_cfg.get('timeframe', '4h')
        self.regime_timerange = regime_det_cfg.get(
            'timerange',
            self.config.get('backtesting', {}).get('timerange', ''),
        )
        self.regime_method = regime_det_cfg.get('method', 'ensemble')
        self.regime_period_days = regime_det_cfg.get('period_days', 60)
        self.regime_holdout_ratio = regime_det_cfg.get('holdout_ratio', 0.20)

        # Score-band config (new default segmentation mode)
        bands_cfg = regime_det_cfg.get('regime_bands', {})
        self.regime_bullish_min = float(bands_cfg.get('bullish_min', 0.40))
        self.regime_bearish_max = float(bands_cfg.get('bearish_max', -0.40))
        # Segment mode: 'score_band' (default) or 'discrete' (legacy)
        self.segment_mode = regime_det_cfg.get('segment_mode', 'score_band')

        # Coverage validation thresholds
        self.min_segments_per_regime = regime_det_cfg.get(
            'min_segments_per_regime', 2
        )
        self.min_bars_per_regime = regime_det_cfg.get(
            'min_bars_per_regime', 500
        )
        self.abort_on_insufficient_data = regime_det_cfg.get(
            'abort_on_insufficient_data', False
        )

        # Phase 1 improvements: auto-calibration, quality report, TF sweep
        phase1_cfg = regime_det_cfg.get('phase1', {})
        self.auto_calibrate_bands = phase1_cfg.get('auto_calibrate', False)
        default_calibration_pairs = ['BTC/USDT', 'ETH/USDT']
        self.calibration_pairs = phase1_cfg.get(
            'calibration_pairs', default_calibration_pairs
        )
        self.calibration_timeframes = phase1_cfg.get(
            'timeframe_sweep', None  # e.g. ['30m', '1h', '4h', '1d']
        )
        self.quality_report_enabled = phase1_cfg.get('quality_report', True)

        # MTF regime detection config (optional)
        self.mtf_enabled = regime_det_cfg.get('mtf_enabled', False)
        self.mtf_timeframes = regime_det_cfg.get(
            'mtf_timeframes', ['1h', '4h', '1d']
        )
        self.mtf_combination = regime_det_cfg.get(
            'mtf_combination', 'hierarchical'
        )
        self.mtf_weights = regime_det_cfg.get('mtf_weights', None)

        # Shared hall of fame
        hof_cfg = self.config.get('hall_of_fame', {})
        hof_dir = hof_cfg.get('directory', 'genetic_algorithm/data/hall_of_fame')
        self.hall_of_fame = HallOfFame(
            directory=hof_dir,
            max_size=hof_cfg.get('max_size', 50),
            min_fitness=hof_cfg.get('min_fitness', 0.0),
        )

        # ── Parallel island evolution ──
        self.parallel_islands = island_cfg.get('parallel_islands', False)
        # Thread lock for shared state (hall_of_fame, island_stats, migration_history)
        self._hof_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._migration_lock = threading.Lock()

        # Runtime state
        self.islands: Dict[str, GeneticAlgorithm] = {}
        self.island_populations: Dict[str, Population] = {}
        self.island_stats: Dict[str, IslandStats] = {}
        self.migration_history: List[MigrationEvent] = []
        self._shutdown_requested = False

        # Generation stats per island (for summary report / visualizer)
        # Maps island_name → list of PopulationStats per generation
        self.generation_stats: Dict[str, List] = {}

        # Terminal monitor (created in evolve → _evolve_inner)
        self.monitor = None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Regime visualization
    # ------------------------------------------------------------------

    def _plot_regime_chart(
        self,
        df: 'pd.DataFrame',
        regimes: 'pd.Series',
        raw_segments: List[RegimeSegment],
        regime_map: Dict[str, Dict[str, List[RegimeSegment]]],
        output_dir: Path,
    ):
        """
        Generate a regime visualization chart showing:
          1. Price with per-candle regime background coloring
          2. Per-candle regime signal bar
          3. 60-day segment classification with optimization/holdout roles
          4. Island assignment summary

        Saved to output_dir/regime_chart.png — produced automatically
        at the start of every island model run.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.patches as mpatches
            import numpy as np
        except ImportError:
            self.logger.warning("matplotlib not available — skipping regime chart")
            return

        regime_colors = {
            RegimeType.BULLISH:   '#2ecc71',  # Green
            RegimeType.BEARISH:   '#e74c3c',  # Red
            RegimeType.SIDEWAYS:  '#f39c12',  # Orange
            RegimeType.VOLATILE:  '#9b59b6',  # Purple
            RegimeType.UNCERTAIN: '#95a5a6',  # Gray
        }

        fig, axes = plt.subplots(
            4, 1, figsize=(22, 18),
            gridspec_kw={'height_ratios': [3, 1, 2.5, 1.5]},
            sharex=True,
        )
        fig.suptitle(
            f'{self.regime_pair} {self.regime_timeframe} — Regime Classification '
            f'(method={self.regime_method})\n{self.regime_timerange}',
            fontsize=16, fontweight='bold',
        )

        ax_price, ax_signal, ax_islands, ax_segments = axes
        dates = df.index
        prices = df['close'].values

        # ── Subplot 1: Price with regime background ──
        ax_price.plot(dates, prices, color='#2c3e50', linewidth=0.8, alpha=0.9, zorder=3)

        regimes_aligned = regimes.reindex(df.index)
        prev_regime = None
        start_idx = 0

        for i in range(len(dates)):
            current = regimes_aligned.iloc[i] if i < len(regimes_aligned) else RegimeType.UNCERTAIN
            if current != prev_regime and prev_regime is not None:
                color = regime_colors.get(prev_regime, '#95a5a6')
                ax_price.axvspan(dates[start_idx], dates[i], alpha=0.15, color=color, zorder=1)
                start_idx = i
            prev_regime = current

        if prev_regime is not None and start_idx < len(dates):
            color = regime_colors.get(prev_regime, '#95a5a6')
            ax_price.axvspan(dates[start_idx], dates[-1], alpha=0.15, color=color, zorder=1)

        ax_price.set_ylabel('Price (USDT)', fontsize=12)
        ax_price.set_title('Price with Per-Candle Regime Background', fontsize=13)
        ax_price.grid(True, alpha=0.3)
        ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        legend_patches = [
            mpatches.Patch(color=regime_colors[RegimeType.BULLISH], alpha=0.4, label='Bullish'),
            mpatches.Patch(color=regime_colors[RegimeType.BEARISH], alpha=0.4, label='Bearish'),
            mpatches.Patch(color=regime_colors[RegimeType.SIDEWAYS], alpha=0.4, label='Sideways'),
        ]
        ax_price.legend(handles=legend_patches, loc='upper left', fontsize=11)

        # ── Subplot 2: Per-candle regime signal ──
        regime_numeric = regimes_aligned.map({
            RegimeType.BULLISH: 1, RegimeType.BEARISH: -1, RegimeType.SIDEWAYS: 0,
            RegimeType.VOLATILE: 0.5, RegimeType.UNCERTAIN: 0,
        }).fillna(0)
        # Use fill_between for performance (bar is too slow with thousands of candles)
        for rtype, val in [(RegimeType.BULLISH, 1), (RegimeType.BEARISH, -1), (RegimeType.SIDEWAYS, 0.15)]:
            mask = regimes_aligned == rtype
            vals = np.where(mask, val, 0)
            ax_signal.fill_between(dates, 0, vals, where=mask,
                                   color=regime_colors[rtype], alpha=0.6,
                                   step='post', linewidth=0)
        ax_signal.set_ylabel('Regime', fontsize=12)
        ax_signal.set_title('Per-Candle Regime (1=Bull, 0=Sideways, -1=Bear)', fontsize=13)
        ax_signal.set_yticks([-1, 0, 1])
        ax_signal.set_yticklabels(['Bearish', 'Sideways', 'Bullish'])
        ax_signal.grid(True, alpha=0.3)
        ax_signal.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # ── Subplot 3: Island Data Assignment (swim lanes) ──
        island_rows = [
            ('Bullish Island', 'bullish', RegimeType.BULLISH, 3),
            ('Bearish Island', 'bearish', RegimeType.BEARISH, 2),
            ('Sideways Island', 'sideways', RegimeType.SIDEWAYS, 1),
            ('Master Island', 'all', None, 0),
        ]

        ax_islands.set_ylim(-0.5, len(island_rows) - 0.5 + 0.5)
        ax_islands.set_title(
            'Island Data Assignment — Which price segments train each island',
            fontsize=13,
        )

        y_labels = []
        y_positions = []

        for label, regime_key, rtype, y_pos in island_rows:
            y_labels.append(label)
            y_positions.append(y_pos)

            data = regime_map.get(regime_key, {})
            opt_list = data.get('optimization', [])

            for seg in opt_list:
                color = regime_colors.get(seg.regime, '#95a5a6')
                ax_islands.barh(
                    y_pos, (seg.end_date - seg.start_date).days,
                    left=seg.start_date, height=0.7,
                    color=color, alpha=0.65, edgecolor='white', linewidth=0.5,
                )
                mid = seg.start_date + (seg.end_date - seg.start_date) / 2
                if seg.duration_days >= 30:
                    ax_islands.text(
                        mid, y_pos, f"{seg.duration_days}d",
                        ha='center', va='center', fontsize=7,
                        fontweight='bold', color='black',
                    )

        ax_islands.set_yticks(y_positions)
        ax_islands.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
        ax_islands.grid(True, alpha=0.3, axis='x')

        # Add segment count annotations on the right
        for label, regime_key, rtype, y_pos in island_rows:
            data = regime_map.get(regime_key, {})
            n_opt = len(data.get('optimization', []))
            total_days = sum(s.duration_days for s in data.get('optimization', []))
            ax_islands.text(
                1.01, y_pos, f"{n_opt} segs, {total_days}d",
                ha='left', va='center', fontsize=8, color='#555',
                transform=ax_islands.get_yaxis_transform(),
            )

        # ── Subplot 4: Segment blocks with optimization/holdout split ──
        ax_segments.set_ylim(-1.2, 1.2)
        ax_segments.set_title(
            'Segment Classification — Optimization (top) / Holdout (bottom)', fontsize=13,
        )

        # Gather all optimization + holdout segments with role labels
        opt_segs = []
        hold_segs = []
        for regime_key in ['bullish', 'bearish', 'sideways']:
            data = regime_map.get(regime_key, {})
            opt_segs.extend(data.get('optimization', []))
            hold_segs.extend(data.get('holdout', []))

        # Draw optimization segments (top half)
        for seg in opt_segs:
            color = regime_colors.get(seg.regime, '#95a5a6')
            ax_segments.axvspan(seg.start_date, seg.end_date, ymin=0.5, ymax=1.0,
                                alpha=0.55, color=color, zorder=2)
            mid = seg.start_date + (seg.end_date - seg.start_date) / 2
            ax_segments.text(mid, 0.7, f"{seg.regime.value}\n{seg.duration_days}d",
                             ha='center', va='center', fontsize=7, fontweight='bold',
                             color='black', zorder=3)

        # Draw holdout segments (bottom half)
        for seg in hold_segs:
            color = regime_colors.get(seg.regime, '#95a5a6')
            ax_segments.axvspan(seg.start_date, seg.end_date, ymin=0.0, ymax=0.5,
                                alpha=0.30, facecolor=color, zorder=2,
                                hatch='///', edgecolor='gray')
            mid = seg.start_date + (seg.end_date - seg.start_date) / 2
            ax_segments.text(mid, -0.7, f"holdout\n{seg.regime.value}",
                             ha='center', va='center', fontsize=6, fontweight='bold',
                             color='#555', zorder=3)

        # Island labels
        ax_segments.axhline(y=0, color='gray', linewidth=1, linestyle='--')
        ax_segments.text(dates[0], 0.95, 'OPTIMIZATION', fontsize=8, fontweight='bold',
                         va='top', ha='left', color='#333')
        ax_segments.text(dates[0], -0.3, 'HOLDOUT', fontsize=8, fontweight='bold',
                         va='top', ha='left', color='#666')
        ax_segments.set_yticks([])
        ax_segments.grid(True, alpha=0.3, axis='x')

        # Island assignment legend
        island_legend = [
            mpatches.Patch(color=regime_colors[RegimeType.BULLISH], alpha=0.55,
                           label=f'Bullish Island ({len([s for s in opt_segs if s.regime == RegimeType.BULLISH])} segs)'),
            mpatches.Patch(color=regime_colors[RegimeType.BEARISH], alpha=0.55,
                           label=f'Bearish Island ({len([s for s in opt_segs if s.regime == RegimeType.BEARISH])} segs)'),
            mpatches.Patch(color=regime_colors[RegimeType.SIDEWAYS], alpha=0.55,
                           label=f'Sideways Island ({len([s for s in opt_segs if s.regime == RegimeType.SIDEWAYS])} segs)'),
            mpatches.Patch(facecolor='white', edgecolor='gray',
                           label=f'Master Island (ALL {len(opt_segs)} segs)'),
            mpatches.Patch(facecolor='white', edgecolor='gray', hatch='///',
                           label=f'Holdout ({len(hold_segs)} segs)'),
        ]
        ax_segments.legend(handles=island_legend, loc='lower right', fontsize=9,
                           ncol=3, framealpha=0.9)

        # X-axis formatting
        ax_segments.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_segments.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output_dir / 'regime_chart.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.info("Regime chart saved to %s", chart_path)
        print(f"\n  📊 Regime chart saved: {chart_path.absolute()}\n")

    # ------------------------------------------------------------------
    # Segment building
    # ------------------------------------------------------------------

    def _detect_regime_segments(self) -> Dict[str, List[RegimeSegment]]:
        """
        Detect regime segments from historical data and split into
        optimization / holdout sets.

        When mtf_enabled is True, uses MTFRegimeDetector to combine
        signals from multiple timeframes (e.g. 1h, 4h, 1d) for richer
        regime classification with continuous scores.

        Returns:
            Dict with keys 'bullish', 'bearish', 'sideways' (and 'all'),
            each containing a dict with 'optimization' and 'holdout' lists.
        """
        backtest_cfg = self.config.get('backtesting', {})
        datadir = Path(backtest_cfg.get('datadir', 'user_data/data/binance'))

        if self.mtf_enabled:
            return self._detect_regime_segments_mtf(datadir)

        self.logger.info(
            "Detecting regimes from %s %s (%s) in %s",
            self.regime_pair, self.regime_timeframe,
            self.regime_timerange, datadir,
        )

        df = load_ohlcv_data(
            pair=self.regime_pair,
            timeframe=self.regime_timeframe,
            datadir=datadir,
            timerange=self.regime_timerange,
        )

        if df.empty:
            raise RuntimeError(
                f"No data loaded for {self.regime_pair} {self.regime_timeframe}. "
                f"Run data download first."
            )

        detector = RegimeDetector(method=self.regime_method)

        # Choose segmentation mode: score-band (default) or legacy discrete
        if self.segment_mode == 'score_band':
            self.logger.info(
                "Using score-band segmentation: bull>=%.2f, bear<=%.2f",
                self.regime_bullish_min, self.regime_bearish_max,
            )
            merge_days = 7
            raw_segments = detector.classify_periods_by_score(
                df=df,
                bullish_min=self.regime_bullish_min,
                bearish_max=self.regime_bearish_max,
                min_segment_days=merge_days,          # aligned with merge to avoid dead zone
                max_segment_days=self.regime_period_days * 3,
                merge_threshold_days=merge_days,
                embargo_days=3,
            )
        else:
            self.logger.info("Using legacy discrete segmentation")
            raw_segments = detector.classify_periods(
                df=df,
                period_days=self.regime_period_days,
                min_period_days=max(14, self.regime_period_days // 2),
            )

        if not raw_segments:
            raise RuntimeError("Regime detection produced no segments.")

        # Get balanced representation
        segments_per_regime = max(3, len(raw_segments) // 4)
        balanced = detector.get_balanced_segments(
            raw_segments,
            segments_per_regime=segments_per_regime,
        )

        # Split into optimization / holdout
        splits = detector.split_segments_by_role(
            balanced,
            optimization_ratio=1.0 - self.regime_holdout_ratio,
            model_selection_ratio=0.0,
            holdout_ratio=self.regime_holdout_ratio,
        )

        # Group by regime type
        regime_map: Dict[str, Dict[str, List[RegimeSegment]]] = {
            'bullish': {'optimization': [], 'holdout': []},
            'bearish': {'optimization': [], 'holdout': []},
            'sideways': {'optimization': [], 'holdout': []},
            'all':     {'optimization': [], 'holdout': []},
        }

        for seg in splits.get('optimization', []):
            regime_key = seg.regime.value.lower()
            if regime_key in regime_map:
                regime_map[regime_key]['optimization'].append(seg)
            regime_map['all']['optimization'].append(seg)

        for seg in splits.get('holdout', []):
            regime_key = seg.regime.value.lower()
            if regime_key in regime_map:
                regime_map[regime_key]['holdout'].append(seg)
            regime_map['all']['holdout'].append(seg)

        # Log summary
        for regime, data in regime_map.items():
            opt_count = len(data['optimization'])
            hold_count = len(data['holdout'])
            self.logger.info(
                "  %-10s: %d optimization, %d holdout segments",
                regime, opt_count, hold_count,
            )

        # Generate regime visualization chart
        try:
            per_candle_regimes = detector.detect(df)
            output_dir = Path(self.config.get('output_dir', 'genetic_algorithm/output'))
            self._plot_regime_chart(
                df=df,
                regimes=per_candle_regimes,
                raw_segments=balanced,
                regime_map=regime_map,
                output_dir=output_dir,
            )
        except Exception as e:
            self.logger.warning("Failed to generate regime chart: %s", e)

        # Print data usage summary
        self._print_data_usage_summary(regime_map)

        # Save regime segments to JSON for post-run analysis
        try:
            self._save_regime_segments_json(regime_map)
        except Exception as e:
            self.logger.warning("Failed to save regime segments JSON: %s", e)

        return regime_map

    def _detect_regime_segments_mtf(
        self,
        datadir: Path,
    ) -> Dict[str, Dict[str, List[RegimeSegment]]]:
        """
        Detect regime segments using Multi-Timeframe (MTF) regime detection.

        Combines signals from multiple timeframes (e.g. 1h, 4h, 1d) using
        hierarchical or weighted_voting fusion to produce richer segments
        with continuous trend/volatility scores.
        """
        self.logger.info(
            "Detecting regimes with MTF: %s %s (%s), combination=%s, in %s",
            self.regime_pair, self.mtf_timeframes,
            self.regime_timerange, self.mtf_combination, datadir,
        )

        # Build MTF config from island regime detection settings
        mtf_config = {
            'regime_aware': {
                'mtf_enabled': True,
                'mtf_timeframes': self.mtf_timeframes,
                'mtf_combination': self.mtf_combination,
                'method': self.regime_method,
                'detection_method': self.regime_method,
            },
        }
        if self.mtf_weights:
            mtf_config['regime_aware']['mtf_weights'] = self.mtf_weights

        try:
            mtf_detector = MTFRegimeDetector(mtf_config)
            result = mtf_detector.detect(
                benchmark_pair=self.regime_pair,
                datadir=datadir,
                timerange=self.regime_timerange or None,
            )
        except Exception as e:
            self.logger.error(
                "MTF detection failed, falling back to single-TF: %s", e
            )
            # Fall back to single-TF detection
            self.mtf_enabled = False
            return self._detect_regime_segments()

        # Load base-TF OHLCV for segment metadata
        target_tf = result.metadata.get(
            'target_timeframe', self.mtf_timeframes[0]
        )
        df = load_ohlcv_data(
            pair=self.regime_pair,
            timeframe=target_tf,
            datadir=datadir,
            timerange=self.regime_timerange or None,
        )

        # Build adaptive segments from MTF result
        # Pass score-band boundaries for consistent regime assignment
        all_segments = mtf_detector.classify_segments(
            result=result,
            df=df,
            min_segment_days=max(14, self.regime_period_days // 2),
            max_segment_days=self.regime_period_days * 2,
            merge_threshold_days=7,
            embargo_days=3,
            bullish_min=self.regime_bullish_min,
            bearish_max=self.regime_bearish_max,
        )

        if not all_segments:
            self.logger.warning(
                "MTF: No segments created, falling back to single-TF"
            )
            self.mtf_enabled = False
            return self._detect_regime_segments()

        self.logger.info(
            "MTF detection produced %d adaptive segments", len(all_segments)
        )

        # Balance & split using standard RegimeDetector utilities
        detector = RegimeDetector(method=self.regime_method)
        segments_per_regime = max(3, len(all_segments) // 4)
        balanced = detector.get_balanced_segments(
            all_segments,
            segments_per_regime=segments_per_regime,
        )

        splits = detector.split_segments_by_role(
            balanced,
            optimization_ratio=1.0 - self.regime_holdout_ratio,
            model_selection_ratio=0.0,
            holdout_ratio=self.regime_holdout_ratio,
        )

        # Group by regime type (same structure as single-TF path)
        regime_map: Dict[str, Dict[str, List[RegimeSegment]]] = {
            'bullish': {'optimization': [], 'holdout': []},
            'bearish': {'optimization': [], 'holdout': []},
            'sideways': {'optimization': [], 'holdout': []},
            'all':     {'optimization': [], 'holdout': []},
        }

        for seg in splits.get('optimization', []):
            regime_key = seg.regime.value.lower()
            if regime_key in regime_map:
                regime_map[regime_key]['optimization'].append(seg)
            regime_map['all']['optimization'].append(seg)

        for seg in splits.get('holdout', []):
            regime_key = seg.regime.value.lower()
            if regime_key in regime_map:
                regime_map[regime_key]['holdout'].append(seg)
            regime_map['all']['holdout'].append(seg)

        # Log summary
        for regime, data in regime_map.items():
            opt_count = len(data['optimization'])
            hold_count = len(data['holdout'])
            self.logger.info(
                "  MTF %-10s: %d optimization, %d holdout segments",
                regime, opt_count, hold_count,
            )

        # Safety check: if any specialist regime has 0 optimization segments,
        # reassign its holdout segments to optimization (better to train on
        # something than to have an empty island).
        for regime_key in ['bullish', 'bearish', 'sideways']:
            data = regime_map.get(regime_key, {})
            if not data.get('optimization') and data.get('holdout'):
                self.logger.warning(
                    "  MTF regime '%s' has 0 optimization segments — "
                    "reassigning %d holdout segments to optimization",
                    regime_key, len(data['holdout']),
                )
                data['optimization'] = list(data['holdout'])
                data['holdout'] = []
                # Also add to 'all' optimization pool
                regime_map['all']['optimization'].extend(data['optimization'])

        # Generate regime chart
        try:
            per_candle_regimes = detector.detect(df)
            output_dir = Path(
                self.config.get('output_dir', 'genetic_algorithm/output')
            )
            self._plot_regime_chart(
                df=df,
                regimes=per_candle_regimes,
                raw_segments=balanced,
                regime_map=regime_map,
                output_dir=output_dir,
            )
        except Exception as e:
            self.logger.warning("Failed to generate MTF regime chart: %s", e)

        # Print data usage summary
        self._print_data_usage_summary(regime_map)

        # Save regime segments JSON
        try:
            self._save_regime_segments_json(regime_map)
        except Exception as e:
            self.logger.warning(
                "Failed to save MTF regime segments JSON: %s", e
            )

        return regime_map

    def _assign_segments_to_islands(
        self,
        regime_map: Dict[str, Dict[str, List[RegimeSegment]]],
    ):
        """Assign detected segments to each island based on its regime."""
        for ic in self.island_configs:
            regime_key = ic.data_regime.lower()
            if regime_key == 'balanced':
                regime_key = 'all'
            data = regime_map.get(regime_key, {'optimization': [], 'holdout': []})
            ic.segments = data['optimization']
            ic.holdout_segments = data['holdout']
            self.logger.info(
                "Island %-10s: %d optimization + %d holdout segments (regime=%s)",
                ic.name, len(ic.segments), len(ic.holdout_segments), ic.data_regime,
            )

    def _validate_regime_coverage(
        self,
        regime_map: Dict[str, Dict[str, List[RegimeSegment]]],
    ) -> bool:
        """
        Validate that each specialist island has sufficient data for
        meaningful backtesting.

        Checks two thresholds per regime (configurable via YAML):
          - ``min_segments_per_regime`` (default 3)
          - ``min_bars_per_regime``     (default 1500)

        When data is insufficient:
          - Always logs a prominent WARNING with actionable advice
          - If ``abort_on_insufficient_data`` is True, raises RuntimeError
            to stop the run before wasting compute
          - Otherwise, marks the island as degraded and continues

        Returns:
            True if all regimes pass, False otherwise.
        """
        all_ok = True
        specialist_regimes = ['bullish', 'bearish', 'sideways']

        self.logger.info("")
        self.logger.info("── Regime Coverage Validation ──")

        for regime_key in specialist_regimes:
            data = regime_map.get(regime_key, {})
            opt_segs = data.get('optimization', [])
            n_segs = len(opt_segs)
            n_bars = sum(s.metadata.get('bar_count', 0) for s in opt_segs)
            total_days = sum(s.duration_days for s in opt_segs)

            seg_ok = n_segs >= self.min_segments_per_regime
            bar_ok = n_bars >= self.min_bars_per_regime

            if seg_ok and bar_ok:
                self.logger.info(
                    "  ✓ %-10s: %d segments, %d bars, %d days — OK",
                    regime_key, n_segs, n_bars, total_days,
                )
            else:
                all_ok = False
                issues = []
                if not seg_ok:
                    issues.append(
                        f"segments={n_segs} < min={self.min_segments_per_regime}"
                    )
                if not bar_ok:
                    issues.append(
                        f"bars={n_bars} < min={self.min_bars_per_regime}"
                    )

                # Estimate how much more data is needed
                if n_bars > 0:
                    bars_per_day = n_bars / max(total_days, 1)
                    needed_bars = self.min_bars_per_regime - n_bars
                    needed_days = int(needed_bars / max(bars_per_day, 1))
                    needed_months = max(1, needed_days // 30)
                else:
                    needed_months = 6

                self.logger.warning(
                    "  ✗ %-10s: %s — INSUFFICIENT DATA. "
                    "Extend timerange by ~%d months or download more data. "
                    "Recommended: use 3-6 years of history for full "
                    "regime coverage.",
                    regime_key, ", ".join(issues), needed_months,
                )

        if not all_ok and self.abort_on_insufficient_data:
            raise RuntimeError(
                "Regime coverage validation FAILED. One or more regimes "
                "have insufficient data for meaningful backtesting. "
                "Set 'abort_on_insufficient_data: false' to continue "
                "anyway, or extend the timerange."
            )

        if all_ok:
            self.logger.info("  All regimes pass coverage validation ✓")
        else:
            self.logger.warning(
                "  Some regimes have insufficient data — evolution will "
                "continue but results for those regimes may be unreliable."
            )

        return all_ok

    def _phase1_auto_calibrate(self) -> None:
        """
        Auto-calibrate score-band boundaries by sweeping bullish_min /
        bearish_max grids and selecting the candidate with the best
        composite quality score (balance + coverage + separation + stability).

        When ``calibration_timeframes`` is set, also sweeps across
        timeframes and picks the best (timeframe, bands) combination.
        Updates ``self.regime_bullish_min``, ``self.regime_bearish_max``,
        and optionally ``self.regime_timeframe`` in-place.
        """
        from genetic_algorithm.tools.calibrate_bands import BandCalibrator

        backtest_cfg = self.config.get('backtesting', {})
        datadir = Path(backtest_cfg.get('datadir', 'user_data/data/binance'))

        self.logger.info("")
        self.logger.info("── Phase 1 Auto-Calibration ──")

        calibrator = BandCalibrator(method=self.regime_method)

        if self.calibration_timeframes:
            # Full timeframe sweep — runs calibrate() per TF on primary pair
            self.logger.info(
                "  Sweeping timeframes: %s for %s",
                self.calibration_timeframes, self.regime_pair,
            )
            tf_results = calibrator.sweep_timeframes(
                pair=self.regime_pair,
                datadir=datadir,
                timerange=self.regime_timerange,
                timeframes=self.calibration_timeframes,
            )

            if tf_results:
                # Pick best by composite score (dict is already sorted)
                best_tf = next(iter(tf_results))
                best_result = tf_results[best_tf]
                self.logger.info(
                    "  Best timeframe: %s (score=%.4f) → "
                    "bull>=%.2f, bear<=%.2f",
                    best_tf, best_result.composite_score,
                    best_result.bullish_min, best_result.bearish_max,
                )
                self.regime_timeframe = best_tf
                self.regime_bullish_min = best_result.bullish_min
                self.regime_bearish_max = best_result.bearish_max
            else:
                self.logger.warning("  Timeframe sweep returned no results")
        elif len(self.calibration_pairs) > 1:
            # Multi-pair calibration on a single timeframe
            self.logger.info(
                "  Calibrating bands for %s across %s",
                self.regime_timeframe, self.calibration_pairs,
            )
            result = calibrator.calibrate_multi_pair(
                pairs=self.calibration_pairs,
                timeframe=self.regime_timeframe,
                datadir=datadir,
                timerange=self.regime_timerange,
            )
            if result is not None:
                self.logger.info(
                    "  Calibrated bands: bull>=%.2f, bear<=%.2f "
                    "(score=%.4f, coverage: bull=%.1f%% side=%.1f%% bear=%.1f%%)",
                    result.bullish_min, result.bearish_max,
                    result.composite_score,
                    result.bull_pct, result.side_pct, result.bear_pct,
                )
                self.regime_bullish_min = result.bullish_min
                self.regime_bearish_max = result.bearish_max
            else:
                self.logger.warning("  Calibration returned no result — keeping defaults")
        else:
            # Single pair, single TF calibration
            self.logger.info(
                "  Calibrating bands for %s %s",
                self.regime_pair, self.regime_timeframe,
            )
            df = load_ohlcv_data(
                self.regime_pair, self.regime_timeframe,
                datadir, self.regime_timerange,
            )
            if df.empty:
                self.logger.warning("  No data for calibration — keeping defaults")
            else:
                result = calibrator.calibrate(
                    df, pair=self.regime_pair, timeframe=self.regime_timeframe,
                )
                self.logger.info(
                    "  Calibrated bands: bull>=%.2f, bear<=%.2f "
                    "(score=%.4f, coverage: bull=%.1f%% side=%.1f%% bear=%.1f%%)",
                    result.bullish_min, result.bearish_max,
                    result.composite_score,
                    result.bull_pct, result.side_pct, result.bear_pct,
                )
                self.regime_bullish_min = result.bullish_min
                self.regime_bearish_max = result.bearish_max

        self.logger.info(
            "  Active bands: bull>=%.2f, bear<=%.2f",
            self.regime_bullish_min, self.regime_bearish_max,
        )

    def _phase1_quality_report(
        self,
        regime_map: Dict[str, Dict[str, List[RegimeSegment]]],
    ) -> Dict[str, Any]:
        """
        Phase 1 Data Quality Report.

        Computes and logs diagnostic metrics that validate the quality of
        the regime segmentation produced by Phase 1:

        1. **Score distribution** — histogram of trend_score across 10 bins
           to detect saturation or skew.
        2. **Conditional statistics** — mean return, volatility, and Sharpe
           per regime to verify segments are economically distinct.
        3. **Cross-pair consistency** — if multiple pairs are configured,
           checks that BTC-derived regimes also make sense on other pairs.
        4. **Segment detail table** — per-segment metrics for inspection.

        Returns the report dict and saves it to JSON + prints summary.
        """
        backtest_cfg = self.config.get('backtesting', {})
        datadir = Path(backtest_cfg.get('datadir', 'user_data/data/binance'))
        pairs = backtest_cfg.get('pairs', [self.regime_pair])

        report: Dict[str, Any] = {
            'pair': self.regime_pair,
            'timeframe': self.regime_timeframe,
            'timerange': self.regime_timerange,
            'method': self.regime_method,
            'segment_mode': self.segment_mode,
            'bands': {
                'bullish_min': self.regime_bullish_min,
                'bearish_max': self.regime_bearish_max,
            },
        }

        # 1. Score distribution
        try:
            df = load_ohlcv_data(
                pair=self.regime_pair,
                timeframe=self.regime_timeframe,
                datadir=datadir,
                timerange=self.regime_timerange,
            )
            detector = RegimeDetector(method=self.regime_method)
            trend_scores = detector._compute_trend_score(df)
            valid = trend_scores.dropna()

            bins = np.linspace(-1, 1, 11)
            hist, _ = np.histogram(valid.values, bins=bins)
            hist_pct = (hist / max(len(valid), 1) * 100).tolist()

            report['score_distribution'] = {
                'bins': [f"[{bins[i]:.1f}, {bins[i+1]:.1f})" for i in range(len(hist))],
                'counts': hist.tolist(),
                'percentages': [round(p, 1) for p in hist_pct],
                'mean': round(float(valid.mean()), 4),
                'std': round(float(valid.std()), 4),
                'skew': round(float(valid.skew()), 4),
                'kurtosis': round(float(valid.kurtosis()), 4),
            }

            # Band coverage
            bull_pct = float((valid >= self.regime_bullish_min).mean() * 100)
            bear_pct = float((valid <= self.regime_bearish_max).mean() * 100)
            side_pct = 100.0 - bull_pct - bear_pct
            report['band_coverage'] = {
                'bullish_pct': round(bull_pct, 1),
                'sideways_pct': round(side_pct, 1),
                'bearish_pct': round(bear_pct, 1),
            }

            self.logger.info("  Score distribution: mean=%.3f std=%.3f skew=%.3f",
                             valid.mean(), valid.std(), valid.skew())
            self.logger.info("  Band coverage: bull=%.1f%% side=%.1f%% bear=%.1f%%",
                             bull_pct, side_pct, bear_pct)
        except Exception as e:
            self.logger.warning("Score distribution analysis failed: %s", e)
            df = None

        # 2. Conditional statistics per regime
        regime_stats: Dict[str, Dict[str, float]] = {}
        specialist_regimes = ['bullish', 'bearish', 'sideways']

        for regime_key in specialist_regimes:
            opt_segs = regime_map.get(regime_key, {}).get('optimization', [])
            if not opt_segs:
                regime_stats[regime_key] = {'n_segments': 0}
                continue

            seg_returns = []
            seg_volatilities = []
            seg_bars = []
            for seg in opt_segs:
                ret = seg.metadata.get('total_return', seg.metadata.get('mean_return', 0.0))
                vol = seg.metadata.get('volatility', 0.0)
                bars = seg.metadata.get('bar_count', 0)
                seg_returns.append(ret)
                seg_volatilities.append(vol)
                seg_bars.append(bars)

            mean_ret = float(np.mean(seg_returns)) if seg_returns else 0.0
            mean_vol = float(np.mean(seg_volatilities)) if seg_volatilities else 0.0
            total_bars = sum(seg_bars)

            # Simplified Sharpe-like ratio (return / volatility)
            sharpe_like = mean_ret / max(mean_vol, 1e-6)

            regime_stats[regime_key] = {
                'n_segments': len(opt_segs),
                'total_bars': total_bars,
                'mean_return': round(mean_ret, 4),
                'mean_volatility': round(mean_vol, 4),
                'sharpe_like': round(sharpe_like, 4),
                'total_days': sum(s.duration_days for s in opt_segs),
                'avg_confidence': round(
                    float(np.mean([s.confidence for s in opt_segs])), 3
                ),
            }

        report['regime_statistics'] = regime_stats

        # Log regime statistics
        self.logger.info("")
        self.logger.info("── Phase 1 Quality: Regime Statistics ──")
        for regime_key, stats in regime_stats.items():
            if stats.get('n_segments', 0) == 0:
                self.logger.info("  %-10s: NO DATA", regime_key)
            else:
                self.logger.info(
                    "  %-10s: %d seg, %d bars, ret=%.4f, vol=%.4f, sharpe=%.2f, conf=%.3f",
                    regime_key,
                    stats['n_segments'], stats['total_bars'],
                    stats['mean_return'], stats['mean_volatility'],
                    stats['sharpe_like'], stats['avg_confidence'],
                )

        # Check return separation (bull should outperform bear)
        bull_ret = regime_stats.get('bullish', {}).get('mean_return', 0.0)
        bear_ret = regime_stats.get('bearish', {}).get('mean_return', 0.0)
        ret_separation = bull_ret - bear_ret
        report['return_separation'] = round(ret_separation, 4)

        if ret_separation > 0:
            self.logger.info("  Return separation: bull-bear = +%.4f ✓", ret_separation)
        else:
            self.logger.warning(
                "  Return separation: bull-bear = %.4f ✗ (bull should outperform bear!)",
                ret_separation,
            )

        # 3. Cross-pair consistency
        if df is not None and len(pairs) > 1:
            cross_pair_results: Dict[str, Dict[str, float]] = {}
            for other_pair in pairs:
                if other_pair == self.regime_pair:
                    continue
                try:
                    other_df = load_ohlcv_data(
                        other_pair, '1h', datadir, self.regime_timerange,
                    )
                    if other_df.empty:
                        continue

                    # For each regime segment, compute other pair's return
                    total_match = 0
                    total_segs = 0
                    for regime_key in specialist_regimes:
                        opt_segs = regime_map.get(regime_key, {}).get('optimization', [])
                        for seg in opt_segs:
                            total_segs += 1
                            mask = (
                                (other_df.index >= seg.start_date)
                                & (other_df.index <= seg.end_date)
                            )
                            sub = other_df.loc[mask]
                            if len(sub) < 10:
                                continue
                            other_ret = (sub['close'].iloc[-1] / sub['close'].iloc[0]) - 1

                            # Check consistency: bull → positive, bear → negative
                            if regime_key == 'bullish' and other_ret > 0:
                                total_match += 1
                            elif regime_key == 'bearish' and other_ret < 0:
                                total_match += 1
                            elif regime_key == 'sideways':
                                total_match += 1  # sideways is always "consistent"

                    consistency = total_match / max(total_segs, 1)
                    cross_pair_results[other_pair] = {
                        'consistency': round(consistency, 3),
                        'matched_segments': total_match,
                        'total_segments': total_segs,
                    }
                    self.logger.info(
                        "  Cross-pair %s: consistency=%.1f%% (%d/%d segments)",
                        other_pair, consistency * 100, total_match, total_segs,
                    )
                except Exception as e:
                    self.logger.debug("Cross-pair %s failed: %s", other_pair, e)

            report['cross_pair_consistency'] = cross_pair_results

        # Save report to JSON
        try:
            output_dir = Path(
                self.config.get('output_dir')
                or os.getenv('GA_OUTPUT_DIR', 'genetic_algorithm/output')
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / 'phase1_quality_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info("  Quality report saved: %s", report_path)
        except Exception as e:
            self.logger.warning("Failed to save quality report: %s", e)

        return report

    # ------------------------------------------------------------------
    # Island GA construction
    # ------------------------------------------------------------------

    def _build_island_config(self, ic: IslandConfig) -> Dict[str, Any]:
        """
        Create a GA config dict for a specific island by overriding
        the base config with island-specific settings.
        """
        cfg = copy.deepcopy(self.config)

        # Override population size
        cfg['genetic_algorithm']['population_size'] = ic.population_size
        cfg['genetic_algorithm']['elite_size'] = max(
            2, ic.population_size // 10
        )
        cfg['genetic_algorithm']['random_immigrants'] = max(
            2, ic.population_size // 10
        )

        # Disable walk-forward (regime segments replace it)
        cfg['walk_forward'] = {'enabled': False}

        # Configure regime-aware evaluation scoped to island's segments
        cfg['regime_aware'] = cfg.get('regime_aware', {})
        cfg['regime_aware']['enabled'] = True
        cfg['regime_aware']['aggregation'] = 'harmonic_mean'

        # Disable island_model in sub-GA to prevent recursion
        cfg['island_model'] = {'enabled': False}

        # Disable terminal monitor for sub-islands (parent orchestrates)
        cfg['terminal_monitor'] = {'enabled': False}

        # Pass through in_strategy_regime settings if configured
        isr_cfg = self.config.get('in_strategy_regime', {})
        if isr_cfg.get('enabled', False):
            cfg['in_strategy_regime'] = copy.deepcopy(isr_cfg)

        # Tag the island name for logging
        cfg['_island_name'] = ic.name

        # Inject regime context for LLM prompt specialization
        llm_cfg = cfg.get('advanced', {}).get('llm', {})
        if llm_cfg.get('enabled'):
            llm_cfg['island_regime'] = ic.data_regime.lower()
            llm_cfg['island_name'] = ic.name

        # ── Parallel island partitioning ──
        # When running islands in parallel, split workers across islands
        # so total process count doesn't exceed CPU count.
        if self.parallel_islands:
            par_cfg = cfg.get('parallel_evaluation', {})
            if par_cfg.get('enabled', False):
                total_workers = par_cfg.get('num_workers') or max(1, os.cpu_count() - 1)
                num_islands = len(self.island_configs)
                workers_per_island = max(1, total_workers // num_islands)
                par_cfg['num_workers'] = workers_per_island
                self.logger.debug(
                    "Island %s: %d workers (total %d / %d islands)",
                    ic.name, workers_per_island, total_workers, num_islands,
                )

        return cfg

    def _create_island_ga(
        self,
        ic: IslandConfig,
    ) -> GeneticAlgorithm:
        """
        Create a GeneticAlgorithm instance for one island.

        The GA's evaluator is configured to use only this island's
        regime-specific segments.
        """
        import tempfile

        island_cfg = self._build_island_config(ic)

        # Write temp config for the GA constructor
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, prefix=f'island_{ic.name}_',
        ) as tmp:
            yaml.dump(island_cfg, tmp)
            tmp_path = tmp.name

        try:
            ga = GeneticAlgorithm(
                config_path=tmp_path,
                visualize=False,
                interactive=False,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Override the evaluator's segments to use island-specific ones
        if hasattr(ga.fitness_evaluator, '_optimization_segments'):
            ga.fitness_evaluator._optimization_segments = list(ic.segments)
            ga.fitness_evaluator._holdout_segments = list(ic.holdout_segments)
            ga.fitness_evaluator.segments = {
                'optimization': list(ic.segments),
                'holdout': list(ic.holdout_segments),
            }
            self.logger.info(
                "Island %s evaluator: %d opt + %d holdout segments",
                ic.name, len(ic.segments), len(ic.holdout_segments),
            )

        # Share the hall of fame
        ga.hall_of_fame = self.hall_of_fame

        return ga

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _get_top_individuals(
        self,
        island_name: str,
        count: int,
    ) -> List[Individual]:
        """Get top-N individuals from an island's population (by raw_fitness)."""
        pop = self.island_populations.get(island_name)
        if pop is None:
            return []

        ranked = sorted(
            [ind for ind in pop.individuals
             if ind.raw_fitness is not None and ind.raw_fitness > 0],
            key=lambda x: x.raw_fitness,
            reverse=True,
        )
        return ranked[:count]

    def _inject_migrants(
        self,
        target_island: str,
        migrants: List[Individual],
        generation: int,
    ):
        """
        Inject migrant individuals into a target island's population,
        replacing the worst individuals.
        """
        pop = self.island_populations.get(target_island)
        if pop is None or not migrants:
            return

        # Sort population worst-first
        sorted_inds = sorted(
            pop.individuals,
            key=lambda x: x.raw_fitness if x.raw_fitness is not None else -1,
        )

        replaced = 0
        for migrant in migrants:
            if replaced >= len(sorted_inds):
                break
            # Copy migrant so it's independent
            gene_copy = migrant.strategy_gene.copy()
            gene_copy.generation = generation
            gene_copy.individual_id = sorted_inds[replaced].strategy_gene.individual_id

            new_ind = Individual(strategy_gene=gene_copy)
            new_ind.evaluated = False  # Force re-evaluation on new data
            new_ind.metrics = {'origin': f'migrant_from_{target_island}'}

            # Replace worst individual in-place
            idx = pop.individuals.index(sorted_inds[replaced])
            pop.individuals[idx] = new_ind
            replaced += 1

        return replaced

    def _migrate_specialists(self, generation: int):
        """
        Migrate top individuals between specialist islands.
        Topology: fully_connected — each specialist sends to ALL others.
        """
        specialist_names = [
            ic.name for ic in self.island_configs
            if ic.data_regime.lower() != 'balanced'
        ]

        if len(specialist_names) < 2:
            return

        self.logger.info(
            "[MIGRATION] Specialist migration at gen %d (top %d, topology=%s)",
            generation + 1,
            self.migration.specialist_count,
            self.migration.specialist_topology,
        )

        for source_name in specialist_names:
            top = self._get_top_individuals(
                source_name, self.migration.specialist_count
            )
            if not top:
                continue

            for target_name in specialist_names:
                if target_name == source_name:
                    continue

                replaced = self._inject_migrants(target_name, top, generation)
                fitnesses = [ind.raw_fitness for ind in top if ind.raw_fitness]

                self.migration_history.append(MigrationEvent(
                    generation=generation,
                    source=source_name,
                    target=target_name,
                    count=replaced or 0,
                    fitnesses=fitnesses,
                ))

                self.island_stats[source_name].migrants_sent += len(top)
                self.island_stats[target_name].migrants_received += replaced or 0

                self.logger.info(
                    "  %s → %s: %d migrants (fitnesses: %s)",
                    source_name, target_name, replaced or 0,
                    [f"{f:.4f}" for f in fitnesses],
                )

    def _migrate_master(self, generation: int):
        """
        Master exchange:
        1. Each specialist sends top-N to master
        2. Master sends top-M back to each specialist
        """
        master_name = None
        specialist_names = []
        for ic in self.island_configs:
            if ic.data_regime.lower() == 'balanced':
                master_name = ic.name
            else:
                specialist_names.append(ic.name)

        if master_name is None or not specialist_names:
            return

        self.logger.info(
            "[MIGRATION] Master exchange at gen %d", generation + 1
        )

        # Phase 1: specialists → master
        for spec_name in specialist_names:
            top = self._get_top_individuals(
                spec_name, self.migration.master_receive_count
            )
            if top:
                replaced = self._inject_migrants(master_name, top, generation)
                fitnesses = [ind.raw_fitness for ind in top if ind.raw_fitness]
                self.migration_history.append(MigrationEvent(
                    generation=generation,
                    source=spec_name,
                    target=master_name,
                    count=replaced or 0,
                    fitnesses=fitnesses,
                ))
                self.island_stats[spec_name].migrants_sent += len(top)
                self.island_stats[master_name].migrants_received += replaced or 0
                self.logger.info(
                    "  %s → %s: %d migrants",
                    spec_name, master_name, replaced or 0,
                )

        # Phase 2: master → specialists
        for spec_name in specialist_names:
            top = self._get_top_individuals(
                master_name, self.migration.master_send_count
            )
            if top:
                replaced = self._inject_migrants(spec_name, top, generation)
                fitnesses = [ind.raw_fitness for ind in top if ind.raw_fitness]
                self.migration_history.append(MigrationEvent(
                    generation=generation,
                    source=master_name,
                    target=spec_name,
                    count=replaced or 0,
                    fitnesses=fitnesses,
                ))
                self.island_stats[master_name].migrants_sent += len(top)
                self.island_stats[spec_name].migrants_received += replaced or 0
                self.logger.info(
                    "  %s → %s: %d migrants",
                    master_name, spec_name, replaced or 0,
                )

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def evolve(self) -> Dict[str, List[Individual]]:
        """
        Run the full island model evolution.

        Returns:
            Dict mapping island name → list of best individuals.
        """
        # Graceful shutdown
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _shutdown(signum, frame):
            if self._shutdown_requested:
                signal.signal(signal.SIGINT, original_sigint)
                raise KeyboardInterrupt
            self._shutdown_requested = True
            self.logger.warning("[SHUTDOWN] Graceful shutdown requested")

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        try:
            return self._evolve_inner()
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def _evolve_inner(self) -> Dict[str, List[Individual]]:
        start_time = time.time()

        # ── Create terminal monitor ──
        from genetic_algorithm.monitor import create_monitor
        # Re-enable monitor at the orchestrator level (sub-islands have it disabled)
        monitor_cfg = copy.deepcopy(self.config)
        monitor_cfg.setdefault('terminal_monitor', {})['enabled'] = (
            self.config.get('terminal_monitor', {}).get('enabled', True)
        )
        self.monitor = create_monitor(monitor_cfg)
        self.monitor.start(monitor_cfg)

        self.logger.info("=" * 70)
        self.logger.info("ISLAND MODEL EVOLUTION STARTING")
        self.logger.info("=" * 70)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: DATA COLLECTION & PREPARATION
        # ═══════════════════════════════════════════════════════════════
        regime_map = self._phase1_data()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: EVOLUTION
        # ═══════════════════════════════════════════════════════════════
        results = self._phase2_evolve()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: VALIDATION
        # ═══════════════════════════════════════════════════════════════
        results = self._phase3_validate(results)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: REPORTING
        # ═══════════════════════════════════════════════════════════════
        total_elapsed = time.time() - start_time
        self._phase4_report(results, total_elapsed)

        # Stop monitor
        self.monitor.on_evolution_complete({
            'total_time': total_elapsed,
            'generations': self.generations,
            'islands': len(self.islands),
            'migrations': len(self.migration_history),
        })

        return results

    # ------------------------------------------------------------------
    # Phase 1: Data Collection & Preparation
    # ------------------------------------------------------------------

    def _phase1_data(self) -> Dict[str, Dict[str, List[RegimeSegment]]]:
        """
        Phase 1 — Data Collection & Preparation.

        Detects market regimes from historical data, builds segments
        using score-band or legacy discrete classification, balances them
        across regimes, splits into optimization/holdout, assigns segments
        to islands, and validates coverage.

        Sub-steps:
          1a. Auto-calibrate score-band boundaries (if enabled)
          1b. Detect regime segments
          1c. Assign segments to islands
          1d. Validate coverage
          1e. Generate quality report (if enabled)

        Raises RuntimeError if data is insufficient and
        ``abort_on_insufficient_data`` is enabled.

        Returns:
            regime_map: Dict with keys per regime → optimization/holdout lists.
        """
        phase_start = time.time()
        self.logger.info("")
        self.logger.info("═" * 70)
        self.logger.info("  PHASE 1: DATA COLLECTION & PREPARATION")
        self.logger.info("═" * 70)

        # 1a. Auto-calibrate bands (optional)
        if self.auto_calibrate_bands and self.segment_mode == 'score_band':
            try:
                self._phase1_auto_calibrate()
            except Exception as e:
                self.logger.warning(
                    "Auto-calibration failed, using configured bands: %s", e
                )

        # 1b. Detect regime segments
        regime_map = self._detect_regime_segments()

        # 1c. Assign segments to islands
        self._assign_segments_to_islands(regime_map)

        # 1d. Validate coverage
        self._validate_regime_coverage(regime_map)

        # 1e. Quality report (optional)
        if self.quality_report_enabled:
            try:
                self._phase1_quality_report(regime_map)
            except Exception as e:
                self.logger.warning("Quality report generation failed: %s", e)

        phase_elapsed = time.time() - phase_start
        self.logger.info(
            "  Phase 1 complete: %.1f seconds. %d total segments assigned.",
            phase_elapsed,
            sum(
                len(ic.segments) + len(ic.holdout_segments)
                for ic in self.island_configs
            ),
        )
        self.logger.info("")

        return regime_map

    # ------------------------------------------------------------------
    # Phase 2: Evolution
    # ------------------------------------------------------------------

    def _phase2_evolve(self) -> Dict[str, List[Individual]]:
        """
        Phase 2 — Evolution.

        Creates one GeneticAlgorithm per island, initializes populations,
        runs the generation loop with periodic migration, and tracks
        statistics + hall of fame.

        Returns:
            Dict mapping island name → top individuals.
        """
        phase_start = time.time()
        self.logger.info("═" * 70)
        self.logger.info("  PHASE 2: EVOLUTION")
        self.logger.info("═" * 70)

        # ── Create GA instances ──
        self.logger.info("Creating %d island GA instances...", len(self.island_configs))

        for ic in self.island_configs:
            ga = self._create_island_ga(ic)
            self.islands[ic.name] = ga
            self.island_stats[ic.name] = IslandStats(
                name=ic.name, regime=ic.data_regime,
            )
            self.generation_stats[ic.name] = []

            # Initialize population
            pop = ga.initialize_population()
            self.island_populations[ic.name] = pop

            self.logger.info(
                "  Island %-10s: pop=%d, regime=%s, segments=%d",
                ic.name, len(pop.individuals), ic.data_regime, len(ic.segments),
            )

        # ── Evolution loop ──
        self.logger.info("")
        self.logger.info("─" * 70)
        self.logger.info(
            "EVOLVING %d ISLANDS × %d GENERATIONS%s",
            len(self.islands), self.generations,
            " (PARALLEL)" if self.parallel_islands else "",
        )
        self.logger.info("─" * 70)

        # Track overall best for monitor
        overall_best_individual = None

        for gen in range(self.generations):
            if self._shutdown_requested:
                self.logger.info("[SHUTDOWN] Stopping at generation %d", gen)
                break

            gen_start = time.time()
            self.logger.info("")
            self.logger.info("─" * 70)
            self.logger.info("GENERATION %d/%d", gen + 1, self.generations)
            self.logger.info("─" * 70)

            # Notify monitor of generation start
            self.monitor.on_generation_start(gen, self.generations)

            # Evolve each island for one generation
            if self.parallel_islands and len(self.island_configs) > 1:
                self._evolve_all_islands_parallel(gen)
            else:
                for ic in self.island_configs:
                    island_name = ic.name
                    ga = self.islands[island_name]
                    pop = self.island_populations[island_name]
                    self._evolve_island_one_generation(ga, pop, island_name, gen)

            # Migration: specialist ↔ specialist
            if (
                self.migration.specialist_interval > 0
                and (gen + 1) % self.migration.specialist_interval == 0
                and gen > 0
            ):
                self._migrate_specialists(gen)

            # Migration: master exchange
            if (
                self.migration.master_interval > 0
                and (gen + 1) % self.migration.master_interval == 0
                and gen > 0
            ):
                self._migrate_master(gen)

            # Log generation summary
            gen_elapsed = time.time() - gen_start
            self._log_generation_summary(gen, gen_elapsed)

            # ── Notify monitor of generation end ──
            agg_best = max(
                (ist.best_fitness for ist in self.island_stats.values()),
                default=0,
            )
            agg_avg = (
                sum(ist.avg_fitness for ist in self.island_stats.values())
                / max(len(self.island_stats), 1)
            )

            # Find overall best individual
            for ic in self.island_configs:
                pop = self.island_populations.get(ic.name)
                if pop:
                    best_list = pop.get_best(1)
                    if best_list:
                        cand = best_list[0]
                        if (cand.raw_fitness and
                                (overall_best_individual is None or
                                 cand.raw_fitness > (overall_best_individual.raw_fitness or 0))):
                            overall_best_individual = cand
                            self.monitor.on_new_best(cand)

            _agg_stats = _AggregateStats(
                best_fitness=agg_best,
                avg_fitness=agg_avg,
                worst_fitness=0,
                genetic_diversity=None,
                generation=gen,
            )
            self.monitor.on_generation_end(
                gen=gen,
                stats=_agg_stats,
                timing=None,
                best_individual=overall_best_individual,
                extras={
                    'island_count': len(self.islands),
                    'migrations': len(self.migration_history),
                },
            )

        # Collect results
        results: Dict[str, List[Individual]] = {}
        for ic in self.island_configs:
            pop = self.island_populations.get(ic.name)
            if pop:
                top5 = sorted(
                    [ind for ind in pop.individuals if ind.raw_fitness is not None],
                    key=lambda x: x.raw_fitness,
                    reverse=True,
                )[:5]
                results[ic.name] = top5

        phase_elapsed = time.time() - phase_start
        self.logger.info("")
        self.logger.info(
            "  Phase 2 complete: %.1f seconds (%.1f minutes). "
            "%d migrations performed.",
            phase_elapsed, phase_elapsed / 60, len(self.migration_history),
        )

        return results

    # ------------------------------------------------------------------
    # Phase 3: Validation
    # ------------------------------------------------------------------

    def _phase3_validate(
        self,
        results: Dict[str, List[Individual]],
    ) -> Dict[str, List[Individual]]:
        """
        Phase 3 — Validation.

        Runs holdout evaluation on top strategies from each island to
        detect overfitting.  Populates ``ind.metrics`` with holdout
        fitness and degradation scores.

        Returns:
            The same results dict with updated metrics.
        """
        phase_start = time.time()
        self.logger.info("")
        self.logger.info("═" * 70)
        self.logger.info("  PHASE 3: VALIDATION")
        self.logger.info("═" * 70)

        # Holdout validation
        self._run_holdout_validation(results)

        phase_elapsed = time.time() - phase_start
        self.logger.info(
            "  Phase 3 complete: %.1f seconds.", phase_elapsed,
        )

        return results

    # ------------------------------------------------------------------
    # Phase 4: Reporting
    # ------------------------------------------------------------------

    def _phase4_report(
        self,
        results: Dict[str, List[Individual]],
        total_elapsed: float,
    ):
        """
        Phase 4 — Reporting.

        Generates charts, saves CSVs/JSONs, logs final summary.
        """
        phase_start = time.time()
        self.logger.info("")
        self.logger.info("═" * 70)
        self.logger.info("  PHASE 4: REPORTING")
        self.logger.info("═" * 70)

        self.logger.info("  Total time: %.1f seconds (%.1f minutes)",
                         total_elapsed, total_elapsed / 60)
        self.logger.info("  Generations: %d", self.generations)
        self.logger.info("  Islands: %d", len(self.islands))
        self.logger.info("  Migrations: %d events", len(self.migration_history))
        self.logger.info("")

        # Log per-island summary
        for ic in self.island_configs:
            ist = self.island_stats[ic.name]
            pop = self.island_populations.get(ic.name)

            self.logger.info("── Island: %s (regime=%s) ──", ic.name, ic.data_regime)
            self.logger.info("  Best fitness:  %.4f", ist.best_fitness)
            self.logger.info("  Best profit:   %.2f%%", ist.best_profit)
            self.logger.info("  Avg fitness:   %.4f", ist.avg_fitness)
            self.logger.info("  Migrants sent: %d  received: %d",
                             ist.migrants_sent, ist.migrants_received)

            top5 = results.get(ic.name, [])
            for rank, ind in enumerate(top5, 1):
                profit = ind.metrics.get('profit', 0)
                sharpe = ind.metrics.get('sharpe_ratio', 0)
                trades = ind.metrics.get('num_trades', 0)
                self.logger.info(
                    "    #%d: fitness=%.4f profit=%.2f%% sharpe=%.2f trades=%d",
                    rank, ind.raw_fitness or 0, profit, sharpe, trades,
                )
            self.logger.info("")

        # Hall of fame summary
        if self.hall_of_fame.entries:
            self.logger.info("── Shared Hall of Fame: %d entries ──",
                             len(self.hall_of_fame.entries))
            for i, entry in enumerate(self.hall_of_fame.entries[:5]):
                self.logger.info(
                    "  #%d: fitness=%.4f (gen %d)",
                    i + 1, entry.fitness, entry.generation_found,
                )

        # Migration effectiveness
        if self.migration_history:
            self.logger.info("")
            self.logger.info("── Migration Summary ──")
            source_counts: Dict[str, int] = {}
            for event in self.migration_history:
                source_counts[event.source] = (
                    source_counts.get(event.source, 0) + event.count
                )
            for source, count in sorted(source_counts.items()):
                self.logger.info("  %s: %d individuals exported", source, count)

        # Generate evolution plots
        try:
            self._plot_island_evolution()
        except Exception as e:
            self.logger.warning("Failed to generate evolution plots: %s", e)

        # Save per-island generation stats to CSV
        try:
            self._save_generation_stats_csv()
        except Exception as e:
            self.logger.warning("Failed to save generation stats CSV: %s", e)

        # Save LLM contribution report
        try:
            self._save_llm_report()
        except Exception as e:
            self.logger.warning("Failed to save LLM report: %s", e)

        # Save results
        self._save_results(results)

        phase_elapsed = time.time() - phase_start
        self.logger.info(
            "  Phase 4 complete: %.1f seconds.", phase_elapsed,
        )
        self.logger.info("")
        self.logger.info("═" * 70)
        self.logger.info("  ALL 4 PHASES COMPLETE — ISLAND MODEL EVOLUTION FINISHED")
        self.logger.info("═" * 70)

    def _evolve_island_one_generation(
        self,
        ga: GeneticAlgorithm,
        population: Population,
        island_name: str,
        generation: int,
    ):
        """
        Run one generation of evolution on a single island.

        This manually executes the core GA steps (evaluate → select →
        crossover/mutate → next gen) without calling ga.evolve(), which
        runs the full loop.
        """
        ga.current_generation = generation

        # Reset LLM generation budget
        if ga.llm_enabled and ga.strategy_designer:
            ga.strategy_designer.reset_generation_budget()

        # Step 1: Evaluate fitness
        ga.evaluate_population(population)

        # Step 2: Fitness sharing
        from genetic_algorithm.core.population import (
            apply_fitness_sharing, calculate_pairwise_distances,
        )
        if ga.fitness_sharing and len(population.individuals) >= 2:
            distance_matrix = calculate_pairwise_distances(
                list(population.individuals)
            )
            apply_fitness_sharing(
                population, sigma_share=ga.sharing_radius,
                distance_matrix=distance_matrix,
            )

        # Step 3: Get stats
        stats = population.get_stats()

        # Update best
        best = population.get_best(1)
        if best:
            best_ind = best[0]
            with self._stats_lock:
                ist = self.island_stats[island_name]
                if best_ind.raw_fitness and best_ind.raw_fitness > ist.best_fitness:
                    ist.best_fitness = best_ind.raw_fitness
                    profit = best_ind.metrics.get('profit', 0)
                    ist.best_profit = profit
                    self.logger.info(
                        "  [%s] NEW BEST: fitness=%.4f profit=%.2f%%",
                        island_name, ist.best_fitness, profit,
                    )
                ist.avg_fitness = stats.avg_fitness
                ist.generations_completed = generation + 1

        # Step 4: Update hall of fame
        try:
            with self._hof_lock:
                ga.hall_of_fame.update(population, generation)
        except Exception as e:
            self.logger.warning("Hall of fame update failed for %s: %s", island_name, e)

        # Step 5: Log island stats
        self.logger.info(
            "  [%-10s] best=%.4f avg=%.4f diversity=%.4f",
            island_name,
            stats.best_fitness,
            stats.avg_fitness,
            stats.genetic_diversity or 0,
        )

        # Step 6: Record generation stats for this island
        stats.generation = generation
        with self._stats_lock:
            self.generation_stats[island_name].append(stats)

        # Step 6b: Record LLM strategy performance for feedback loop
        if ga.llm_enabled and ga.strategy_designer and ga.strategy_designer.enabled:
            try:
                ga.strategy_designer.record_llm_performance(generation, population)
            except Exception as e:
                self.logger.warning(
                    "LLM performance recording failed for %s gen %d: %s",
                    island_name, generation, e,
                )

        # Step 6c: Save best strategy snapshot for this generation
        try:
            self._save_strategy_snapshot(population, island_name, generation)
        except Exception as e:
            self.logger.debug("Strategy snapshot failed for %s gen %d: %s",
                              island_name, generation, e)

        # Step 7: Create next generation
        if generation < self.generations - 1:
            next_pop = ga.create_next_generation(population)
            self.island_populations[island_name] = next_pop

    def _log_generation_summary(self, gen: int, elapsed: float):
        """Log a compact summary of all islands for this generation."""
        parts = []
        for ic in self.island_configs:
            ist = self.island_stats[ic.name]
            parts.append(f"{ic.name}={ist.best_fitness:.4f}")

        self.logger.info(
            "[SUMMARY] Gen %d/%d (%.1fs): %s",
            gen + 1, self.generations, elapsed, " | ".join(parts),
        )

    def _evolve_all_islands_parallel(self, generation: int):
        """
        Evolve ALL islands for one generation concurrently using threads.

        Each island's GA already uses its own ``ProcessPoolExecutor`` for
        backtesting, so we use threads here (not processes) to orchestrate
        island-level parallelism.  The per-island worker pools are partitioned
        in ``_build_island_config`` so that total process count stays within
        CPU core limits.

        Migration is NOT done here — it runs synchronously on the main
        thread after this method returns (requires all populations to be
        evaluated).
        """
        def _evolve_single(ic: IslandConfig):
            """Thread target: evolve one island for one generation."""
            island_name = ic.name
            ga = self.islands[island_name]
            pop = self.island_populations[island_name]
            try:
                self._evolve_island_one_generation(ga, pop, island_name, generation)
            except Exception as e:
                self.logger.error(
                    "[PARALLEL-ISLAND] Island %s gen %d failed: %s",
                    island_name, generation, e,
                )

        with ThreadPoolExecutor(
            max_workers=len(self.island_configs),
            thread_name_prefix="island",
        ) as executor:
            futures = {
                executor.submit(_evolve_single, ic): ic.name
                for ic in self.island_configs
            }
            for future in as_completed(futures):
                island_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(
                        "[PARALLEL-ISLAND] Island %s raised: %s",
                        island_name, e,
                    )

    # ------------------------------------------------------------------
    # Final report (legacy — kept for backward compatibility)
    # ------------------------------------------------------------------

    def _final_report(
        self,
        total_elapsed: float,
    ) -> Dict[str, List[Individual]]:
        """Legacy wrapper — now split into _phase3_validate + _phase4_report."""
        results: Dict[str, List[Individual]] = {}
        for ic in self.island_configs:
            pop = self.island_populations.get(ic.name)
            if pop:
                top5 = sorted(
                    [ind for ind in pop.individuals if ind.raw_fitness is not None],
                    key=lambda x: x.raw_fitness,
                    reverse=True,
                )[:5]
                results[ic.name] = top5
        results = self._phase3_validate(results)
        self._phase4_report(results, total_elapsed)
        return results

    # ------------------------------------------------------------------
    # Holdout Validation
    # ------------------------------------------------------------------

    def _run_holdout_validation(
        self,
        results: Dict[str, List[Individual]],
    ):
        """
        Evaluate top strategies on holdout segments to detect overfitting.

        Uses the master island's evaluator (which holds ALL holdout segments)
        to get a regime-diversified out-of-sample score.  If the master island
        has no holdout segments (smart-skip), falls back to the first island
        that does.

        Populates ``ind.metrics`` with ``holdout_fitness``,
        ``holdout_degradation``, etc. so that ``classify_overfitting()`` in
        ``overfit_analysis.py`` can produce a real verdict instead of UNKNOWN.
        """
        # Find an evaluator that has holdout segments
        evaluator = None
        evaluator_source = None

        # Prefer master / balanced island (has all holdout segments)
        for ic in self.island_configs:
            if ic.data_regime.lower() in ('balanced', 'all') and ic.holdout_segments:
                ga = self.islands.get(ic.name)
                if ga and hasattr(ga, 'fitness_evaluator'):
                    evaluator = ga.fitness_evaluator
                    evaluator_source = ic.name
                    break

        # Fallback: any island with holdout segments
        if evaluator is None:
            for ic in self.island_configs:
                if ic.holdout_segments:
                    ga = self.islands.get(ic.name)
                    if ga and hasattr(ga, 'fitness_evaluator'):
                        evaluator = ga.fitness_evaluator
                        evaluator_source = ic.name
                        break

        if evaluator is None:
            self.logger.warning(
                "No holdout segments available on any island — "
                "skipping holdout validation (overfitting will be UNKNOWN)"
            )
            return

        n_holdout = len(getattr(evaluator, '_holdout_segments', []))
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("HOLDOUT VALIDATION  (evaluator from '%s', %d holdout segments)",
                         evaluator_source, n_holdout)
        self.logger.info("=" * 70)

        # Pool all top individuals across islands (deduplicate by id)
        seen_ids: set = set()
        all_top: List[Individual] = []
        for island_name, individuals in results.items():
            for ind in individuals:
                ind_id = id(ind)
                if ind_id not in seen_ids:
                    seen_ids.add(ind_id)
                    all_top.append(ind)

        # Sort by optimization fitness (descending)
        all_top.sort(key=lambda x: x.raw_fitness or 0, reverse=True)

        evaluated = 0
        for ind in all_top:
            if ind.strategy_gene is None:
                continue
            try:
                holdout_fitness, holdout_metrics = evaluator.evaluate_holdout(
                    ind.strategy_gene, auto_unlock=True,
                )
            except (ValueError, RuntimeError) as exc:
                self.logger.debug("Holdout eval failed for ind %s: %s", ind, exc)
                continue

            # Compute degradation: how much worse is holdout vs optimization
            opt_fitness = ind.raw_fitness or 0
            if opt_fitness > 0:
                degradation = (opt_fitness - holdout_fitness) / opt_fitness
            else:
                degradation = 0.0

            # Store in metrics for overfit_analysis.classify_overfitting()
            ind.metrics['holdout_fitness'] = holdout_fitness
            ind.metrics['holdout_degradation'] = degradation
            ind.metrics['holdout_profit'] = holdout_metrics.get('profit', 0)
            ind.metrics['holdout_sharpe'] = holdout_metrics.get('sharpe_ratio', 0)
            ind.metrics['holdout_drawdown'] = holdout_metrics.get('max_drawdown', 0)
            ind.metrics['holdout_trades'] = holdout_metrics.get('num_trades', 0)

            status = "✓" if degradation < 0.30 else "⚠"
            self.logger.info(
                "  %s fitness=%.4f → holdout=%.4f  degradation=%.1f%%  "
                "holdout_profit=%.2f%%  trades=%d",
                status, opt_fitness, holdout_fitness, degradation * 100,
                holdout_metrics.get('profit', 0),
                holdout_metrics.get('num_trades', 0),
            )
            evaluated += 1

        self.logger.info("  Evaluated %d / %d individuals on holdout", evaluated, len(all_top))
        self.logger.info("")

    def _plot_island_evolution(self):
        """
        Generate a multi-panel evolution chart showing per-island fitness
        curves, diversity, and migration events.

        Saved to genetic_algorithm/output/island_evolution.png
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
        except ImportError:
            self.logger.warning("matplotlib not available — skipping evolution plots")
            return

        island_colors = {
            'bullish': '#2ecc71',
            'bearish': '#e74c3c',
            'sideways': '#f39c12',
            'balanced': '#3498db',
        }

        fig, axes = plt.subplots(
            3, 1, figsize=(18, 14),
            gridspec_kw={'height_ratios': [3, 2, 1.5]},
        )
        fig.suptitle(
            'Island Model Evolution Progress',
            fontsize=16, fontweight='bold',
        )

        ax_fitness, ax_diversity, ax_migration = axes

        # ── Panel 1: Best & Average fitness per island ──
        for ic in self.island_configs:
            stats_list = self.generation_stats.get(ic.name, [])
            if not stats_list:
                continue

            regime = ic.data_regime.lower()
            color = island_colors.get(regime, '#95a5a6')
            gens = [getattr(s, 'generation', i) + 1 for i, s in enumerate(stats_list)]
            best_fits = [s.best_fitness or 0 for s in stats_list]
            avg_fits = [s.avg_fitness or 0 for s in stats_list]

            ax_fitness.plot(
                gens, best_fits, '-o', color=color, linewidth=2,
                markersize=4, label=f'{ic.name} best', alpha=0.9,
            )
            ax_fitness.plot(
                gens, avg_fits, '--', color=color, linewidth=1,
                alpha=0.5, label=f'{ic.name} avg',
            )

        ax_fitness.set_ylabel('Fitness', fontsize=12)
        ax_fitness.set_title('Per-Island Fitness Evolution', fontsize=13)
        ax_fitness.legend(loc='upper left', fontsize=9, ncol=2)
        ax_fitness.grid(True, alpha=0.3)
        ax_fitness.set_xlim(left=1)

        # ── Panel 2: Diversity per island ──
        for ic in self.island_configs:
            stats_list = self.generation_stats.get(ic.name, [])
            if not stats_list:
                continue

            regime = ic.data_regime.lower()
            color = island_colors.get(regime, '#95a5a6')
            gens = [getattr(s, 'generation', i) + 1 for i, s in enumerate(stats_list)]
            divs = [s.genetic_diversity or 0 for s in stats_list]

            ax_diversity.plot(
                gens, divs, '-s', color=color, linewidth=1.5,
                markersize=3, label=ic.name, alpha=0.8,
            )

        ax_diversity.set_ylabel('Genetic Diversity', fontsize=12)
        ax_diversity.set_title('Per-Island Diversity', fontsize=13)
        ax_diversity.legend(loc='upper right', fontsize=9, ncol=2)
        ax_diversity.grid(True, alpha=0.3)
        ax_diversity.set_xlim(left=1)

        # ── Panel 3: Migration events timeline ──
        if self.migration_history:
            mig_gens = [e.generation + 1 for e in self.migration_history]
            mig_counts = [e.count for e in self.migration_history]
            mig_sources = [e.source for e in self.migration_history]

            # Color by source regime
            mig_colors = []
            source_regimes = {}
            for ic in self.island_configs:
                source_regimes[ic.name] = ic.data_regime.lower()
            for src in mig_sources:
                regime = source_regimes.get(src, 'balanced')
                mig_colors.append(island_colors.get(regime, '#95a5a6'))

            ax_migration.bar(
                mig_gens, mig_counts, color=mig_colors, alpha=0.7, width=0.6,
            )
            ax_migration.set_ylabel('Migrants', fontsize=12)
            ax_migration.set_title('Migration Events', fontsize=13)
            ax_migration.grid(True, alpha=0.3, axis='y')
        else:
            ax_migration.text(
                0.5, 0.5, 'No migration events',
                ha='center', va='center', fontsize=14, color='gray',
                transform=ax_migration.transAxes,
            )

        ax_migration.set_xlabel('Generation', fontsize=12)
        ax_migration.set_xlim(left=0.5)

        # Island legend
        legend_patches = []
        for ic in self.island_configs:
            regime = ic.data_regime.lower()
            color = island_colors.get(regime, '#95a5a6')
            legend_patches.append(
                mpatches.Patch(color=color, alpha=0.7,
                               label=f'{ic.name} ({ic.data_regime})')
            )
        ax_migration.legend(handles=legend_patches, loc='upper right', fontsize=9)

        plt.tight_layout()

        output_dir = Path(self.config.get('output_dir', 'genetic_algorithm/output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output_dir / 'island_evolution.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.info("Evolution plot saved to %s", chart_path)
        print(f"\n  📊 Island evolution plot saved: {chart_path.absolute()}\n")

    def _save_generation_stats_csv(self):
        """Save per-island generation stats to CSV for post-run analysis."""
        import csv

        output_dir = Path("genetic_algorithm/output/island_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "island_generation_stats.csv"

        fieldnames = [
            'island', 'generation', 'size', 'best_fitness', 'avg_fitness',
            'worst_fitness', 'median_fitness', 'best_raw_fitness',
            'avg_raw_fitness', 'genetic_diversity', 'diversity_score',
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for island_name, stats_list in self.generation_stats.items():
                for stats in stats_list:
                    writer.writerow({
                        'island': island_name,
                        'generation': stats.generation,
                        'size': stats.size,
                        'best_fitness': f"{stats.best_fitness:.6f}" if stats.best_fitness is not None else '',
                        'avg_fitness': f"{stats.avg_fitness:.6f}" if stats.avg_fitness is not None else '',
                        'worst_fitness': f"{stats.worst_fitness:.6f}" if stats.worst_fitness is not None else '',
                        'median_fitness': f"{stats.median_fitness:.6f}" if stats.median_fitness is not None else '',
                        'best_raw_fitness': f"{stats.best_raw_fitness:.6f}" if stats.best_raw_fitness is not None else '',
                        'avg_raw_fitness': f"{stats.avg_raw_fitness:.6f}" if stats.avg_raw_fitness is not None else '',
                        'genetic_diversity': f"{stats.genetic_diversity:.6f}" if stats.genetic_diversity is not None else '',
                        'diversity_score': f"{stats.diversity_score:.6f}" if stats.diversity_score is not None else '',
                    })

        self.logger.info("Generation stats CSV saved to %s", csv_path)
        print(f"  📊 Generation stats CSV: {csv_path.absolute()}")

    def _save_llm_report(self):
        """Generate and save a comprehensive LLM contribution report."""
        output_dir = Path("genetic_algorithm/output/island_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            'per_island': {},
            'provider_stats': {},
            'origin_breakdown': {},
            'summary': {},
        }

        total_llm = 0
        total_ga = 0
        total_elite = 0
        provider_totals: Dict[str, int] = {}

        for ic in self.island_configs:
            island_name = ic.name
            ga = self.islands.get(island_name)
            pop = self.island_populations.get(island_name)

            island_report: Dict[str, Any] = {
                'regime': ic.data_regime,
                'llm_enabled': False,
                'designer_stats': {},
                'origin_counts': {},
                'provider_counts': {},
                'top_llm_individuals': [],
            }

            # Gather designer stats if available
            if ga and ga.llm_enabled and ga.strategy_designer:
                island_report['llm_enabled'] = True
                island_report['designer_stats'] = dict(ga.strategy_designer.stats)
                island_report['designer_stats']['calls_by_type'] = dict(
                    ga.strategy_designer.stats.get('calls_by_type', {})
                )

                # Router-level stats
                provider = ga.strategy_designer.provider
                if hasattr(provider, 'get_router_stats'):
                    router_stats = provider.get_router_stats()
                    island_report['router_stats'] = router_stats
                    # Accumulate provider stats globally
                    for pname, pstats in router_stats.get('stats', {}).items():
                        if pname not in provider_totals:
                            provider_totals[pname] = 0
                        provider_totals[pname] += pstats.get('successes', 0)

            # Count origins in current population
            if pop:
                origin_counts: Dict[str, int] = {}
                provider_counts: Dict[str, int] = {}
                llm_individuals = []

                for ind in pop.individuals:
                    origin = ind.metrics.get('origin', 'unknown')
                    origin_counts[origin] = origin_counts.get(origin, 0) + 1

                    if origin.startswith('llm_'):
                        total_llm += 1
                        prov = ind.metrics.get('llm_provider', 'unknown')
                        provider_counts[prov] = provider_counts.get(prov, 0) + 1
                        llm_individuals.append(ind)
                    elif origin == 'elite':
                        total_elite += 1
                    else:
                        total_ga += 1

                island_report['origin_counts'] = origin_counts
                island_report['provider_counts'] = provider_counts

                # Top LLM individuals by fitness
                llm_individuals.sort(
                    key=lambda x: x.raw_fitness or 0, reverse=True,
                )
                island_report['top_llm_individuals'] = [
                    {
                        'fitness': ind.raw_fitness,
                        'origin': ind.metrics.get('origin', ''),
                        'provider': ind.metrics.get('llm_provider', ''),
                        'profit': ind.metrics.get('profit', 0),
                        'sharpe': ind.metrics.get('sharpe_ratio', 0),
                        'indicators': [
                            str(i) for i in (ind.strategy_gene.indicators[:5]
                                              if ind.strategy_gene else [])
                        ],
                    }
                    for ind in llm_individuals[:3]
                ]

            report['per_island'][island_name] = island_report

        # Global summary
        report['provider_stats'] = {
            name: count for name, count in sorted(provider_totals.items())
        }
        report['summary'] = {
            'total_llm_in_final_pop': total_llm,
            'total_ga_in_final_pop': total_ga,
            'total_elite_in_final_pop': total_elite,
            'llm_ratio': f"{total_llm / max(1, total_llm + total_ga + total_elite):.1%}",
        }

        # Save JSON report
        json_path = output_dir / "llm_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary to console
        print("")
        print("=" * 70)
        print("  LLM CONTRIBUTION REPORT")
        print("=" * 70)
        for island_name, ir in report['per_island'].items():
            origins = ir.get('origin_counts', {})
            llm_count = sum(v for k, v in origins.items() if k.startswith('llm_'))
            total = sum(origins.values()) or 1
            print(f"  {island_name:12s} (regime={ir['regime']:10s}): "
                  f"LLM={llm_count}/{total} ({llm_count/total:.0%})")
            if ir.get('provider_counts'):
                for prov, cnt in sorted(ir['provider_counts'].items()):
                    print(f"    {prov}: {cnt} individuals")
        if provider_totals:
            print(f"\n  Provider Success Counts (from router):")
            for pname, count in sorted(provider_totals.items()):
                print(f"    {pname}: {count} successful calls")
        print(f"\n  Final Population Composition: "
              f"LLM={total_llm} GA={total_ga} elite={total_elite} "
              f"(LLM ratio: {report['summary']['llm_ratio']})")
        print("=" * 70)

        self.logger.info("LLM report saved to %s", json_path)
        print(f"  📊 LLM report: {json_path.absolute()}\n")

    def _save_strategy_snapshot(
        self,
        population: Population,
        island_name: str,
        generation: int,
    ):
        """Save the best strategy from this island/generation as a JSON snapshot."""
        best = population.get_best(1)
        if not best:
            return

        ind = best[0]
        if ind.strategy_gene is None:
            return

        snapshot_dir = Path("genetic_algorithm/output/island_strategies") / island_name
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot = {
            'island': island_name,
            'generation': generation,
            'fitness': ind.raw_fitness,
            'metrics': {
                k: v for k, v in ind.metrics.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
            'strategy': ind.strategy_gene.to_dict(),
        }

        filepath = snapshot_dir / f"gen_{generation:03d}_best.json"
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)

    def _save_results(self, results: Dict[str, List[Individual]]):
        """Save island results to JSON file."""
        output_dir = Path("genetic_algorithm/output/island_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            'island_stats': {
                name: {
                    'regime': ist.regime,
                    'best_fitness': ist.best_fitness,
                    'best_profit': ist.best_profit,
                    'avg_fitness': ist.avg_fitness,
                    'generations': ist.generations_completed,
                    'migrants_sent': ist.migrants_sent,
                    'migrants_received': ist.migrants_received,
                }
                for name, ist in self.island_stats.items()
            },
            'migration_events': len(self.migration_history),
            'generations': self.generations,
            'islands': len(self.islands),
        }

        filepath = output_dir / "island_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info("Results saved to %s", filepath)

    def _print_data_usage_summary(
        self,
        regime_map: Dict[str, Dict[str, List['RegimeSegment']]],
    ):
        """Print a detailed text summary of how data is assigned to each island."""
        print("")
        print("=" * 70)
        print("  ISLAND DATA USAGE SUMMARY")
        print("=" * 70)

        for regime_key in ['bullish', 'bearish', 'sideways']:
            data = regime_map.get(regime_key, {})
            opt_segs = data.get('optimization', [])
            hold_segs = data.get('holdout', [])

            island_name = f"{regime_key.capitalize()} Island"
            print(f"\n  {island_name}")
            print(f"  {'─' * 40}")

            if opt_segs:
                total_days = sum(s.duration_days for s in opt_segs)
                print(f"    Optimization segments: {len(opt_segs)} ({total_days} days)")
                for i, seg in enumerate(opt_segs, 1):
                    conf = f" (confidence={seg.confidence:.0%})" if hasattr(seg, 'confidence') and seg.confidence else ""
                    print(f"      {i}. {seg.start_date.strftime('%Y-%m-%d')} → "
                          f"{seg.end_date.strftime('%Y-%m-%d')} "
                          f"({seg.duration_days}d){conf}")
            else:
                print("    Optimization segments: none")

            if hold_segs:
                total_days = sum(s.duration_days for s in hold_segs)
                print(f"    Holdout segments:      {len(hold_segs)} ({total_days} days)")
            else:
                print("    Holdout segments:      none")

        # Master island
        all_opt = []
        for regime_key in ['bullish', 'bearish', 'sideways']:
            all_opt.extend(regime_map.get(regime_key, {}).get('optimization', []))
        total_days = sum(s.duration_days for s in all_opt)
        print(f"\n  Master Island (all regimes)")
        print(f"  {'─' * 40}")
        print(f"    Uses ALL {len(all_opt)} optimization segments ({total_days} days)")
        print(f"    Trains a generalist strategy across all market conditions")

        print("")
        print("  HOW DATA IS USED PER ISLAND:")
        print("  " + "─" * 40)
        print("  • Specialist islands (bullish/bearish/sideways) receive ONLY")
        print("    segments matching their regime type for training.")
        print("  • The Master island receives ALL optimization segments,")
        print("    training a generalist that works across all conditions.")
        print("  • During migration, top performers from specialists are")
        print("    injected into the Master island (and vice versa),")
        print("    allowing cross-pollination of good gene combinations.")
        print("  • Holdout segments are reserved for future validation")
        print("    (not currently used during evolution).")
        print("=" * 70)
        print("")

    def _save_regime_segments_json(
        self,
        regime_map: Dict[str, Dict[str, List['RegimeSegment']]],
    ):
        """Save regime segments data to JSON for post-run analysis."""
        output_dir = Path("genetic_algorithm/output/island_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        serialized = {}
        for regime_key, data in regime_map.items():
            serialized[regime_key] = {}
            for split_type in ['optimization', 'holdout']:
                segments = data.get(split_type, [])
                serialized[regime_key][split_type] = [
                    {
                        'start_date': seg.start_date.strftime('%Y-%m-%d'),
                        'end_date': seg.end_date.strftime('%Y-%m-%d'),
                        'duration_days': seg.duration_days,
                        'regime': seg.regime.value if hasattr(seg.regime, 'value') else str(seg.regime),
                        'confidence': getattr(seg, 'confidence', None),
                    }
                    for seg in segments
                ]

        json_path = output_dir / "regime_segments.json"
        with open(json_path, 'w') as f:
            json.dump(serialized, f, indent=2)

        self.logger.info("Regime segments saved to %s", json_path)
        print(f"  📊 Regime segments JSON: {json_path.absolute()}")
