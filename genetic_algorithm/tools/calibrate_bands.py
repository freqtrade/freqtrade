"""
Score-Band Calibration Tool
===========================

Automatically finds the optimal ``bullish_min`` / ``bearish_max`` band
boundaries for score-band regime segmentation.

For a given pair + timeframe + timerange it:

1. Computes the continuous trend_score across the full dataset.
2. Sweeps band boundaries over a configurable grid (symmetric and/or
   asymmetric).
3. For each candidate, runs ``classify_periods_by_score()`` and evaluates
   a composite **data-quality score** made of:
   - **Balance**  – KL divergence from a uniform ⅓ distribution (lower = better)
   - **Coverage** – minimum bars across the three regimes (higher = better)
   - **Segment count** – number of segments per regime (≥3 preferred)
   - **Return separation** – mean return gap between bull & bear segments
   - **Stability** – how much the distribution changes on shifted sub-windows
4. Ranks candidates, prints a summary table, and saves the best config to
   JSON.

Usage (CLI)::

    python -m genetic_algorithm.tools.calibrate_bands \\
        --pair BTC/USDT \\
        --timeframe 4h \\
        --timerange 20230101-20260228 \\
        --method advanced_ensemble \\
        --sweep-timeframes          # optional: also sweep detection TFs

    # Multi-pair consensus:
    python -m genetic_algorithm.tools.calibrate_bands \\
        --pair BTC/USDT ETH/USDT SOL/USDT \\
        --timeframe 4h \\
        --timerange 20230101-20260228
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data-quality metric
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BandCandidate:
    """Result for one band-boundary candidate."""
    bullish_min: float
    bearish_max: float
    # Per-regime stats
    bull_segments: int = 0
    bear_segments: int = 0
    side_segments: int = 0
    bull_bars: int = 0
    bear_bars: int = 0
    side_bars: int = 0
    bull_pct: float = 0.0
    bear_pct: float = 0.0
    side_pct: float = 0.0
    # Return separation
    bull_mean_return: float = 0.0
    bear_mean_return: float = 0.0
    return_separation: float = 0.0
    # Composite quality score (0-1, higher = better)
    quality_score: float = 0.0
    # Sub-scores
    balance_score: float = 0.0
    coverage_score: float = 0.0
    segment_score: float = 0.0
    separation_score: float = 0.0
    stability_score: float = 0.0
    # Metadata
    pair: str = ''
    timeframe: str = ''
    method: str = ''


@dataclass
class CalibrationResult:
    """Full calibration result."""
    best: BandCandidate
    all_candidates: List[BandCandidate] = field(default_factory=list)
    timeframe_ranking: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience properties for quick access to best band settings
    @property
    def bullish_min(self) -> float:
        return self.best.bullish_min

    @property
    def bearish_max(self) -> float:
        return self.best.bearish_max

    @property
    def composite_score(self) -> float:
        return self.best.quality_score

    @property
    def bull_pct(self) -> float:
        return self.best.bull_pct

    @property
    def side_pct(self) -> float:
        return self.best.side_pct

    @property
    def bear_pct(self) -> float:
        return self.best.bear_pct


# ──────────────────────────────────────────────────────────────────────
# Core calibration engine
# ──────────────────────────────────────────────────────────────────────

class BandCalibrator:
    """
    Sweeps score-band boundaries and evaluates data quality for each.

    Parameters
    ----------
    method : str
        Regime detection method (e.g. 'advanced_ensemble').
    min_segment_days : int
        Passed to ``classify_periods_by_score()``.  Default 14.
    max_segment_days : int
        Passed to ``classify_periods_by_score()``.  Default 180.
    min_quality_segments : int
        Minimum segments per regime for a "good" score.  Default 3.
    min_quality_bars : int
        Minimum bars per regime for a "good" score.  Default 1500.
    """

    def __init__(
        self,
        method: str = 'advanced_ensemble',
        min_segment_days: int = 14,
        max_segment_days: int = 180,
        min_quality_segments: int = 3,
        min_quality_bars: int = 1500,
    ):
        self.method = method
        self.min_segment_days = min_segment_days
        self.max_segment_days = max_segment_days
        self.min_quality_segments = min_quality_segments
        self.min_quality_bars = min_quality_bars

    # ─────────────────────── public API ───────────────────────

    def calibrate(
        self,
        df: pd.DataFrame,
        pair: str = 'BTC/USDT',
        timeframe: str = '4h',
        symmetric: bool = True,
        asymmetric: bool = False,
        bull_range: Tuple[float, float] = (0.15, 0.60),
        bear_range: Tuple[float, float] = (-0.60, -0.15),
        step: float = 0.05,
    ) -> CalibrationResult:
        """
        Sweep band boundaries and rank candidates by data-quality score.

        Parameters
        ----------
        df : DataFrame
            OHLCV data for the pair/timeframe being calibrated.
        pair, timeframe : str
            Used for labeling only.
        symmetric : bool
            When True, sweeps ``bullish_min`` = ``-bearish_max`` together.
        asymmetric : bool
            When True, independently sweeps ``bullish_min`` and ``bearish_max``.
        bull_range : (min, max)
            Range for ``bullish_min`` sweep (applied as positive values).
        bear_range : (min, max)
            Range for ``bearish_max`` sweep (applied as negative values).
        step : float
            Grid step size.

        Returns
        -------
        CalibrationResult with ranked candidates and best pick.
        """
        from genetic_algorithm.utils.regime_detector import RegimeDetector

        detector = RegimeDetector(method=self.method)

        # Precompute the continuous trend score once — reused for all sweep points
        trend_scores = detector._compute_trend_score(df)
        vol_scores = detector._compute_volatility_score(df)

        candidates: List[BandCandidate] = []

        # Build grid
        grid_points: List[Tuple[float, float]] = []

        if symmetric:
            for bm in np.arange(bull_range[0], bull_range[1] + step / 2, step):
                bm = round(bm, 3)
                grid_points.append((bm, -bm))

        if asymmetric:
            for bm in np.arange(bull_range[0], bull_range[1] + step / 2, step):
                for bx in np.arange(bear_range[0], bear_range[1] + step / 2, step):
                    bm_r = round(bm, 3)
                    bx_r = round(bx, 3)
                    if (bm_r, bx_r) not in grid_points:
                        grid_points.append((bm_r, bx_r))

        logger.info(
            "Calibrating %s %s %s: %d grid points...",
            pair, timeframe, self.method, len(grid_points),
        )

        for bullish_min, bearish_max in grid_points:
            try:
                candidate = self._evaluate_candidate(
                    df=df,
                    detector=detector,
                    trend_scores=trend_scores,
                    vol_scores=vol_scores,
                    bullish_min=bullish_min,
                    bearish_max=bearish_max,
                    pair=pair,
                    timeframe=timeframe,
                )
                candidates.append(candidate)
            except Exception as e:
                logger.debug("Candidate (%.2f, %.2f) failed: %s", bullish_min, bearish_max, e)

        # Sort by quality score descending
        candidates.sort(key=lambda c: c.quality_score, reverse=True)

        best = candidates[0] if candidates else BandCandidate(0.35, -0.35)

        return CalibrationResult(
            best=best,
            all_candidates=candidates,
            metadata={
                'pair': pair,
                'timeframe': timeframe,
                'method': self.method,
                'n_bars': len(df),
                'grid_points': len(grid_points),
                'symmetric': symmetric,
                'asymmetric': asymmetric,
            },
        )

    def calibrate_multi_pair(
        self,
        pairs: List[str],
        timeframe: str,
        datadir: Path,
        timerange: Optional[str] = None,
        **kwargs,
    ) -> CalibrationResult:
        """
        Run calibration across multiple pairs and find consensus bands.

        Averages quality scores per candidate across pairs, so the optimal
        bands work well on *all* configured pairs.
        """
        from genetic_algorithm.utils.regime_detector import load_ohlcv_data

        # Collect per-pair results
        pair_results: Dict[str, CalibrationResult] = {}
        for pair in pairs:
            logger.info("Loading %s %s ...", pair, timeframe)
            df = load_ohlcv_data(pair, timeframe, datadir, timerange)
            if df.empty:
                logger.warning("No data for %s, skipping", pair)
                continue
            result = self.calibrate(df, pair=pair, timeframe=timeframe, **kwargs)
            pair_results[pair] = result

        if not pair_results:
            raise ValueError("No pairs produced data for calibration")

        # Build consensus: average quality score per band config
        # Key: (bullish_min, bearish_max)  →  list of quality scores
        consensus: Dict[Tuple[float, float], List[float]] = {}
        all_candidates_map: Dict[Tuple[float, float], BandCandidate] = {}

        for pair, result in pair_results.items():
            for c in result.all_candidates:
                key = (c.bullish_min, c.bearish_max)
                consensus.setdefault(key, []).append(c.quality_score)
                all_candidates_map[key] = c  # last pair's candidate for metadata

        # Rank by mean quality score across pairs
        ranked = sorted(
            consensus.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True,
        )

        best_key = ranked[0][0] if ranked else (0.35, -0.35)
        best_candidate = all_candidates_map.get(best_key, BandCandidate(0.35, -0.35))
        best_candidate.quality_score = float(np.mean(consensus.get(best_key, [0.0])))

        # Build full consensus list
        consensus_candidates = []
        for key, scores in ranked:
            c = BandCandidate(
                bullish_min=key[0],
                bearish_max=key[1],
                quality_score=float(np.mean(scores)),
            )
            consensus_candidates.append(c)

        return CalibrationResult(
            best=best_candidate,
            all_candidates=consensus_candidates,
            metadata={
                'pairs': pairs,
                'timeframe': timeframe,
                'method': self.method,
                'n_pairs': len(pair_results),
                'consensus': True,
            },
        )

    def sweep_timeframes(
        self,
        pair: str,
        datadir: Path,
        timerange: Optional[str] = None,
        timeframes: Optional[List[str]] = None,
        bullish_min: float = 0.35,
        bearish_max: float = -0.35,
    ) -> Dict[str, CalibrationResult]:
        """
        Run full band calibration on each timeframe and return a
        ``CalibrationResult`` per TF.

        Returns dict of ``{timeframe: CalibrationResult}``, sorted
        descending by composite quality score.  The best TF's
        ``CalibrationResult`` contains the optimal bands *for that TF*.
        """
        from genetic_algorithm.utils.regime_detector import load_ohlcv_data

        if timeframes is None:
            timeframes = ['30m', '1h', '4h', '1d']

        results: Dict[str, CalibrationResult] = {}

        for tf in timeframes:
            try:
                df = load_ohlcv_data(pair, tf, datadir, timerange)
                if df.empty or len(df) < 200:
                    logger.warning("Insufficient data for %s %s, skipping", pair, tf)
                    continue

                result = self.calibrate(df, pair=pair, timeframe=tf)
                results[tf] = result
                logger.info(
                    "  %s: score=%.4f  bands=(bull>=%.2f, bear<=%.2f)",
                    tf, result.composite_score,
                    result.bullish_min, result.bearish_max,
                )
            except Exception as e:
                logger.warning("Timeframe %s failed: %s", tf, e)

        # Sort descending by composite score
        results = dict(
            sorted(results.items(),
                   key=lambda x: x[1].composite_score, reverse=True)
        )
        return results

    # ─────────────────────── internals ────────────────────────

    def _evaluate_candidate(
        self,
        df: pd.DataFrame,
        detector,
        trend_scores: pd.Series,
        vol_scores: pd.Series,
        bullish_min: float,
        bearish_max: float,
        pair: str = '',
        timeframe: str = '',
    ) -> BandCandidate:
        """Evaluate a single band-boundary candidate."""
        segments = detector.classify_periods_by_score(
            df=df,
            bullish_min=bullish_min,
            bearish_max=bearish_max,
            min_segment_days=self.min_segment_days,
            max_segment_days=self.max_segment_days,
            merge_threshold_days=7,
            embargo_days=3,
        )

        c = BandCandidate(
            bullish_min=bullish_min,
            bearish_max=bearish_max,
            pair=pair,
            timeframe=timeframe,
            method=self.method,
        )

        if not segments:
            return c

        # Aggregate per-regime stats
        for seg in segments:
            regime = seg.regime.value.lower()
            bars = seg.metadata.get('bar_count', 0)
            ret = seg.metadata.get('total_return', 0.0)
            if regime == 'bullish':
                c.bull_segments += 1
                c.bull_bars += bars
                c.bull_mean_return += ret
            elif regime == 'bearish':
                c.bear_segments += 1
                c.bear_bars += bars
                c.bear_mean_return += ret
            else:
                c.side_segments += 1
                c.side_bars += bars

        total_bars = c.bull_bars + c.bear_bars + c.side_bars
        if total_bars > 0:
            c.bull_pct = c.bull_bars / total_bars
            c.bear_pct = c.bear_bars / total_bars
            c.side_pct = c.side_bars / total_bars

        # Average returns
        if c.bull_segments > 0:
            c.bull_mean_return /= c.bull_segments
        if c.bear_segments > 0:
            c.bear_mean_return /= c.bear_segments

        c.return_separation = c.bull_mean_return - c.bear_mean_return

        # ── Sub-scores ──

        # 1. Balance: KL-divergence from uniform 1/3 (lower = better)
        c.balance_score = self._score_balance(c.bull_pct, c.bear_pct, c.side_pct)

        # 2. Coverage: min bars per regime vs threshold
        c.coverage_score = self._score_coverage(c.bull_bars, c.bear_bars, c.side_bars)

        # 3. Segment count: reward ≥ min_quality_segments per regime
        c.segment_score = self._score_segments(c.bull_segments, c.bear_segments, c.side_segments)

        # 4. Return separation: bull should outperform bear
        c.separation_score = self._score_separation(c.return_separation)

        # 5. Stability: consistency across sub-windows
        c.stability_score = self._score_stability(
            df, detector, trend_scores, bullish_min, bearish_max
        )

        # ── Composite ──
        # Weights chosen to emphasize balance and coverage (the main problems)
        c.quality_score = (
            0.25 * c.balance_score
            + 0.25 * c.coverage_score
            + 0.20 * c.segment_score
            + 0.15 * c.separation_score
            + 0.15 * c.stability_score
        )

        return c

    def _score_balance(self, bull_pct: float, bear_pct: float, side_pct: float) -> float:
        """Score how balanced the distribution is (1.0 = perfect ⅓ each)."""
        if bull_pct == 0 and bear_pct == 0 and side_pct == 0:
            return 0.0

        target = 1.0 / 3.0
        # KL divergence from uniform — but can be infinite for zero entries
        # Use a smoothed version
        eps = 0.01
        p = np.array([max(bull_pct, eps), max(bear_pct, eps), max(side_pct, eps)])
        p = p / p.sum()
        q = np.array([target, target, target])

        kl = float(np.sum(p * np.log(p / q)))
        # Map KL to [0, 1]: KL=0 → 1.0, KL=1 → ~0.37
        return float(np.exp(-kl))

    def _score_coverage(self, bull_bars: int, bear_bars: int, side_bars: int) -> float:
        """Score based on minimum bars across regimes."""
        min_bars = min(bull_bars, bear_bars, side_bars)
        # Sigmoid mapping: min_bars at threshold → 0.5, above → approaches 1
        threshold = self.min_quality_bars
        if threshold == 0:
            return 1.0
        return float(1.0 / (1.0 + np.exp(-5.0 * (min_bars / threshold - 1.0))))

    def _score_segments(self, bull_seg: int, bear_seg: int, side_seg: int) -> float:
        """Score based on segment count per regime."""
        min_seg = min(bull_seg, bear_seg, side_seg)
        threshold = self.min_quality_segments
        if threshold == 0:
            return 1.0
        # Linear ramp: 0 at 0, 1.0 at threshold*2, capped
        return float(min(1.0, min_seg / max(threshold, 1)))

    def _score_separation(self, return_sep: float) -> float:
        """Score based on return separation between bull/bear."""
        # We want positive separation (bull > bear in returns)
        # Tanh mapping: 0 → 0.5, positive gap → approaches 1.0
        # Typical separation might be 5-20% over segment duration
        return float(0.5 + 0.5 * np.tanh(return_sep * 5))

    def _score_stability(
        self,
        df: pd.DataFrame,
        detector,
        trend_scores: pd.Series,
        bullish_min: float,
        bearish_max: float,
        n_windows: int = 3,
    ) -> float:
        """
        Score how stable the distribution is across sub-windows.

        Splits the data into ``n_windows`` overlapping sub-windows and checks
        that each window has all 3 regimes represented.
        """
        n = len(df)
        window_size = n // 2  # 50% overlapping windows
        if window_size < 500 or n_windows < 2:
            return 0.5  # not enough data for stability estimate

        step = (n - window_size) // (n_windows - 1)
        distributions = []

        for i in range(n_windows):
            start = i * step
            end = start + window_size
            sub_df = df.iloc[start:end]

            if len(sub_df) < 200:
                continue

            try:
                sub_segments = detector.classify_periods_by_score(
                    df=sub_df,
                    bullish_min=bullish_min,
                    bearish_max=bearish_max,
                    min_segment_days=max(7, self.min_segment_days // 2),
                    max_segment_days=self.max_segment_days,
                    merge_threshold_days=5,
                    embargo_days=2,
                )
                dist = {'bullish': 0, 'bearish': 0, 'sideways': 0}
                for seg in sub_segments:
                    r = seg.regime.value.lower()
                    if r in dist:
                        dist[r] += seg.metadata.get('bar_count', 0)
                total = sum(dist.values())
                if total > 0:
                    distributions.append({k: v / total for k, v in dist.items()})
            except Exception:
                continue

        if len(distributions) < 2:
            return 0.5

        # Stability = 1 - mean pairwise L1 distance between distributions
        l1_distances = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                d = sum(
                    abs(distributions[i][k] - distributions[j][k])
                    for k in ['bullish', 'bearish', 'sideways']
                )
                l1_distances.append(d)

        mean_l1 = np.mean(l1_distances)
        # L1 max is 2.0 (completely different), 0.0 is identical
        return float(max(0.0, 1.0 - mean_l1 / 2.0))


# ──────────────────────────────────────────────────────────────────────
# Pretty-printing
# ──────────────────────────────────────────────────────────────────────

def print_calibration_table(result: CalibrationResult, top_n: int = 10) -> None:
    """Print a formatted summary table of top candidates."""
    print("\n" + "=" * 95)
    print("  SCORE-BAND CALIBRATION RESULTS")
    print("=" * 95)
    meta = result.metadata
    print(f"  Method: {meta.get('method', '?')}  |  Bars: {meta.get('n_bars', '?')}  |  "
          f"Grid: {meta.get('grid_points', '?')} points")
    if 'pairs' in meta:
        print(f"  Pairs: {', '.join(meta['pairs'])}  (consensus)")
    else:
        print(f"  Pair: {meta.get('pair', '?')} {meta.get('timeframe', '?')}")
    print("-" * 95)
    print(f"  {'bull_min':>8s} {'bear_max':>8s} │ {'quality':>7s} │ "
          f"{'balance':>7s} {'cover':>7s} {'segs':>7s} {'sep':>7s} {'stab':>7s} │ "
          f"{'bull%':>6s} {'side%':>6s} {'bear%':>6s} │ "
          f"{'b_seg':>5s} {'s_seg':>5s} {'r_seg':>5s}")
    print("-" * 95)

    for i, c in enumerate(result.all_candidates[:top_n]):
        marker = " ★" if i == 0 else ""
        print(
            f"  {c.bullish_min:8.3f} {c.bearish_max:8.3f} │ "
            f"{c.quality_score:7.4f} │ "
            f"{c.balance_score:7.4f} {c.coverage_score:7.4f} "
            f"{c.segment_score:7.4f} {c.separation_score:7.4f} "
            f"{c.stability_score:7.4f} │ "
            f"{c.bull_pct * 100:5.1f}% {c.side_pct * 100:5.1f}% "
            f"{c.bear_pct * 100:5.1f}% │ "
            f"{c.bull_segments:5d} {c.side_segments:5d} "
            f"{c.bear_segments:5d}{marker}"
        )

    print("-" * 95)
    best = result.best
    print(f"\n  ★ BEST: bullish_min={best.bullish_min:.3f}, "
          f"bearish_max={best.bearish_max:.3f}  "
          f"(quality={best.quality_score:.4f})")
    print(f"    Distribution: bull={best.bull_pct * 100:.1f}%, "
          f"sideways={best.side_pct * 100:.1f}%, "
          f"bear={best.bear_pct * 100:.1f}%")
    print(f"    Segments: bull={best.bull_segments}, "
          f"sideways={best.side_segments}, "
          f"bear={best.bear_segments}")
    print()


def save_calibration_result(
    result: CalibrationResult,
    output_path: Path,
) -> None:
    """Save calibration result to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'best': {
            'bullish_min': result.best.bullish_min,
            'bearish_max': result.best.bearish_max,
            'quality_score': result.best.quality_score,
            'balance_score': result.best.balance_score,
            'coverage_score': result.best.coverage_score,
            'segment_score': result.best.segment_score,
            'separation_score': result.best.separation_score,
            'stability_score': result.best.stability_score,
            'bull_pct': result.best.bull_pct,
            'bear_pct': result.best.bear_pct,
            'side_pct': result.best.side_pct,
            'bull_segments': result.best.bull_segments,
            'bear_segments': result.best.bear_segments,
            'side_segments': result.best.side_segments,
            'return_separation': result.best.return_separation,
        },
        'top_10': [
            {
                'bullish_min': c.bullish_min,
                'bearish_max': c.bearish_max,
                'quality_score': c.quality_score,
            }
            for c in result.all_candidates[:10]
        ],
        'timeframe_ranking': result.timeframe_ranking,
        'metadata': result.metadata,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Calibration saved to %s", output_path)


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score-Band Calibration Tool — find optimal regime band boundaries",
    )
    parser.add_argument('--pair', nargs='+', default=['BTC/USDT'],
                        help='Trading pair(s) to calibrate on')
    parser.add_argument('--timeframe', default='4h',
                        help='Detection timeframe (default: 4h)')
    parser.add_argument('--timerange', default=None,
                        help='Timerange in YYYYMMDD-YYYYMMDD format')
    parser.add_argument('--datadir', default='user_data/data/binance',
                        help='Path to OHLCV data directory')
    parser.add_argument('--method', default='advanced_ensemble',
                        help='Regime detection method (default: advanced_ensemble)')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Grid step size (default: 0.05)')
    parser.add_argument('--asymmetric', action='store_true',
                        help='Also sweep asymmetric band boundaries')
    parser.add_argument('--sweep-timeframes', action='store_true',
                        help='Also evaluate detection quality across timeframes')
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: auto)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top candidates to display')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    from genetic_algorithm.utils.regime_detector import load_ohlcv_data

    datadir = Path(args.datadir)

    calibrator = BandCalibrator(method=args.method)

    if len(args.pair) == 1:
        # Single-pair calibration
        pair = args.pair[0]
        logger.info("Loading %s %s from %s ...", pair, args.timeframe, datadir)
        df = load_ohlcv_data(pair, args.timeframe, datadir, args.timerange)
        if df.empty:
            print(f"ERROR: No data for {pair} {args.timeframe}")
            sys.exit(1)

        logger.info("Loaded %d bars", len(df))

        result = calibrator.calibrate(
            df=df,
            pair=pair,
            timeframe=args.timeframe,
            symmetric=True,
            asymmetric=args.asymmetric,
            step=args.step,
        )
    else:
        # Multi-pair consensus
        result = calibrator.calibrate_multi_pair(
            pairs=args.pair,
            timeframe=args.timeframe,
            datadir=datadir,
            timerange=args.timerange,
            symmetric=True,
            asymmetric=args.asymmetric,
            step=args.step,
        )

    # Print results
    print_calibration_table(result, top_n=args.top_n)

    # Optional: sweep timeframes
    if args.sweep_timeframes:
        print("\n" + "=" * 50)
        print("  TIMEFRAME QUALITY RANKING")
        print("=" * 50)
        tf_ranking = calibrator.sweep_timeframes(
            pair=args.pair[0],
            datadir=datadir,
            timerange=args.timerange,
            bullish_min=result.best.bullish_min,
            bearish_max=result.best.bearish_max,
        )
        result.timeframe_ranking = tf_ranking
        for rank, (tf, score) in enumerate(tf_ranking.items(), 1):
            marker = " ★" if rank == 1 else ""
            print(f"  {rank}. {tf:>4s}: quality={score:.4f}{marker}")
        print()

    # Save
    output_path = Path(
        args.output or f"genetic_algorithm/output/calibration_{args.method}.json"
    )
    save_calibration_result(result, output_path)

    print(f"  Results saved to: {output_path}")
    print()


if __name__ == '__main__':
    main()
