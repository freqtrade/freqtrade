#!/usr/bin/env python3
"""
Regime Detection Comparison — Rule-Based vs ML (LightGBM)

Generates side-by-side plots comparing the old rule-based regime detection
methods with the new Phase 1B ML classifier on real market data.

Plots generated:
  1. Price chart with regime overlay (rule-based vs ML) per pair
  2. Agreement heatmap: how often each method pair agrees
  3. Regime distribution bar chart (rule vs ML vs ensemble)
  4. Confidence distribution for the ML classifier
  5. Transition matrix comparison
  6. Summary statistics table

Usage:
    python genetic_algorithm/visualization/compare_regime_methods.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from genetic_algorithm.ml.regime_detector import (
    INT_TO_REGIME,
    REGIME_TO_INT,
    MLRegimeDetector,
)
from genetic_algorithm.utils.regime_detector import (
    RegimeDetector,
    RegimeType,
    load_ohlcv_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────
DATA_DIR = Path("user_data/data/binance")
MODEL_PATH = "genetic_algorithm/ml/models/regime_lgbm.pkl"
OUTPUT_DIR = Path("genetic_algorithm/visualization/regime_comparison_output")
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAME = "4h"
TIMERANGE = "20240101-20260101"  # last 2 years for clarity

# Rule-based methods to compare
RULE_METHODS = [
    "sma_adx",
    "adx_di_hysteresis",
    "rolling_returns",
    "bollinger",
    "volatility_cluster",
    "ensemble",
]

# Color scheme for regimes
REGIME_COLORS = {
    RegimeType.BULLISH:   "#2ecc71",  # green
    RegimeType.BEARISH:   "#e74c3c",  # red
    RegimeType.SIDEWAYS:  "#f39c12",  # amber
    RegimeType.VOLATILE:  "#9b59b6",  # purple
    RegimeType.UNCERTAIN: "#95a5a6",  # gray
}
REGIME_LABELS = {
    RegimeType.BULLISH:   "Bullish",
    RegimeType.BEARISH:   "Bearish",
    RegimeType.SIDEWAYS:  "Sideways",
    RegimeType.VOLATILE:  "Volatile",
    RegimeType.UNCERTAIN: "Uncertain",
}


def load_data(pair: str) -> pd.DataFrame:
    """Load OHLCV data for a pair."""
    from freqtrade.configuration import TimeRange
    tr = TimeRange.parse_timerange(TIMERANGE) if TIMERANGE else None
    return load_ohlcv_data(pair=pair, timeframe=TIMEFRAME, datadir=DATA_DIR, timerange=TIMERANGE)


def run_all_methods(df: pd.DataFrame) -> dict:
    """
    Run all rule-based methods + ML detector on the data.
    Returns dict: method_name -> pd.Series of RegimeType.
    """
    results = {}

    # Rule-based methods
    for method in RULE_METHODS:
        logger.info(f"  Running rule-based method: {method}")
        try:
            detector = RegimeDetector(method=method)
            results[f"rule_{method}"] = detector.detect(df)
        except Exception as e:
            logger.warning(f"  Method {method} failed: {e}")

    # ML LightGBM
    logger.info("  Running ML (LightGBM) detector...")
    try:
        ml_detector = MLRegimeDetector(
            model_path=MODEL_PATH,
            feature_mode="raw_only",
        )
        regimes_ml, confidence_ml = ml_detector.detect_with_confidence(df)
        results["ml_lgbm"] = regimes_ml
        results["_ml_confidence"] = confidence_ml  # store confidence separately
    except Exception as e:
        logger.error(f"  ML detector failed: {e}")

    return results


def regime_to_int_series(series: pd.Series) -> pd.Series:
    """Convert RegimeType series to integer for numerical operations."""
    mapping = {
        RegimeType.BULLISH: 0,
        RegimeType.BEARISH: 1,
        RegimeType.SIDEWAYS: 2,
        RegimeType.VOLATILE: 3,
        RegimeType.UNCERTAIN: -1,
    }
    return series.map(mapping)


def _fill_regime_background(ax, dates, regimes, alpha=0.3):
    """
    Efficient regime background coloring using contiguous segment fill_between.
    Groups consecutive bars with the same regime into segments and draws one
    axvspan per segment instead of one per bar.
    """
    if len(dates) == 0:
        return

    segments = []
    current_regime = regimes.iloc[0]
    seg_start = 0

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current_regime:
            segments.append((seg_start, i - 1, current_regime))
            current_regime = regimes.iloc[i]
            seg_start = i
    segments.append((seg_start, len(regimes) - 1, current_regime))

    for s_start, s_end, regime in segments:
        color = REGIME_COLORS.get(regime, "#cccccc")
        left = dates[s_start]
        right = dates[min(s_end + 1, len(dates) - 1)]
        ax.axvspan(left, right, alpha=alpha, color=color, linewidth=0)


# ── Plot 1: Price + Regime Overlay ──────────────────────────────────

def plot_price_with_regimes(df: pd.DataFrame, results: dict, pair: str, output_dir: Path):
    """
    Price chart with colored background for regime classification.
    Shows the top 3 rule-based methods + ML side by side.
    """
    # Choose methods to show
    methods_to_show = []
    for name in ["rule_ensemble", "rule_adx_di_hysteresis", "rule_rolling_returns", "ml_lgbm"]:
        if name in results:
            methods_to_show.append(name)

    n_methods = len(methods_to_show)
    if n_methods == 0:
        return

    fig, axes = plt.subplots(n_methods, 1, figsize=(20, 4 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]

    close = df["close"] if "close" in df.columns else df["Close"]
    dates = df.index

    for ax, method_name in zip(axes, methods_to_show):
        regimes = results[method_name]

        # Plot price
        ax.plot(dates, close, color="black", linewidth=0.6, alpha=0.9)

        # Efficient colored background: group contiguous regime segments
        _fill_regime_background(ax, dates, regimes, alpha=0.25)

        # Title
        display_name = method_name.replace("rule_", "Rule: ").replace("ml_lgbm", "ML (LightGBM)")
        ax.set_title(display_name, fontsize=13, fontweight="bold", loc="left")
        ax.set_ylabel("Price ($)")
        ax.grid(True, alpha=0.2)

        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Legend
    patches = [mpatches.Patch(color=REGIME_COLORS[r], label=REGIME_LABELS[r], alpha=0.5)
               for r in [RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS, RegimeType.VOLATILE]]
    fig.legend(handles=patches, loc="upper center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 1.02), frameon=True, fancybox=True)

    fig.suptitle(f"Regime Classification Comparison — {pair} ({TIMEFRAME})",
                 fontsize=16, fontweight="bold", y=1.06)
    plt.tight_layout()

    out = output_dir / f"regime_overlay_{pair.replace('/', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Plot 2: Agreement Heatmap ──────────────────────────────────────

def plot_agreement_heatmap(all_results: dict, output_dir: Path):
    """
    Heatmap showing pairwise agreement % between all methods across all pairs.
    """
    # Collect all method-pair regime series
    all_series = {}
    for pair_name, results in all_results.items():
        for method, series in results.items():
            if method.startswith("_"):
                continue
            key = f"{method}"
            if key not in all_series:
                all_series[key] = []
            all_series[key].append(regime_to_int_series(series))

    # Concatenate across pairs
    method_names = list(all_series.keys())
    concat_series = {}
    for m in method_names:
        concat_series[m] = pd.concat(all_series[m], ignore_index=True)

    n = len(method_names)
    agreement_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            s1 = concat_series[method_names[i]]
            s2 = concat_series[method_names[j]]
            valid = (s1 >= 0) & (s2 >= 0)
            if valid.sum() > 0:
                agreement_matrix[i, j] = (s1[valid] == s2[valid]).mean() * 100
            else:
                agreement_matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    display_names = [m.replace("rule_", "").replace("ml_lgbm", "ML LightGBM").title()
                     for m in method_names]

    im = ax.imshow(agreement_matrix, cmap="RdYlGn", vmin=30, vmax=100, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(display_names, fontsize=10)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = agreement_matrix[i, j]
            color = "white" if val < 50 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Agreement %", shrink=0.8)
    ax.set_title("Pairwise Method Agreement (across all pairs)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = output_dir / "method_agreement_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Plot 3: Regime Distribution ───────────────────────────────────

def plot_regime_distribution(all_results: dict, output_dir: Path):
    """
    Bar chart showing regime distribution for each method (aggregated across pairs).
    """
    regime_types = [RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS, RegimeType.VOLATILE]
    regime_names = [REGIME_LABELS[r] for r in regime_types]

    # Aggregate counts
    method_counts = {}
    for pair_name, results in all_results.items():
        for method, series in results.items():
            if method.startswith("_"):
                continue
            if method not in method_counts:
                method_counts[method] = {r: 0 for r in regime_types}
            for r in regime_types:
                method_counts[method][r] += (series == r).sum()

    methods = list(method_counts.keys())
    display_names = [m.replace("rule_", "").replace("ml_lgbm", "ML LightGBM").title()
                     for m in methods]

    # Compute percentages
    n_methods = len(methods)
    x = np.arange(len(regime_types))
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, method in enumerate(methods):
        total = sum(method_counts[method].values())
        pcts = [method_counts[method][r] / max(total, 1) * 100 for r in regime_types]
        bars = ax.bar(x + i * width - (n_methods - 1) * width / 2, pcts,
                      width * 0.9, label=display_names[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(regime_names, fontsize=12)
    ax.set_ylabel("% of Bars", fontsize=12)
    ax.set_title("Regime Distribution by Method", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = output_dir / "regime_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Plot 4: ML Confidence Distribution ────────────────────────────

def plot_ml_confidence(all_results: dict, output_dir: Path):
    """
    Histogram of ML classifier confidence, split by predicted regime.
    """
    all_conf = []
    all_regimes = []

    for pair_name, results in all_results.items():
        if "_ml_confidence" in results and "ml_lgbm" in results:
            conf = results["_ml_confidence"]
            regimes = results["ml_lgbm"]
            all_conf.append(conf)
            all_regimes.append(regimes)

    if not all_conf:
        return

    conf_cat = pd.concat(all_conf, ignore_index=True)
    regime_cat = pd.concat(all_regimes, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: overall confidence histogram
    ax = axes[0]
    valid_mask = conf_cat > 0
    ax.hist(conf_cat[valid_mask], bins=50, color="#3498db", alpha=0.75, edgecolor="white")
    ax.axvline(conf_cat[valid_mask].median(), color="red", linestyle="--",
               label=f"Median: {conf_cat[valid_mask].median():.3f}")
    ax.set_xlabel("Confidence (max class probability)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("ML Classifier Confidence Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: confidence per regime
    ax = axes[1]
    regime_types = [RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS, RegimeType.VOLATILE]
    box_data = []
    labels = []
    for r in regime_types:
        mask = (regime_cat == r) & valid_mask
        if mask.sum() > 0:
            box_data.append(conf_cat[mask].values)
            labels.append(REGIME_LABELS[r])

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    colors = [REGIME_COLORS[r] for r in regime_types if REGIME_LABELS[r] in labels]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title("Confidence by Predicted Regime", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "ml_confidence_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Plot 5: Transition Matrix ────────────────────────────────────

def plot_transition_matrices(all_results: dict, output_dir: Path):
    """
    Compare regime transition matrices: rule-ensemble vs ML.
    """
    regime_types = [RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS, RegimeType.VOLATILE]
    n = len(regime_types)
    r2i = {r: i for i, r in enumerate(regime_types)}

    methods_to_compare = ["rule_ensemble", "ml_lgbm"]
    existing = [m for m in methods_to_compare
                if any(m in results for results in all_results.values())]

    if not existing:
        return

    fig, axes = plt.subplots(1, len(existing), figsize=(8 * len(existing), 7))
    if len(existing) == 1:
        axes = [axes]

    for ax, method in zip(axes, existing):
        trans = np.zeros((n, n))

        for pair_name, results in all_results.items():
            if method not in results:
                continue
            series = results[method]
            for i in range(len(series) - 1):
                from_r = series.iloc[i]
                to_r = series.iloc[i + 1]
                if from_r in r2i and to_r in r2i:
                    trans[r2i[from_r], r2i[to_r]] += 1

        # Normalize rows
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_pct = trans / row_sums * 100

        im = ax.imshow(trans_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        labels = [REGIME_LABELS[r] for r in regime_types]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("To")
        ax.set_ylabel("From")

        for i in range(n):
            for j in range(n):
                val = trans_pct[i, j]
                color = "white" if val > 60 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

        display = method.replace("rule_", "Rule: ").replace("ml_lgbm", "ML LightGBM")
        ax.set_title(f"Transition Matrix — {display}", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out = output_dir / "transition_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Plot 6: Summary Stats Table ──────────────────────────────────

def plot_summary_table(all_results: dict, output_dir: Path):
    """
    Visual table with key statistics: segment counts, mean duration,
    agreement with ensemble, etc.
    """
    regime_types = [RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS, RegimeType.VOLATILE]
    regime_names = [REGIME_LABELS[r] for r in regime_types]

    rows = []
    for pair_name, results in all_results.items():
        for method, series in results.items():
            if method.startswith("_"):
                continue
            total = len(series)
            counts = {r: (series == r).sum() for r in regime_types}

            # Count regime transitions (segments)
            transitions = 0
            for i in range(1, len(series)):
                if series.iloc[i] != series.iloc[i - 1]:
                    transitions += 1

            # Agreement with ensemble
            ensemble_key = "rule_ensemble"
            if ensemble_key in results and method != ensemble_key:
                ens = results[ensemble_key]
                s1 = regime_to_int_series(series)
                s2 = regime_to_int_series(ens)
                valid = (s1 >= 0) & (s2 >= 0)
                agreement = (s1[valid] == s2[valid]).mean() * 100 if valid.sum() > 0 else 0
            else:
                agreement = 100.0 if method == ensemble_key else 0.0

            display_method = method.replace("rule_", "").replace("ml_lgbm", "ML LightGBM").title()
            rows.append({
                "Pair": pair_name,
                "Method": display_method,
                "Bull%": f"{counts[RegimeType.BULLISH] / total * 100:.1f}",
                "Bear%": f"{counts[RegimeType.BEARISH] / total * 100:.1f}",
                "Side%": f"{counts[RegimeType.SIDEWAYS] / total * 100:.1f}",
                "Vol%": f"{counts[RegimeType.VOLATILE] / total * 100:.1f}",
                "Segments": transitions + 1,
                "Avg Seg Bars": f"{total / max(transitions + 1, 1):.0f}",
                "Ens Agree%": f"{agreement:.1f}",
            })

    df_table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(18, max(4, 0.5 * len(rows) + 2)))
    ax.axis("off")

    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(df_table.columns)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(df_table.columns)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")

    ax.set_title("Regime Detection — Summary Statistics",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    out = output_dir / "summary_statistics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Plot 7: Zoomed Comparison (interesting window) ───────────────

def plot_zoomed_comparison(df: pd.DataFrame, results: dict, pair: str, output_dir: Path):
    """
    Zoomed-in 3-month window showing detailed regime differences
    between rule-ensemble and ML.
    """
    if "rule_ensemble" not in results or "ml_lgbm" not in results:
        return

    # Pick a window where methods disagree most
    ens = regime_to_int_series(results["rule_ensemble"])
    ml = regime_to_int_series(results["ml_lgbm"])
    valid = (ens >= 0) & (ml >= 0)

    disagree = (ens != ml) & valid
    # Rolling disagreement rate over 30-bar windows
    disagree_rate = disagree.astype(float).rolling(30, min_periods=10).mean()
    if disagree_rate.max() == 0:
        # Methods fully agree — still plot but from the middle
        center = len(df) // 2
    else:
        center = disagree_rate.idxmax()
        if isinstance(center, pd.Timestamp):
            center = df.index.get_loc(center)

    # Window: ~500 bars around the most interesting point
    window = 250
    start = max(0, center - window)
    end = min(len(df), center + window)

    df_zoom = df.iloc[start:end]
    close = df_zoom["close"] if "close" in df_zoom.columns else df_zoom["Close"]
    dates = df_zoom.index

    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

    # Top: price
    axes[0].plot(dates, close, color="black", linewidth=1)
    axes[0].set_title(f"{pair} — Price", fontsize=12, fontweight="bold", loc="left")
    axes[0].set_ylabel("Price ($)")
    axes[0].grid(True, alpha=0.3)

    # Middle: Rule ensemble
    ens_zoom = results["rule_ensemble"].iloc[start:end]
    _fill_regime_background(axes[1], dates, ens_zoom, alpha=0.5)
    axes[1].plot(dates, close, color="black", linewidth=0.5, alpha=0.5)
    axes[1].set_title("Rule-Based Ensemble", fontsize=12, fontweight="bold", loc="left")
    axes[1].set_ylabel("Price ($)")

    # Bottom: ML
    ml_zoom = results["ml_lgbm"].iloc[start:end]
    _fill_regime_background(axes[2], dates, ml_zoom, alpha=0.5)
    axes[2].plot(dates, close, color="black", linewidth=0.5, alpha=0.5)
    axes[2].set_title("ML (LightGBM)", fontsize=12, fontweight="bold", loc="left")
    axes[2].set_ylabel("Price ($)")

    # Shared formatting
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Legend
    patches = [mpatches.Patch(color=REGIME_COLORS[r], label=REGIME_LABELS[r], alpha=0.6)
               for r in [RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS, RegimeType.VOLATILE]]
    fig.legend(handles=patches, loc="upper center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"Zoomed Comparison — {pair} (highest disagreement window)",
                 fontsize=15, fontweight="bold", y=1.06)
    plt.tight_layout()

    out = output_dir / f"zoomed_comparison_{pair.replace('/', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Plot 8: ML vs Ensemble Timeline Disagreement ─────────────────

def plot_disagreement_timeline(all_results: dict, output_dir: Path):
    """
    Rolling disagreement rate between ML and rule-ensemble over time.
    """
    fig, axes = plt.subplots(len(all_results), 1,
                             figsize=(18, 4 * len(all_results)), sharex=False)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (pair_name, results) in zip(axes, all_results.items()):
        if "rule_ensemble" not in results or "ml_lgbm" not in results:
            continue

        ens = regime_to_int_series(results["rule_ensemble"])
        ml = regime_to_int_series(results["ml_lgbm"])
        valid = (ens >= 0) & (ml >= 0)
        disagree = ((ens != ml) & valid).astype(float)

        # Rolling 50-bar disagreement rate
        roll = disagree.rolling(50, min_periods=10).mean() * 100

        ax.fill_between(roll.index, roll.values, alpha=0.4, color="#e74c3c")
        ax.plot(roll.index, roll.values, color="#c0392b", linewidth=1)
        ax.axhline(y=roll.mean(), color="blue", linestyle="--", alpha=0.6,
                   label=f"Mean: {roll.mean():.1f}%")
        ax.set_ylabel("Disagreement %")
        ax.set_title(f"{pair_name} — Rolling Disagreement (ML vs Ensemble, 50-bar window)",
                     fontsize=12, fontweight="bold", loc="left")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    out = output_dir / "disagreement_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Regime Detection Comparison: Rule-Based vs ML (LightGBM)")
    logger.info("=" * 60)

    all_results = {}
    all_dfs = {}

    for pair in PAIRS:
        logger.info(f"\nProcessing {pair}...")
        df = load_data(pair)
        if df.empty:
            logger.warning(f"  No data for {pair}, skipping")
            continue

        logger.info(f"  Loaded {len(df)} bars ({df.index[0]} → {df.index[-1]})")
        results = run_all_methods(df)
        all_results[pair] = results
        all_dfs[pair] = df

        logger.info(f"  Methods completed: {[k for k in results if not k.startswith('_')]}")

    if not all_results:
        logger.error("No data processed. Exiting.")
        return

    # Generate plots
    logger.info("\n" + "=" * 60)
    logger.info("Generating plots...")

    # Per-pair plots
    for pair in all_results:
        logger.info(f"\n  Plotting {pair}...")
        plot_price_with_regimes(all_dfs[pair], all_results[pair], pair, OUTPUT_DIR)
        plot_zoomed_comparison(all_dfs[pair], all_results[pair], pair, OUTPUT_DIR)

    # Cross-pair plots
    logger.info("\n  Generating cross-pair analysis...")
    plot_agreement_heatmap(all_results, OUTPUT_DIR)
    plot_regime_distribution(all_results, OUTPUT_DIR)
    plot_ml_confidence(all_results, OUTPUT_DIR)
    plot_transition_matrices(all_results, OUTPUT_DIR)
    plot_disagreement_timeline(all_results, OUTPUT_DIR)
    plot_summary_table(all_results, OUTPUT_DIR)

    logger.info("\n" + "=" * 60)
    logger.info(f"All plots saved to: {OUTPUT_DIR}/")
    logger.info("=" * 60)

    # Print quick summary
    for pair, results in all_results.items():
        if "rule_ensemble" in results and "ml_lgbm" in results:
            ens = regime_to_int_series(results["rule_ensemble"])
            ml = regime_to_int_series(results["ml_lgbm"])
            valid = (ens >= 0) & (ml >= 0)
            agree = (ens[valid] == ml[valid]).mean() * 100
            logger.info(f"  {pair}: ML-Ensemble agreement = {agree:.1f}%")


if __name__ == "__main__":
    main()
