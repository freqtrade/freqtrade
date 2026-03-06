#!/usr/bin/env python3
"""
CLI training script for the ML regime classifier.

Usage:
    python -m genetic_algorithm.ml.train_regime \\
        --config genetic_algorithm/config/ga_config.yaml \\
        --pairs BTC/USDT ETH/USDT \\
        --timeframe 4h \\
        --timerange 20230101-20260101

    # Use price-based self-labeling instead of rule distillation:
    python -m genetic_algorithm.ml.train_regime \\
        --config genetic_algorithm/config/ga_config.yaml \\
        --pairs BTC/USDT \\
        --label-mode price

    # Override feature mode on CLI:
    python -m genetic_algorithm.ml.train_regime \\
        --pairs BTC/USDT \\
        --feature-mode raw_only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so genetic_algorithm is importable
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml

from genetic_algorithm.ml.regime_trainer import RegimeTrainer

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the LightGBM market-regime classifier for Phase 1B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="genetic_algorithm/config/ga_config.yaml",
        help="Path to the GA config YAML (default: genetic_algorithm/config/ga_config.yaml).",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Trading pairs for training data (e.g. BTC/USDT ETH/USDT).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="4h",
        help="Candle timeframe (default: 4h).",
    )
    parser.add_argument(
        "--timerange",
        type=str,
        default=None,
        help="Date range in YYYYMMDD-YYYYMMDD format (optional).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override FreqTrade data directory.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        choices=["rules", "price"],
        default=None,
        help="Override label mode from config (rules = distill existing ensemble, price = self-label).",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["combined", "raw_only", "rules_only"],
        default=None,
        help="Override feature mode from config.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override output model path (default from config).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Number of walk-forward CV folds (default from config).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    return parser.parse_args(argv)


def load_config(config_path: str) -> dict:
    """Load GA YAML config file; return empty dict if missing."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {path}  — using defaults.")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Merge CLI-supplied overrides into the config dict."""
    ml_cfg = config.setdefault("regime_aware", {}).setdefault("ml_regime", {})

    if args.label_mode is not None:
        ml_cfg["label_mode"] = args.label_mode
    if args.feature_mode is not None:
        ml_cfg["feature_mode"] = args.feature_mode
    if args.model_path is not None:
        ml_cfg["model_path"] = args.model_path
    if args.cv_folds is not None:
        ml_cfg["cv_folds"] = args.cv_folds

    return config


def main(argv=None) -> int:
    args = parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== Phase 1B — ML Regime Trainer ===")

    # Load & merge config
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    ml_cfg = config.get("regime_aware", {}).get("ml_regime", {})
    logger.info(f"Feature mode : {ml_cfg.get('feature_mode', 'combined')}")
    logger.info(f"Label mode   : {ml_cfg.get('label_mode', 'rules')}")
    logger.info(f"Model path   : {ml_cfg.get('model_path', 'genetic_algorithm/ml/models/regime_lgbm.pkl')}")

    # Train
    trainer = RegimeTrainer(config)
    try:
        report = trainer.train(
            pairs=args.pairs,
            timeframe=args.timeframe,
            timerange=args.timerange,
            data_path=args.data_path,
        )
    except Exception:
        logger.exception("Training failed")
        return 1

    # Summary
    logger.info("=== Training Complete ===")
    logger.info(f"Samples          : {report['training_samples']}")
    logger.info(f"Features         : {report['feature_count']}")
    logger.info(f"CV Accuracy      : {report['overall_accuracy']:.4f}")
    logger.info(f"CV F1 (macro)    : {report['overall_f1_macro']:.4f}")
    logger.info(f"Label distribution: {json.dumps(report['label_distribution'])}")
    logger.info(f"Model saved to   : {report['model_path']}")

    # Print top-10 features
    top = list(report["feature_importance"].items())[:10]
    logger.info("Top features:")
    for name, imp in top:
        logger.info(f"  {name:30s} {imp}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
