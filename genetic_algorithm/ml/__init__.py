"""
ML Module for Genetic Algorithm — Phase 1B

Market regime classification using supervised ML (LightGBM) as a meta-ensemble
on top of existing rule-based RegimeDetector methods.

Main components:
- MLRegimeDetector: LightGBM-based regime classifier (detect → Series[RegimeType])
- RegimeTrainer: Training pipeline with walk-forward validation + two labeling modes
"""

from genetic_algorithm.ml.regime_detector import MLRegimeDetector
from genetic_algorithm.ml.regime_trainer import RegimeTrainer

__all__ = ['MLRegimeDetector', 'RegimeTrainer']
