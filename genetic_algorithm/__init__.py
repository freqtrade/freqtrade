"""
Genetic Algorithm Package for FreqTrade Strategy Evolution

This package provides tools for autonomously developing and optimizing
trading strategies using genetic algorithms.
"""

__version__ = "0.1.0"
__author__ = "FreqTrade GA Project"

from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.fitness import FitnessEvaluator

__all__ = [
    'GeneticAlgorithm',
    'StrategyGenerator', 
    'FitnessEvaluator',
]
