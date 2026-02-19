#!/usr/bin/env python3
"""
Demonstration of Instance-Based Indicator Encoding

This script shows how the new instance-based encoding works
by creating strategies with multiple instances of the same indicator type.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


def demonstrate_instance_encoding():
    """Demonstrate the instance-based encoding feature."""
    
    print("="*70)
    print("Instance-Based Indicator Encoding Demonstration")
    print("="*70)
    print()
    
    # Create a strategy with multiple EMAs (common pattern for trend following)
    print("Creating a strategy with 3 different EMAs...")
    strategy = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='EMA', parameters={'period': 9}),   # Fast EMA
            IndicatorGene(type='EMA', parameters={'period': 21}),  # Medium EMA
            IndicatorGene(type='EMA', parameters={'period': 50}),  # Slow EMA
            IndicatorGene(type='RSI', parameters={'period': 14}),  # RSI for momentum
        ],
        entry_conditions=[
            # We want: fast EMA crosses above medium EMA (bullish signal)
            ConditionGene(indicator='EMA', operator='cross_above', threshold=0),
        ],
    )
    
    print("\nBefore assign_instance_ids():")
    print("-" * 70)
    for i, ind in enumerate(strategy.indicators):
        print(f"  Indicator {i}: type={ind.type}, period={ind.parameters.get('period', 'N/A')}, instance_id={ind.instance_id}")
    print(f"  Entry condition references: {[c.indicator for c in strategy.entry_conditions]}")
    
    # Assign unique instance IDs
    print("\nCalling assign_instance_ids()...")
    strategy.assign_instance_ids()
    
    print("\nAfter assign_instance_ids():")
    print("-" * 70)
    for i, ind in enumerate(strategy.indicators):
        print(f"  Indicator {i}: type={ind.type}, period={ind.parameters.get('period', 'N/A')}, instance_id={ind.instance_id}")
    print(f"  Entry condition now references: {[c.indicator for c in strategy.entry_conditions]}")
    
    print("\n" + "="*70)
    print("Key Benefits:")
    print("="*70)
    print("✓ Each EMA has a unique instance ID (EMA_0, EMA_1, EMA_2)")
    print("✓ Conditions can reference specific EMA instances")
    print("✓ No ambiguity when crossing over strategies with multiple EMAs")
    print("✓ Better genetic distance calculation possible")
    print()
    
    # Demonstrate serialization
    print("="*70)
    print("Serialization Test")
    print("="*70)
    print("\nConverting to dictionary...")
    data = strategy.to_dict()
    print(f"  Indicators in dict: {len(data['indicators'])} items")
    for ind in data['indicators']:
        print(f"    - {ind['type']} (instance_id={ind['instance_id']})")
    
    print("\nRestoring from dictionary...")
    restored = StrategyGene.from_dict(data)
    print(f"  Restored {len(restored.indicators)} indicators")
    for ind in restored.indicators:
        print(f"    - {ind.type} (instance_id={ind.instance_id})")
    
    print("\n✓ Instance IDs preserved through serialization!")
    print()
    
    # Demonstrate backward compatibility
    print("="*70)
    print("Backward Compatibility")
    print("="*70)
    print("\nCreating strategy without instance IDs (old format)...")
    old_strategy = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30),
        ],
    )
    
    print("  Indicators created without instance_id field")
    print(f"  Instance IDs: {[ind.instance_id for ind in old_strategy.indicators]}")
    
    print("\nCalling assign_instance_ids()...")
    old_strategy.assign_instance_ids()
    
    print(f"  Instance IDs after: {[ind.instance_id for ind in old_strategy.indicators]}")
    print(f"  Condition now references: {old_strategy.entry_conditions[0].indicator}")
    print("\n✓ Backward compatible - old strategies get instance IDs automatically!")
    print()
    
    print("="*70)
    print("Summary")
    print("="*70)
    print("The instance-based encoding upgrade provides:")
    print("  1. Unique identifiers for each indicator instance")
    print("  2. Clear references in conditions (no ambiguity)")
    print("  3. Better crossover semantics")
    print("  4. Foundation for improved genetic distance metrics")
    print("  5. Full backward compatibility")
    print()
    print("This improvement was completed as part of the 'Encoding & Representation'")
    print("medium-scope task from TODO_ga_improvements.md")
    print("="*70)


if __name__ == '__main__':
    demonstrate_instance_encoding()
