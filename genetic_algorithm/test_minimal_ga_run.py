#!/usr/bin/env python3
"""
Minimal GA Test

Runs the GA for 1 generation with 3 individuals to verify it works.
"""

import sys
import logging
from pathlib import Path
import tempfile
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.evolution import GeneticAlgorithm

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("=" * 70)
    print("Minimal GA Test - Verify Fix")
    print("=" * 70)
    print()
    
    # Load config
    config_path = Path("genetic_algorithm/config/ga_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override for minimal test
    config['genetic_algorithm']['population_size'] = 3
    config['genetic_algorithm']['num_generations'] = 1
    
    print(f"Config: {config['genetic_algorithm']['population_size']} individuals, "
          f"{config['genetic_algorithm']['num_generations']} generation")
    print()
    
    # Save modified config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_config_path = tmp.name
    
    try:
        # Create GA
        ga = GeneticAlgorithm(tmp_config_path, visualize=False, interactive=False)
        
        # Run evolution
        print("Running evolution...")
        best_individuals = ga.evolve()
        
        print()
        print("=" * 70)
        print("Results:")
        print("=" * 70)
        
        if best_individuals:
            for i, ind in enumerate(best_individuals[:3], 1):
                trades = ind.metrics.get('trades', 0)
                profit = ind.metrics.get('profit_pct', 0.0)
                print(f"{i}. {ind.id}: fitness={ind.fitness:.4f}, trades={trades}, profit={profit:.2f}%")
            
            # Check if any strategies made trades
            total_trades = sum(ind.metrics.get('trades', 0) for ind in best_individuals)
            print()
            if total_trades > 0:
                print(f"✓ SUCCESS: Strategies generated {total_trades} total trades")
                print("✓ The fix is working - strategies are producing entry/exit signals")
                return 0
            else:
                print("⚠ WARNING: No trades generated")
                print("  This might be due to:")
                print("  - Market conditions in the test data")
                print("  - Very restrictive strategy conditions")
                print("  - Check if strategies have valid entry/exit conditions")
                # Print a sample strategy to debug
                if best_individuals:
                    ind = best_individuals[0]
                    print()
                    print(f"Sample strategy ({ind.id}):")
                    print(f"  Indicators: {[f'{i.type}({i.instance_id})' for i in ind.strategy_gene.indicators]}")
                    print(f"  Entry conditions: {[(c.indicator, c.operator, c.threshold) for c in ind.strategy_gene.entry_conditions]}")
                    print(f"  Exit conditions: {[(c.indicator, c.operator, c.threshold) for c in ind.strategy_gene.exit_conditions]}")
                return 1
        else:
            print("✗ FAILED: No individuals returned")
            return 1
    finally:
        # Clean up temp file
        import os
        if os.path.exists(tmp_config_path):
            os.unlink(tmp_config_path)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        sys.exit(1)
