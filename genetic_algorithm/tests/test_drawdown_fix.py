#!/usr/bin/env python3
"""
Quick test to verify max_drawdown and other FreqTrade metric extraction.
"""
import sys
import yaml
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format='%(message)s')

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genetic_algorithm.core.evolution import GeneticAlgorithm


def main():
    # Load base config
    config_path = Path(__file__).parent.parent / "config" / "ga_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override for quick test
    config['genetic_algorithm']['population_size'] = 5
    config['genetic_algorithm']['generations'] = 2
    config['genetic_algorithm']['elite_size'] = 1
    config['genetic_algorithm']['random_immigrants'] = 1
    config['backtesting']['enable_cache'] = False

    # Write temp config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp_path = f.name

    print("=" * 60)
    print("QUICK GA TEST - Verifying FreqTrade Metric Extraction")
    print("=" * 60)

    ga = GeneticAlgorithm(config_path=tmp_path)
    best = ga.evolve()

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for i, ind in enumerate(best[:3]):
        m = ind.metrics
        print(f"\nRank {i+1}: {ind.id}")
        print(f"  Fitness:       {ind.fitness:.4f}")
        print(f"  Profit:        {m.get('profit', 0):.2f}%")
        print(f"  Sharpe:        {m.get('sharpe_ratio', 0):.2f}")
        dd = m.get('max_drawdown', 'NOT SET')
        if isinstance(dd, (int, float)):
            print(f"  Max Drawdown:  {dd:.4f} ({dd:.2%})")
        else:
            print(f"  Max Drawdown:  {dd}")
        print(f"  Win Rate:      {m.get('win_rate', 0):.2%}")
        print(f"  Trades:        {m.get('num_trades', 0)}")
        print(f"  Profit Factor: {m.get('profit_factor', 0):.2f}")
        print(f"  Sortino:       {m.get('sortino_ratio', 0):.2f}")

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    has_trades_with_losses = any(
        ind.metrics.get('num_trades', 0) > 0 and ind.metrics.get('win_rate', 1) < 1.0
        for ind in best
    )
    all_zero_dd = all(
        ind.metrics.get('max_drawdown', 0) == 0
        for ind in best
        if ind.metrics.get('num_trades', 0) > 0
    )

    if has_trades_with_losses and all_zero_dd:
        print("FAIL: All drawdowns are 0 despite having losing trades!")
        return 1
    elif has_trades_with_losses:
        non_zero = sum(1 for ind in best if ind.metrics.get('max_drawdown', 0) > 0)
        print(f"PASS: {non_zero}/{len(best)} individuals have non-zero drawdown")
        return 0
    else:
        print("WARN: No individuals had losing trades - cannot verify drawdown")
        return 0

    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    sys.exit(main())
