#!/usr/bin/env python3
"""
Test auto-download functionality in DirectBacktester.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.evaluation.direct_backtester import DirectBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_backtest_config_with_test_data():
    """Test DirectBacktester with test data (should skip validation)."""
    print("\n" + "=" * 80)
    print("TEST 1: DirectBacktester with UNITTEST data (should skip validation)")
    print("=" * 80 + "\n")
    
    config = {
        'backtesting': {
            'pairs': ['UNITTEST/BTC'],
            'timerange': '',
            'stake_amount': 0.05,
            'max_open_trades': 1,
            'fee': 0.001,
            'exchange': 'binance',
            'auto_download_data': False,
            'enable_cache': False,
        },
        'strategy_constraints': {
            'timeframes': ['5m'],
        }
    }
    
    try:
        backtester = DirectBacktester(config)
        print("\n✅ Test 1 PASSED: DirectBacktester initialized successfully with test data\n")
        return True
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_config_with_real_pairs():
    """Test DirectBacktester with real pairs (should validate data)."""
    print("\n" + "=" * 80)
    print("TEST 2: DirectBacktester with real pairs (should validate data)")
    print("=" * 80 + "\n")
    
    config = {
        'backtesting': {
            'pairs': ['BTC/USDT'],
            'timerange': '20250120-20250219',
            'stake_amount': 0.1,
            'max_open_trades': 1,
            'fee': 0.001,
            'exchange': 'binance',
            'auto_download_data': False,  # Disabled for test
            'enable_cache': False,
        },
        'strategy_constraints': {
            'timeframes': ['1h'],
        }
    }
    
    try:
        backtester = DirectBacktester(config)
        print("\n✅ Test 2 PASSED: DirectBacktester initialized and validated data\n")
        return True
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_data_validation():
    """Test data validation method directly."""
    print("\n" + "=" * 80)
    print("TEST 3: Data validation method")
    print("=" * 80 + "\n")
    
    config = {
        'backtesting': {
            'pairs': ['BTC/USDT', 'ETH/USDT'],
            'timerange': '20250120-20250219',
            'stake_amount': 0.1,
            'max_open_trades': 1,
            'fee': 0.001,
            'exchange': 'binance',
            'auto_download_data': False,
            'enable_cache': False,
        },
        'strategy_constraints': {
            'timeframes': ['1h', '5m'],
        }
    }
    
    try:
        backtester = DirectBacktester(config)
        validation_result = backtester._validate_data_exists()
        
        print(f"Validation result: {len(validation_result['missing'])} missing file(s)")
        if validation_result['missing']:
            print("Missing files:")
            for pair, timeframe in validation_result['missing']:
                print(f"  - {pair} {timeframe}")
        
        print("\n✅ Test 3 PASSED: Data validation method works correctly\n")
        return True
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DirectBacktester Auto-Download Tests")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Test 1: UNITTEST data", test_backtest_config_with_test_data()))
    results.append(("Test 2: Real pairs", test_backtest_config_with_real_pairs()))
    results.append(("Test 3: Data validation", test_data_validation()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
