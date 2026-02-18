#!/usr/bin/env python3
"""
Test _get_mock_markets() to verify it supports both test and real pairs.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from file to avoid package initialization
import importlib.util
spec = importlib.util.spec_from_file_location(
    "direct_backtester",
    Path(__file__).parent / "evaluation" / "direct_backtester.py"
)
direct_backtester = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direct_backtester)
DirectBacktester = direct_backtester.DirectBacktester

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_mock_markets_with_test_pairs():
    """Test that _get_mock_markets() works with test pairs."""
    print("\n" + "=" * 80)
    print("TEST 1: Mock markets with test pairs (UNITTEST/BTC)")
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
        mock_markets = backtester._get_mock_markets()
        
        # Verify UNITTEST/BTC is in mock markets
        assert 'UNITTEST/BTC' in mock_markets, "UNITTEST/BTC should be in mock markets"
        
        # Verify structure
        market = mock_markets['UNITTEST/BTC']
        assert market['symbol'] == 'UNITTEST/BTC'
        assert market['base'] == 'UNITTEST'
        assert market['quote'] == 'BTC'
        assert market['active'] is True
        assert market['spot'] is True
        
        print(f"✅ Test 1 PASSED: Mock markets created for {len(mock_markets)} pairs")
        print(f"   Includes: {list(mock_markets.keys())[:5]}...")
        print()
        return True
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_markets_with_real_pairs():
    """Test that _get_mock_markets() works with real pairs."""
    print("\n" + "=" * 80)
    print("TEST 2: Mock markets with real pairs (BTC/USDT, ETH/USDT)")
    print("=" * 80 + "\n")
    
    config = {
        'backtesting': {
            'pairs': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'timerange': '20250120-20250219',
            'stake_amount': 0.1,
            'max_open_trades': 1,
            'fee': 0.001,
            'exchange': 'binance',
            'auto_download_data': False,
            'enable_cache': False,
        },
        'strategy_constraints': {
            'timeframes': ['1h'],
        }
    }
    
    try:
        backtester = DirectBacktester(config)
        mock_markets = backtester._get_mock_markets()
        
        # Verify all configured pairs are in mock markets
        assert 'BTC/USDT' in mock_markets, "BTC/USDT should be in mock markets"
        assert 'ETH/USDT' in mock_markets, "ETH/USDT should be in mock markets"
        assert 'BNB/USDT' in mock_markets, "BNB/USDT should be in mock markets"
        
        # Verify structure for BTC/USDT
        market = mock_markets['BTC/USDT']
        assert market['symbol'] == 'BTC/USDT'
        assert market['base'] == 'BTC'
        assert market['quote'] == 'USDT'
        assert market['active'] is True
        assert market['spot'] is True
        
        # Verify test pairs are still included for backward compatibility
        assert 'UNITTEST/BTC' in mock_markets, "Test pairs should still be included"
        
        print(f"✅ Test 2 PASSED: Mock markets created for {len(mock_markets)} pairs")
        print(f"   Real pairs: BTC/USDT, ETH/USDT, BNB/USDT ✓")
        print(f"   Test pairs: UNITTEST/BTC, ETH/BTC, etc. ✓")
        print()
        return True
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_markets_with_mixed_pairs():
    """Test that _get_mock_markets() works with mixed test and real pairs."""
    print("\n" + "=" * 80)
    print("TEST 3: Mock markets with mixed pairs")
    print("=" * 80 + "\n")
    
    config = {
        'backtesting': {
            'pairs': ['BTC/USDT', 'UNITTEST/BTC', 'SOL/USDT'],
            'timerange': '',
            'stake_amount': 0.1,
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
        mock_markets = backtester._get_mock_markets()
        
        # Verify all configured pairs are in mock markets
        assert 'BTC/USDT' in mock_markets
        assert 'UNITTEST/BTC' in mock_markets
        assert 'SOL/USDT' in mock_markets
        
        print(f"✅ Test 3 PASSED: Mock markets handles mixed pairs correctly")
        print(f"   Created markets for {len(mock_markets)} pairs")
        print()
        return True
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_markets_with_invalid_pair():
    """Test that _get_mock_markets() handles invalid pairs gracefully."""
    print("\n" + "=" * 80)
    print("TEST 4: Mock markets with invalid pair format")
    print("=" * 80 + "\n")
    
    config = {
        'backtesting': {
            'pairs': ['BTC/USDT', 'INVALID_PAIR', 'ETH/USDT'],
            'timerange': '',
            'stake_amount': 0.1,
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
        mock_markets = backtester._get_mock_markets()
        
        # Verify valid pairs are included
        assert 'BTC/USDT' in mock_markets
        assert 'ETH/USDT' in mock_markets
        
        # Verify invalid pair is skipped (not included)
        assert 'INVALID_PAIR' not in mock_markets
        
        print(f"✅ Test 4 PASSED: Invalid pairs are handled gracefully")
        print(f"   Valid pairs included, invalid pairs skipped")
        print()
        return True
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Testing _get_mock_markets() Fix")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Test 1: Test pairs", test_mock_markets_with_test_pairs()))
    results.append(("Test 2: Real pairs", test_mock_markets_with_real_pairs()))
    results.append(("Test 3: Mixed pairs", test_mock_markets_with_mixed_pairs()))
    results.append(("Test 4: Invalid pairs", test_mock_markets_with_invalid_pair()))
    
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
        print("\nThe _get_mock_markets() fix is working correctly!")
        print("- Test pairs (UNITTEST/BTC) are supported ✓")
        print("- Real pairs (BTC/USDT, ETH/USDT, etc.) are supported ✓")
        print("- Mixed pairs are supported ✓")
        print("- Invalid pairs are handled gracefully ✓")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
