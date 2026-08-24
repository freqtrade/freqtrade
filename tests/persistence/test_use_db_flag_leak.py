"""
Regression test for a leaked-global-state bug: several tests toggle
``Trade.use_db = False`` (and friends) with a bare assignment instead of the
exception-safe ``FtNoDBContext``. If anything between the ``False`` and the
matching ``= True`` raises, the flag is never restored -- and because
``Trade.use_db`` is a class attribute, not a per-test fixture, it stays
``False`` for every later test that shares the same pytest-xdist worker
process. That single leaked flag was the root cause of dozens of unrelated
CI failures (``Trade.get_trades() not supported in backtesting mode``, empty
query results, stale trades from other tests) once CI started running with
the project's real ``--random-order -n auto`` recipe.

This uses pytest's own ``pytester`` fixture to spawn an isolated sub-run
that imports the real ``reset_use_db_flags`` autouse fixture from
``tests/conftest.py``, because the property under test -- "state leaked by
a failing test does not survive into the next test in the same process" --
can only be observed across a test boundary, not within a single test body.
"""


def test_reset_use_db_flags_fixture_prevents_the_leak(pytester):
    pytester.makepyfile(
        test_leak="""
            # Importing this registers the autouse fixture in this module.
            from tests.conftest import reset_use_db_flags  # noqa: F401

            from freqtrade.persistence import Trade

            def test_a_disables_use_db_then_fails():
                Trade.use_db = False
                assert False, "simulates a test that fails before resetting the flag"

            def test_b_expects_use_db_enabled():
                assert Trade.use_db is True
        """
    )

    result = pytester.runpytest()

    # test_a still fails as designed; test_b must see a properly reset flag,
    # not the False left behind by test_a.
    result.assert_outcomes(failed=1, passed=1)


def test_reset_use_db_flags_fixture_also_clears_backtest_trade_state(pytester):
    """
    Sibling leak: LocalTrade.bt_trades / bt_trades_open / bt_trades_open_pp /
    bt_open_open_trade_count / bt_total_profit are the same kind of class-level
    mutable state as Trade.use_db, only cleared by an explicit (and equally
    unguarded) Trade.reset_trades() call in a handful of tests.
    """
    pytester.makepyfile(
        test_leak="""
            # Importing this registers the autouse fixture in this module.
            from tests.conftest import reset_use_db_flags  # noqa: F401

            from freqtrade.persistence import LocalTrade

            def test_a_adds_a_backtest_trade_then_fails():
                LocalTrade.bt_trades.append(object())
                assert False, "simulates a test that fails before resetting bt state"

            def test_b_expects_clean_backtest_trade_state():
                assert LocalTrade.bt_trades == []
        """
    )

    result = pytester.runpytest()

    result.assert_outcomes(failed=1, passed=1)
