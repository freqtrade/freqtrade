"""
Regression test for a leaked-global-state bug: ``freqtrade.optimize.backtesting.Backtesting``
unconditionally sets ``LoggingMixin.show_output = False`` in both ``__init__`` and
``reset_backtest()`` (``freqtrade/optimize/backtesting.py:144`` and ``:495``), and is only
ever restored by the separate ``Backtesting.cleanup()`` staticmethod
(``freqtrade/optimize/backtesting.py:284-286``) -- which nothing in ``research/`` (or any
test that constructs a ``Backtesting`` instance directly, bypassing the CLI's
``start_backtesting`` command) ever calls. Because ``LoggingMixin.show_output`` is a class
attribute, not per-test state, any test in the same pytest-xdist worker process that runs
*after* one that touches ``Backtesting`` silently has every ``LoggingMixin.log_once()`` call
become a no-op for the rest of that worker's life -- the same class of bug PR #5 already
fixed once for ``Trade.use_db``/``LocalTrade.bt_trades`` (see ``reset_use_db_flags`` above).

Confirmed as the real root cause of two CI failures that looked unrelated to each other on
the surface (``tests/exchange/test_exchange.py::test__async_kucoin_get_candle_history`` and
``tests/plugins/test_pairlist.py::test_log_cached``/``test_remove_logs_for_pairs_already_in_blacklist``)
-- both assert a ``log_once``-driven message was captured/called and got 0 instead.

Uses pytest's own ``pytester`` fixture, same technique as ``test_use_db_flag_leak.py``,
since the property under test -- state leaked by one test surviving into the next test in
the same process -- can only be observed across a test boundary.
"""


def test_reset_show_output_fixture_prevents_the_leak(pytester):
    pytester.makepyfile(
        test_leak="""
            # Importing this registers the autouse fixture in this module.
            from tests.conftest import reset_logging_mixin_show_output  # noqa: F401

            from freqtrade.mixins import LoggingMixin

            def test_a_disables_show_output_then_fails():
                LoggingMixin.show_output = False
                assert False, "simulates Backtesting.__init__ without a matching cleanup()"

            def test_b_expects_show_output_enabled():
                assert LoggingMixin.show_output is True
        """
    )

    result = pytester.runpytest()

    # test_a still fails as designed; test_b must see a properly reset flag, not the
    # False left behind by test_a.
    result.assert_outcomes(failed=1, passed=1)
