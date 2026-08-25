import pytest

from freqtrade.mixins import LoggingMixin


@pytest.fixture(autouse=True)
def reset_logging_mixin_show_output():
    """
    Mirrors tests/conftest.py's fixture of the same name -- `tests/conftest.py`'s autouse
    fixtures only apply to tests under `tests/`, not to this sibling `research/tests/`
    directory, so the same reset is needed here too. research/walkforward.py and
    research/cost_stress.py construct freqtrade.optimize.backtesting.Backtesting instances
    directly (never calling the separate Backtesting.cleanup() staticmethod that's the only
    thing that ever restores LoggingMixin.show_output = True), so every real backtest this
    package's own tests run leaves that class attribute (not per-test state) disabled for
    every later test sharing this pytest-xdist worker process -- including tests outside
    this package entirely. See tests/conftest.py's reset_logging_mixin_show_output for the
    full explanation and the two CI failures (in tests/exchange/ and tests/plugins/) this
    was confirmed to be the real root cause of.
    """
    yield
    LoggingMixin.show_output = True
