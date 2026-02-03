import pytest

from freqtrade.persistence import FtNoDBContext, PairLocks, Trade


@pytest.mark.parametrize("timeframe", ["", "5m", "1d"])
def test_FtNoDBContext(timeframe):
    PairLocks.timeframe = ""
    assert Trade.use_db
    assert PairLocks.use_db
    assert PairLocks.timeframe == ""

    with FtNoDBContext(timeframe):
        assert not Trade.use_db
        assert not PairLocks.use_db
        assert PairLocks.timeframe == timeframe

    with FtNoDBContext():
        assert not Trade.use_db
        assert not PairLocks.use_db
        assert PairLocks.timeframe == ""

    assert Trade.use_db
    assert PairLocks.use_db
